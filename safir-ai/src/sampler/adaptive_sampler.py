"""02 - VLM Oncesi Katman: OpenCV tabanli Adaptive Frame Sampler (CPU ONLY).

Bu modul, ham video akisini agir bir tespit modeli (YOLO/ByteTrack) ve GPU
kullanmadan tarar. Ardisik gri-tonlama kareler arasindaki mutlak fark ile
degisimi olcer, dinamik (medyan tabanli) bir gurultu tabani (noise floor)
ile gereksiz titremeleri eler ve yalnizca anlamli degisim iceren kareleri
"Kanit Karesi" (Evidence Frame) olarak isaretler. Ardindan zaman acisindan
yakin kanit kareleri "Olay Grubu"na (Event Cluster) kumeler ve her grubun en
yuksek degisim skoruna sahip "zirve karesini" (peak frame) secer; boylece VLM
katmanina yalnizca gercekten anlamli ve tekrarsiz kareler gonderilir.
"""

from __future__ import annotations

import base64
import gc
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Deque, List, Optional, Tuple

import cv2
import numpy as np

from src.sampler.context.peak_frame_exporter import PeakFrameExporter
from src.sampler.schema import EventCluster, EvidenceFrame, RepresentativeFrame

if TYPE_CHECKING:
    # Yalnizca tip ipucu icin gereklidir (bkz. `sampler_from_config`); modul
    # ustunde zaten `from __future__ import annotations` oldugundan tip
    # ipuclari calisma zamaninda hic degerlendirilmez. Bu importu calisma
    # zamanindan (runtime) TYPE_CHECKING'e tasimak, yalnizca
    # `AdaptiveFrameSampler`/`EvidenceFrame`/`EventCluster`'i kullanan (ve
    # `sampler_from_config`'e hic ihtiyaci olmayan) cagiranlar icin gereksiz
    # `pydantic`/`PyYAML` (config_loader) import maliyetini ortadan kaldirir;
    # davranista HICBIR degisiklik yaratmaz.
    from src.utils.config_loader import SamplerConfig

logger = logging.getLogger(__name__)

_DEFAULT_EVIDENCE_OUTPUT_DIR = "outputs/evidence_frames"


@dataclass
class SamplerRunStats:
    """Tek bir `process_video` cagrisi icin CPU suzgec/GPU tasarruf istatistikleri."""

    total_frames_scanned: int
    sampled_frames_evaluated: int
    evidence_frame_count: int
    eliminated_frame_count: int
    eliminated_ratio_pct: float
    elapsed_sec: float


@dataclass
class ClusterMergeStats:
    """Tek bir `cluster_events` cagrisi icin ham grup -> nihai Olay Grubu birlestirme istatistikleri."""

    raw_group_count: int
    final_cluster_count: int
    merged_raw_group_count: int


class AdaptiveFrameSampler:
    """OpenCV tabanli, CPU uzerinde ultra hizli calisan Uyarlanabilir Kare Ornekleyici.

    Gereksiz/durgun kareleri buyuk oranda eleyerek yalnizca anlamli degisim
    iceren kareleri VLM katmanina iletir. Katman tamamen CPU uzerinde calisir;
    GPU yuku sifirdir.
    """

    def __init__(
        self,
        min_change_threshold: float = 0.001,
        blur_kernel_size: Tuple[int, int] = (21, 21),
        history_window: int = 30,
        min_event_interval_sec: float = 2.0,
        evidence_output_dir: str = _DEFAULT_EVIDENCE_OUTPUT_DIR,
        max_evidence_buffer: Optional[int] = None,
        cluster_merge_gap_sec: float = 20.0,
        bbox_iou_merge_threshold: float = 0.10,
        temporal_vote_window: int = 1,
        temporal_vote_min_count: int = 1,
        pre_peak_offset_sec: float = 2.0,
        post_peak_offset_sec: float = 2.0,
        max_sampling_gap_sec: Optional[float] = 4.0,
        ref_reset_interval_sec: float = 10.0,
        ref_change_threshold: float = 0.0015,
        enable_contrast_check: bool = True,
        contrast_change_threshold: float = 0.05,
        contrast_check_interval_sec: float = 1.0,
    ) -> None:
        """AdaptiveFrameSampler'i esik ve pencere parametreleriyle baslatir."""
        if temporal_vote_window < 1:
            raise ValueError(
                f"temporal_vote_window en az 1 olmalidir, verilen: {temporal_vote_window}"
            )
        if temporal_vote_min_count < 1:
            raise ValueError(
                f"temporal_vote_min_count en az 1 olmalidir, verilen: {temporal_vote_min_count}"
            )
        if temporal_vote_min_count > temporal_vote_window:
            raise ValueError(
                f"temporal_vote_min_count ({temporal_vote_min_count}) "
                f"temporal_vote_window'dan ({temporal_vote_window}) buyuk olamaz."
            )

        self.min_change_threshold = min_change_threshold
        self.blur_kernel_size = blur_kernel_size
        self.history_window = history_window
        self.min_event_interval_sec = min_event_interval_sec
        self.evidence_output_dir = Path(evidence_output_dir)
        self.max_evidence_buffer = max_evidence_buffer
        self.cluster_merge_gap_sec = cluster_merge_gap_sec
        self.bbox_iou_merge_threshold = bbox_iou_merge_threshold
        self.temporal_vote_window = temporal_vote_window
        self.temporal_vote_min_count = temporal_vote_min_count
        self.pre_peak_offset_sec = pre_peak_offset_sec
        self.post_peak_offset_sec = post_peak_offset_sec

        # Kademeli / Yavas-baslangicli olay tespiti parametreleri (Duman / Sızıntı / Drift)
        self.max_sampling_gap_sec = max_sampling_gap_sec
        self.ref_reset_interval_sec = ref_reset_interval_sec
        self.ref_change_threshold = ref_change_threshold
        self.enable_contrast_check = enable_contrast_check
        self.contrast_change_threshold = contrast_change_threshold
        self.contrast_check_interval_sec = contrast_check_interval_sec

        self.prev_gray: np.ndarray | None = None
        self.ref_gray: np.ndarray | None = None
        self.ref_timestamp: float = 0.0
        self.ref_mean: float = 0.0
        self.ref_std: float = 0.0
        self.last_sampled_timestamp: float = -1.0
        self.last_contrast_check_timestamp: float = -1.0

        self.contrast_history: List[float] = []
        self.brightness_history: List[float] = []
        self.noise_floor_history: List[float] = []
        self.last_run_stats: Optional[SamplerRunStats] = None
        self.last_cluster_merge_stats: Optional[ClusterMergeStats] = None
        self._recent_threshold_decisions: Deque[bool] = deque(maxlen=temporal_vote_window)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Goruntuyu gri tonlamaya cevirip Gauss bulaniklastirma ile gurultuyu yumusatir.

        Args:
            frame: BGR formatinda ham video karesi.

        Returns:
            Gri tonlamali, bulaniklastirilmis kare.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        return blurred

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Ikili bir hareket maskesinden sinirlayici kutuyu (bounding box) cikarir.

        Args:
            mask: `cv2.threshold` ile uretilmis ikili (0/255) hareket maskesi.

        Returns:
            `(x_min, y_min, x_max, y_max)` — maskede hic hareketli piksel
            yoksa `None`.
        """
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    @staticmethod
    def _bbox_iou(box_a: Optional[Tuple[int, int, int, int]], box_b: Optional[Tuple[int, int, int, int]]) -> float:
        """Iki sinirlayici kutu arasindaki kesisim/birlesim (IoU) oranini hesaplar.

        Args:
            box_a: `(x_min, y_min, x_max, y_max)` ya da `None`.
            box_b: `(x_min, y_min, x_max, y_max)` ya da `None`.

        Returns:
            0.0-1.0 arasi IoU degeri; iki kutudan biri `None` ise `0.0`
            (temkinli: bilgi eksikse ayni olay sayilmaz).
        """
        if box_a is None or box_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = inter_w * inter_h
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _is_same_physical_event(self, group_a_last: EvidenceFrame, group_b_first: EvidenceFrame) -> bool:
        """Iki ham grubun sinir kareleri arasinda konumsal (mekansal) sureklilik olup olmadigini belirler.

        Bir grubun SON karesinin `motion_bbox`'u ile bir sonraki grubun ILK
        karesinin `motion_bbox`'u arasindaki IoU, `bbox_iou_merge_threshold`i
        gecerse ayni fiziksel olayin devami sayilir. Olay/nesne turunden
        bagimsizdir; yalnizca hareket bolgesinin piksel-uzayindaki konumuna
        bakar (bkz. `__init__` docstring'i: `bbox_iou_merge_threshold`).

        Args:
            group_a_last: Onceki ham grubun son `EvidenceFrame`'i.
            group_b_first: Sonraki ham grubun ilk `EvidenceFrame`'i.

        Returns:
            `True` ise iki grup konumsal olarak ayni fiziksel olayin devami
            sayilir (birlestirmeye aday); `False` ise farkli olay.
        """
        iou = self._bbox_iou(group_a_last.motion_bbox, group_b_first.motion_bbox)
        return iou >= self.bbox_iou_merge_threshold

    def _confirm_candidate(self, is_suspicious: bool) -> bool:
        """Temporal voting: bir karenin GERCEK candidate olarak onaylanip
        onaylanmayacagina, son `temporal_vote_window` esik-testi sonucuna
        bakarak karar verir.

        Mevcut `min_change_threshold` formulunu DEGISTIRMEZ; yalnizca onun
        ciktisini (`is_suspicious`) girdi olarak kullanan ek bir sureklilik
        filtresidir. `is_suspicious` her cagrida (kare esigi gecsin ya da
        gecmesin) pencereye eklenir, boylece pencere daima son N kararin
        GERCEK gecmisini yansitir. Varsayilan `temporal_vote_window=1`,
        `temporal_vote_min_count=1` ile bu metod `is_suspicious` ile
        birebir ayni sonucu doner (voting-oncesi davranisla tam uyumlu).

        Args:
            is_suspicious: Bu karenin `net_change_score >= min_change_threshold`
                testini gecip gecmedigi (mevcut, degismemis esik sonucu).

        Returns:
            `True` ise kare GERCEK candidate olarak onaylanir (hem kendisi
            supheli HEM DE pencerede yeterli sureklilik var); `False` ise
            (esigi gecmedi ya da izole/yetersiz sureklilik) reddedilir.
        """
        self._recent_threshold_decisions.append(is_suspicious)
        if not is_suspicious:
            return False
        vote_count = sum(self._recent_threshold_decisions)
        return vote_count >= self.temporal_vote_min_count

    def _encode_frame_jpeg(self, frame: np.ndarray) -> bytes:
        """Kareyi tek seferde JPEG'e kodlar (disk ve base64 tarafindan ortak kullanilir).

        Args:
            frame: BGR formatinda video karesi.

        Returns:
            JPEG-kodlu ham kare baytlari.

        Raises:
            RuntimeError: Kare JPEG formatina kodlanamazsa.
        """
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("Kare JPEG formatina kodlanamadi.")
        return buffer.tobytes()

    def _persist_frame(self, frame_id: int, time_str: str, image_bytes: bytes) -> str:
        """Bir Kanit Karesini `evidence_output_dir` altina `.jpg` olarak yazar.

        Args:
            frame_id: Karenin video icindeki sirasi (dosya adi icin kullanilir).
            time_str: `MM:SS` formatinda zaman damgasi (dosya adinda `-` ile).
            image_bytes: JPEG-kodlu ham kare baytlari.

        Returns:
            Yazilan dosyanin yolu (string).
        """
        self.evidence_output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"evt_{frame_id:06d}_{time_str.replace(':', '-')}.jpg"
        path = self.evidence_output_dir / filename
        path.write_bytes(image_bytes)
        return str(path)

    def process_video(self, video_path: str, sample_fps: int = 5) -> List[EvidenceFrame]:
        """Videoyu okur ve her tam saniye [T.00 - T.99] icin istisnasiz tam 1 adet en net kare uretir (Uniform PTS).

        Hareket/delta veya optik akis filtreleri tamamen kaldirilmistir; boylece
        hareketli anlarda gereksiz kare yigilmasi (burst), oda bosaldiginda ise
        kare atlamasi yasanmaz.

        Args:
            video_path: `.mp4` dosya yolu veya RTSP URI'si.
            sample_fps: Geriye donuk arayuz uyumlulugu icin tutulur (Uniform akista saniyede 1 en net kare uretilir).

        Returns:
            Zaman sirali, saniye saniye duzenli `EvidenceFrame` listesi (00:00, 00:01, 00:02...).
        """
        is_live = video_path.strip().lower().startswith(("rtsp://", "http://", "https://"))
        if not is_live and not Path(video_path).exists():
            raise FileNotFoundError(
                f"Video dosyasi sunucuda bulunamadi: '{video_path}'. "
                "Lutfen dosyanin sunucuya yuklendiginden veya gecerli bir mutlak yol girildiginden emin olun."
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(
                f"Video dosyasi OpenCV ile acilamadi veya gecerli bir video formati degil: '{video_path}'"
            )

        started_at = time.perf_counter()
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Her tam saniye kovasi icin yalnizca EN NET kareyi bellekte tut (Streaming Memory Optimization):
        # sec_idx -> (best_pts, best_frame, best_sharpness, best_frame_id)
        best_by_second: Dict[int, Tuple[float, np.ndarray, float, int]] = {}
        frame_id = 0
        total_scanned = 0
        self.baseline_clean_frame = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                total_scanned += 1

                # SENKRON PTS OKUMA: read() sonrasinda CAP_PROP_POS_MSEC
                pts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                pts_sec = (pts_msec / 1000.0) if pts_msec > 0 else (frame_id / native_fps)

                # YUKSEK COZUNURLUK KORUMASI: Pus/duman piksellerini korumak icin 1280px limit
                if frame.shape[1] > 1280:
                    h = int(frame.shape[0] * (1280.0 / float(frame.shape[1])))
                    frame = cv2.resize(frame, (1280, h), interpolation=cv2.INTER_AREA)

                sec_idx = int(pts_sec)

                # Laplacian varyansi ile netlik/kalite hesabi
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                # Akış esnasında yalnızca o saniyenin en net karesini sakla (Bellek tasarrufu)
                if sec_idx not in best_by_second or sharpness > best_by_second[sec_idx][2]:
                    best_by_second[sec_idx] = (pts_sec, frame, sharpness, frame_id)

                frame_id += 1
        finally:
            cap.release()

        evidence_frames: List[EvidenceFrame] = []
        for sec in sorted(best_by_second.keys()):
            best_pts, best_frame, best_sharpness, best_frame_id = best_by_second[sec]

            ef = self._build_evidence_frame(
                frame=best_frame,
                frame_id=best_frame_id,
                timestamp_sec=best_pts,
                change_score=0.0100,  # Uniform timeline temel skoru
                has_gradual_trend=False,
            )
            evidence_frames.append(ef)
            if self.baseline_clean_frame is None:
                self.baseline_clean_frame = ef

        elapsed_sec = time.perf_counter() - started_at
        eliminated = max(0, total_scanned - len(evidence_frames))
        eliminated_ratio = (100.0 * eliminated / total_scanned) if total_scanned else 0.0

        self.last_run_stats = SamplerRunStats(
            total_frames_scanned=total_scanned,
            sampled_frames_evaluated=total_scanned,
            evidence_frame_count=len(evidence_frames),
            eliminated_frame_count=eliminated,
            eliminated_ratio_pct=round(eliminated_ratio, 2),
            elapsed_sec=round(elapsed_sec, 3),
        )
        logger.info(
            "Uniform 1-Sec Time-Bucket Sampler tamamlandi: %d ham kare tarandi, "
            "%d Kanit Karesi uretildi (Saniyede tam 1 en net kare, 0 mukerrer, 0 atlama), sure=%.3fs",
            total_scanned,
            len(evidence_frames),
            elapsed_sec,
        )
        curr_gray = None
        self.ref_gray = None
        self.prev_gray = None
        gc.collect()
        return evidence_frames

    def _build_evidence_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_sec: float,
        change_score: float,
        motion_bbox: Optional[Tuple[int, int, int, int]] = None,
        has_gradual_trend: bool = False,
    ) -> EvidenceFrame:
        image_bytes = self._encode_frame_jpeg(frame)
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        minutes, seconds = divmod(int(timestamp_sec), 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        return EvidenceFrame(
            frame_id=frame_id,
            timestamp_sec=round(timestamp_sec, 2),
            timestamp_str=time_str,
            change_score=round(change_score, 4),
            image_bytes=image_bytes,
            base64_image=f"data:image/jpeg;base64,{base64_str}",
            image_shape=frame.shape,
            saved_path=None,
            has_gradual_trend=has_gradual_trend,
            motion_bbox=motion_bbox,
        )

    def cluster_events(
        self, evidence_frames: List[EvidenceFrame], video_path: Optional[str] = None
    ) -> List[EventCluster]:
        """Suzulen kareleri zaman araligina gore kumeleyip zirve karelerini secer.

        Iki gecisli calisir:

        1) HAM gruplama: ardisik Kanit Kareleri arasindaki zaman farki
           `min_event_interval_sec` degerini asmadigi surece ayni ham gruba
           dahil edilir (ince taneli, kisa patlama/titreme birlestirme).
        2) BIRLESTIRME (merge): ardisik ham gruplar, YALNIZCA hem (a)
           aralarindaki bosluk `cluster_merge_gap_sec` degerini asmiyorsa
           HEM DE (b) sinir kareleri konumsal olarak ayni fiziksel olayin
           devami sayilacak kadar benzer ise (`_is_same_physical_event`,
           bkz. `bbox_iou_merge_threshold`) TEK bir nihai gruba birlestirilir.
           Yalnizca zaman yakinligi YETERLI DEGILDIR: zaman olarak yakin ama
           hareketin gerceklestigi bolge farkli iki grup, gercekten farkli
           iki olay sayilip AYRI tutulur. Bu iki kosul birlikte, surekli tek
           bir olayin (gurultu-tabani adaptasyonu yuzunden arada kisa sessiz
           birakan) yanlislikla birden fazla ayri Olay Grubuna bolunmesini
           ONLERKEN, zaman olarak yakin ama konumsal olarak farkli gercek
           olaylarin birbirine karisip kaybolmasini da ENGELLER.

        Her nihai grup icin en yuksek degisim skoruna sahip kare (peak frame)
        VLM'e gonderilecek temsilci olarak secilir ve YALNIZCA bu kare diske
        yazilir; boylece merge sonrasi elenen ham gruplarin zirve adaylari
        icin gereksiz disk I/O yapilmaz.

        Args:
            evidence_frames: `process_video` tarafindan uretilen, zaman sirali
                Kanit Kareleri listesi.
            video_path: Verilirse, her nihai Olay Grubu icin peak'e ek olarak
                pre_peak/post_peak kareleri de kaynak videodan seek edilip
                `evidence_output_dir` altina (`event_XXXX/` alt klasoruyle)
                diske yazilir ve sonucu ozetleyen bir log satiri basilir
                (bkz. `export_event_frames`). `None` ise (varsayilan) bu
                adim hic calismaz ve davranis, bu parametre eklenmeden
                onceki mevcut peak-only akisla BIREBIR AYNIDIR.

        Returns:
            Zaman sirali, birlestirilmis `EventCluster` listesi. Girdi bossa
            bos liste doner.
        """
        if not evidence_frames:
            return []

        # 1) 20 saniyelik sabit periyodik zamansal pencereler (Fixed 20-Sec Uniform Windowing)
        chunk_sec = 20.0
        raw_groups: List[List[EvidenceFrame]] = []
        current_group: List[EvidenceFrame] = [evidence_frames[0]]
        chunk_start_t = evidence_frames[0].timestamp_sec

        for ef in evidence_frames[1:]:
            if (ef.timestamp_sec - chunk_start_t) < chunk_sec:
                current_group.append(ef)
            else:
                raw_groups.append(current_group)
                current_group = [ef]
                chunk_start_t = ef.timestamp_sec
        raw_groups.append(current_group)

        clusters: List[EventCluster] = [
            self._close_group(group, event_id=i + 1) for i, group in enumerate(raw_groups)
        ]

        self.last_cluster_merge_stats = ClusterMergeStats(
            raw_group_count=len(raw_groups),
            final_cluster_count=len(clusters),
            merged_raw_group_count=0,
        )

        logger.info(
            "EventCluster tamamlandi: %d Kanit Karesi -> %d nihai Olay Grubu (20s sabit pencereler)",
            len(evidence_frames),
            len(clusters),
        )

        if video_path is not None:
            self._export_and_log_pre_post_peak_frames(video_path, clusters)

        return clusters

    def _export_and_log_pre_post_peak_frames(self, video_path: str, clusters: List[EventCluster]) -> None:
        """Her Olay Grubu icin pre_peak/peak/post_peak `.jpg`lerini diske yazar ve ozetler.

        `evidence_output_dir` altina (peak'in zaten yazildigi ayni kok
        klasor) her olay icin bir `event_XXXX/` alt klasoru olusturur (bkz.
        `PeakFrameExporter`/`export_event_frames`). Diske yazma basarisiz
        olursa (ör. video acilamadi) hata sessizce loglanir; `cluster_events`
        cokmez ve zaten hesaplanmis `clusters` degismeden donulur.

        Args:
            video_path: Kaynak video dosyasinin yolu.
            clusters: `cluster_events` tarafindan az once uretilen Olay Gruplari.
        """
        try:
            event_dirs = self.export_event_frames(
                video_path, clusters, output_dir=str(self.evidence_output_dir)
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "Olay basina pre_peak/post_peak kareleri diske yazilamadi (%s): %s",
                video_path,
                exc,
            )
            return

        for cluster, event_dir in zip(clusters, event_dirs):
            peak = cluster.peak_frame
            logger.info(
                "Olay #%d kareleri diske yazildi | pre_peak=%s/pre_peak.jpg | "
                "peak=%s/peak.jpg (t=%.2fs) | post_peak=%s/post_peak.jpg",
                cluster.event_id,
                event_dir,
                event_dir,
                peak.timestamp_sec,
                event_dir,
            )

    def _close_group(self, group: List[EvidenceFrame], event_id: int) -> EventCluster:
        """Bir Kanit Karesi grubunu kapatip zirve karesini secerek `EventCluster` uretir.

        Yalnizca burada secilen zirve (peak) kare diske kalici olarak yazilir;
        gruptaki diger adaylar hic diske yazilmadan atlanmis olur. Boylece
        `cluster_events` sonrasinda hicbir zaman VLM'e gonderilmeyecek/kullanilmayacak
        kareler icin gereksiz disk I/O yapilmaz.

        Args:
            group: Ayni zaman araligina dusen Kanit Kareleri.
            event_id: Bu gruba atanacak olay kimligi.

        Returns:
            `peak_frame.saved_path` doldurulmus, grubun ozetini tasiyan `EventCluster`.
        """
        peak = max(group, key=lambda ef: ef.change_score)
        peak.saved_path = self._persist_frame(peak.frame_id, peak.timestamp_str, peak.image_bytes)
        start_time = group[0].timestamp_sec
        end_time = group[-1].timestamp_sec
        has_gradual_trend = any(getattr(ef, "has_gradual_trend", False) for ef in group)

        rep_frames = []
        if self.baseline_clean_frame is not None and group[0].timestamp_sec >= 10.0:
            rep_frames.append(
                RepresentativeFrame(
                    label="REFERANS_TEMIZ_KARE_00:00",
                    timestamp_sec=self.baseline_clean_frame.timestamp_sec,
                    timestamp_str=self.baseline_clean_frame.timestamp_str,
                    base64_image=self.baseline_clean_frame.base64_image,
                )
            )

        for ef in group:
            rep_frames.append(
                RepresentativeFrame(
                    label=f"kare_{ef.timestamp_str}",
                    timestamp_sec=ef.timestamp_sec,
                    timestamp_str=ef.timestamp_str,
                    base64_image=ef.base64_image,
                )
            )

        return EventCluster(
            event_id=event_id,
            start_time=start_time,
            end_time=end_time,
            peak_frame=peak,
            total_candidate_frames=len(group),
            has_gradual_trend=has_gradual_trend,
            duration_sec=round(end_time - start_time, 2),
            representative_frames=rep_frames,
        )

    def export_event_frames(
        self, video_path: str, clusters: List[EventCluster], output_dir: str = "outputs/sampler"
    ) -> List[str]:
        """Her Olay Grubu icin `pre_peak.jpg`/`peak.jpg`/`post_peak.jpg` + `metadata.json` yazar.

        `PeakFrameExporter`'a ince bir sarmalayicidir; `self.pre_peak_offset_sec`/
        `self.post_peak_offset_sec` kullanir. `context.mp4` gibi hicbir video
        klibi URETMEZ (bkz. modul docstring'i). Cluster/peak secim mantigina
        (bu metod cagrilmadan once `cluster_events` ile uretilmis olmalidir)
        dokunmaz.

        Args:
            video_path: `cluster_events`i ureten `EventCluster`lerin ait
                oldugu kaynak video dosyasinin yolu.
            clusters: `cluster_events` ciktisi Olay Gruplari.
            output_dir: `event_XXXX/` alt klasorlerinin olusturulacagi kok dizin.

        Returns:
            Yazilan her `event_XXXX` klasorunun yolu (string listesi).

        Raises:
            ValueError: Video acilamazsa.
            RuntimeError: Video FPS degeri gecersizse.
        """
        exporter = PeakFrameExporter(
            pre_peak_offset_sec=self.pre_peak_offset_sec,
            post_peak_offset_sec=self.post_peak_offset_sec,
        )
        return exporter.export(video_path, clusters, output_dir=output_dir)


def sampler_from_config(
    config: SamplerConfig, min_change_threshold_override: Optional[float] = None
) -> AdaptiveFrameSampler:
    """`configs/config.yaml` icindeki `sampler` blogundan bir `AdaptiveFrameSampler` uretir.

    Her cagri taze bir `AdaptiveFrameSampler` orneği doner (paylasimli/kalici
    durum tutmaz); boylece operator panelinden gelen ayarlar cagrilar arasinda
    birbirine karismaz ve her analiz temiz bir `prev_gray`/gurultu gecmisiyle
    baslar.

    Args:
        config: Dogrulanmis `SamplerConfig` nesnesi.
        min_change_threshold_override: Verilirse, config degeri yerine bu
            hassasiyet esigi kullanilir (operator panelindeki slider icin).

    Returns:
        Config (ve varsa override) parametreleriyle ilklendirilmis, taze
        `AdaptiveFrameSampler` orneği.
    """
    return AdaptiveFrameSampler(
        min_change_threshold=(
            min_change_threshold_override
            if min_change_threshold_override is not None
            else config.min_change_threshold
        ),
        blur_kernel_size=tuple(config.blur_kernel_size),
        history_window=config.history_window,
        min_event_interval_sec=config.min_event_interval_sec,
        max_evidence_buffer=config.max_evidence_buffer,
        cluster_merge_gap_sec=config.cluster_merge_gap_sec,
        bbox_iou_merge_threshold=config.bbox_iou_merge_threshold,
        temporal_vote_window=config.temporal_vote_window,
        temporal_vote_min_count=config.temporal_vote_min_count,
        pre_peak_offset_sec=config.pre_peak_offset_sec,
        post_peak_offset_sec=config.post_peak_offset_sec,
        max_sampling_gap_sec=config.max_sampling_gap_sec,
        ref_reset_interval_sec=config.ref_reset_interval_sec,
        ref_change_threshold=config.ref_change_threshold,
        enable_contrast_check=config.enable_contrast_check,
        contrast_change_threshold=config.contrast_change_threshold,
        contrast_check_interval_sec=config.contrast_check_interval_sec,
    )


if __name__ == "__main__":
    # Modul 1'in bagimsiz calistirilabilirlik testi:
    #   python -m src.sampler.adaptive_sampler [video_yolu]
    # Varsayilan olarak data/test.mp4 uzerinde calisir; VLM/GPU'ya veya
    # projenin geri kalanina hicbir bagimliligi yoktur.
    import json
    import sys

    logging.basicConfig(level=logging.INFO)

    demo_video_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.mp4"
    if not Path(demo_video_path).exists():
        print(
            f"Video bulunamadi: {demo_video_path}\n"
            "Kullanim: python -m src.sampler.adaptive_sampler [video_yolu]\n"
            "(Varsayilan olarak 'data/test.mp4' aranir.)"
        )
        sys.exit(1)

    demo_sampler = AdaptiveFrameSampler()
    demo_evidence_frames = demo_sampler.process_video(demo_video_path)
    demo_clusters = demo_sampler.cluster_events(demo_evidence_frames, video_path=demo_video_path)
    demo_stats = demo_sampler.last_run_stats

    print(f"Video: {demo_video_path}")
    print(f"Taranan ham kare: {demo_stats.total_frames_scanned}")
    print(f"Degerlendirilen ornek kare: {demo_stats.sampled_frames_evaluated}")
    print(f"Uretilen Kanit Karesi: {demo_stats.evidence_frame_count}")
    print(
        f"Elenen kare sayisi: {demo_stats.eliminated_frame_count} "
        f"(%{demo_stats.eliminated_ratio_pct:.1f} eleme orani)"
    )
    print(f"Olusan Olay Grubu sayisi: {len(demo_clusters)}")
    for demo_cluster in demo_clusters:
        demo_event_dir = demo_sampler.evidence_output_dir / f"event_{demo_cluster.event_id:04d}"
        print(
            f"Olay #{demo_cluster.event_id} | pre_peak={demo_event_dir}/pre_peak.jpg | "
            f"peak={demo_event_dir}/peak.jpg (t={demo_cluster.peak_frame.timestamp_sec:.2f}s) | "
            f"post_peak={demo_event_dir}/post_peak.jpg"
        )

    demo_output_path = Path("data/mock_evidence.json")
    demo_output_path.parent.mkdir(parents=True, exist_ok=True)
    demo_payload = {
        "video_source": demo_video_path,
        "stats": {
            "total_frames_scanned": demo_stats.total_frames_scanned,
            "sampled_frames_evaluated": demo_stats.sampled_frames_evaluated,
            "evidence_frame_count": demo_stats.evidence_frame_count,
            "eliminated_frame_count": demo_stats.eliminated_frame_count,
            "eliminated_ratio_pct": demo_stats.eliminated_ratio_pct,
            "elapsed_sec": demo_stats.elapsed_sec,
        },
        "evidence_frames": [
            ef.model_dump(exclude={"image_bytes"}) for ef in demo_evidence_frames
        ],
        "clusters": [
            {
                "event_id": cluster.event_id,
                "start_time": cluster.start_time,
                "end_time": cluster.end_time,
                "total_candidate_frames": cluster.total_candidate_frames,
                "peak_frame": cluster.peak_frame.model_dump(exclude={"image_bytes"}),
            }
            for cluster in demo_clusters
        ],
    }
    demo_output_path.write_text(
        json.dumps(demo_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Kanit verisi yazildi: {demo_output_path}")
