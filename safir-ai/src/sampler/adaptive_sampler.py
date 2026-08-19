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
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Deque, List, Optional, Tuple

import cv2
import numpy as np

from src.sampler.context.frame_archiver import FrameArchiver
from src.sampler.context.frame_selector import FrameSelector
from src.sampler.schema import EventCluster, EvidenceFrame

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
        cluster_merge_gap_sec: float = 20.0,
        bbox_iou_merge_threshold: float = 0.10,
        max_cluster_duration_sec: float = 120.0,
        temporal_vote_window: int = 1,
        temporal_vote_min_count: int = 1,
    ) -> None:
        """AdaptiveFrameSampler'i esik ve pencere parametreleriyle baslatir.

        Args:
            min_change_threshold: Bir karenin Kanit Karesi sayilmasi icin
                gereken, gurultu tabani dusulmus minimum piksel degisim orani.
            blur_kernel_size: Gurultu gidermek icin uygulanan Gauss bulaniklastirma
                cekirdek boyutu (tek sayilardan olusan (genislik, yukseklik) ikilisi).
            history_window: Dinamik gurultu tabanini (medyan) hesaplamak icin
                tutulan son degisim orani sayisi.
            min_event_interval_sec: Ardisik Kanit Karelerini ayni HAM (raw) Olay
                Grubuna dahil etmek icin izin verilen maksimum zaman farki (saniye).
                Bu, tek bir surekli olayin kisa sureli titremeleri/patlamalarini
                bir arada tutan INCE taneli esiktir; birbirinden ayri ham gruplarin
                yine de ayni fiziksel olaya ait olup olmadigina KARAR VERMEZ
                (bkz. `cluster_merge_gap_sec`).
            evidence_output_dir: Her Kanit Karesinin `.jpg` olarak diske
                yazilacagi klasor (UI ve denetim/log amacli kalicilik icin).
                ONEMLI: Kanit Karesi sayisinda sabit bir ust sinir YOKTUR
                (eski `max_evidence_buffer` kaldirildi); uzun/surekli bir
                olay ne kadar sürerse sürsün hicbir Kanit Karesi
                `process_video` seviyesinde sessizce atlanmaz. VLM'e giden
                kare sayisi, bu asamada DEGIL, `cluster_events` icinde
                `FrameSelector` tarafindan olay basina (video geneli DEGIL)
                sinirlanir (bkz. `FrameSelector.TARGET_FRAME_COUNT`).
            cluster_merge_gap_sec: `cluster_events` icinde, `min_event_interval_sec`
                ile olusturulan ardisik HAM Olay Gruplarini, aralarindaki bosluk bu
                degeri asmadigi surece TEK bir nihai Olay Grubunda (EventCluster)
                birlestirmek icin kullanilan, `min_event_interval_sec`den BAGIMSIZ
                ve KASITLI OLARAK DAHA GENIS bir esik. Amaci: surekli tek bir olayin
                (ör. yanan bir ates), gurultu tabaninin zamanla adapte olmasi
                nedeniyle arada sirali kisa sessiz araliklar birakarak birden fazla
                sahte/kopya olaya bolunmesini onlemek. `min_event_interval_sec`den
                KUCUK verilirse bu adim etkisiz kalir (pass 1 zaten her seyi
                birlestirmis olur).
            bbox_iou_merge_threshold: `cluster_merge_gap_sec` zaman kosulunu
                gecen iki ham grubun GERCEKTEN ayni fiziksel olayin devami
                olup olmadigini ayirt eden konumsal (mekansal) esik. Bir
                grubun SON karesinin hareket kutusu (`motion_bbox`) ile bir
                sonraki grubun ILK karesinin hareket kutusu arasindaki IoU
                (kesisim/birlesim orani) bu degerin altindaysa, iki grup
                zaman olarak yakin olsa bile FARKLI bir olay sayilir ve
                birlestirilmez. Bu, "zaman olarak yakin ama konumsal olarak
                farkli -> farkli olay" kuralini uygular; nesne/olay turunden
                bagimsizdir (yalnizca hareket bolgesinin piksel-uzayindaki
                konumuna bakar). `motion_bbox` mevcut degilse (ör. fallback
                kare, bos hareket maskesi) TEMKINLI davranilir: birlestirme
                yapilmaz. Birlestirme kararinda YALNIZCA komsu siniri DEGIL,
                mevcut nihai grubun ILK karesiyle olan konumsal sureklilik de
                ayrica kontrol edilir (bkz. `cluster_events` icindeki
                "transitive chaining" onlemi): bu, A-B-C gibi ardisik
                birlestirmelerin, her adim komsusuyla benzer gorunse bile
                zamanla tamamen farkli bir bolgeye "kaymasini" (mega-cluster
                olusumunu) engeller.
            max_cluster_duration_sec: Bir nihai Olay Grubunun (`EventCluster`)
                izin verilen azami toplam suresi (saniye), ilk kareden son
                karenin zaman damgasina kadar. Bu sinira ulasan bir
                birlestirme reddedilir (aday grup, birlesmek yerine YENI bir
                nihai grup baslatir); hicbir Kanit Karesi bu nedenle
                SILINMEZ, yalnizca farkli/daha kucuk Olay Gruplarina
                bolusturulur. Amaci: gercekte tek fiziksel olay olmayan ama
                zaman/konum kriterlerini zincirleme (transitive chaining)
                gecerek birlesen kareleri makul bir sure ile sinirlamak.
            temporal_vote_window: Bir karenin candidate olarak onaylanip
                onaylanmayacagina karar verirken dikkate alinan, o anki karar
                dahil, en son kac esik-testi sonucunun tutulacagi (temporal
                voting penceresi). Varsayilan `1` ile pencere yalnizca o anki
                karari icerir; bu, mevcut (voting oncesi) davranisla BIREBIR
                AYNIDIR. `min_change_threshold` formulunu, gurultu tabanini
                veya cluster/merge mantigini DEGISTIRMEZ; yalnizca esigi
                gecen bir karenin GERCEK candidate sayilip sayilmayacagina
                ek bir "sureklilik" filtresi uygular (bkz.
                `temporal_vote_min_count`).
            temporal_vote_min_count: `temporal_vote_window` penceresi
                icinde, bir karenin candidate olarak onaylanmasi icin gereken
                minimum "supheli" (esigi gecen) karar sayisi. Varsayilan `1`
                ile tek bir supheli karar yeterlidir; bu da mevcut davranisla
                BIREBIR AYNIDIR. Daha yuksek bir deger (ör. `3`), tek karelik
                kamera titremesi/isik degisimi/sikistirma artefakti gibi
                izole supheli kareleri elemek icin `temporal_vote_window`i
                artirmakla birlikte kullanilir (ör. window=5, min_count=3).
                `temporal_vote_window`den buyuk olamaz.

        Raises:
            ValueError: `temporal_vote_window` veya `temporal_vote_min_count`
                1'den kucukse, `temporal_vote_min_count` `temporal_vote_window`den
                buyukse, ya da `max_cluster_duration_sec` 0'dan kucuk/esitse.
        """
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
        if max_cluster_duration_sec <= 0:
            raise ValueError(
                f"max_cluster_duration_sec pozitif olmalidir, verilen: {max_cluster_duration_sec}"
            )

        self.min_change_threshold = min_change_threshold
        self.blur_kernel_size = blur_kernel_size
        self.history_window = history_window
        self.min_event_interval_sec = min_event_interval_sec
        self.evidence_output_dir = Path(evidence_output_dir)
        self.cluster_merge_gap_sec = cluster_merge_gap_sec
        self.bbox_iou_merge_threshold = bbox_iou_merge_threshold
        self.max_cluster_duration_sec = max_cluster_duration_sec
        self.temporal_vote_window = temporal_vote_window
        self.temporal_vote_min_count = temporal_vote_min_count

        self.prev_gray: np.ndarray | None = None
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
        """Videoyu okur, kare farklarini hesaplar ve suzulmus Kanit Karelerini dondurur.

        Args:
            video_path: `.mp4` dosya yolu veya RTSP URI'si.
            sample_fps: Videonun kac saniyede bir kare kontrol edilecegini
                belirleyen ornekleme hizi (native FPS'ten dusuk olmalidir).

        Returns:
            Zaman sirali `EvidenceFrame` listesi.

        Raises:
            ValueError: Video dosyasi acilamazsa.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Video dosyasi acilamadi: {video_path}")

        started_at = time.perf_counter()
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_step = max(1, int(native_fps / sample_fps))

        # Temporal voting durumu (bkz. `_confirm_candidate`) her yeni video
        # icin sifirlanir; onceki bir `process_video` cagrisinin (varsa) son
        # kararlari bu videoya tasinmaz.
        self._recent_threshold_decisions.clear()

        evidence_frames: List[EvidenceFrame] = []
        frame_id = 0
        sampled_frame_count = 0
        first_frame_raw: Optional[np.ndarray] = None
        first_frame_timestamp: float = 0.0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % frame_step != 0:
                    frame_id += 1
                    continue

                sampled_frame_count += 1
                timestamp_sec = frame_id / native_fps
                curr_gray = self._preprocess_frame(frame)

                if first_frame_raw is None:
                    first_frame_raw = frame.copy()
                    first_frame_timestamp = timestamp_sec

                if self.prev_gray is not None:
                    frame_diff = cv2.absdiff(curr_gray, self.prev_gray)
                    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                    change_ratio = np.sum(thresh > 0) / float(thresh.size)

                    self.noise_floor_history.append(change_ratio)
                    if len(self.noise_floor_history) > self.history_window:
                        self.noise_floor_history.pop(0)

                    adaptive_noise_floor = np.median(self.noise_floor_history)
                    net_change_score = max(0.0, change_ratio - adaptive_noise_floor)

                    # Temporal voting: mevcut esik sonucunu (degismedi) girdi olarak
                    # kullanip, izole/tek karelik supheli kareleri (kamera titremesi,
                    # isik degisimi, sikistirma artefakti) eleyen ek sureklilik kontrolu.
                    # Varsayilan ayarlarda (window=1, min_count=1) bu, ESKI davranisla
                    # BIREBIR AYNIDIR (bkz. `_confirm_candidate` docstring'i).
                    is_suspicious = net_change_score >= self.min_change_threshold
                    if self._confirm_candidate(is_suspicious):
                        # Sabit bir ust sinir YOKTUR (eski max_evidence_buffer kaldirildi):
                        # video ne kadar uzun/olay ne kadar surekli olursa olsun, hicbir
                        # Kanit Karesi burada sessizce atlanmaz. Kare sayisi kontrolu,
                        # bunun yerine cluster_events icinde OLAY BASINA (video geneli
                        # DEGIL) FrameSelector tarafindan yapilir.
                        motion_bbox = self._bbox_from_mask(thresh)
                        evidence_frames.append(
                            self._build_evidence_frame(
                                frame, frame_id, timestamp_sec, net_change_score, motion_bbox
                            )
                        )

                self.prev_gray = curr_gray
                frame_id += 1
        finally:
            cap.release()

        if not evidence_frames:
            if first_frame_raw is None:
                raise ValueError(f"Video kaynagindan hic kare okunamadi: {video_path}")

            logger.warning(
                "Esigi gecen Kanit Karesi bulunamadi; sistemin cokmemesi icin frame 0 "
                "varsayilan Kanit Karesi olarak kabul ediliyor (fallback)."
            )
            fallback = self._build_evidence_frame(
                first_frame_raw, frame_id=0, timestamp_sec=first_frame_timestamp, change_score=0.0
            )
            fallback.is_fallback = True
            evidence_frames.append(fallback)

        elapsed_sec = time.perf_counter() - started_at
        eliminated = sampled_frame_count - len(evidence_frames)
        eliminated_ratio = (100.0 * eliminated / sampled_frame_count) if sampled_frame_count else 0.0
        self.last_run_stats = SamplerRunStats(
            total_frames_scanned=frame_id,
            sampled_frames_evaluated=sampled_frame_count,
            evidence_frame_count=len(evidence_frames),
            eliminated_frame_count=eliminated,
            eliminated_ratio_pct=round(eliminated_ratio, 2),
            elapsed_sec=round(elapsed_sec, 3),
        )
        logger.info(
            "AdaptiveFrameSampler tamamlandi: %d ham kare tarandi, %d kare ornekleme icin "
            "degerlendirildi, %d Kanit Karesi uretildi (%d kare elendi, %%%.1f eleme orani), "
            "sure=%.3fs",
            frame_id,
            sampled_frame_count,
            len(evidence_frames),
            eliminated,
            eliminated_ratio,
            elapsed_sec,
        )
        return evidence_frames

    def _build_evidence_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_sec: float,
        change_score: float,
        motion_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> EvidenceFrame:
        """Bir kareden `EvidenceFrame` uretir ve JPEG/base64'e cevirir.

        Diske yazma burada YAPILMAZ: bu kare `cluster_events` tarafindan bir
        Olay Grubunun zirve karesi olarak secilmedigi surece hic kullanilmayabilir.
        Kalici diske kayit, yalnizca zirve olarak secilen kareler icin
        `_close_group` tarafindan yapilir (bkz. o metodun docstring'i).

        Args:
            frame: BGR formatinda ham video karesi.
            frame_id: Karenin video icindeki sirasi.
            timestamp_sec: Karenin saniye cinsinden zaman damgasi.
            change_score: Hesaplanan (gurultu-tabani-dusulmus) degisim skoru.
            motion_bbox: Bu kareyi ureten hareket maskesinin sinirlayici kutusu
                (`_bbox_from_mask`); yoksa (ör. fallback kare) `None`.

        Returns:
            `image_bytes`/`base64_image` doldurulmus, `saved_path=None` olan `EvidenceFrame`.
        """
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
            motion_bbox=motion_bbox,
        )

    def cluster_events(
        self, evidence_frames: List[EvidenceFrame], export_to_disk: bool = False
    ) -> List[EventCluster]:
        """Suzulen kareleri zaman araligina gore kumeleyip zirve karelerini ve temsili kareleri secer.

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

        Her nihai grup icin en yuksek degisim skoruna sahip kare secilir ve
        diske kalici olarak yazilir; ardindan AYNI grubun Kanit Kareleri
        havuzundan `FrameSelector` ile (video YENIDEN ACILMADAN, hicbir kare
        yeniden JPEG'e KODLANMADAN) en fazla 5 benzersiz, kronolojik evidence
        karesi secilip `EventCluster.representative_frames`e yazilir (en
        yuksek skorlu kare dahil, ama HICBIR konumsal 'pre'/'peak'/'post'
        rolu tasimadan). Bu, VLM'e giden kareler ile diske/rapora yansiyan
        kareler icin TEK ortak kaynaktir.

        Args:
            evidence_frames: `process_video` tarafindan uretilen, zaman sirali
                Kanit Kareleri listesi.
            export_to_disk: `True` verilirse, her nihai Olay Grubu icin
                `FrameSelector`in sectigi kareler (bagimsiz bir secim
                YAPILMADAN) `FrameArchiver` ile `evidence_output_dir` altina
                (`event_XXXX/` alt klasoruyle) diske yazilir. Varsayilan
                `False`: bu adim VLM'e gonderilecek `representative_frames`i
                ETKILEMEZ (o her zaman doldurulur), yalnizca ekstra disk
                arsivlemesini acar/kapatir.

        Returns:
            Zaman sirali, birlestirilmis `EventCluster` listesi. Girdi bossa
            bos liste doner.
        """
        if not evidence_frames:
            return []

        # 1) HAM gruplama (mevcut ince taneli mantik, DEGISMEDI).
        raw_groups: List[List[EvidenceFrame]] = []
        current_group: List[EvidenceFrame] = [evidence_frames[0]]
        for ef in evidence_frames[1:]:
            if ef.timestamp_sec - current_group[-1].timestamp_sec <= self.min_event_interval_sec:
                current_group.append(ef)
            else:
                raw_groups.append(current_group)
                current_group = [ef]
        raw_groups.append(current_group)

        # 2) BIRLESTIRME: komsu ham gruplari, YALNIZCA UCU DE saglanirsa tek
        # nihai grupta topla:
        #   (a) aralarindaki zaman bosluğu `cluster_merge_gap_sec`i asmiyor,
        #   (b) sinir kareleri (onceki grubun SON karesi <-> aday grubun ILK
        #       karesi) konumsal olarak ayni fiziksel olayin devami sayilacak
        #       kadar benzer (`_is_same_physical_event`),
        #   (c) aday grubun ILK karesi, nihai grubun kendi ILK karesiyle de
        #       (yalnizca en son eklenen komsuyla DEGIL) konumsal sureklilik
        #       gosteriyor - bu, "transitive chaining" onlemidir: her adim
        #       yalnizca komsusuyla benzer olsa bile, A->B->C->D zinciri
        #       zamanla tamamen farkli bir bolgeye kayabilir (mega-cluster).
        #       Sabit bir ANKOR (nihai grubun ilk karesi) ile karsilastirmak
        #       bu kaymayi sinirlar,
        #   (d) birlestirme sonrasi toplam sure `max_cluster_duration_sec`i
        #       ASMIYOR (asiri uzun/mega cluster onlemi). Hicbir Kanit Karesi
        #       bu kontroller nedeniyle SILINMEZ; yalnizca birlestirilmeyip
        #       AYRI bir nihai gruba (event'e) aktarilir.
        merged_groups: List[List[EvidenceFrame]] = [raw_groups[0]]
        merge_count = 0
        for group in raw_groups[1:]:
            previous_group = merged_groups[-1]
            cluster_anchor = previous_group[0]
            gap = group[0].timestamp_sec - previous_group[-1].timestamp_sec
            candidate_duration = group[-1].timestamp_sec - cluster_anchor.timestamp_sec
            can_merge = (
                gap <= self.cluster_merge_gap_sec
                and self._is_same_physical_event(previous_group[-1], group[0])
                and self._is_same_physical_event(cluster_anchor, group[0])
                and candidate_duration <= self.max_cluster_duration_sec
            )
            if can_merge:
                previous_group.extend(group)
                merge_count += 1
            else:
                merged_groups.append(group)

        clusters: List[EventCluster] = [
            self._close_group(group, event_id=i + 1) for i, group in enumerate(merged_groups)
        ]

        self.last_cluster_merge_stats = ClusterMergeStats(
            raw_group_count=len(raw_groups),
            final_cluster_count=len(clusters),
            merged_raw_group_count=merge_count,
        )

        logger.info(
            "EventCluster tamamlandi: %d Kanit Karesi -> %d ham grup -> %d nihai Olay Grubu "
            "(%d ham grup surekli olay olarak birlestirildi)",
            len(evidence_frames),
            len(raw_groups),
            len(clusters),
            merge_count,
        )

        if export_to_disk:
            self._export_and_log_representative_frames(clusters)

        return clusters

    def _export_and_log_representative_frames(self, clusters: List[EventCluster]) -> None:
        """`FrameSelector`in ZATEN sectigi temsili kareleri diske yazar ve ozetler.

        Bagimsiz bir kare secimi YAPMAZ: yalnizca her `EventCluster.
        representative_frames`i (VLM'e giden AYNI kareler) `FrameArchiver` ile
        `evidence_output_dir` altina yazar. Diske yazma basarisiz olursa hata
        sessizce loglanir; `cluster_events` cokmez.

        Args:
            clusters: `cluster_events` tarafindan az once uretilen, `representative_frames`i
                zaten dolu Olay Gruplari.
        """
        try:
            event_dirs = self.export_event_frames(clusters, output_dir=str(self.evidence_output_dir))
        except OSError as exc:
            logger.warning("Temsili kareler diske yazilamadi: %s", exc)
            return

        for cluster, event_dir in zip(clusters, event_dirs):
            logger.info(
                "Olay #%d icin %d temsili kare diske yazildi: %s",
                cluster.event_id,
                len(cluster.representative_frames),
                event_dir,
            )

    def _close_group(self, group: List[EvidenceFrame], event_id: int) -> EventCluster:
        """Bir Kanit Karesi grubunu kapatip zirve/temsili kareleri secerek `EventCluster` uretir.

        Yalnizca zirve (peak) kare `evidence_output_dir` altina kalici olarak
        tekil `.jpg` yazilir. Ardindan `FrameSelector`, AYNI grubun (video
        YENIDEN acilmadan, hicbir kare yeniden JPEG'e kodlanmadan) bellekteki
        Kanit Karelerinden en fazla 5 benzersiz temsili kare secer; bu secim
        hem VLM payload'ina hem (istege bagli) disk arsivine tek kaynaktir.

        Args:
            group: Ayni zaman araligina dusen Kanit Kareleri.
            event_id: Bu gruba atanacak olay kimligi.

        Returns:
            `peak_frame.saved_path` ve `representative_frames` doldurulmus,
            grubun ozetini tasiyan `EventCluster`.
        """
        peak = max(group, key=lambda ef: ef.change_score)
        peak.saved_path = self._persist_frame(peak.frame_id, peak.timestamp_str, peak.image_bytes)
        start_time = group[0].timestamp_sec
        end_time = group[-1].timestamp_sec
        representative_frames = FrameSelector.select(peak, group, event_id=event_id)
        return EventCluster(
            event_id=event_id,
            start_time=start_time,
            end_time=end_time,
            peak_frame=peak,
            total_candidate_frames=len(group),
            duration_sec=round(end_time - start_time, 2),
            representative_frames=representative_frames,
        )

    def export_event_frames(
        self, clusters: List[EventCluster], output_dir: str = "outputs/sampler"
    ) -> List[str]:
        """Her Olay Grubu icin `FrameSelector`in ZATEN sectigi kareleri + `metadata.json` yazar.

        `FrameArchiver`e ince bir sarmalayicidir; bagimsiz bir kare secimi
        YAPMAZ ve kaynak videoya erismez (bkz. modul docstring'i). Cluster/
        peak/temsili-kare secim mantigina (bu metod cagrilmadan once
        `cluster_events` ile uretilmis olmalidir) dokunmaz.

        Args:
            clusters: `cluster_events` ciktisi, `representative_frames`i
                zaten dolu Olay Gruplari.
            output_dir: `event_XXXX/` alt klasorlerinin olusturulacagi kok dizin.

        Returns:
            Yazilan her `event_XXXX` klasorunun yolu (string listesi).
        """
        return FrameArchiver.export(clusters, output_dir=output_dir)


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
        cluster_merge_gap_sec=config.cluster_merge_gap_sec,
        bbox_iou_merge_threshold=config.bbox_iou_merge_threshold,
        max_cluster_duration_sec=config.max_cluster_duration_sec,
        temporal_vote_window=config.temporal_vote_window,
        temporal_vote_min_count=config.temporal_vote_min_count,
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
    demo_clusters = demo_sampler.cluster_events(demo_evidence_frames, export_to_disk=True)
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
            f"Olay #{demo_cluster.event_id} | {len(demo_cluster.representative_frames)} temsili kare "
            f"-> {demo_event_dir}/ | zirve t={demo_cluster.peak_frame.timestamp_sec:.2f}s"
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
