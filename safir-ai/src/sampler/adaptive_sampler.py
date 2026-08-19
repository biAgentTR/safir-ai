"""02 - VLM Oncesi Katman: OpenCV tabanli Adaptive Frame Sampler (CPU ONLY).

Bu modul, ham video akisini agir bir tespit modeli (YOLO/ByteTrack) ve GPU
kullanmadan tarar. Ardisik gri-tonlama kareler arasindaki mutlak fark ile
degisimi olcer, dinamik (medyan tabanli) bir gurultu tabani (noise floor)
ile gereksiz titremeleri eler ve yalnizca anlamli degisim iceren kareleri
"Kanit Karesi" (Evidence Frame) olarak isaretler.

ONEMLI (mimari): Sampler burada DURUR - hicbir olay kumelemesi (event
clustering), zirve/temsili kare secimi veya `event_id`/`cluster_id` uretimi
YAPMAZ. Esik-gecmis TUM Kanit Kareleri, video geneli SIRALI (kronolojik) ve
KAYIPSIZ olarak `process_video()`den doner; hicbir global buffer/kare
limiti, temporal voting/clustering/deduplication veya liste kesme nedeniyle
bir kare burada elenmez. Kumeleme + olay analizi artik VLM katmaninda
yapilir (bkz. `src/vlm/base_vlm.py::BaseVLM.analyze_evidence_batched` +
`reconcile_events`).

ONEMLI (zamansal kapsama / temporal coverage): Esik tabanli secim, uzun bir
sessiz araliktaki (ör. 00:15 -> 01:45) esik-alti kareleri DOGASI GEREGI hic
evidence yapmaz; bu, gercek bir olayin tamamen kacirilmasina yol acabilir.
Bunu onlemek icin `process_video()`, son evidence karesinden bu yana gecen
sure `max_temporal_gap_sec`i asarsa, o pencerede degerlendirilen esik-alti
adaylar arasindan `net_change_score`'u EN YUKSEK olani (rastgele veya sabit
periyodik bir kare DEGIL) `selection_reason="temporal_coverage"` ile
evidence listesine ekler. Bu bir OLAY KUMELEMESI DEGILDIR ve pre/peak/post
gibi bir konumsal rol getirmez - yalnizca "bu kare neden secildi" bilgisini
tasiyan bir metadata alanidir (bkz. `EvidenceFrame.selection_reason`).
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

from src.sampler.schema import EvidenceFrame

_SELECTION_REASON_THRESHOLD = "threshold_exceeded"
_SELECTION_REASON_COVERAGE = "temporal_coverage"
_SELECTION_REASON_FALLBACK = "fallback"

if TYPE_CHECKING:
    # Yalnizca tip ipucu icin gereklidir (bkz. `sampler_from_config`); modul
    # ustunde zaten `from __future__ import annotations` oldugundan tip
    # ipuclari calisma zamaninda hic degerlendirilmez. Bu importu calisma
    # zamanindan (runtime) TYPE_CHECKING'e tasimak, yalnizca
    # `AdaptiveFrameSampler`/`EvidenceFrame`'i kullanan (ve
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
class _PendingCoverageCandidate:
    """Zamansal kapsama penceresi icinde su ana kadar goeruelen EN IYI (en yuksek
    `net_change_score`'lu) esik-alti aday; pencere kapanana kadar tek bir ornek
    olarak tutulur (tum pencereyi bellekte biriktirmez)."""

    frame: np.ndarray
    frame_id: int
    timestamp_sec: float
    net_change_score: float
    motion_bbox: Optional[Tuple[int, int, int, int]]


class AdaptiveFrameSampler:
    """OpenCV tabanli, CPU uzerinde ultra hizli calisan Uyarlanabilir Kare Ornekleyici.

    Gereksiz/durgun kareleri buyuk oranda eleyerek yalnizca anlamli degisim
    iceren kareleri VLM katmanina iletir. Katman tamamen CPU uzerinde calisir;
    GPU yuku sifirdir. Olay kumelemesi YAPMAZ (bkz. modul docstring'i).
    """

    def __init__(
        self,
        min_change_threshold: float = 0.001,
        blur_kernel_size: Tuple[int, int] = (21, 21),
        history_window: int = 30,
        evidence_output_dir: str = _DEFAULT_EVIDENCE_OUTPUT_DIR,
        temporal_vote_window: int = 1,
        temporal_vote_min_count: int = 1,
        max_temporal_gap_sec: float = 15.0,
    ) -> None:
        """AdaptiveFrameSampler'i esik parametreleriyle baslatir.

        Args:
            min_change_threshold: Bir karenin Kanit Karesi sayilmasi icin
                gereken, gurultu tabani dusulmus minimum piksel degisim orani.
            blur_kernel_size: Gurultu gidermek icin uygulanan Gauss bulaniklastirma
                cekirdek boyutu (tek sayilardan olusan (genislik, yukseklik) ikilisi).
            history_window: Dinamik gurultu tabanini (medyan) hesaplamak icin
                tutulan son degisim orani sayisi.
            evidence_output_dir: Her Kanit Karesinin `.jpg` olarak diske
                yazilacagi klasor (UI ve denetim/log amacli kalicilik icin).
                ONEMLI: Kanit Karesi sayisinda sabit bir ust sinir YOKTUR
                (eski `max_evidence_buffer` kaldirildi); uzun/surekli bir
                olay ne kadar sürerse sürsün hicbir Kanit Karesi burada
                sessizce atlanmaz.
            temporal_vote_window: Bir karenin candidate olarak onaylanip
                onaylanmayacagina karar verirken dikkate alinan, o anki karar
                dahil, en son kac esik-testi sonucunun tutulacagi (temporal
                voting penceresi). Varsayilan `1` ile pencere yalnizca o anki
                karari icerir; bu, mevcut (voting oncesi) davranisla BIREBIR
                AYNIDIR. `min_change_threshold` formulunu veya gurultu
                tabanini DEGISTIRMEZ; yalnizca esigi gecen bir karenin GERCEK
                candidate sayilip sayilmayacagina ek bir "sureklilik" filtresi
                uygular (bkz. `temporal_vote_min_count`). NOT: Bu, esigi
                GECEN bir kareyi asla "kumeleme" amaciyla elemez; yalnizca
                izole/tek karelik gurultuyu (kamera titremesi, sikistirma
                artefakti) esik testinin BIR PARCASI olarak filtreler.
            temporal_vote_min_count: `temporal_vote_window` penceresi
                icinde, bir karenin candidate olarak onaylanmasi icin gereken
                minimum "supheli" (esigi gecen) karar sayisi. Varsayilan `1`
                ile tek bir supheli karar yeterlidir; bu da mevcut davranisla
                BIREBIR AYNIDIR.
            max_temporal_gap_sec: Son evidence karesinden (esik-gecen VEYA
                coverage) bu yana izin verilen azami sessizlik suresi
                (saniye). Bu sure asilirsa, o ana kadar degerlendirilen
                esik-alti adaylar arasindan `net_change_score`'u en yuksek
                olan kare `selection_reason="temporal_coverage"` ile
                evidence listesine eklenir (bkz. modul docstring'i). Bu bir
                OLAY KUMELEMESI DEGILDIR; yalnizca uzun zamansal kor
                noktalari onleyen bir guvenlik agidir.

        Raises:
            ValueError: `temporal_vote_window` veya `temporal_vote_min_count`
                1'den kucukse, `temporal_vote_min_count`,
                `temporal_vote_window`den buyukse, ya da
                `max_temporal_gap_sec` 0'dan kucuk/esitse.
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
        if max_temporal_gap_sec <= 0:
            raise ValueError(
                f"max_temporal_gap_sec 0'dan buyuk olmalidir, verilen: {max_temporal_gap_sec}"
            )

        self.min_change_threshold = min_change_threshold
        self.blur_kernel_size = blur_kernel_size
        self.history_window = history_window
        self.evidence_output_dir = Path(evidence_output_dir)
        self.temporal_vote_window = temporal_vote_window
        self.temporal_vote_min_count = temporal_vote_min_count
        self.max_temporal_gap_sec = max_temporal_gap_sec

        self.prev_gray: np.ndarray | None = None
        self.noise_floor_history: List[float] = []
        self.last_run_stats: Optional[SamplerRunStats] = None
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

    def process_video(self, video_path: str, sample_fps: int = 5) -> List[EvidenceFrame]:
        """Videoyu okur, kare farklarini hesaplar ve suzulmus Kanit Karelerini dondurur.

        Esik-gecmis HICBIR kare burada elenmez: sabit bir ust sinir yoktur
        (eski `max_evidence_buffer` kaldirildi), kumeleme/deduplication/liste
        kesme YAPILMAZ. Her kare tam olarak BIR kez okunur ve BIR kez JPEG'e
        kodlanir (bkz. `_encode_frame_jpeg`); VLM katmani ayni baytlari
        yeniden kullanir (bkz. `src/sampler/payload_builder.py`).

        Ayrica, son evidence karesinden (esik-gecen VEYA coverage) bu yana
        `max_temporal_gap_sec` asilirsa, o pencerede degerlendirilen
        esik-alti adaylar arasindan `net_change_score`'u EN YUKSEK olan kare
        `selection_reason="temporal_coverage"` ile evidence listesine
        eklenir (bkz. modul docstring'i, `_PendingCoverageCandidate`); bu
        kumeleme degildir, yalnizca zamansal kor noktalari onler.

        Args:
            video_path: `.mp4` dosya yolu veya RTSP URI'si.
            sample_fps: Videonun kac saniyede bir kare kontrol edilecegini
                belirleyen ornekleme hizi (native FPS'ten dusuk olmalidir).

        Returns:
            Zaman sirali (kronolojik), kayipsiz `EvidenceFrame` listesi.

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

        # Zamansal kapsama durumu: `last_evidence_timestamp`, gap sayacinin
        # baslangic referansidir (video basindan itibaren izlenir, boylece
        # video BASLARKEN gecen sessiz sure de kapsama tetigine dahildir).
        # `pending_best`, mevcut pencerede su ana kadar goeruelen EN IYI
        # esik-alti adaydir (TUM pencere bellekte tutulmaz, yalnizca bu tek
        # ornek - bkz. `_PendingCoverageCandidate`).
        last_evidence_timestamp: Optional[float] = None
        pending_best: Optional[_PendingCoverageCandidate] = None

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
                    last_evidence_timestamp = timestamp_sec

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
                        # Sabit bir ust sinir YOKTUR: video ne kadar uzun/olay ne kadar
                        # surekli olursa olsun, hicbir Kanit Karesi burada sessizce
                        # atlanmaz. Kumeleme artik VLM katmaninda yapilir.
                        motion_bbox = self._bbox_from_mask(thresh)
                        evidence_frames.append(
                            self._build_evidence_frame(
                                frame,
                                frame_id,
                                timestamp_sec,
                                net_change_score,
                                motion_bbox,
                                selection_reason=_SELECTION_REASON_THRESHOLD,
                            )
                        )
                        # Esik-gecen bir kare eklendiginde gap sayaci ve pencere
                        # sifirlanir: bu kare zaten o anki en guncel "gorulme"dir.
                        last_evidence_timestamp = timestamp_sec
                        pending_best = None
                    else:
                        # Esik-alti aday: zamansal kapsama penceresinin bir
                        # parcasi olarak degerlendirilir (kumeleme DEGIL -
                        # yalnizca "bu pencerede en yuksek skorlu kare hangisi"
                        # sorusuna cevap arar). Esitlikte (ör. tamamen durgun
                        # bir sahnede tum adaylar net_change_score=0.0) EN
                        # GUNCEL aday kazanir (`>=`); aksi halde pending_best
                        # pencerenin basindaki eski bir karede "sikisip kalir"
                        # ve coverage karesinin zaman damgasi asiri geriye
                        # sarkar - bu da bir sonraki pencerede gap sayacinin
                        # hemen tekrar dolup ARKA ARKAYA, gereksiz bir ikinci
                        # coverage karesi uretmesine yol acar (duplicate-benzeri
                        # yigilma). Guncel-aday tercihi bunu engeller ve
                        # coverage karelerini gercekten `max_temporal_gap_sec`
                        # araliklarla, duzgun dagitir.
                        motion_bbox = self._bbox_from_mask(thresh)
                        if pending_best is None or net_change_score >= pending_best.net_change_score:
                            pending_best = _PendingCoverageCandidate(
                                frame=frame.copy(),
                                frame_id=frame_id,
                                timestamp_sec=timestamp_sec,
                                net_change_score=net_change_score,
                                motion_bbox=motion_bbox,
                            )

                        if (
                            last_evidence_timestamp is not None
                            and (timestamp_sec - last_evidence_timestamp) >= self.max_temporal_gap_sec
                        ):
                            evidence_frames.append(
                                self._build_evidence_frame(
                                    pending_best.frame,
                                    pending_best.frame_id,
                                    pending_best.timestamp_sec,
                                    pending_best.net_change_score,
                                    pending_best.motion_bbox,
                                    selection_reason=_SELECTION_REASON_COVERAGE,
                                )
                            )
                            # Gap sayaci, TETIKLEYEN ornegin degil, EKLENEN
                            # coverage karesinin KENDI zaman damgasindan devam
                            # eder (o kare artik "en son gorulen" evidence'tir).
                            last_evidence_timestamp = pending_best.timestamp_sec
                            pending_best = None

                self.prev_gray = curr_gray
                frame_id += 1
        finally:
            cap.release()

        # Video sonu: pencere kapanmadan (bir sonraki ornek gelmeden) video
        # bitmis olabilir. Su ana kadarki en iyi aday varsa VE gap zaten
        # asilmissa, kayipsizlik icin GUVENLI sekilde son bir coverage
        # karesi olarak eklenir; asilmadiysa (video coverage'i gerektirecek
        # kadar uzun surmedi) hicbir seye elenmez/hata verilmez - sessizce
        # birakilir (bu veri kaybi degildir, esik zaten hic gecilmedi).
        if (
            pending_best is not None
            and last_evidence_timestamp is not None
            and (pending_best.timestamp_sec - last_evidence_timestamp) >= self.max_temporal_gap_sec
        ):
            evidence_frames.append(
                self._build_evidence_frame(
                    pending_best.frame,
                    pending_best.frame_id,
                    pending_best.timestamp_sec,
                    pending_best.net_change_score,
                    pending_best.motion_bbox,
                    selection_reason=_SELECTION_REASON_COVERAGE,
                )
            )

        if not evidence_frames:
            if first_frame_raw is None:
                raise ValueError(f"Video kaynagindan hic kare okunamadi: {video_path}")

            logger.warning(
                "Esigi gecen Kanit Karesi bulunamadi; sistemin cokmemesi icin frame 0 "
                "varsayilan Kanit Karesi olarak kabul ediliyor (fallback)."
            )
            fallback = self._build_evidence_frame(
                first_frame_raw,
                frame_id=0,
                timestamp_sec=first_frame_timestamp,
                change_score=0.0,
                selection_reason=_SELECTION_REASON_FALLBACK,
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
        selection_reason: str = _SELECTION_REASON_THRESHOLD,
    ) -> EvidenceFrame:
        """Bir kareden `EvidenceFrame` uretir ve JPEG/base64'e cevirir (tek seferlik encode).

        Args:
            frame: BGR formatinda ham video karesi.
            frame_id: Karenin video icindeki sirasi.
            timestamp_sec: Karenin saniye cinsinden zaman damgasi.
            change_score: Hesaplanan (gurultu-tabani-dusulmus) evidence/degisim skoru.
            motion_bbox: Bu kareyi ureten hareket maskesinin sinirlayici kutusu
                (`_bbox_from_mask`); yoksa (ör. fallback kare) `None`.
            selection_reason: Bu karenin evidence listesine GIRME NEDENI
                (`"threshold_exceeded"` | `"temporal_coverage"` |
                `"fallback"`) - konumsal bir rol DEGILDIR, bkz.
                `EvidenceFrame.selection_reason`.

        Returns:
            `image_bytes`/`base64_image` doldurulmus, `saved_path=None` olan `EvidenceFrame`.
        """
        image_bytes = self._encode_frame_jpeg(frame)
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        minutes, seconds = divmod(int(timestamp_sec), 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        return EvidenceFrame(
            evidence_id=f"ev{frame_id}",
            frame_id=frame_id,
            timestamp_sec=round(timestamp_sec, 2),
            timestamp_str=time_str,
            change_score=round(change_score, 4),
            image_bytes=image_bytes,
            base64_image=f"data:image/jpeg;base64,{base64_str}",
            image_shape=frame.shape,
            saved_path=None,
            motion_bbox=motion_bbox,
            selection_reason=selection_reason,
        )


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
        temporal_vote_window=config.temporal_vote_window,
        temporal_vote_min_count=config.temporal_vote_min_count,
        max_temporal_gap_sec=config.max_temporal_gap_sec,
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
    demo_stats = demo_sampler.last_run_stats

    print(f"Video: {demo_video_path}")
    print(f"Taranan ham kare: {demo_stats.total_frames_scanned}")
    print(f"Degerlendirilen ornek kare: {demo_stats.sampled_frames_evaluated}")
    print(f"Uretilen Kanit Karesi: {demo_stats.evidence_frame_count}")
    print(
        f"Elenen kare sayisi: {demo_stats.eliminated_frame_count} "
        f"(%{demo_stats.eliminated_ratio_pct:.1f} eleme orani)"
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
    }
    demo_output_path.write_text(
        json.dumps(demo_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Kanit verisi yazildi: {demo_output_path}")
