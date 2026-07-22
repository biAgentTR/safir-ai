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
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from src.utils.config_loader import SamplerConfig

logger = logging.getLogger(__name__)


@dataclass
class EvidenceFrame:
    """VLM'e gonderilecek Kanit Karesi veri modeli."""

    frame_id: int
    timestamp_sec: float
    timestamp_str: str
    change_score: float
    image_bytes: bytes
    base64_image: str
    image_shape: Tuple[int, int, int]


@dataclass
class EventCluster:
    """Arka arkaya gerceklesen degisim karelerinin kumelenmis olay grubu."""

    event_id: int
    start_time: float
    end_time: float
    peak_frame: EvidenceFrame
    total_candidate_frames: int


class AdaptiveFrameSampler:
    """OpenCV tabanli, CPU uzerinde ultra hizli calisan Uyarlanabilir Kare Ornekleyici.

    Gereksiz/durgun kareleri buyuk oranda eleyerek yalnizca anlamli degisim
    iceren kareleri VLM katmanina iletir. Katman tamamen CPU uzerinde calisir;
    GPU yuku sifirdir.
    """

    def __init__(
        self,
        min_change_threshold: float = 0.011,
        blur_kernel_size: Tuple[int, int] = (21, 21),
        history_window: int = 30,
        min_event_interval_sec: float = 2.0,
    ) -> None:
        """AdaptiveFrameSampler'i esik ve pencere parametreleriyle baslatir.

        Args:
            min_change_threshold: Bir karenin Kanit Karesi sayilmasi icin
                gereken, gurultu tabani dusulmus minimum piksel degisim orani.
            blur_kernel_size: Gurultu gidermek icin uygulanan Gauss bulaniklastirma
                cekirdek boyutu (tek sayilardan olusan (genislik, yukseklik) ikilisi).
            history_window: Dinamik gurultu tabanini (medyan) hesaplamak icin
                tutulan son degisim orani sayisi.
            min_event_interval_sec: Ardisik Kanit Karelerini ayni Olay Grubuna
                dahil etmek icin izin verilen maksimum zaman farki (saniye).
        """
        self.min_change_threshold = min_change_threshold
        self.blur_kernel_size = blur_kernel_size
        self.history_window = history_window
        self.min_event_interval_sec = min_event_interval_sec

        self.prev_gray: np.ndarray | None = None
        self.noise_floor_history: List[float] = []

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

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Kareyi VLM OpenAI API uyumlu Base64 string'e donusturur.

        Args:
            frame: BGR formatinda video karesi.

        Returns:
            JPEG-kodlu, base64 metne cevrilmis kare.

        Raises:
            RuntimeError: Kare JPEG formatina kodlanamazsa.
        """
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("Kare JPEG formatina kodlanamadi.")
        return base64.b64encode(buffer).decode("utf-8")

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

        evidence_frames: List[EvidenceFrame] = []
        frame_id = 0
        sampled_frame_count = 0

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

                if self.prev_gray is not None:
                    frame_diff = cv2.absdiff(curr_gray, self.prev_gray)
                    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                    change_ratio = np.sum(thresh > 0) / float(thresh.size)

                    self.noise_floor_history.append(change_ratio)
                    if len(self.noise_floor_history) > self.history_window:
                        self.noise_floor_history.pop(0)

                    adaptive_noise_floor = np.median(self.noise_floor_history)
                    net_change_score = max(0.0, change_ratio - adaptive_noise_floor)

                    if net_change_score >= self.min_change_threshold:
                        base64_str = self._frame_to_base64(frame)
                        minutes, seconds = divmod(int(timestamp_sec), 60)
                        time_str = f"{minutes:02d}:{seconds:02d}"

                        evidence = EvidenceFrame(
                            frame_id=frame_id,
                            timestamp_sec=round(timestamp_sec, 2),
                            timestamp_str=time_str,
                            change_score=round(net_change_score, 4),
                            image_bytes=cv2.imencode(".jpg", frame)[1].tobytes(),
                            base64_image=f"data:image/jpeg;base64,{base64_str}",
                            image_shape=frame.shape,
                        )
                        evidence_frames.append(evidence)

                self.prev_gray = curr_gray
                frame_id += 1
        finally:
            cap.release()

        elapsed_sec = time.perf_counter() - started_at
        eliminated = sampled_frame_count - len(evidence_frames)
        eliminated_ratio = (100.0 * eliminated / sampled_frame_count) if sampled_frame_count else 0.0
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

    def cluster_events(self, evidence_frames: List[EvidenceFrame]) -> List[EventCluster]:
        """Suzulen kareleri zaman araligina gore kumeleyip zirve karelerini secer.

        Ardisik Kanit Kareleri arasindaki zaman farki `min_event_interval_sec`
        degerini asmadigi surece ayni Olay Grubuna dahil edilir. Her grup icin
        en yuksek degisim skoruna sahip kare (peak frame) VLM'e gonderilecek
        temsilci olarak secilir; boylece VLM'e gereksiz tekrar giden kareler
        onlenmis olur.

        Args:
            evidence_frames: `process_video` tarafindan uretilen, zaman sirali
                Kanit Kareleri listesi.

        Returns:
            Zaman sirali `EventCluster` listesi. Girdi bossa bos liste doner.
        """
        if not evidence_frames:
            return []

        clusters: List[EventCluster] = []
        current_group: List[EvidenceFrame] = [evidence_frames[0]]

        for ef in evidence_frames[1:]:
            if ef.timestamp_sec - current_group[-1].timestamp_sec <= self.min_event_interval_sec:
                current_group.append(ef)
            else:
                clusters.append(self._close_group(current_group, event_id=len(clusters) + 1))
                current_group = [ef]

        if current_group:
            clusters.append(self._close_group(current_group, event_id=len(clusters) + 1))

        logger.info(
            "EventCluster tamamlandi: %d Kanit Karesi -> %d Olay Grubu",
            len(evidence_frames),
            len(clusters),
        )
        return clusters

    @staticmethod
    def _close_group(group: List[EvidenceFrame], event_id: int) -> EventCluster:
        """Bir Kanit Karesi grubunu kapatip zirve karesini secerek `EventCluster` uretir.

        Args:
            group: Ayni zaman araligina dusen Kanit Kareleri.
            event_id: Bu gruba atanacak olay kimligi.

        Returns:
            Grubun ozetini tasiyan `EventCluster`.
        """
        peak = max(group, key=lambda ef: ef.change_score)
        return EventCluster(
            event_id=event_id,
            start_time=group[0].timestamp_sec,
            end_time=group[-1].timestamp_sec,
            peak_frame=peak,
            total_candidate_frames=len(group),
        )


def sampler_from_config(config: SamplerConfig) -> AdaptiveFrameSampler:
    """`configs/config.yaml` icindeki `sampler` blogundan bir `AdaptiveFrameSampler` uretir.

    Args:
        config: Dogrulanmis `SamplerConfig` nesnesi.

    Returns:
        Config parametreleriyle ilklendirilmis `AdaptiveFrameSampler`.
    """
    return AdaptiveFrameSampler(
        min_change_threshold=config.min_change_threshold,
        blur_kernel_size=tuple(config.blur_kernel_size),
        history_window=config.history_window,
        min_event_interval_sec=config.min_event_interval_sec,
    )
