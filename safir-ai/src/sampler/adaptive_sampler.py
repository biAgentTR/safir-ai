"""02 - Hafif Suzgec Katmani: OpenCV tabanli Adaptive Frame Sampler (CPU ONLY).

Bu modul, ham video akisini agir bir tespit modeli (YOLO/ByteTrack) kullanmadan
hareket ve zamansal esikleme ile suzer. Sakin bolgelerde dusuk frekansta
(orn. 10 sn/kare), hareket/degisim anlarinda ise aktif frekansta (orn. 1 fps)
kare yakalayarak "Evidence Frames" (Kanit Kareleri) listesi uretir. Katman
tamamen CPU uzerinde calisir; GPU yuku sifirdir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import cv2
import numpy as np

from src.utils.config_loader import SamplerConfig

logger = logging.getLogger(__name__)


@dataclass
class EvidenceFrame:
    """Suzgecten gecmis, VLM katmanina gonderilmeye deger bulunan tek bir kare."""

    frame: np.ndarray
    timestamp: float
    frame_index: int
    motion_score: float
    trigger: str  # "motion" | "scene_change" | "idle_keyframe"


@dataclass
class _SamplerState:
    """Sampler'in kareler arasinda tasidigi ic durum."""

    prev_gray: Optional[np.ndarray] = None
    last_capture_ts: float = 0.0
    frame_index: int = 0
    evidence_frames: List[EvidenceFrame] = field(default_factory=list)


class AdaptiveSampler:
    """Hareket ve zamansal eskikleme ile CPU uzerinde kare suzen ornekleyici.

    Motion detection icin ardisik gri-tonlama kareler arasindaki mutlak fark
    kullanilir; sahne degisimi icin gri-tonlama histogram korelasyonu esik
    olarak alinir. Her iki tetikleyici de config uzerinden ayarlanabilir.
    """

    def __init__(self, config: SamplerConfig) -> None:
        """AdaptiveSampler'i verilen konfigurasyon ile baslatir.

        Args:
            config: `configs/config.yaml` icindeki `sampler` blogundan uretilen
                dogrulanmis `SamplerConfig` nesnesi.
        """
        self._config = config
        self._state = _SamplerState()

    def reset(self) -> None:
        """Ic durumu (onceki kare, zaman damgasi, tampon) sifirlar."""
        self._state = _SamplerState()

    def process_video(self, source: str) -> Iterator[EvidenceFrame]:
        """Bir video dosyasi veya RTSP akisini okuyup Evidence Frame uretir.

        Args:
            source: `.mp4` dosya yolu veya RTSP URI'si.

        Yields:
            Suzgecten gecen her kare icin bir `EvidenceFrame`.

        Raises:
            IOError: Video kaynagi acilamazsa.
        """
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise IOError(f"Video kaynagi acilamadi: {source}")

        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            frame_interval = 1.0 / fps
            wall_clock = 0.0

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                evidence = self._ingest_frame(frame, timestamp=wall_clock)
                if evidence is not None:
                    yield evidence

                wall_clock += frame_interval
        finally:
            capture.release()

    def _ingest_frame(self, frame: np.ndarray, timestamp: float) -> Optional[EvidenceFrame]:
        """Tek bir ham kareyi degerlendirip gerekirse Evidence Frame uretir.

        Args:
            frame: BGR formatinda ham video karesi.
            timestamp: Karenin akis icindeki saniye cinsinden zaman damgasi.

        Returns:
            Kare "anlamli" bulunduysa `EvidenceFrame`, aksi halde `None`.
        """
        state = self._state
        state.frame_index += 1

        small = cv2.resize(
            frame,
            (self._config.resize_width, self._config.resize_width * frame.shape[0] // frame.shape[1]),
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if state.prev_gray is None:
            state.prev_gray = gray
            state.last_capture_ts = timestamp
            return self._emit(frame, timestamp, motion_score=0.0, trigger="idle_keyframe")

        motion_score = self._compute_motion_score(state.prev_gray, gray)
        scene_changed = self._detect_scene_change(state.prev_gray, gray)
        state.prev_gray = gray

        elapsed_since_capture = timestamp - state.last_capture_ts
        is_motion = motion_score > self._config.motion_threshold

        if is_motion or scene_changed:
            active_interval = 1.0 / self._config.active_fps
            if elapsed_since_capture >= active_interval:
                trigger = "scene_change" if scene_changed else "motion"
                return self._emit(frame, timestamp, motion_score, trigger)
            return None

        if elapsed_since_capture >= self._config.idle_interval_sec:
            return self._emit(frame, timestamp, motion_score, trigger="idle_keyframe")

        return None

    def _compute_motion_score(self, prev_gray: np.ndarray, gray: np.ndarray) -> float:
        """Ardisik iki gri-tonlama kare arasindaki hareket yuzdesini hesaplar.

        Args:
            prev_gray: Bir onceki gri-tonlama kare.
            gray: Mevcut gri-tonlama kare.

        Returns:
            Gurultu tabani sonrasi degisen piksellerin yuzdesi (0-100).
        """
        diff = cv2.absdiff(prev_gray, gray)
        _, thresholded = cv2.threshold(
            diff, self._config.noise_floor, 255, cv2.THRESH_BINARY
        )
        changed_pixels = int(np.count_nonzero(thresholded))
        total_pixels = thresholded.size
        return 100.0 * changed_pixels / total_pixels if total_pixels else 0.0

    def _detect_scene_change(self, prev_gray: np.ndarray, gray: np.ndarray) -> bool:
        """Histogram korelasyonuna dayanarak ani sahne degisimini tespit eder.

        Args:
            prev_gray: Bir onceki gri-tonlama kare.
            gray: Mevcut gri-tonlama kare.

        Returns:
            Korelasyon, config esiginin altindaysa `True`.
        """
        hist_prev = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
        hist_curr = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist_prev, hist_prev)
        cv2.normalize(hist_curr, hist_curr)
        correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
        return correlation < self._config.scene_change_threshold

    def _emit(
        self, frame: np.ndarray, timestamp: float, motion_score: float, trigger: str
    ) -> EvidenceFrame:
        """Bir Evidence Frame olusturur, tamponlar ve ust sinira gore budar.

        Args:
            frame: Kaydedilecek ham (orijinal cozunurlukte) kare.
            timestamp: Karenin zaman damgasi.
            motion_score: Hesaplanan hareket skoru.
            trigger: Karenin secilme nedeni.

        Returns:
            Olusturulan `EvidenceFrame` nesnesi.
        """
        state = self._state
        state.last_capture_ts = timestamp

        evidence = EvidenceFrame(
            frame=frame,
            timestamp=timestamp,
            frame_index=state.frame_index,
            motion_score=motion_score,
            trigger=trigger,
        )
        state.evidence_frames.append(evidence)
        if len(state.evidence_frames) > self._config.max_evidence_buffer:
            state.evidence_frames.pop(0)

        logger.debug(
            "Evidence frame uretildi: idx=%d ts=%.2f trigger=%s motion=%.2f",
            evidence.frame_index,
            evidence.timestamp,
            evidence.trigger,
            evidence.motion_score,
        )
        return evidence

    @property
    def evidence_buffer(self) -> List[EvidenceFrame]:
        """Su ana kadar biriktirilen Evidence Frame tamponunu dondurur."""
        return list(self._state.evidence_frames)
