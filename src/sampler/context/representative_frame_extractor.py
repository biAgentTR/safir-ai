from __future__ import annotations

import base64
import logging
from typing import List, Optional

import cv2

from src.sampler.schema import EvidenceFrame, RepresentativeFrame

logger = logging.getLogger(__name__)

_JPEG_QUALITY = 85


class RepresentativeFrameExtractor:git add .
    """Zirve kare etrafinda pre-event/post-event temsili kareleri seek ile cikarir."""

    def __init__(self, pre_event_sec: float = 3.0, post_event_sec: float = 3.0) -> None:
        """RepresentativeFrameExtractor'i zaman penceresi parametreleriyle baslatir.

        Args:
            pre_event_sec: Zirve karenin kac saniye oncesinden kare alinacagi.
            post_event_sec: Zirve karenin kac saniye sonrasindan kare alinacagi.

        Raises:
            ValueError: `pre_event_sec` veya `post_event_sec` negatifse.
        """
        if pre_event_sec < 0 or post_event_sec < 0:
            raise ValueError("pre_event_sec ve post_event_sec negatif olamaz.")
        self.pre_event_sec = pre_event_sec
        self.post_event_sec = post_event_sec

    def extract(self, video_path: str, peak_frame: EvidenceFrame) -> List[RepresentativeFrame]:
        """Zirve kare icin pre/peak/post temsili karelerin zaman sirali listesini uretir.

        Pre/post kareler video sinirlarinin disina tasarsa gecerli araliga
        (`0`/`total_frames - 1`) kenetlenir; kenetleme zirve kareyle ayni
        kareye denk gelirse (video basinda/sonunda olay) o kare tekrar
        eklenmeyip atlanir. Tek bir pre veya post karenin okunmasi basarisiz
        olursa hata firlatilmaz, yalnizca loglanip atlanir; kalan gecerli
        karelerle devam edilir.

        Args:
            video_path: Kaynak video dosyasinin yolu.
            peak_frame: Bu olay grubunun zirve `EvidenceFrame`i.

        Returns:
            Zaman sirali (`timestamp_sec` artan) `RepresentativeFrame` listesi;
            en az zirve karesini icerir, en fazla 3 kare (pre + peak + post).

        Raises:
            ValueError: Video acilamazsa.
            RuntimeError: Video FPS degeri gecersizse.
        """
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Video acilamadi: {video_path}")

            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if native_fps is None or native_fps <= 0:
                raise RuntimeError(f"Video FPS degeri gecersiz: {native_fps}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pre_frames = int(round(self.pre_event_sec * native_fps))
            post_frames = int(round(self.post_event_sec * native_fps))

            representative_frames: List[RepresentativeFrame] = [
                RepresentativeFrame(
                    label="peak",
                    timestamp_sec=peak_frame.timestamp_sec,
                    timestamp_str=peak_frame.timestamp_str,
                    base64_image=peak_frame.base64_image,
                )
            ]

            pre_frame_id = max(0, peak_frame.frame_id - pre_frames)
            if pre_frame_id < peak_frame.frame_id:
                pre_rep = self._read_frame_at(cap, pre_frame_id, native_fps, "pre-event")
                if pre_rep is not None:
                    representative_frames.append(pre_rep)

            if total_frames > 0:
                post_frame_id = min(peak_frame.frame_id + post_frames, total_frames - 1)
            else:
                post_frame_id = peak_frame.frame_id + post_frames
            if post_frame_id > peak_frame.frame_id:
                post_rep = self._read_frame_at(cap, post_frame_id, native_fps, "post-event")
                if post_rep is not None:
                    representative_frames.append(post_rep)

            representative_frames.sort(key=lambda rf: rf.timestamp_sec)
            return representative_frames
        finally:
            cap.release()

    @staticmethod
    def _read_frame_at(
        cap: cv2.VideoCapture, frame_id: int, native_fps: float, label: str
    ) -> Optional[RepresentativeFrame]:
        """Belirtilen kare indeksine seek edip okur; basarisizlikta `None` doner (hata firlatmaz)."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.warning("%s karesi (frame_id=%d) okunamadi, atlaniyor.", label, frame_id)
            return None

        encode_ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not encode_ok:
            logger.warning("%s karesi (frame_id=%d) JPEG'e kodlanamadi, atlaniyor.", label, frame_id)
            return None

        timestamp_sec = frame_id / native_fps
        minutes, seconds = divmod(int(timestamp_sec), 60)
        timestamp_str = f"{minutes:02d}:{seconds:02d}"
        base64_image = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return RepresentativeFrame(
            label=label,
            timestamp_sec=timestamp_sec,
            timestamp_str=timestamp_str,
            base64_image=base64_image,
        )