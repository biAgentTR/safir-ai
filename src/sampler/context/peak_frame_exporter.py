

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from src.sampler.schema import EventCluster, EvidenceFrame

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = "outputs/sampler"
_JPEG_QUALITY = 85
_MAX_NEAREST_FRAME_PROBE = 5


class PeakFrameExporter:
    """Her `EventCluster` icin pre_peak/peak/post_peak JPEG'lerini + metadata.json'i diske yazar."""

    def __init__(self, pre_peak_offset_sec: float = 2.0, post_peak_offset_sec: float = 2.0) -> None:
        """PeakFrameExporter'i zirve-oncesi/sonrasi zaman penceresiyle baslatir.

        Args:
            pre_peak_offset_sec: Zirve karenin kac saniye oncesinden kare alinacagi.
            post_peak_offset_sec: Zirve karenin kac saniye sonrasindan kare alinacagi.

        Raises:
            ValueError: `pre_peak_offset_sec` veya `post_peak_offset_sec` negatifse.
        """
        if pre_peak_offset_sec < 0 or post_peak_offset_sec < 0:
            raise ValueError("pre_peak_offset_sec ve post_peak_offset_sec negatif olamaz.")
        self.pre_peak_offset_sec = pre_peak_offset_sec
        self.post_peak_offset_sec = post_peak_offset_sec

    def export(
        self, video_path: str, clusters: List[EventCluster], output_dir: str = _DEFAULT_OUTPUT_DIR
    ) -> List[str]:
        """Her Olay Grubu icin pre_peak/peak/post_peak `.jpg` + `metadata.json` yazar.

        Video sinirlarinda (baslangic/bitis) `pre_peak_time`/`post_peak_time`
        `[0, video_duration]` araligina kenetlenir. Istenen tam zaman
        damgasinda kare okunamazsa (ör. bozuk kare) en yakin gecerli kare
        denenir; o da basarisiz olursa zirve karenin kendisiyle doldurulur —
        boylece her olay icin HER ZAMAN tam olarak birer `pre_peak.jpg`,
        `peak.jpg`, `post_peak.jpg` uretilir ve hicbir hata firlatilmaz.

        Zirve kare videonun basina/sonuna yeterince yakinsa (ör. olay video
        bitmeden hemen once zirve yapiyorsa), kenetleme nedeniyle
        `pre_peak.jpg`/`post_peak.jpg` icin istenen `pre_peak_offset_sec`/
        `post_peak_offset_sec` kadar zaman KULLANILAMAZ; bu durumda uretilen
        kare, zirve karesine gorsel olarak yakin/ozdes olabilir (videoda o
        andan once/sonra baska goruntu YOKTUR — bu bir hata degildir).
        `metadata.json`daki `pre_peak_window_limited`/`post_peak_window_limited`
        (bool) ve `pre_peak_available_window_sec`/`post_peak_available_window_sec`
        alanlari bu durumu acikca isaretler; tuketiciler bu alanlara bakarak
        "post_peak, peak ile ayni gorunuyor" durumunun bir kenetlenme mi
        yoksa beklenmedik bir davranis mi oldugunu ayirt edebilir.

        Args:
            video_path: Kaynak video dosyasinin yolu.
            clusters: `cluster_events` ciktisi Olay Gruplari.
            output_dir: `event_XXXX/` alt klasorlerinin olusturulacagi kok dizin.

        Returns:
            Yazilan her `event_XXXX` klasorunun yolu (string listesi).

        Raises:
            ValueError: Video acilamazsa.
            RuntimeError: Video FPS degeri gecersizse.
        """
        if not clusters:
            return []

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Video acilamadi: {video_path}")

            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if native_fps is None or native_fps <= 0:
                raise RuntimeError(f"Video FPS degeri gecersiz: {native_fps}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = (total_frames / native_fps) if total_frames > 0 else None

            event_dirs: List[str] = []
            for cluster in clusters:
                event_dirs.append(
                    self._export_one_event(cap, cluster, native_fps, total_frames, video_duration, output_dir)
                )
            return event_dirs
        finally:
            cap.release()

    def _export_one_event(
        self,
        cap: cv2.VideoCapture,
        cluster: EventCluster,
        native_fps: float,
        total_frames: int,
        video_duration: Optional[float],
        output_dir: str,
    ) -> str:
        """Tek bir Olay Grubu icin pre_peak/peak/post_peak/metadata.json yazar."""
        peak = cluster.peak_frame
        peak_time = peak.timestamp_sec

        pre_peak_time_wanted = peak_time - self.pre_peak_offset_sec
        pre_peak_time = max(0.0, pre_peak_time_wanted)
        pre_peak_window_limited = pre_peak_time_wanted < 0.0

        post_peak_time_wanted = peak_time + self.post_peak_offset_sec
        if video_duration is not None:
            post_peak_time = min(video_duration, post_peak_time_wanted)
            post_peak_window_limited = post_peak_time_wanted > video_duration
        else:
            post_peak_time = post_peak_time_wanted
            post_peak_window_limited = False

        pre_bytes, pre_actual_time = self._read_jpeg_near(cap, pre_peak_time, native_fps, total_frames, peak)
        post_bytes, post_actual_time = self._read_jpeg_near(cap, post_peak_time, native_fps, total_frames, peak)

        event_dir = Path(output_dir) / f"event_{cluster.event_id:04d}"
        event_dir.mkdir(parents=True, exist_ok=True)

        (event_dir / "pre_peak.jpg").write_bytes(pre_bytes)
        (event_dir / "peak.jpg").write_bytes(peak.image_bytes)
        (event_dir / "post_peak.jpg").write_bytes(post_bytes)

        metadata = {
            "event_id": cluster.event_id,
            "pre_peak_offset_sec": self.pre_peak_offset_sec,
            "post_peak_offset_sec": self.post_peak_offset_sec,
            "pre_peak_time_requested": round(pre_peak_time, 2),
            "pre_peak_time_actual": round(pre_actual_time, 2),
            "peak_time": round(peak_time, 2),
            "post_peak_time_requested": round(post_peak_time, 2),
            "post_peak_time_actual": round(post_actual_time, 2),
            "peak_change_score": peak.change_score,
            "total_candidate_frames": cluster.total_candidate_frames,
            # Video sinirina cakildigi icin istenen tam pre/post_peak_offset_sec
            # kadar zaman kullanilamadiginda True: pre_peak.jpg/post_peak.jpg,
            # peak.jpg'e (video basi/sonuna cok yakin bir olay oldugu icin)
            # gorsel olarak yakin/ozdes olabilir - bu bir hata degil, videonun
            # o olaydan once/sonra yeterli goruntu icermemesidir.
            "pre_peak_window_limited": pre_peak_window_limited,
            "post_peak_window_limited": post_peak_window_limited,
            "pre_peak_available_window_sec": round(peak_time - pre_actual_time, 2),
            "post_peak_available_window_sec": round(post_actual_time - peak_time, 2),
        }
        (event_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return str(event_dir)

    def _read_jpeg_near(
        self,
        cap: cv2.VideoCapture,
        target_time_sec: float,
        native_fps: float,
        total_frames: int,
        peak: EvidenceFrame,
    ) -> Tuple[bytes, float]:
        """Istenen zamana en yakin okunabilir kareyi JPEG olarak doner; hicbiri okunamazsa zirveye duser."""
        target_frame_id = round(target_time_sec * native_fps)
        max_frame_id = (total_frames - 1) if total_frames > 0 else target_frame_id
        clamped_frame_id = max(0, min(target_frame_id, max_frame_id))

        for delta in range(_MAX_NEAREST_FRAME_PROBE + 1):
            for frame_id in {clamped_frame_id - delta, clamped_frame_id + delta}:
                if frame_id < 0 or (total_frames > 0 and frame_id > max_frame_id):
                    continue
                jpeg_bytes = self._try_read_frame(cap, frame_id)
                if jpeg_bytes is not None:
                    return jpeg_bytes, frame_id / native_fps

        logger.warning(
            "Zaman=%.2fs (kare=%d) civarinda okunabilir kare bulunamadi; zirve kareyle dolduruluyor.",
            target_time_sec,
            clamped_frame_id,
        )
        return peak.image_bytes, peak.timestamp_sec

    @staticmethod
    def _try_read_frame(cap: cv2.VideoCapture, frame_id: int) -> Optional[bytes]:
        """Belirtilen kare indeksine seek edip JPEG'e kodlar; basarisizlikta `None` doner."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        encode_ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not encode_ok:
            return None
        return buffer.tobytes()