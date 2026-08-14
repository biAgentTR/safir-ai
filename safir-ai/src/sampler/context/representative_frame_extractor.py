"""Kayitli (VOD) videolar icin seek-tabanli pre/peak/post temsili kare cikarici.

`AdaptiveFrameSampler`'in evidence/clustering/peak-secim mantigindan tamamen
ayri, opsiyonel bir katmandir. Iki modda calisir:

- KISA Olay Grubu (suresi `pre_event_sec + post_event_sec` toplamini asmiyor,
  ya da cagiran cluster zaman araligini hic vermiyor): mevcut, degismemis
  davranis — zirve karenin sabit `pre_event_sec`/`post_event_sec` penceresinde
  tek bir pre-event ve tek bir post-event karesi (toplam en fazla 3 kare).
- UZUN Olay Grubu (cagiran `start_time`/`end_time` verir ve suresi bu
  pencereyi asar): zirve kare yine korunur, ancak pre/post kareler ARTIK
  sabit ±pencerede degil, olayin TAMAMINA (`start_time` -> `end_time`)
  dengeli/esit araliklarla yayilir; boylece VLM yalnizca zirve anini degil,
  uzun bir olayin baslangic/gelisim/sonuc akisinin tamamini gorebilir.
  Kare sayisi `max_representative_frames` ile ust sinirlanir (GPU/API
  maliyeti kontrol altinda kalir).

Hicbir modda `context.mp4` klibi URETMEZ; VLM'e dogrudan `image_url`
bloklari olarak gonderilecek JPEG+base64 kareler uretir. `AdaptiveFrameSampler`in
clustering/merge parametrelerine (`cluster_merge_gap_sec`,
`bbox_iou_merge_threshold`, `min_change_threshold`, `max_evidence_buffer`)
HICBIR sekilde dokunmaz; yalnizca zaten olusmus bir `EventCluster` icin
VLM'e gonderilecek kare SETINI zenginlestirir.
"""

from __future__ import annotations

import base64
import logging
import math
from typing import List, Optional

import cv2

from src.sampler.schema import EvidenceFrame, RepresentativeFrame

logger = logging.getLogger(__name__)

_JPEG_QUALITY = 85
# Gemini payload/gecikme/kota baskisini sinirli tutmak icin ust sinir (bkz.
# Hata #5: 8'e cikarilmasi payload'i buyutup VLM cagrisini yavaslatmis/
# basarisiz kilmis olabilir; 5'e dusuruldu). Uzun olaylarda etkilidir;
# kisa olaylarda hicbir etkisi yoktur (o modda zaten en fazla 3 kare uretilir).
_DEFAULT_MAX_REPRESENTATIVE_FRAMES = 5


class RepresentativeFrameExtractor:
    """Zirve kare etrafinda (kisa olaylarda) veya olayin tamaminda (uzun olaylarda) temsili kareleri seek ile cikarir."""

    def __init__(
        self,
        pre_event_sec: float = 3.0,
        post_event_sec: float = 3.0,
        max_representative_frames: int = _DEFAULT_MAX_REPRESENTATIVE_FRAMES,
    ) -> None:
        """RepresentativeFrameExtractor'i zaman penceresi parametreleriyle baslatir.

        Args:
            pre_event_sec: KISA olaylarda zirve karenin kac saniye oncesinden
                kare alinacagi; UZUN olaylarda ayrica "kisa mi uzun mu"
                kararini veren esik penceresinin (`pre_event_sec + post_event_sec`)
                bir parcasi olarak kullanilir.
            post_event_sec: KISA olaylarda zirve karenin kac saniye
                sonrasindan kare alinacagi (bkz. `pre_event_sec`).
            max_representative_frames: UZUN olaylarda VLM'e gonderilecek
                temsili kare sayisinin ust siniri (zirve dahil). GPU/API
                maliyetini kontrol altinda tutmak icin sabit bir tavan
                saglar; KISA olaylarda etkisizdir (o modda zaten en fazla 3
                kare uretilir).

        Raises:
            ValueError: `pre_event_sec`/`post_event_sec` negatifse veya
                `max_representative_frames` 3'ten kucukse (zirve + en az bir
                pre + bir post icin minimum).
        """
        if pre_event_sec < 0 or post_event_sec < 0:
            raise ValueError("pre_event_sec ve post_event_sec negatif olamaz.")
        if max_representative_frames < 3:
            raise ValueError("max_representative_frames en az 3 olmalidir (pre + peak + post).")
        self.pre_event_sec = pre_event_sec
        self.post_event_sec = post_event_sec
        self.max_representative_frames = max_representative_frames

    def extract(
        self,
        video_path: str,
        peak_frame: EvidenceFrame,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[RepresentativeFrame]:
        """Bu Olay Grubu icin zaman sirali temsili kare listesini uretir.

        `start_time`/`end_time` verilmezse VEYA olayin suresi
        (`end_time - start_time`) `pre_event_sec + post_event_sec` toplamini
        asmiyorsa, MEVCUT (degismemis) sabit-pencere davranisi kullanilir:
        zirve karenin ±`pre_event_sec`/`post_event_sec` saniyesinden birer
        kare (en fazla 3 kare toplam).

        Suresi bu pencereyi ASAN uzun olaylarda, kareler zirve etrafinda
        SIKISMAZ: `start_time` -> `end_time` araligina dengeli/esit
        araliklarla yayilir (zirve karenin kendisi HER ZAMAN dahil edilir),
        `max_representative_frames` ile sinirlanir ve ayni video karesi
        (frame_id) iki kez secilmez.

        Pre/post kareler video sinirlarinin disina tasarsa gecerli araliga
        (`0`/`total_frames - 1`) kenetlenir. Tek bir karenin okunmasi
        basarisiz olursa hata firlatilmaz, yalnizca loglanip atlanir; kalan
        gecerli karelerle devam edilir.

        Args:
            video_path: Kaynak video dosyasinin yolu.
            peak_frame: Bu olay grubunun zirve `EvidenceFrame`i.
            start_time: Olay Grubunun baslangic zaman damgasi (sn); verilmezse
                sabit-pencere davranisina dusulur.
            end_time: Olay Grubunun bitis zaman damgasi (sn); verilmezse
                sabit-pencere davranisina dusulur.

        Returns:
            Zaman sirali (`timestamp_sec` artan) `RepresentativeFrame` listesi;
            zirve karesini HER ZAMAN icerir.

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

            duration = (end_time - start_time) if (start_time is not None and end_time is not None) else 0.0
            window_span = self.pre_event_sec + self.post_event_sec

            if start_time is None or end_time is None or duration <= window_span:
                return self._extract_fixed_window(cap, peak_frame, native_fps, total_frames)

            return self._extract_spread_across_event(
                cap, peak_frame, native_fps, total_frames, start_time, end_time
            )
        finally:
            cap.release()

    def _extract_fixed_window(
        self,
        cap: "cv2.VideoCapture",
        peak_frame: EvidenceFrame,
        native_fps: float,
        total_frames: int,
    ) -> List[RepresentativeFrame]:
        """KISA olaylar icin mevcut (degismemis) sabit ±pencere davranisi: en fazla 3 kare (pre + peak + post)."""
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

    def _extract_spread_across_event(
        self,
        cap: "cv2.VideoCapture",
        peak_frame: EvidenceFrame,
        native_fps: float,
        total_frames: int,
        start_time: float,
        end_time: float,
    ) -> List[RepresentativeFrame]:
        """UZUN olaylar icin `start_time` -> `end_time` araligina dengeli yayilmis kareler; zirve dahil, ust sinirli, tekrarsiz."""
        window_span = self.pre_event_sec + self.post_event_sec
        target_count = min(
            self.max_representative_frames,
            max(3, math.ceil((end_time - start_time) / window_span) + 1),
        )

        clamped_start = max(0.0, start_time)
        clamped_end = end_time
        if total_frames > 0:
            clamped_end = min(end_time, (total_frames - 1) / native_fps)
        clamped_end = max(clamped_start, clamped_end)

        if target_count <= 1 or clamped_end <= clamped_start:
            timestamps = [peak_frame.timestamp_sec]
        else:
            step = (clamped_end - clamped_start) / (target_count - 1)
            timestamps = [clamped_start + i * step for i in range(target_count)]
            # Zirve karenin, esit-aralikli setin en yakin nokasinin YERINE
            # gecmesini saglar; boylece zirve HER ZAMAN secilen kareler
            # arasinda kalir (ekstra kare eklemeden, ust siniri asmadan).
            nearest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - peak_frame.timestamp_sec))
            timestamps[nearest_idx] = peak_frame.timestamp_sec

        representative_frames: List[RepresentativeFrame] = []
        seen_frame_ids: set[int] = set()
        for ts in timestamps:
            is_peak_slot = ts == peak_frame.timestamp_sec
            if is_peak_slot:
                frame_id = peak_frame.frame_id
            else:
                frame_id = int(round(ts * native_fps))
                if total_frames > 0:
                    frame_id = max(0, min(frame_id, total_frames - 1))
                else:
                    frame_id = max(0, frame_id)

            # Ayni video karesinin (frame_id) iki kez secilmesini engelle
            # (farkli hedef zaman damgalari ayni kareye yuvarlanabilir).
            if frame_id in seen_frame_ids:
                continue
            seen_frame_ids.add(frame_id)

            if is_peak_slot:
                representative_frames.append(
                    RepresentativeFrame(
                        label="peak",
                        timestamp_sec=peak_frame.timestamp_sec,
                        timestamp_str=peak_frame.timestamp_str,
                        base64_image=peak_frame.base64_image,
                    )
                )
                continue

            label = "pre-event" if frame_id < peak_frame.frame_id else "post-event"
            rep = self._read_frame_at(cap, frame_id, native_fps, label)
            if rep is not None:
                representative_frames.append(rep)

        representative_frames.sort(key=lambda rf: rf.timestamp_sec)
        return representative_frames

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
