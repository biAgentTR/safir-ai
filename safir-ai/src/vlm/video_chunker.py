"""03 - VLM Katmani: EVREN "video cozunurluk zarfi" sorunu icin video parcalama (chunking).

Sorun (EVREN dokumantasyonu): sistem, gonderilen videonun TAMAMINA TEK bir
toplam piksel butcesi uygular - 180 saniyelik 720p bir klip ile 60 saniyelik
ayni cozunurlukteki bir klip AYNI oranda kucultulmez; uzun klip cok daha
agresif kucultulur ve detaylar (baret, kucuk alev/duman baslangici vb.)
kaybolabilir. EVREN dokumantasyonu "klibin kisa parcalara bolunmesini"
onerir.

Cozum: `split_video_into_chunks`, bir video dosyasini (yerel, .mp4) sabit
sureli, KRONOLOJIK ve KAYIPSIZ parcalara boler. Yeni bir sistem bagimliligi
(ffmpeg ikili dosyasi) EKLEMEZ - proje zaten `opencv-python-headless`e
(cv2) bagimli (bkz. `src/sampler/adaptive_sampler.py`nin ayni VideoCapture
deseni); bu modul SADECE bunu kullanir.

Video kisa ise (toplam sure <= `chunk_duration_sec`), TEK bir "chunk"
(orijinal dosyanin kendisi, KOPYALANMADAN) dondurulur - kisa videolarda
davranis/performans HIC degismez.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import cv2

logger = logging.getLogger(__name__)

_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


@dataclass
class VideoChunk:
    """Bolunmus bir video parcasi: dosya yolu + orijinal videodaki zaman araligi."""

    path: str
    start_offset_sec: float
    end_offset_sec: float
    index: int
    is_original: bool = False
    """`True` ise `path`, GECICI bir parca DEGIL - orijinal video dosyasinin
    kendisidir (video zaten `chunk_duration_sec`den kisa oldugu icin bolunmedi).
    Cagiran taraf bu dosyayi ASLA silmemeli (bkz. `cleanup_chunks`)."""


def split_video_into_chunks(
    video_path: str, chunk_duration_sec: float, out_dir: Optional[str] = None
) -> List[VideoChunk]:
    """Bir video dosyasini sabit sureli, kronolojik/kayipsiz parcalara boler.

    Args:
        video_path: Yerel bir `.mp4` dosyasinin yolu.
        chunk_duration_sec: Her parcanin hedef suresi (saniye); 0/negatifse
            bolme YAPILMAZ (video oldugu gibi TEK parca olarak doner).
        out_dir: Uretilen parca dosyalarinin yazilacagi dizin; verilmezse
            gecici bir dizin olusturulur (cagiran taraf `cleanup_chunks` ile
            temizlemelidir).

    Returns:
        Zaman sirali `VideoChunk` listesi. Video, `chunk_duration_sec`den
        KISA veya esitse (veya `chunk_duration_sec<=0`), TEK elemanli bir
        liste doner ve o eleman ORIJINAL dosyayi (`is_original=True`,
        kopyalanmadan) isaret eder.

    Raises:
        ValueError: Video dosyasi acilamazsa.
    """
    if chunk_duration_sec <= 0:
        return [VideoChunk(path=video_path, start_offset_sec=0.0, end_offset_sec=0.0, index=0, is_original=True)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video dosyasi acilamadi: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0.0

        if total_frames <= 0 or total_duration <= chunk_duration_sec:
            return [
                VideoChunk(
                    path=video_path, start_offset_sec=0.0, end_offset_sec=total_duration, index=0, is_original=True
                )
            ]

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_chunk = max(1, int(round(chunk_duration_sec * fps)))
        out_dir = out_dir or tempfile.mkdtemp(prefix="safir_vlm_chunks_")
        os.makedirs(out_dir, exist_ok=True)

        chunks: List[VideoChunk] = []
        writer: Optional[cv2.VideoWriter] = None
        chunk_path = ""
        chunk_start_frame = 0
        chunk_idx = 0
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frames_per_chunk == 0:
                if writer is not None:
                    writer.release()
                    chunks.append(
                        VideoChunk(
                            path=chunk_path,
                            start_offset_sec=chunk_start_frame / fps,
                            end_offset_sec=frame_idx / fps,
                            index=chunk_idx,
                        )
                    )
                    chunk_idx += 1
                chunk_start_frame = frame_idx
                chunk_path = os.path.join(out_dir, f"chunk_{chunk_idx:03d}.mp4")
                writer = cv2.VideoWriter(chunk_path, _FOURCC, fps, (width, height))
            assert writer is not None
            writer.write(frame)
            frame_idx += 1

        if writer is not None:
            writer.release()
            chunks.append(
                VideoChunk(
                    path=chunk_path,
                    start_offset_sec=chunk_start_frame / fps,
                    end_offset_sec=frame_idx / fps,
                    index=chunk_idx,
                )
            )

        logger.info(
            "Video %s parcalara bolundu: %.1fs -> %d parca (%.1fs/parca)",
            video_path,
            total_duration,
            len(chunks),
            chunk_duration_sec,
        )
        return chunks
    finally:
        cap.release()


def cleanup_chunks(chunks: List[VideoChunk]) -> None:
    """`split_video_into_chunks`in urettigi GECICI parca dosyalarini siler.

    `is_original=True` olan (orijinal video dosyasini isaret eden) girdiler
    ASLA silinmez - yalnizca bu modulun urettigi gecici `.mp4` dosyalari
    temizlenir. Bir dosya zaten silinmisse/erisilemezse SESSIZCE gecilir
    (temizlik best-effort'tur, pipeline'i KESMEZ).
    """
    for chunk in chunks:
        if chunk.is_original:
            continue
        try:
            os.remove(chunk.path)
        except OSError:
            logger.debug("Gecici video parcasi silinemedi (zaten yok olabilir): %s", chunk.path)
