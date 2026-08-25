"""`src/vlm/video_chunker.py` icin GPU/ag bagimliligi gerektirmeyen birim testleri.

`tests/test_sampler.py`daki ile ayni desende, tamamen sentetik (cv2 ile
uretilmis) kucuk videolar uzerinde calisir - EVREN'e/aga hicbir bagimlilik yoktur.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vlm.video_chunker import VideoChunk, cleanup_chunks, split_video_into_chunks


def _write_video(path: Path, num_frames: int, fps: float = 10.0, size: tuple[int, int] = (32, 24)) -> None:
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _video_duration_sec(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total_frames / fps


def test_short_video_returns_single_original_chunk_not_copied(tmp_path: Path) -> None:
    """Video, chunk_duration_sec'ten KISA/ESITSE bolme YAPILMAZ - orijinal dosya (kopyalanmadan) tek parca olarak doner."""
    video_path = tmp_path / "short.mp4"
    _write_video(video_path, num_frames=50, fps=10.0)  # 5.0s

    chunks = split_video_into_chunks(str(video_path), chunk_duration_sec=60.0)

    assert len(chunks) == 1
    assert chunks[0].path == str(video_path)
    assert chunks[0].is_original is True
    assert chunks[0].start_offset_sec == 0.0


def test_zero_or_negative_chunk_duration_disables_splitting(tmp_path: Path) -> None:
    video_path = tmp_path / "any.mp4"
    _write_video(video_path, num_frames=200, fps=10.0)  # 20.0s

    for duration in (0.0, -5.0):
        chunks = split_video_into_chunks(str(video_path), chunk_duration_sec=duration)
        assert len(chunks) == 1
        assert chunks[0].is_original is True


def test_long_video_splits_into_expected_number_of_chunks(tmp_path: Path) -> None:
    """20 saniyelik video, 5 saniyelik parcalara bolununce 4 parca uretmeli."""
    video_path = tmp_path / "long.mp4"
    _write_video(video_path, num_frames=200, fps=10.0)  # 20.0s

    chunks = split_video_into_chunks(str(video_path), chunk_duration_sec=5.0, out_dir=str(tmp_path / "chunks"))

    assert len(chunks) == 4
    assert all(not c.is_original for c in chunks)
    for chunk in chunks:
        assert os.path.exists(chunk.path)


def test_chunks_are_chronological_contiguous_and_lossless(tmp_path: Path) -> None:
    """Parcalar zaman sirali, birbirini takip eder (bosluk/cakisma yok) ve toplam sure orijinaliyle eslesir."""
    video_path = tmp_path / "long.mp4"
    _write_video(video_path, num_frames=237, fps=10.0)  # 23.7s (esit bolunmeyen bir sure - kayip testi icin)
    original_duration = _video_duration_sec(str(video_path))

    chunks = split_video_into_chunks(str(video_path), chunk_duration_sec=5.0, out_dir=str(tmp_path / "chunks"))

    assert chunks == sorted(chunks, key=lambda c: c.start_offset_sec)
    for i in range(1, len(chunks)):
        assert chunks[i].start_offset_sec == pytest.approx(chunks[i - 1].end_offset_sec, abs=1e-6)
    assert chunks[0].start_offset_sec == 0.0
    assert chunks[-1].end_offset_sec == pytest.approx(original_duration, abs=0.2)

    # Hicbir kare kaybolmadi: parcalarin toplam kare sayisi orijinaliyle AYNI.
    total_chunk_frames = 0
    for chunk in chunks:
        cap = cv2.VideoCapture(chunk.path)
        total_chunk_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    assert total_chunk_frames == 237


def test_missing_video_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        split_video_into_chunks(str(tmp_path / "yok.mp4"), chunk_duration_sec=10.0)


def test_cleanup_chunks_removes_generated_files_but_never_the_original(tmp_path: Path) -> None:
    video_path = tmp_path / "long.mp4"
    _write_video(video_path, num_frames=200, fps=10.0)  # 20.0s
    chunks = split_video_into_chunks(str(video_path), chunk_duration_sec=5.0, out_dir=str(tmp_path / "chunks"))
    assert all(os.path.exists(c.path) for c in chunks)

    cleanup_chunks(chunks)

    assert all(not os.path.exists(c.path) for c in chunks)


def test_cleanup_chunks_never_deletes_the_original_video_for_short_videos(tmp_path: Path) -> None:
    video_path = tmp_path / "short.mp4"
    _write_video(video_path, num_frames=50, fps=10.0)  # 5.0s
    chunks = split_video_into_chunks(str(video_path), chunk_duration_sec=60.0)

    cleanup_chunks(chunks)

    assert video_path.exists()


def test_cleanup_chunks_is_safe_to_call_twice(tmp_path: Path) -> None:
    """Zaten silinmis bir parca icin ikinci `cleanup_chunks` cagrisi PATLAMAMALI (best-effort)."""
    video_path = tmp_path / "long.mp4"
    _write_video(video_path, num_frames=200, fps=10.0)
    chunks = split_video_into_chunks(str(video_path), chunk_duration_sec=5.0, out_dir=str(tmp_path / "chunks"))

    cleanup_chunks(chunks)
    cleanup_chunks(chunks)  # ikinci cagri hata FIRLATMAMALI
