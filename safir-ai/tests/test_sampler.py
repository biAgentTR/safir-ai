"""Modul 1 (src/sampler/) icin GPU gerektirmeyen birim testleri.

Tamamen sentetik, testin kendi urettigi kucuk videolar uzerinde calisir;
GPU'ya, aga veya baska bir module bagimliligi yoktur.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.sampler.adaptive_sampler import AdaptiveFrameSampler, sampler_from_config
from src.sampler.schema import EventCluster, EvidenceFrame


def _write_video(path: Path, frames: list) -> None:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.fixture
def static_video(tmp_path: Path) -> str:
    """Hic hareket icermeyen, esik ustu degisim uretmeyecek bir video."""
    frames = [np.full((48, 64, 3), 100, dtype=np.uint8) for _ in range(30)]
    path = tmp_path / "static.mp4"
    _write_video(path, frames)
    return str(path)


@pytest.fixture
def motion_video(tmp_path: Path) -> str:
    """20-30. kareler arasinda yuksek kontrastli bir hareket iceren video."""
    frames = []
    for i in range(60):
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        if 20 <= i < 30:
            cv2.rectangle(frame, (5, 5), (58, 42), (255, 255, 255), -1)
        frames.append(frame)
    path = tmp_path / "motion.mp4"
    _write_video(path, frames)
    return str(path)


def test_fallback_frame_used_when_no_threshold_crossed(tmp_path: Path, static_video: str) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001, evidence_output_dir=str(tmp_path / "evidence")
    )
    frames = sampler.process_video(static_video, sample_fps=5)

    assert len(frames) >= 1
    assert isinstance(frames[0], EvidenceFrame)
    assert frames[0].frame_id == 0
    assert frames[0].saved_path is None
    assert frames[0].base64_image.startswith("data:image/jpeg;base64,")
    clusters = sampler.cluster_events(frames)
    assert clusters and Path(clusters[0].peak_frame.saved_path).exists()


def test_fallback_still_produces_a_valid_cluster(tmp_path: Path, static_video: str) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001, evidence_output_dir=str(tmp_path / "evidence")
    )
    frames = sampler.process_video(static_video, sample_fps=5)
    clusters = sampler.cluster_events(frames)

    assert len(clusters) == 1
    assert isinstance(clusters[0], EventCluster)
    assert clusters[0].peak_frame is not None


def test_motion_produces_real_evidence_frames(tmp_path: Path, motion_video: str) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001, evidence_output_dir=str(tmp_path / "evidence")
    )
    frames = sampler.process_video(motion_video, sample_fps=5)

    assert len(frames) >= 1
    assert all(not f.is_fallback for f in frames)
    assert all(f.saved_path is None for f in frames)
    assert all(f.base64_image.startswith("data:image/jpeg;base64,") for f in frames)
    assert all(f.image_bytes for f in frames)
    clusters = sampler.cluster_events(frames)
    assert all(Path(c.peak_frame.saved_path).exists() for c in clusters)


def test_cluster_events_groups_nearby_frames(tmp_path: Path, motion_video: str) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        min_event_interval_sec=2.0,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = sampler.process_video(motion_video, sample_fps=5)
    clusters = sampler.cluster_events(frames)

    assert len(clusters) >= 1
    total_grouped = sum(c.total_candidate_frames for c in clusters)
    assert total_grouped == len(frames)


def test_last_run_stats_reflect_real_counts(tmp_path: Path, motion_video: str) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001, evidence_output_dir=str(tmp_path / "evidence")
    )
    frames = sampler.process_video(motion_video, sample_fps=5)
    stats = sampler.last_run_stats

    assert stats is not None
    assert stats.total_frames_scanned == 60
    assert stats.evidence_frame_count == len(frames)
    assert stats.eliminated_frame_count == stats.sampled_frames_evaluated - len(frames)
    assert 0.0 <= stats.eliminated_ratio_pct <= 100.0
    assert stats.elapsed_sec >= 0.0


def test_sampler_from_config_creates_fresh_isolated_instances(
    safir_config, tmp_path: Path, motion_video: str
) -> None:
    """Ayni config'ten uretilen iki sampler orneği birbirinin durumunu paylasmamalidir."""
    first = sampler_from_config(safir_config.sampler)
    first.evidence_output_dir = tmp_path / "first"
    first.process_video(motion_video, sample_fps=5)
    assert first.last_run_stats is not None

    second = sampler_from_config(safir_config.sampler, min_change_threshold_override=0.02)
    assert second.last_run_stats is None
    assert second.min_change_threshold == 0.02
    assert second.min_change_threshold != first.min_change_threshold


def test_process_video_raises_for_missing_file(tmp_path: Path) -> None:
    sampler = AdaptiveFrameSampler(evidence_output_dir=str(tmp_path / "evidence"))
    with pytest.raises((FileNotFoundError, ValueError)):
        sampler.process_video(str(tmp_path / "does_not_exist.mp4"))


def test_slow_onset_smoke_drift_detection(tmp_path: Path) -> None:
    """Kademeli/yavaş duman yayılması senaryosu: Ardışık fark küçük kalsa bile kümülatif drift ve kontrast ile yakalar."""
    frames = []
    # 0-25. kareler: sabit gri
    for i in range(25):
        frames.append(np.full((48, 64, 3), 120, dtype=np.uint8))
    # 25-100. kareler: kademeli kararma / duman yayılması (kare başına 1-2 birim kademeli değişim)
    for i in range(75):
        val = max(10, 120 - int(i * 1.2))
        frames.append(np.full((48, 64, 3), val, dtype=np.uint8))

    video_path = tmp_path / "slow_smoke.mp4"
    _write_video(video_path, frames)

    # Yeni Sampler (Kümülatif Drift + Kontrast aktif)
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_sampling_gap_sec=4.0,
        ref_change_threshold=0.0015,
        enable_contrast_check=True,
        contrast_change_threshold=0.03,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    ev_frames = sampler.process_video(str(video_path), sample_fps=5)

    # Kademeli olay başladıktan sonra (kare 25 / ~1.0s) yeni kareler üretildiğini doğrula
    assert len(ev_frames) >= 2
    timestamps = [f.timestamp_sec for f in ev_frames]
    # Olayın başlangıç zamanına yakın (1.0s - 3.0s arası) kare yakalandığını doğrula
    assert any(1.0 <= t <= 3.5 for t in timestamps)


def test_hard_cap_max_sampling_gap(tmp_path: Path) -> None:
    """Maksimum örnekleme boşluğu üst sınırı (hard cap): 10 saniyelik statik videoda 4.0s'de bir garanti kare alınır."""
    frames = [np.full((48, 64, 3), 100, dtype=np.uint8) for _ in range(250)]  # 10s video
    video_path = tmp_path / "static_10s.mp4"
    _write_video(video_path, frames)

    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_sampling_gap_sec=3.0,  # 3 saniyede bir garanti kare
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    ev_frames = sampler.process_video(str(video_path), sample_fps=5)

    # 10 saniyelik videoda en az 3-4 adet keyframe alınmalı (büyük boşluk imkansız)
    assert len(ev_frames) >= 3
    gaps = [ev_frames[i+1].timestamp_sec - ev_frames[i].timestamp_sec for i in range(len(ev_frames)-1)]
    # Hiçbir iki kare arasındaki zaman boşluğunun max_sampling_gap_sec'i belirgin şekilde aşmadığını doğrula
    assert all(gap <= 3.5 for gap in gaps)
