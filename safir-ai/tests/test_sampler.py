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

    assert len(frames) == 1
    assert isinstance(frames[0], EvidenceFrame)
    assert frames[0].is_fallback is True
    assert frames[0].frame_id == 0
    # Tasarim (bkz. _build_evidence_frame docstring): Kanit Kareleri
    # process_video'da diske YAZILMAZ (saved_path=None); yalnizca zirve kareler
    # cluster_events/_close_group'ta kalici yazilir. Kare gecerli goruntu tasir.
    assert frames[0].saved_path is None
    assert frames[0].base64_image.startswith("data:image/jpeg;base64,")
    # Zirve olarak secildiginde kalici diske yazilir:
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
    assert clusters[0].peak_frame.is_fallback is True


def test_motion_produces_real_evidence_frames(tmp_path: Path, motion_video: str) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001, evidence_output_dir=str(tmp_path / "evidence")
    )
    frames = sampler.process_video(motion_video, sample_fps=5)

    assert len(frames) >= 1
    assert all(not f.is_fallback for f in frames)
    # Tasarim: process_video diske yazmaz (saved_path=None); kareler gecerli
    # goruntu verisi tasir. Kalici kayit, zirve karelerde cluster_events'te olur.
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
    assert first.prev_gray is not None

    second = sampler_from_config(safir_config.sampler, min_change_threshold_override=0.02)
    assert second.prev_gray is None
    assert second.min_change_threshold == 0.02
    assert second.min_change_threshold != first.min_change_threshold


def test_process_video_raises_for_missing_file(tmp_path: Path) -> None:
    sampler = AdaptiveFrameSampler(evidence_output_dir=str(tmp_path / "evidence"))
    with pytest.raises(ValueError):
        sampler.process_video(str(tmp_path / "does_not_exist.mp4"))


# =============================================================================
# max_evidence_buffer kaldirildi: sabit bir ust sinir nedeniyle hicbir Kanit
# Karesi/Olay Grubu artik atlanmamali.
# =============================================================================


def test_adaptive_frame_sampler_no_longer_accepts_max_evidence_buffer(tmp_path: Path) -> None:
    """Eski `max_evidence_buffer` parametresi kaldirildi; verilirse TypeError beklenir."""
    with pytest.raises(TypeError):
        AdaptiveFrameSampler(max_evidence_buffer=5, evidence_output_dir=str(tmp_path / "evidence"))


def test_sampler_config_no_longer_requires_max_evidence_buffer(safir_config) -> None:
    """`SamplerConfig`in artik `max_evidence_buffer` alani olmamali (config.yaml'dan kaldirildi)."""
    assert not hasattr(safir_config.sampler, "max_evidence_buffer")


@pytest.fixture
def long_multi_event_video(tmp_path: Path) -> str:
    """Baslangicta, ortada ve sonda ayri hareket patlamalari iceren, aralari sessiz uzun bir video.

    3 ayri hareket bolgesi (her biri 10 kare), aralarinda 40'ar karelik durgun
    bolgeler ile ayrilir; boylece `min_event_interval_sec` kucuk tutuldugunda
    3 AYRI ham/nihai Olay Grubu olusur (bkz. ilgili testler).
    """
    frames = []
    total = 220
    motion_windows = [(0, 10), (100, 110), (200, 210)]
    for i in range(total):
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        if any(start <= i < end for start, end in motion_windows):
            cv2.rectangle(frame, (5, 5), (58, 42), (255, 255, 255), -1)
        frames.append(frame)
    path = tmp_path / "long_multi_event.mp4"
    _write_video(path, frames)
    return str(path)


def test_long_video_no_events_dropped_due_to_buffer_limit(
    tmp_path: Path, long_multi_event_video: str
) -> None:
    """Eskiden `max_evidence_buffer` asilirsa kareler sessizce atlanirdi; artik hicbir kare atlanmamali."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        min_event_interval_sec=0.5,
        cluster_merge_gap_sec=1.0,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = sampler.process_video(long_multi_event_video, sample_fps=25)
    clusters = sampler.cluster_events(frames)

    # Uc ayri hareket bolgesinin UCU DE Kanit Karesi/Olay Grubu olarak yakalanmali.
    assert len(clusters) == 3
    # Hicbir kare "buffer dolusu" nedeniyle atlanmamali: gruplanan toplam kare
    # sayisi, uretilen tum Kanit Karesi sayisina esit olmalidir (kayip yok).
    assert sum(c.total_candidate_frames for c in clusters) == len(frames)


def test_earliest_event_is_not_evicted_by_later_events(
    tmp_path: Path, long_multi_event_video: str
) -> None:
    """Videonun BASINDAKI olay, sonraki olaylar geldiginde silinmemeli/kaybolmamali."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        min_event_interval_sec=0.5,
        cluster_merge_gap_sec=1.0,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = sampler.process_video(long_multi_event_video, sample_fps=25)
    clusters = sampler.cluster_events(frames)

    assert len(clusters) == 3
    first_cluster = clusters[0]
    # Ilk olay, videonun basina (t~0s) yakin olmali; sonraki iki olay yuzunden
    # atilmamis/kaymamis olmali.
    assert first_cluster.start_time < 1.0


# =============================================================================
# Ortak FrameSelector: `cluster_events` artik her Olay Grubu icin VLM'e giden
# temsili kareleri de doldurur (eski bagimsiz RepresentativeFrameExtractor/
# PeakFrameExporter mekanizmalari kaldirildi).
# =============================================================================


def test_cluster_events_populates_representative_frames(tmp_path: Path, motion_video: str) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001, evidence_output_dir=str(tmp_path / "evidence")
    )
    frames = sampler.process_video(motion_video, sample_fps=5)
    clusters = sampler.cluster_events(frames)

    assert clusters
    for cluster in clusters:
        assert cluster.representative_frames
        assert len(cluster.representative_frames) <= 5


def test_representative_frames_include_peak_and_are_chronological(
    tmp_path: Path, long_multi_event_video: str
) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        min_event_interval_sec=0.5,
        cluster_merge_gap_sec=1.0,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = sampler.process_video(long_multi_event_video, sample_fps=25)
    clusters = sampler.cluster_events(frames)

    for cluster in clusters:
        peak_entries = [rf for rf in cluster.representative_frames if rf.label == "peak"]
        assert len(peak_entries) == 1
        assert peak_entries[0].frame_id == cluster.peak_frame.frame_id

        timestamps = [rf.timestamp_sec for rf in cluster.representative_frames]
        assert timestamps == sorted(timestamps)

        frame_ids = [rf.frame_id for rf in cluster.representative_frames]
        assert len(frame_ids) == len(set(frame_ids))


def test_short_event_does_not_produce_duplicate_or_fabricated_frames(
    tmp_path: Path, static_video: str
) -> None:
    """Kisa/tekil-kareli olayda (fallback) sahte/tekrarlanan kare uretilmemeli."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001, evidence_output_dir=str(tmp_path / "evidence")
    )
    frames = sampler.process_video(static_video, sample_fps=5)
    clusters = sampler.cluster_events(frames)

    assert len(clusters) == 1
    rep_frames = clusters[0].representative_frames
    frame_ids = [rf.frame_id for rf in rep_frames]
    assert len(frame_ids) == len(set(frame_ids))
    # Fallback tek kareyle kapanan bir olay -> tek temsili kare (cogaltma yok).
    assert len(rep_frames) == 1


def test_vlm_and_disk_output_share_identical_frame_identity(
    tmp_path: Path, long_multi_event_video: str
) -> None:
    """VLM'e giden (`representative_frames`) ve diske yazilan kareler AYNI frame_id/timestamp setini kullanmali."""
    from src.sampler.context.frame_archiver import FrameArchiver

    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        min_event_interval_sec=0.5,
        cluster_merge_gap_sec=1.0,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = sampler.process_video(long_multi_event_video, sample_fps=25)
    clusters = sampler.cluster_events(frames, export_to_disk=True)

    output_dir = tmp_path / "evidence"
    for cluster in clusters:
        event_dir = output_dir / f"event_{cluster.event_id:04d}"
        metadata = __import__("json").loads((event_dir / "metadata.json").read_text(encoding="utf-8"))
        disk_frame_ids = {f["frame_id"] for f in metadata["frames"]}
        vlm_frame_ids = {rf.frame_id for rf in cluster.representative_frames}
        assert disk_frame_ids == vlm_frame_ids

    # FrameArchiver de ayni ciktiyi (bagimsiz secim yapmadan) uretebilmeli.
    event_dirs = FrameArchiver.export(clusters, output_dir=str(tmp_path / "manual_export"))
    assert len(event_dirs) == len(clusters)


def test_independent_representative_frame_extractor_module_no_longer_exists() -> None:
    """Eski bagimsiz VLM-kare-secim yolu (RepresentativeFrameExtractor) artik mevcut olmamali."""
    with pytest.raises(ModuleNotFoundError):
        __import__("src.sampler.context.representative_frame_extractor")


def test_independent_peak_frame_exporter_module_no_longer_exists() -> None:
    """Eski bagimsiz disk-arsiv kare-secim yolu (PeakFrameExporter) artik mevcut olmamali."""
    with pytest.raises(ModuleNotFoundError):
        __import__("src.sampler.context.peak_frame_exporter")
