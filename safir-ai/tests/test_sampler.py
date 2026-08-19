"""Modul 1 (src/sampler/) icin GPU gerektirmeyen birim testleri.

Tamamen sentetik, testin kendi urettigi kucuk videolar uzerinde calisir;
GPU'ya, aga veya baska bir module bagimliligi yoktur.

ONEMLI (mimari): Sampler artik hicbir olay kumelemesi (event clustering)
YAPMAZ - yalnizca evidence esigini gecen kareleri, hicbir global buffer/kare
limiti, temporal voting/clustering/deduplication veya liste kesme nedeniyle
elemeden, kronolojik sirayla dondurur. Kumeleme VLM katmaninda yapilir.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.sampler.adaptive_sampler import AdaptiveFrameSampler, sampler_from_config
from src.sampler.schema import EvidenceFrame


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


@pytest.fixture
def long_multi_event_video(tmp_path: Path) -> str:
    """Baslangicta, ortada ve sonda ayri hareket patlamalari iceren, aralari sessiz uzun bir video."""
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


def test_fallback_frame_used_when_no_threshold_crossed(tmp_path: Path, static_video: str) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001, evidence_output_dir=str(tmp_path / "evidence")
    )
    frames = sampler.process_video(static_video, sample_fps=5)

    assert len(frames) == 1
    assert isinstance(frames[0], EvidenceFrame)
    assert frames[0].is_fallback is True
    assert frames[0].frame_id == 0
    assert frames[0].evidence_id == "ev0"
    assert frames[0].saved_path is None
    assert frames[0].base64_image.startswith("data:image/jpeg;base64,")


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


def test_evidence_frames_are_chronological_and_have_stable_ids(tmp_path: Path, motion_video: str) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001, evidence_output_dir=str(tmp_path / "evidence")
    )
    frames = sampler.process_video(motion_video, sample_fps=5)

    timestamps = [f.timestamp_sec for f in frames]
    assert timestamps == sorted(timestamps)

    frame_ids = [f.frame_id for f in frames]
    assert frame_ids == sorted(frame_ids)
    assert len(frame_ids) == len(set(frame_ids))

    for f in frames:
        assert f.evidence_id == f"ev{f.frame_id}"


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
# Karesi artik atlanmamali.
# =============================================================================


def test_adaptive_frame_sampler_no_longer_accepts_max_evidence_buffer(tmp_path: Path) -> None:
    """Eski `max_evidence_buffer` parametresi kaldirildi; verilirse TypeError beklenir."""
    with pytest.raises(TypeError):
        AdaptiveFrameSampler(max_evidence_buffer=5, evidence_output_dir=str(tmp_path / "evidence"))


def test_sampler_config_no_longer_requires_max_evidence_buffer(safir_config) -> None:
    """`SamplerConfig`in artik `max_evidence_buffer` alani olmamali (config.yaml'dan kaldirildi)."""
    assert not hasattr(safir_config.sampler, "max_evidence_buffer")


def test_sampler_no_longer_accepts_clustering_params(tmp_path: Path) -> None:
    """Eski kumeleme parametreleri (min_event_interval_sec vb.) kaldirildi; verilirse TypeError beklenir."""
    for kwarg in (
        "min_event_interval_sec",
        "cluster_merge_gap_sec",
        "bbox_iou_merge_threshold",
        "max_cluster_duration_sec",
    ):
        with pytest.raises(TypeError):
            AdaptiveFrameSampler(**{kwarg: 1.0}, evidence_output_dir=str(tmp_path / "evidence"))


def test_long_video_no_events_dropped_due_to_buffer_limit(
    tmp_path: Path, long_multi_event_video: str
) -> None:
    """Eskiden `max_evidence_buffer` asilirsa kareler sessizce atlanirdi; artik hicbir kare atlanmamali."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = sampler.process_video(long_multi_event_video, sample_fps=25)
    stats = sampler.last_run_stats

    # Kayipsizlik: uretilen Kanit Karesi sayisi + elenen kare sayisi ==
    # degerlendirilen ornek kare sayisi (hicbir kare "buffer dolusu"
    # nedeniyle sessizce atlanmamis).
    assert stats.evidence_frame_count == len(frames)
    assert stats.evidence_frame_count + stats.eliminated_frame_count == stats.sampled_frames_evaluated


def test_earliest_evidence_frames_not_evicted_by_later_ones(
    tmp_path: Path, long_multi_event_video: str
) -> None:
    """Videonun BASINDAKI kanit kareleri, sonraki hareket patlamalari geldiginde silinmemeli/kaybolmamali."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = sampler.process_video(long_multi_event_video, sample_fps=25)

    assert frames
    # Ilk hareket patlamasina (t~0s) ait en az bir kanit karesi hala listede olmali.
    assert any(f.timestamp_sec < 1.0 for f in frames)


# =============================================================================
# Sampler artik `EventCluster`/`cluster_events` uretmez; eski kumeleme API'si
# tamamen kaldirilmis olmali.
# =============================================================================


def test_sampler_has_no_clustering_api() -> None:
    """`cluster_events`/`export_event_frames` gibi eski kumeleme metotlari artik mevcut olmamali."""
    assert not hasattr(AdaptiveFrameSampler, "cluster_events")
    assert not hasattr(AdaptiveFrameSampler, "export_event_frames")


def test_schema_has_no_clustering_models() -> None:
    """`EventCluster`/`RepresentativeFrame` gibi eski kumeleme modelleri kaldirilmis olmali."""
    import src.sampler.schema as schema_module

    assert not hasattr(schema_module, "EventCluster")
    assert not hasattr(schema_module, "RepresentativeFrame")


def test_sampler_context_module_no_longer_exists() -> None:
    """Eski `src.sampler.context` (FrameSelector/FrameArchiver) artik mevcut olmamali."""
    with pytest.raises(ModuleNotFoundError):
        __import__("src.sampler.context.frame_selector")


def test_evidence_frame_produces_no_event_or_cluster_id() -> None:
    """`EvidenceFrame`in hicbir alani `event_id`/`cluster_id` uretmemeli (sampler'in gorevi degil)."""
    field_names = set(EvidenceFrame.model_fields.keys())
    assert "event_id" not in field_names
    assert "cluster_id" not in field_names


# =============================================================================
# Pre/peak/post konumsal bilgisi TAMAMEN kaldirilmis olmali: sampler ciktisinda
# hicbir konumsal rol/etiket bulunmamali.
# =============================================================================


def test_evidence_frame_has_no_pre_peak_post_role_labels() -> None:
    field_names = set(EvidenceFrame.model_fields.keys())
    assert "label" not in field_names
    assert "frame_role" not in field_names
    assert "frame_type" not in field_names
    assert "position_type" not in field_names
    assert "selection_reason" not in field_names


def test_vlm_payload_contains_no_pre_peak_post_tags(tmp_path: Path, long_multi_event_video: str) -> None:
    from src.sampler.payload_builder import VLMPayloadBuilder

    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = sampler.process_video(long_multi_event_video, sample_fps=25)

    content = VLMPayloadBuilder.build_content_blocks(frames, prompt="test")
    text_blocks = " ".join(b["text"].lower() for b in content if b["type"] == "text")

    assert "pre-event" not in text_blocks
    assert "post-event" not in text_blocks
    assert "kare rolu" not in text_blocks
    assert "pre_context" not in text_blocks
    assert "post_context" not in text_blocks
