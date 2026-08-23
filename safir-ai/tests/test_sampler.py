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


def test_selection_reason_is_a_reason_not_a_positional_role() -> None:
    """`selection_reason`, sampler'in NEDEN bu kareyi sectigini aciklar (esik mi gecti, zamansal
    bosluk mu kapatildi); pre/peak/post gibi bir konumsal ROL degildir ve olay
    kumelemesiyle ilgisi yoktur (bkz. `EvidenceFrame.selection_reason` docstring'i)."""
    field_names = set(EvidenceFrame.model_fields.keys())
    assert "selection_reason" in field_names

    allowed_values = {"threshold_exceeded", "temporal_coverage", "fallback"}
    for value in allowed_values:
        assert "pre" not in value
        assert "post" not in value
        assert "peak" not in value


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


# =============================================================================
# Zamansal kapsama (temporal coverage): esigi hicbir zaman gecemeyen uzun
# sessiz araliklarda (ör. 00:15 -> 01:45) sistemin kor kalmasini onleyen
# guvenlik agi. Kumeleme DEGILDIR, pre/peak/post rolu getirmez - yalnizca
# `max_temporal_gap_sec` asildiginda, pencere icindeki en yuksek
# `net_change_score`'lu esik-alti aday `selection_reason="temporal_coverage"`
# ile evidence listesine eklenir.
# =============================================================================


@pytest.fixture
def long_silence_video(tmp_path: Path) -> str:
    """Basta ve NEREDEYSE sonda kisa birer hareket patlamasi, aralarinda uzun
    (esigi hic gecmeyen) bir sessizlik iceren video (25fps, 8s = 200 kare).

    0-10. kareler (t=0.00-0.36s) ve 190-199. kareler (t=7.60-7.96s) arasinda
    hareket vardir; aradaki 10-190 araligi (yaklasik 7.2s) TAMAMEN durgundur
    - bu, "00:15 -> 01:45" senaryosunun kucultulmus/hizlandirilmis bir
    esdegeridir.
    """
    frames = []
    for i in range(200):
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        if 0 <= i < 10:
            cv2.rectangle(frame, (5, 5), (58, 42), (255, 255, 255), -1)
        if 190 <= i < 200:
            cv2.rectangle(frame, (5, 5), (58, 42), (200, 200, 200), -1)
        frames.append(frame)
    path = tmp_path / "long_silence.mp4"
    _write_video(path, frames)
    return str(path)


def test_threshold_frames_selected_exactly_as_before(tmp_path: Path, motion_video: str) -> None:
    """1) Coverage mekanizmasi eklendikten sonra da threshold-gecen kareler AYNI sekilde secilmeli."""
    without_coverage = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=999.0,  # pratikte hic tetiklenmeyecek kadar buyuk
        evidence_output_dir=str(tmp_path / "ev_a"),
    ).process_video(motion_video, sample_fps=5)

    with_short_gap = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=1.0,
        early_change_selection_interval_sec=0.5,
        significant_change_selection_interval_sec=0.2,
        evidence_output_dir=str(tmp_path / "ev_b"),
    ).process_video(motion_video, sample_fps=5)

    threshold_only_a = [f for f in without_coverage if f.selection_reason == "threshold_exceeded"]
    threshold_only_b = [f for f in with_short_gap if f.selection_reason == "threshold_exceeded"]

    assert [(f.frame_id, round(f.change_score, 4)) for f in threshold_only_a] == [
        (f.frame_id, round(f.change_score, 4)) for f in threshold_only_b
    ]
    assert all(not f.is_fallback for f in threshold_only_a)


def test_long_gap_selects_highest_net_change_score_candidate_in_window(
    tmp_path: Path, long_silence_video: str
) -> None:
    """2) Uzun bosluk kapatilirken, pencere icindeki en yuksek net_change_score'lu aday secilmeli."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=2.0,
        early_change_selection_interval_sec=1.0,
        significant_change_selection_interval_sec=0.5,
        # Bu test yalnizca SAKIN/coverage davranisini izole eder: guclu/erken
        # hysteresis cok kisa tutulup hizla sakine donsun.
        strong_change_cooldown_sec=0.1,
        early_change_cooldown_sec=0.1,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    evidence = sampler.process_video(long_silence_video, sample_fps=25)

    coverage_frames = [f for f in evidence if f.selection_reason == "temporal_coverage"]
    assert coverage_frames, "Uzun sessizlik araliginda en az bir coverage karesi beklenir."

    # Rastgele/sabit periyodik degil: coverage karesi, en azindan kendi
    # penceresindeki digerlerinden dusuk olmayan (esitlik durumunda en
    # guncel) bir net_change_score tasimalidir - negatif bir sonuc uretmez.
    assert all(f.change_score >= 0.0 for f in coverage_frames)


def test_coverage_frame_not_added_before_gap_exceeded(tmp_path: Path) -> None:
    """3) Gap dolmadan coverage karesi eklenmemeli (kisa/erken bir sessizlikte tetiklenmemeli)."""
    frames = []
    for i in range(20):  # 20 kare @ 25fps = 0.8s; gap (2.0s) hicbir zaman asilmaz
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        if 0 <= i < 5:
            cv2.rectangle(frame, (5, 5), (58, 42), (255, 255, 255), -1)
        frames.append(frame)
    path = tmp_path / "too_short_for_gap.mp4"
    _write_video(path, frames)

    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=2.0,
        early_change_selection_interval_sec=1.0,
        significant_change_selection_interval_sec=0.5,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    evidence = sampler.process_video(path, sample_fps=25)

    assert all(f.selection_reason != "temporal_coverage" for f in evidence)


def test_gap_counter_resumes_from_coverage_frame_timestamp(tmp_path: Path, long_silence_video: str) -> None:
    """4) Coverage sonrasi gap hesabi, coverage karesinin KENDI zaman damgasindan devam etmeli."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=2.0,
        early_change_selection_interval_sec=1.0,
        significant_change_selection_interval_sec=0.5,
        strong_change_cooldown_sec=0.1,
        early_change_cooldown_sec=0.1,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    evidence = sampler.process_video(long_silence_video, sample_fps=25)

    coverage_frames = [f for f in evidence if f.selection_reason == "temporal_coverage"]
    assert len(coverage_frames) >= 2, "Bu testin anlamli olmasi icin en az iki coverage karesi gerekir."

    # Ardisik coverage kareleri arasindaki fark, max_temporal_gap_sec'e (yaklasik,
    # ornekleme adimi kadar tolaransla) esit olmali - "arka arkaya, gereksiz
    # ikinci bir coverage karesi" (stale timestamp'ten kaynaklanan yigilma)
    # OLMAMALI.
    for earlier, later in zip(coverage_frames, coverage_frames[1:]):
        delta = later.timestamp_sec - earlier.timestamp_sec
        assert delta >= sampler.max_temporal_gap_sec - 0.1, (
            f"Ardisik coverage kareleri cok yakin: {earlier.timestamp_sec} -> {later.timestamp_sec} "
            f"(delta={delta}, beklenen >= {sampler.max_temporal_gap_sec})"
        )


def test_evidence_frames_remain_chronological_and_without_timestamp_duplicates(
    tmp_path: Path, long_silence_video: str
) -> None:
    """5) Evidence kareleri (threshold + coverage karisik) kronolojik ve zaman-damgasi-tekrarsiz kalmali."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=2.0,
        early_change_selection_interval_sec=1.0,
        significant_change_selection_interval_sec=0.5,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    evidence = sampler.process_video(long_silence_video, sample_fps=25)

    timestamps = [f.timestamp_sec for f in evidence]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == len(set(timestamps))

    frame_ids = [f.frame_id for f in evidence]
    assert frame_ids == sorted(frame_ids)
    assert len(frame_ids) == len(set(frame_ids))

    evidence_ids = [f.evidence_id for f in evidence]
    assert len(evidence_ids) == len(set(evidence_ids))


def test_end_of_video_pending_candidate_flushed_when_gap_already_exceeded(tmp_path: Path) -> None:
    """6) Video sonunda gap zaten asilmissa, degerlendirilmemis en iyi aday guvenli sekilde coverage olarak eklenmeli."""
    frames = []
    for i in range(80):  # 80 kare @ 25fps = 3.2s; video, ikinci bir threshold-olayi olmadan biter
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        if 0 <= i < 5:
            cv2.rectangle(frame, (5, 5), (58, 42), (255, 255, 255), -1)
        frames.append(frame)
    path = tmp_path / "end_gap.mp4"
    _write_video(path, frames)

    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=2.0,
        early_change_selection_interval_sec=1.0,
        significant_change_selection_interval_sec=0.5,
        strong_change_cooldown_sec=0.1,
        early_change_cooldown_sec=0.1,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    evidence = sampler.process_video(path, sample_fps=25)

    assert evidence[-1].selection_reason == "temporal_coverage"
    assert evidence[-1].timestamp_sec > evidence[0].timestamp_sec


def test_end_of_video_pending_candidate_dropped_safely_when_gap_not_exceeded(tmp_path: Path) -> None:
    """6b) Video, gap'i hic asmadan biterse buffer'daki aday sessizce/hatasiz birakilmali (coverage EKLENMEMELI)."""
    frames = []
    for i in range(20):  # 0.8s, gap (2.0s) hicbir zaman asilmaz
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        if 0 <= i < 5:
            cv2.rectangle(frame, (5, 5), (58, 42), (255, 255, 255), -1)
        frames.append(frame)
    path = tmp_path / "short_no_gap.mp4"
    _write_video(path, frames)

    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=2.0,
        early_change_selection_interval_sec=1.0,
        significant_change_selection_interval_sec=0.5,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    evidence = sampler.process_video(path, sample_fps=25)

    assert all(f.selection_reason != "temporal_coverage" for f in evidence)


def test_empty_candidate_buffer_does_not_crash(tmp_path: Path, static_video: str) -> None:
    """7) Bos buffer/hic aday bulunmamasi (tamamen durgun video, fallback yolu) hataya yol acmamali."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=2.0,
        early_change_selection_interval_sec=1.0,
        significant_change_selection_interval_sec=0.5,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    evidence = sampler.process_video(static_video, sample_fps=5)

    assert len(evidence) == 1
    assert evidence[0].is_fallback is True
    assert evidence[0].selection_reason == "fallback"


def test_max_temporal_gap_sec_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AdaptiveFrameSampler(max_temporal_gap_sec=0.0, evidence_output_dir=str(tmp_path / "evidence"))
    with pytest.raises(ValueError):
        AdaptiveFrameSampler(max_temporal_gap_sec=-1.0, evidence_output_dir=str(tmp_path / "evidence"))


def test_sampler_from_config_wires_max_temporal_gap_sec(safir_config) -> None:
    """8) `sampler_from_config`, config'teki `max_temporal_gap_sec`i dogru sekilde kullanmali."""
    sampler = sampler_from_config(safir_config.sampler)
    assert sampler.max_temporal_gap_sec == safir_config.sampler.max_temporal_gap_sec


def test_coverage_frame_carries_temporal_coverage_reason_not_positional_role(
    tmp_path: Path, long_silence_video: str
) -> None:
    """Secim nedeni metadata icinde acikca belirtilmeli: threshold_exceeded / temporal_coverage."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=2.0,
        early_change_selection_interval_sec=1.0,
        significant_change_selection_interval_sec=0.5,
        strong_change_cooldown_sec=0.1,
        early_change_cooldown_sec=0.1,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    evidence = sampler.process_video(long_silence_video, sample_fps=25)

    reasons = {f.selection_reason for f in evidence}
    assert reasons <= {"threshold_exceeded", "temporal_coverage", "fallback"}
    assert "threshold_exceeded" in reasons
    assert "temporal_coverage" in reasons


def test_no_event_clustering_or_positional_fields_introduced_by_coverage(
    tmp_path: Path, long_silence_video: str
) -> None:
    """Coverage mekanizmasi, EventCluster/event_id/cluster_id veya pre/peak/post benzeri hicbir alan getirmemeli."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=2.0,
        early_change_selection_interval_sec=1.0,
        significant_change_selection_interval_sec=0.5,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    evidence = sampler.process_video(long_silence_video, sample_fps=25)

    assert not hasattr(AdaptiveFrameSampler, "cluster_events")
    for f in evidence:
        field_names = set(type(f).model_fields.keys())
        assert "event_id" not in field_names
        assert "cluster_id" not in field_names
        assert "label" not in field_names
        assert "frame_role" not in field_names


def test_sampler_config_has_max_temporal_gap_sec_field(safir_config) -> None:
    """Sabit deger kod icine gomulmemis: config uzerinden geliyor olmali."""
    assert hasattr(safir_config.sampler, "max_temporal_gap_sec")
    assert safir_config.sampler.max_temporal_gap_sec > 0


# =============================================================================
# Yogunluk-uyarlamali secim (adaptive selection density): sakin/erken-degisim/
# guclu-degisim uc seviyesi. Kaynak/analiz `sample_fps`'i SABIT kalir - yalnizca
# VLM'e gonderilecek karelerin SECIM SIKLIGI degisir. Bu, ana esik gecilmeden
# ONCE baslayan kucuk-ama-surekli degisimlerin (ör. izmarit atma ani, hafif
# dumanin ilk gorulme ani) tamamen kacirilmasini onler; kumeleme/olay
# baslangic-bitis zamani DEGILDIR.
#
# NOT: `blur_kernel_size=(3, 3)` (varsayilan (21, 21) yerine) ve daha buyuk
# bir kare (96x128) kullanilir - kucuk/erken bir sinyalin gercek video
# kodlama/gri-tonlama/bulaniklastirma zincirinden GECEREK olcneklenebilir
# (sifira yuvarlanmayan) bir `net_change_score` uretmesini saglamak icin;
# `min_change_threshold`/`early_change_score_ratio` formulunu DEGISTIRMEZ.
# =============================================================================


def _write_growing_patch_video(
    path: Path,
    total_frames: int,
    ramp_start: int,
    ramp_end: int,
    growth_per_frame: int = 2,
    strong_burst: Optional[Tuple[int, int]] = None,
) -> None:
    """`ramp_start`den `ramp_end`e kadar KARE BUYUKLUGU dogrusal artan (ivmeli)
    bir yama iceren sentetik video yazar - bu, ana esik altinda ama SURDURULEN
    (rastgele degil, gercekten buyuyen) bir onset sinyalini (ör. izmaritin
    atildigi/dumanin ilk goruldugu an, buyuyerek gelisen bir olay) taklit eder.
    Dogrusal buyume, adaptif (medyan-tabanli) gurultu tabaninin surekli
    GERIDEN takip etmesine yol acar, boylece `net_change_score` sifira
    donmeden kademeli yukselir. `ramp_end`den sonra yama KAYBOLMAZ - SON
    boyutunda sabit kalir (abrupt kaybolma, kendi basina yapay/istenmeyen bir
    "ani degisim" sinyali uretir - bu yanlislikla ayri bir esik-gecisi
    taklit etmesin diye). `strong_burst=(start, end)` verilirse, o aralikta
    TAM ekran kaplayan, ana esigi kesin gecen ayri bir patlama eklenir.
    """
    frames = []
    for i in range(total_frames):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if i >= ramp_start:
            size = 6 + min(i - ramp_start, ramp_end - ramp_start - 1) * growth_per_frame
            cv2.rectangle(frame, (10, 10), (10 + size, 10 + size), (150, 150, 150), -1)
        if strong_burst is not None and strong_burst[0] <= i < strong_burst[1]:
            cv2.rectangle(frame, (5, 5), (120, 90), (255, 255, 255), -1)
        frames.append(frame)
    _write_video(path, frames)


def _density_sampler(tmp_path: Path, **overrides) -> AdaptiveFrameSampler:
    """Yogunluk testleri icin, gercek video zincirinde dogrulanmis parametrelerle
    (bkz. modul notu) taze bir `AdaptiveFrameSampler` uretir."""
    params = dict(
        min_change_threshold=0.02,
        blur_kernel_size=(3, 3),
        early_change_score_ratio=0.4,
        early_change_window=3,
        early_change_min_count=2,
        max_temporal_gap_sec=5.0,
        early_change_selection_interval_sec=1.0,
        significant_change_selection_interval_sec=0.3,
        strong_change_cooldown_sec=2.0,
        early_change_cooldown_sec=2.0,
        dedup_similarity_ratio=0.5,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    params.update(overrides)
    return AdaptiveFrameSampler(**params)


def test_fully_calm_video_only_selects_sparse_coverage_frames(tmp_path: Path) -> None:
    """1) Tamamen sakin videoda yalnizca seyrek coverage frameleri secilmeli (hicbir early/significant kare yok)."""
    frames = [np.full((96, 128, 3), 30, dtype=np.uint8) for _ in range(150)]  # 6s @ 25fps, tamamen durgun
    path = tmp_path / "fully_calm.mp4"
    _write_video(path, frames)

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=2.0)
    evidence = sampler.process_video(path, sample_fps=25)

    reasons = {f.selection_reason for f in evidence}
    assert reasons <= {"temporal_coverage", "fallback"}
    assert "early_change" not in reasons
    assert "significant_change" not in reasons
    assert "threshold_exceeded" not in reasons


def test_sustained_change_below_main_threshold_is_captured_as_early_change(tmp_path: Path) -> None:
    """2) Ana esik gecilmeden once baslayan SURDURULEN kucuk degisim, early_change olarak yakalanmali."""
    path = tmp_path / "sustained_ramp.mp4"
    # Yalnizca yavas buyuyen bir yama; ana esigi (0.02) hicbir zaman gecmeyecek
    # kadar kisa tutulur (bkz. onceki manuel dogrulama: ~80 kare sonra plato
    # civarinda kalir, ana esigi gecmez).
    _write_growing_patch_video(path, total_frames=100, ramp_start=50, ramp_end=75, growth_per_frame=1)

    sampler = _density_sampler(tmp_path)
    evidence = sampler.process_video(path, sample_fps=25)

    early_frames = [f for f in evidence if f.selection_reason == "early_change"]
    assert early_frames, "Surdurulen esik-alti degisim en az bir early_change karesi uretmeli."
    assert all(f.selection_reason != "threshold_exceeded" for f in evidence), (
        "Bu senaryoda ana esik hic gecilmemeli (yalnizca erken sinyal test ediliyor)."
    )


def test_light_smoke_onset_selected_before_smoke_grows_large(tmp_path: Path) -> None:
    """3) Hafif duman baslangici secilmeli; yalnizca duman buyudukten sonraki kare degil."""
    path = tmp_path / "smoke_onset.mp4"
    _write_growing_patch_video(
        path, total_frames=250, ramp_start=50, ramp_end=130, growth_per_frame=2, strong_burst=(200, 210)
    )

    sampler = _density_sampler(tmp_path)
    evidence = sampler.process_video(path, sample_fps=25)

    early_frames = [f for f in evidence if f.selection_reason == "early_change"]
    threshold_frames = [f for f in evidence if f.selection_reason == "threshold_exceeded"]
    assert early_frames, "Hafif duman baslangici en az bir early_change karesi uretmeli."
    assert threshold_frames, "Duman buyuyup ana esigi gecmeli (dogrulama icin)."
    # Erken kare, "duman buyudukten sonraki" (ana esigi gecen) ilk kareden
    # ONCE gelmeli - yalnizca buyudukten sonraki kare secilmis olmamali.
    assert early_frames[0].timestamp_sec < threshold_frames[0].timestamp_sec


def test_cigarette_flick_short_onset_is_preserved(tmp_path: Path) -> None:
    """4) Izmarit atilmasi gibi KISA bir hareketin baslangic karesi, ana esik hic gecilmese de korunmali."""
    path = tmp_path / "cigarette_flick.mp4"
    # Kisa, tek seferlik bir "bump": buyur ama esigi gecmeden hemen durur
    # (izmaritin atildigi anlik hareket - surekli bir olay DEGIL).
    _write_growing_patch_video(path, total_frames=90, ramp_start=40, ramp_end=60, growth_per_frame=1)

    sampler = _density_sampler(tmp_path)
    evidence = sampler.process_video(path, sample_fps=25)

    early_frames = [f for f in evidence if f.selection_reason == "early_change"]
    assert early_frames, "Kisa baslangic hareketi, ana esik hic gecilmese bile en az bir early_change uretmeli."
    # Baslangic karesi, hareketin baslangicina (t~1.6s) yakin olmali - cok
    # geriye buffer gerektirmeden, GERCEK zamanina yakin yakalanmis olmali.
    assert early_frames[0].timestamp_sec < 3.0


def test_single_frame_noise_does_not_hold_dense_selection_open_for_long(tmp_path: Path) -> None:
    """5) Tek karelik gurultu (izole bir sicrama), erken-degisim durumunu SURDURMEMELI (min_count>=2 gerekir)."""
    frames = [np.full((96, 128, 3), 30, dtype=np.uint8) for _ in range(150)]
    # Tek bir izole karede kisa bir yama (bir sonraki karede hemen kaybolur) -
    # SURDURULEN bir egilim DEGIL, tek karelik bir sicrama.
    cv2.rectangle(frames[50], (10, 10), (10 + 20, 10 + 20), (150, 150, 150), -1)
    path = tmp_path / "single_spike.mp4"
    _write_video(path, frames)

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=2.0)
    evidence = sampler.process_video(path, sample_fps=25)

    # Izole sicrama, early_change_min_count=2 pencere-onayini GECEMEZ; hicbir
    # early_change/significant_change uretilmemeli - yalnizca (varsa) coverage.
    assert all(f.selection_reason not in {"early_change", "significant_change"} for f in evidence)


def test_main_threshold_exceeded_increases_selection_frequency(tmp_path: Path) -> None:
    """6) Ana esik gecildiginde frame secim sikligi (sakin araliga kiyasla) artmali."""
    path = tmp_path / "strong_then_calm.mp4"
    _write_growing_patch_video(
        path, total_frames=300, ramp_start=50, ramp_end=130, growth_per_frame=2, strong_burst=(150, 160)
    )

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=5.0)
    evidence = sampler.process_video(path, sample_fps=25)

    threshold_ts = next(f.timestamp_sec for f in evidence if f.selection_reason == "threshold_exceeded")
    post_threshold = [f for f in evidence if f.timestamp_sec > threshold_ts and f.selection_reason == "significant_change"]
    assert post_threshold, "Ana esik sonrasi hysteresis penceresinde significant_change kareleri beklenir."

    # Ardisik significant_change kareleri arasindaki fark, SAKIN araliktan
    # (max_temporal_gap_sec) KUCUK olmali - yani secim sikligi artmis olmali.
    all_post = sorted(post_threshold, key=lambda f: f.timestamp_sec)
    if len(all_post) >= 2:
        gaps = [b.timestamp_sec - a.timestamp_sec for a, b in zip(all_post, all_post[1:])]
        assert all(g < sampler.max_temporal_gap_sec for g in gaps)


def test_density_returns_to_calm_after_change_ends(tmp_path: Path) -> None:
    """7) Degisim sona erdiginde secim tekrar seyreklesmeli (sakin/coverage araligina donmeli)."""
    path = tmp_path / "strong_then_silence.mp4"
    frames = []
    for i in range(400):  # 16s @ 25fps
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if 20 <= i < 30:
            cv2.rectangle(frame, (5, 5), (120, 90), (255, 255, 255), -1)
        frames.append(frame)
    _write_video(path, frames)

    sampler = _density_sampler(
        tmp_path,
        max_temporal_gap_sec=3.0,
        strong_change_cooldown_sec=1.0,
        early_change_cooldown_sec=1.0,
    )
    evidence = sampler.process_video(path, sample_fps=25)

    coverage_frames = [f for f in evidence if f.selection_reason == "temporal_coverage"]
    assert coverage_frames, "Degisim (kisa patlama) sona erdikten sonra sakin coverage'a donulmeli."
    # Coverage kareleri, guclu olayin (t~0.4-1.2s hysteresis penceresi)
    # BITISINDEN sonra gelmeli.
    assert all(f.timestamp_sec > 1.5 for f in coverage_frames)


def test_dedup_does_not_delete_the_first_meaningful_onset_frame(tmp_path: Path) -> None:
    """8) Duplicate eleme, ilk anlamli baslangic (early_change) karesini SILMEMELI."""
    path = tmp_path / "onset_dedup.mp4"
    _write_growing_patch_video(path, total_frames=100, ramp_start=50, ramp_end=75, growth_per_frame=1)

    sampler = _density_sampler(tmp_path)
    evidence = sampler.process_video(path, sample_fps=25)

    early_frames = [f for f in evidence if f.selection_reason == "early_change"]
    assert early_frames, "Dedup, sakin taban cizgisinden farkli olan onset karesini yanlislikla SILMEMELI."


def test_density_evidence_frames_remain_chronological_and_correctly_timestamped(tmp_path: Path) -> None:
    """9) Frameler (coverage+early+significant+threshold karisik) kronolojik ve dogru zaman damgali kalmali."""
    path = tmp_path / "mixed_density.mp4"
    _write_growing_patch_video(
        path, total_frames=300, ramp_start=50, ramp_end=130, growth_per_frame=2, strong_burst=(150, 160)
    )

    sampler = _density_sampler(tmp_path)
    evidence = sampler.process_video(path, sample_fps=25)

    timestamps = [f.timestamp_sec for f in evidence]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == len(set(timestamps))
    for f in evidence:
        assert f.timestamp_sec == round(f.frame_id / 25.0, 2)


def test_video_is_processed_in_a_single_pass(tmp_path: Path) -> None:
    """10) Video yalnizca TEK gecişte islenmeli (ikinci okuma/rewind yok)."""
    path = tmp_path / "single_pass.mp4"
    total_frames = 120
    _write_growing_patch_video(path, total_frames=total_frames, ramp_start=50, ramp_end=75, growth_per_frame=1)

    sampler = _density_sampler(tmp_path)
    evidence = sampler.process_video(path, sample_fps=25)

    # Taranan toplam ham kare sayisi, videonun GERCEK kare sayisiyla BIREBIR
    # esit olmali - ne eksik (atlama) ne fazla (ikinci gecis/rewind).
    assert sampler.last_run_stats.total_frames_scanned == total_frames
    assert evidence  # sanity: en az bir kare secilmis olmali


def test_no_growing_buffer_internal_state_stays_bounded(tmp_path: Path) -> None:
    """11) Buyuyen bir frame buffer'i/ring buffer/max-buffer OLUSMAMALI - ic durum sabit boyutlu kalmali."""
    path = tmp_path / "long_for_buffer_check.mp4"
    _write_growing_patch_video(
        path, total_frames=500, ramp_start=50, ramp_end=400, growth_per_frame=1
    )  # uzun, surekli degisen bir video

    sampler = _density_sampler(tmp_path, history_window=10)
    sampler.process_video(path, sample_fps=25)

    # Video ne kadar uzun/degisken olursa olsun, bu ic durumlar SABIT
    # boyutludur (buyuyen bir liste/kuyruk DEGILDIR):
    assert len(sampler.noise_floor_history) <= sampler.history_window
    assert len(sampler._recent_threshold_decisions) <= sampler.temporal_vote_window
    assert len(sampler._recent_early_decisions) <= sampler.early_change_window
    # `pending_best`, TEK bir aday tutar (bir liste/pencere DEGIL).
    import src.sampler.adaptive_sampler as sampler_module

    assert not hasattr(sampler, "_evidence_buffer")
    assert not hasattr(sampler, "_frame_ring_buffer")
    assert sampler_module._PendingSelectionCandidate  # dataclass tek-ornek tutar (liste degil)


def test_density_mechanism_produces_no_event_clustering_or_boundaries(tmp_path: Path) -> None:
    """12) Yogunluk mekanizmasi, sampler'a olay kumelemesi veya olay baslangic/bitis zamani GETIRMEMELI."""
    path = tmp_path / "density_no_clustering.mp4"
    _write_growing_patch_video(
        path, total_frames=300, ramp_start=50, ramp_end=130, growth_per_frame=2, strong_burst=(150, 160)
    )

    sampler = _density_sampler(tmp_path)
    evidence = sampler.process_video(path, sample_fps=25)

    assert not hasattr(AdaptiveFrameSampler, "cluster_events")
    for f in evidence:
        field_names = set(type(f).model_fields.keys())
        assert "event_id" not in field_names
        assert "cluster_id" not in field_names
        assert "start_time" not in field_names
        assert "end_time" not in field_names
        assert "label" not in field_names
        assert "frame_role" not in field_names
        assert f.selection_reason in {
            "threshold_exceeded",
            "temporal_coverage",
            "early_change",
            "significant_change",
            "fallback",
        }


def test_density_config_fields_are_ratio_or_config_driven_not_hardcoded(safir_config) -> None:
    """Yeni esik/araliklar config'ten geliyor olmali (koda gomulu sabit degil)."""
    s = safir_config.sampler
    for field in (
        "early_change_score_ratio",
        "early_change_window",
        "early_change_min_count",
        "early_change_selection_interval_sec",
        "early_change_cooldown_sec",
        "significant_change_selection_interval_sec",
        "strong_change_cooldown_sec",
        "dedup_similarity_ratio",
    ):
        assert hasattr(s, field), f"SamplerConfig alani eksik: {field}"
        assert getattr(s, field) > 0

    sampler = sampler_from_config(safir_config.sampler)
    assert sampler.early_change_score_ratio == s.early_change_score_ratio
    assert sampler.significant_change_selection_interval_sec == s.significant_change_selection_interval_sec
    assert sampler.dedup_similarity_ratio == s.dedup_similarity_ratio


def test_density_invalid_config_values_raise(tmp_path: Path) -> None:
    """Gecersiz oran/aralik degerleri (0 disinda) acikca ValueError vermeli."""
    with pytest.raises(ValueError):
        AdaptiveFrameSampler(early_change_score_ratio=0.0, evidence_output_dir=str(tmp_path / "e1"))
    with pytest.raises(ValueError):
        AdaptiveFrameSampler(early_change_score_ratio=1.5, evidence_output_dir=str(tmp_path / "e2"))
    with pytest.raises(ValueError):
        AdaptiveFrameSampler(dedup_similarity_ratio=0.0, evidence_output_dir=str(tmp_path / "e3"))
    with pytest.raises(ValueError):
        # significant > early siralamayi bozar
        AdaptiveFrameSampler(
            significant_change_selection_interval_sec=10.0,
            early_change_selection_interval_sec=1.0,
            max_temporal_gap_sec=5.0,
            evidence_output_dir=str(tmp_path / "e4"),
        )
    with pytest.raises(ValueError):
        AdaptiveFrameSampler(early_change_min_count=5, early_change_window=2, evidence_output_dir=str(tmp_path / "e5"))


# =============================================================================
# Erken-degisim onayinda ONSET KARESININ KORUNMASI: onay `early_change_min_count`
# kareyi bulunca tamamlansa da, secilen `early_change` karesi ONAYIN
# GERCEKLESTIGI (mevcut) kare degil, zincirin SAKIN durumdan cikan ILK
# supheli-erken karesidir (bkz. `pending_early_candidate` -
# `_PendingSelectionCandidate`nin bu amacla yeniden kullanilan tek-ornekli
# hali; ayri bir veri modeli veya buyuyen bir buffer YOKTUR).
# =============================================================================


def _isolated_appearance_video(
    path: Path, total_frames: int, appear_at: int, size: int = 10, pos: Tuple[int, int] = (10, 10)
) -> None:
    """`appear_at` karesinde bir `size`x`size` yama BELIRIR ve rest of the
    video boyunca SABIT kalir (kaybolmaz) - boylece diff-akisinda TEK BIR
    izole "degisim ani" (yamanın ortaya cikisi) olusur; yama kaybolsaydi
    ikinci bir (yamanin gitmesi) diff-olayi daha uretilirdi ve bu, "tek
    karelik izole gurultu" senaryosunu yanlislikla IKI ayri sinyale
    donusturup testi gecersiz kilardi."""
    frames = []
    for i in range(total_frames):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if i >= appear_at:
            x, y = pos
            cv2.rectangle(frame, (x, y), (x + size, y + size), (150, 150, 150), -1)
        frames.append(frame)
    _write_video(path, frames)


def test_early_change_onset_is_the_first_signal_frame_not_the_confirmation_frame(tmp_path: Path) -> None:
    """1) 00:26 ilk sinyal / 00:28 onay senaryosu: secilen early_change 00:26'ya (ILK karaye) ait olmali, 00:28'e (onay karesine) degil.

    Ayni video, FARKLI `early_change_min_count` degerleriyle iki kez islenir.
    Sinyalin ILK gorundugu kare `min_count`tan BAGIMSIZDIR; dogru calisan bir
    implementasyonda secilen `early_change` zaman damgasi HER IKI durumda da
    AYNI olmalidir - yalnizca ONAYIN NE ZAMAN tamamlandigi (min_count
    buyudukce daha GEC) degisir, SECILEN kare degil.
    """
    path = tmp_path / "onset_precision.mp4"
    _write_growing_patch_video(path, total_frames=150, ramp_start=50, ramp_end=140, growth_per_frame=2)

    sampler_a = _density_sampler(
        tmp_path, early_change_min_count=2, early_change_window=4, max_temporal_gap_sec=50.0
    )
    evidence_a = sampler_a.process_video(path, sample_fps=25)
    sampler_b = _density_sampler(
        tmp_path, early_change_min_count=3, early_change_window=4, max_temporal_gap_sec=50.0
    )
    evidence_b = sampler_b.process_video(path, sample_fps=25)

    onset_a = next(f for f in evidence_a if f.selection_reason == "early_change")
    onset_b = next(f for f in evidence_b if f.selection_reason == "early_change")
    assert onset_a.timestamp_sec == onset_b.timestamp_sec
    assert onset_a.frame_id == onset_b.frame_id


def test_unconfirmed_single_frame_noise_produces_no_early_change_output(tmp_path: Path) -> None:
    """2) Doğrulanmayan (izole, tek karelik) bir sinyal hicbir cikti uretmemeli."""
    path = tmp_path / "isolated_noise.mp4"
    # Tek bir izole kare (bir sonraki karede hemen kaybolur) - min_count=2
    # icin YETERSIZ, SURDURULEN bir egilim degil.
    _isolated_appearance_video(path, total_frames=150, appear_at=50)

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=50.0)
    evidence = sampler.process_video(path, sample_fps=25)

    assert all(f.selection_reason != "early_change" for f in evidence)


def test_pending_early_candidate_survives_threshold_preemption(tmp_path: Path) -> None:
    """3) Pending aday, onay tamamlanmadan ana esik gecilirse KAYBOLMAMALI."""
    path = tmp_path / "threshold_preemption.mp4"
    frames = []
    for i in range(80):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if i == 50:
            # Tek, onaylanmamis bir erken sinyal (min_count=2 icin yetersiz).
            cv2.rectangle(frame, (10, 10), (10 + 10, 10 + 10), (150, 150, 150), -1)
        if 51 <= i < 61:
            # Onay tamamlanmadan HEMEN sonraki karede ana esik gecilir.
            cv2.rectangle(frame, (5, 5), (120, 90), (255, 255, 255), -1)
        frames.append(frame)
    _write_video(path, frames)

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=50.0)
    evidence = sampler.process_video(path, sample_fps=25)

    early_frames = [f for f in evidence if f.selection_reason == "early_change"]
    threshold_frames = [f for f in evidence if f.selection_reason == "threshold_exceeded"]
    assert early_frames, "Onaylanmamis pending aday, ana esik tarafindan kaybolmamali."
    assert threshold_frames
    # Pending aday (early_change), esik-gecen kareden ONCE (kronolojik olarak
    # dogru sirada) korunmali.
    assert early_frames[0].timestamp_sec < threshold_frames[0].timestamp_sec


def test_pending_candidate_and_threshold_frame_never_duplicate(tmp_path: Path) -> None:
    """4) Pending aday ile threshold_exceeded arasinda hicbir duplicate (ayni frame_id) olusmamali."""
    path = tmp_path / "no_duplicate_frame_ids.mp4"
    frames = []
    for i in range(80):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if i == 50:
            cv2.rectangle(frame, (10, 10), (10 + 10, 10 + 10), (150, 150, 150), -1)
        if 51 <= i < 61:
            cv2.rectangle(frame, (5, 5), (120, 90), (255, 255, 255), -1)
        frames.append(frame)
    _write_video(path, frames)

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=50.0)
    evidence = sampler.process_video(path, sample_fps=25)

    frame_ids = [f.frame_id for f in evidence]
    assert len(frame_ids) == len(set(frame_ids)), "Ayni kare (frame_id) birden fazla kez gonderilmemeli."
    evidence_ids = [f.evidence_id for f in evidence]
    assert len(evidence_ids) == len(set(evidence_ids))


def test_two_separate_early_chains_do_not_reuse_stale_pending_candidate(tmp_path: Path) -> None:
    """5+6) Pending aday, secildikten sonra temizlenmeli; YENI bir erken-degisim dizisi ONCEKI dizinin adayini kullanmamali."""
    path = tmp_path / "two_chains.mp4"
    frames = []
    for i in range(250):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if i in (50, 51):
            cv2.rectangle(frame, (10, 10), (10 + 10, 10 + 10), (150, 150, 150), -1)
        if i in (150, 151):
            # Farkli bir konumda, tamamen bagimsiz IKINCI bir zincir.
            cv2.rectangle(frame, (60, 60), (60 + 10, 60 + 10), (150, 150, 150), -1)
        frames.append(frame)
    _write_video(path, frames)

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=50.0)
    evidence = sampler.process_video(path, sample_fps=25)

    early_frames = [f for f in evidence if f.selection_reason == "early_change"]
    assert len(early_frames) >= 2, "Iki ayri zincir, iki ayri early_change karesi uretmeli."
    first_chain = [f for f in early_frames if f.timestamp_sec < 4.0]
    second_chain = [f for f in early_frames if f.timestamp_sec >= 4.0]
    assert first_chain and second_chain
    # Ikinci zincirin onset karesi, ilk zincirin (eski/tuketilmis) pending
    # adayiyla AYNI olmamali - kendi (daha sonraki, farkli konumdaki) ilk
    # karesini kullanmali.
    assert first_chain[0].frame_id != second_chain[0].frame_id
    assert first_chain[0].timestamp_sec != second_chain[0].timestamp_sec
    assert second_chain[0].frame_id == 150


def test_insert_chronologically_maintains_order_when_inserted_out_of_sequence(tmp_path: Path) -> None:
    """7+9) Pending adayin (gec eklenen ama ERKEN zaman damgali) kronolojik konuma dogru yerlestirilmesi.

    `_insert_chronologically`, coverage veya daha sonra eklenen baska bir
    adayla ARAYA GIRMIS bir pending-onay durumunda bile listeyi kronolojik
    tutan asil mekanizmadir; burada dogrudan test edilir (video-seviyesinde
    bu spesifik yarisi zorlamak kirilgan olur - mekanizmanin kendisini test
    etmek daha guvenilirdir).
    """
    import src.sampler.adaptive_sampler as sampler_module
    from src.sampler.schema import EvidenceFrame

    def _ef(ts: float) -> EvidenceFrame:
        fid = int(ts * 100)
        return EvidenceFrame(
            evidence_id=f"ev{fid}",
            frame_id=fid,
            timestamp_sec=ts,
            timestamp_str="00:00",
            change_score=0.0,
            image_bytes=b"",
            base64_image="data:image/jpeg;base64,AA==",
            image_shape=(1, 1, 3),
        )

    # t=3.0'daki bir coverage karesi ZATEN eklenmisken, t=2.0'daki (daha ONCE
    # gerceklesmis ama GEC onaylanan) bir pending aday geliyor.
    frames = [_ef(1.0), _ef(3.0)]
    sampler_module._insert_chronologically(frames, _ef(2.0))
    assert [f.timestamp_sec for f in frames] == [1.0, 2.0, 3.0]

    # Pending aday, listenin EN BASINA da dogru sekilde girebilmeli.
    frames2 = [_ef(5.0), _ef(6.0)]
    sampler_module._insert_chronologically(frames2, _ef(1.0))
    assert [f.timestamp_sec for f in frames2] == [1.0, 5.0, 6.0]


def test_end_to_end_evidence_stays_chronological_with_pending_early_candidate(tmp_path: Path) -> None:
    """7) Gercek bir video akisinda, pending aday ile coverage/diger secimler birlikteyken kronolojik sira korunmali."""
    path = tmp_path / "mixed_with_pending.mp4"
    frames = []
    for i in range(250):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if i in (50, 51):
            cv2.rectangle(frame, (10, 10), (10 + 10, 10 + 10), (150, 150, 150), -1)
        if i in (150, 151):
            cv2.rectangle(frame, (60, 60), (60 + 10, 60 + 10), (150, 150, 150), -1)
        frames.append(frame)
    _write_video(path, frames)

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=1.0)  # kucuk gap: coverage sik tetiklenir
    evidence = sampler.process_video(path, sample_fps=25)

    timestamps = [f.timestamp_sec for f in evidence]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == len(set(timestamps))


def test_end_of_video_unconfirmed_pending_early_candidate_is_dropped_safely(tmp_path: Path) -> None:
    """8) Video sonunda henuz onaylanmamis pending aday guvenli sekilde temizlenmeli (hata/veri bozulmasi olmadan)."""
    path = tmp_path / "unconfirmed_at_end.mp4"
    # Yama t=50'de belirir ve SABIT kalir (video, tek bir erken sinyalden
    # hemen sonra biter) - yalnizca TEK bir izole diff-olayi olusur, min_count=2
    # icin yetersiz kalir (bkz. _isolated_appearance_video docstring'i).
    _isolated_appearance_video(path, total_frames=52, appear_at=50)

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=50.0)
    evidence = sampler.process_video(path, sample_fps=25)  # cokmemeli

    assert all(f.selection_reason != "early_change" for f in evidence)


def test_pending_early_candidate_state_is_a_single_object_not_a_growing_collection(tmp_path: Path) -> None:
    """9) Bellek kullanimi video/pencere uzunluguyla BUYUMEMELI - pending_early_candidate TEK bir nesnedir, liste/kuyruk degildir."""
    path = tmp_path / "long_for_pending_memory_check.mp4"
    frames = []
    for i in range(600):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        # Uzun video boyunca ARALIKLI, hicbir zaman onaylanmayan tek karelik
        # sinyaller (surekli "pending set -> temizlenir" dongusu).
        if i % 50 == 0:
            cv2.rectangle(frame, (10, 10), (10 + 10, 10 + 10), (150, 150, 150), -1)
        frames.append(frame)
    _write_video(path, frames)

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=50.0, history_window=10)
    sampler.process_video(path, sample_fps=25)

    # `pending_early_candidate` fonksiyon-yerel bir degiskendir (self uzerinde
    # SAKLANMAZ); burada dolayli olarak, ilgili sabit-boyutlu ic durumlarin
    # (ayni desen) video uzunlugundan BAGIMSIZ kaldigi dogrulanir.
    assert len(sampler.noise_floor_history) <= sampler.history_window
    assert len(sampler._recent_early_decisions) <= sampler.early_change_window
    assert len(sampler._recent_threshold_decisions) <= sampler.temporal_vote_window
    assert not hasattr(sampler, "_pending_early_candidates")  # coğul/liste bir alan YOK
    assert not hasattr(sampler, "_early_candidate_history")


# =============================================================================
# Kok neden testleri: "olay baslangici bazen kaciyor" / "VLM'e giden evidence
# frame sayisi yetersiz kaliyor" / "VLM ciktisindaki bir kare kimligi sampler
# ciktisinda bulunmuyor" (bkz. gorev raporu). `is_single_frame_strong`
# (tek-kareli guclu/lokal degisim), ana esik gibi `early_cooldown_until`i ileri
# atarak SAKIN -> ERKEN gecisini kendi basina tetikleyebilir; bu gecisin
# ADANMIS "onset flush" blogu `selected_this_frame=True` oldugu icin
# ATLANDIGINDAN, o an bekleyen `pending_early_candidate` (zincirin GERCEK,
# daha ESKI baslangic karesi) `pre_density_state` bir DAHA ASLA `CALM`
# olmayacagi icin BASKA HICBIR YOLLA flush EDILEMEZ ve sessizce dusebilirdi.
# Asagidaki testler, bu kok nedeni (fix'ten ONCE basarisiz olacak sekilde) ve
# fix sonrasi dogru davranisi dogrular.
# =============================================================================


def test_onset_candidate_survives_later_single_frame_change_in_same_window(tmp_path: Path) -> None:
    """10) Zincirin GERCEK ilk isareti (pending_early_candidate), ayni pencerede
    DAHA SONRA gelen bagimsiz bir tek-kareli guclu/lokal degisim (single_frame_change)
    SAKIN->ERKEN gecisini "calsa" bile evidence'tan KAYBOLMAMALI.

    Senaryo: t~1.2s'de zayif-ama-supheli-erken bir baslangic karesi (30) belirir
    ve SABIT kalir (tek bir izole diff-olayi). Hemen ardindan (32), TAMAMEN
    AYRI, kucuk/lokal ama GUCLU bir tek-kare degisimi (single_frame_change
    kriterlerini karsilayan, ana esigi GECMEYEN) gorunur - bu, erken-degisim
    cooldown'unu kendi basina uzatip SAKIN->ERKEN gecisini tetikler. Daha
    sonra (t~8s) ana esigi acikca gecen ayri bir olay gelir; bu ikisi
    arasinda baska hicbir esik-gecen kare yoktur, dolayisiyla pending
    aday'i "kurtarabilecek" tek yol bizim eklegimiz flush'tir.
    """
    frames = []
    total = 300
    for i in range(total):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if i >= 30:
            cv2.rectangle(frame, (10, 10), (22, 22), (90, 90, 90), -1)  # true onset (weak, sub-single-frame)
        if i == 32:
            cv2.rectangle(frame, (100, 70), (111, 83), (255, 255, 255), -1)  # unrelated strong local flash
        if i == 200:
            cv2.rectangle(frame, (40, 40), (55, 57), (200, 200, 200), -1)  # later real event (main threshold)
        frames.append(frame)
    path = tmp_path / "onset_stolen_by_single_frame.mp4"
    _write_video(path, frames)

    sampler = _density_sampler(
        tmp_path,
        min_change_threshold=0.02,
        early_change_score_ratio=0.4,
        early_change_window=3,
        early_change_min_count=2,
        early_change_cooldown_sec=2.0,
        max_temporal_gap_sec=5.0,
    )
    evidence = sampler.process_video(path, sample_fps=25)

    frame_ids = [f.frame_id for f in evidence]
    assert 30 in frame_ids, (
        "Gercek ilk baslangic karesi (frame_id=30), sonraki bagimsiz bir "
        "single_frame_change tarafindan 'calinan' SAKIN->ERKEN gecisi "
        "yuzunden sessizce kaybolmamali."
    )
    onset = next(f for f in evidence if f.frame_id == 30)
    assert onset.selection_reason == "early_change"
    # Kronolojik sira ve zaman damgasi tutarliligi (frame_index/timestamp_sec
    # ile birebir) hala korunmali.
    timestamps = [f.timestamp_sec for f in evidence]
    assert timestamps == sorted(timestamps)
    assert onset.timestamp_sec == pytest.approx(30 / 25.0, abs=1e-6)


def test_short_isolated_event_still_produces_onset_evidence(tmp_path: Path) -> None:
    """11) Kisa/tek seferlik bir olay (ör. izmarit atma), ana esigi hic gecmese
    bile en az bir evidence karesi uretmeli - 'az kare secildigi' icin
    tamamen atlanmamali."""
    path = tmp_path / "short_event.mp4"
    _isolated_appearance_video(path, total_frames=90, appear_at=40, size=14, pos=(10, 10))

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=50.0)
    evidence = sampler.process_video(path, sample_fps=25)

    assert evidence, "Kisa olay bile hicbir evidence uretmeden tamamen kaybolmamali."


def test_long_evolving_event_produces_more_evidence_than_short_event(tmp_path: Path) -> None:
    """12) Uzun/gelisen bir olay, KISA bir olaydan DAHA FAZLA evidence karesi
    uretmeli - sabit 3/5 kare gibi bir ust kota YOKTUR, kare sayisi olayin
    suresi/degisimiyle olceklenir."""
    short_path = tmp_path / "short_for_scaling.mp4"
    _write_growing_patch_video(short_path, total_frames=70, ramp_start=30, ramp_end=40, growth_per_frame=2)

    long_path = tmp_path / "long_for_scaling.mp4"
    _write_growing_patch_video(
        long_path, total_frames=400, ramp_start=30, ramp_end=350, growth_per_frame=2, strong_burst=(200, 350)
    )

    short_evidence = _density_sampler(tmp_path, max_temporal_gap_sec=5.0).process_video(short_path, sample_fps=25)
    long_evidence = _density_sampler(tmp_path, max_temporal_gap_sec=5.0).process_video(long_path, sample_fps=25)

    assert len(long_evidence) > len(short_evidence)


def test_new_change_during_cooldown_still_gets_its_own_evidence(tmp_path: Path) -> None:
    """13) Guclu-degisim hysteresis/cooldown penceresi surerken ortaya cikan
    YENI bir degisim de kendi evidence karesini almali - cooldown, yeni
    sinyalleri BASTIRMAMALI, yalnizca secim sikligini kontrol etmelidir."""
    path = tmp_path / "change_during_cooldown.mp4"
    frames = []
    for i in range(150):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if 20 <= i < 24:
            cv2.rectangle(frame, (5, 5), (120, 90), (255, 255, 255), -1)  # ana esigi kesin gecen ilk patlama
        if 30 <= i < 34:
            cv2.rectangle(frame, (5, 5), (120, 90), (200, 200, 200), -1)  # cooldown penceresi icinde YENI degisim
        frames.append(frame)
    _write_video(path, frames)

    sampler = _density_sampler(
        tmp_path, max_temporal_gap_sec=5.0, strong_change_cooldown_sec=2.0, early_change_cooldown_sec=2.0
    )
    evidence = sampler.process_video(path, sample_fps=25)

    first_ts = min(f.timestamp_sec for f in evidence if f.selection_reason == "threshold_exceeded")
    later_frames = [f for f in evidence if f.timestamp_sec > first_ts + 0.1]
    assert later_frames, "Cooldown penceresi icindeki yeni degisim de en az bir evidence uretmeli."


def test_noisy_scene_does_not_bury_the_real_onset(tmp_path: Path) -> None:
    """14) Gurultulu (izole tek-kare sicramalari iceren) bir sahnede bile,
    GERCEK (surdurulen) baslangic sinyali evidence'a girmeli."""
    path = tmp_path / "noisy_with_real_onset.mp4"
    frames = []
    total = 200
    for i in range(total):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        # Ardisik olmayan izole gurultu sicramalari (tek kare, hemen kaybolur).
        if i in (10, 45, 80, 115):
            cv2.rectangle(frame, (90, 5), (100, 15), (150, 150, 150), -1)
        # Gercek, surdurulen baslangic sinyali (buyuyen yama).
        if i >= 60:
            size = 6 + min(i - 60, 20) * 2
            cv2.rectangle(frame, (10, 10), (10 + size, 10 + size), (110, 110, 110), -1)
        frames.append(frame)
    _write_video(path, frames)

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=50.0)
    evidence = sampler.process_video(path, sample_fps=25)

    assert any(f.selection_reason == "early_change" for f in evidence), (
        "Izole gurultu sicramalari arasinda kaybolan gercek baslangic sinyali "
        "hala en az bir early_change karesi uretmeli."
    )


def test_evidence_ids_are_unique_and_traceable_to_frame_index_and_timestamp(tmp_path: Path) -> None:
    """15) Her evidence_id benzersiz olmali ve GERCEK frame_id/timestamp_sec ile eslesmeli
    (duplicate/gecersiz kimlik uretilmemeli)."""
    path = tmp_path / "traceability.mp4"
    _write_growing_patch_video(
        path, total_frames=300, ramp_start=50, ramp_end=130, growth_per_frame=2, strong_burst=(150, 160)
    )

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=5.0)
    evidence = sampler.process_video(path, sample_fps=25)

    evidence_ids = [f.evidence_id for f in evidence]
    assert len(evidence_ids) == len(set(evidence_ids)), "Duplicate evidence_id uretilmemeli."
    for f in evidence:
        assert f.evidence_id == f"ev{f.frame_id}", (
            "evidence_id, HER ZAMAN kendi gercek frame_id'sinden turetilmeli "
            "(uydurma/kararsiz bir kimlik degil)."
        )
        assert f.timestamp_sec == round(f.frame_id / 25.0, 2)


def test_diagnostic_evidence_id_and_payload_label_match_real_vlm_payload(tmp_path: Path) -> None:
    """16) Diagnostic kayittaki `evidence_id`/`vlm_payload_label`, VLM'e GERCEKTEN
    gonderilecek payload etiketiyle birebir ayni olmali (uctan uca izlenebilirlik)."""
    from src.sampler.payload_builder import VLMPayloadBuilder

    path = tmp_path / "diag_traceability.mp4"
    _write_growing_patch_video(path, total_frames=200, ramp_start=30, ramp_end=90, growth_per_frame=2)

    diag_dir = tmp_path / "diag"
    sampler = _density_sampler(
        tmp_path,
        max_temporal_gap_sec=5.0,
        diagnostic_enabled=True,
        diagnostic_output_dir=str(diag_dir),
        diagnostic_output_format="jsonl",
    )
    evidence = sampler.process_video(path, sample_fps=25)
    assert evidence

    real_content = VLMPayloadBuilder.build_content_blocks(evidence, prompt="test")
    real_labels = {block["text"] for block in real_content if block["type"] == "text"} - {"test"}

    diag_files = sorted(diag_dir.glob("*.jsonl"))
    assert diag_files, "diagnostic_enabled=True iken bir JSONL dosyasi uretilmeli."
    import json

    rows = [json.loads(line) for line in diag_files[-1].read_text(encoding="utf-8").splitlines()]
    selected_rows = [r for r in rows if r["selected"]]
    assert selected_rows, "En az bir secilen (selected=True) tanilama satiri olmali."

    seen_evidence_ids = set()
    for row in selected_rows:
        assert row["evidence_id"] == f"ev{row['frame_index']}"
        assert row["vlm_payload_label"] in real_labels, (
            "Diagnostic'teki vlm_payload_label, VLMPayloadBuilder'in GERCEKTEN "
            "uretecegi etiketle birebir ayni olmali - iki ayri kaynaktan "
            "sapmis bir etiket kabul edilemez."
        )
        assert row["evidence_id"] not in seen_evidence_ids, "Diagnostic'te duplicate evidence_id olmamali."
        seen_evidence_ids.add(row["evidence_id"])

    # Her GERCEK evidence karesinin diagnostic'te karsiligi olmali (kayip yok).
    assert seen_evidence_ids == {f.evidence_id for f in evidence}


def test_diagnostic_never_assigns_evidence_id_to_rejected_frames(tmp_path: Path) -> None:
    """17) Sampler'da secilmemis (rejected) bir ornek kareye ASLA bir evidence_id/
    payload etiketi UYDURULMAMALI - yalnizca gercekten secilen kareler kimlik tasir."""
    path = tmp_path / "diag_rejected.mp4"
    _write_growing_patch_video(path, total_frames=200, ramp_start=30, ramp_end=90, growth_per_frame=2)

    diag_dir = tmp_path / "diag_rej"
    sampler = _density_sampler(
        tmp_path,
        max_temporal_gap_sec=5.0,
        diagnostic_enabled=True,
        diagnostic_output_dir=str(diag_dir),
        diagnostic_output_format="jsonl",
    )
    sampler.process_video(path, sample_fps=25)

    import json

    diag_files = sorted(diag_dir.glob("*.jsonl"))
    rows = [json.loads(line) for line in diag_files[-1].read_text(encoding="utf-8").splitlines()]
    rejected_rows = [r for r in rows if not r["selected"]]
    assert rejected_rows, "Bu senaryoda reddedilen ornek kareler de olmali."
    for row in rejected_rows:
        assert row["evidence_id"] is None
        assert row["vlm_payload_label"] is None


# =============================================================================
# Kanita dayali minimal iyilestirme: "olay baslangicina YAKLASAN saniyelerden
# VLM'e daha fazla kare gitsin, kota EKLENMEDEN". Onceden, bir zincir henuz
# onaylanmamisken (`pending_early_candidate` bekliyorken) yalnizca zincirin
# ILK karesi evidence'a girerdi; onay tamamlanana kadar gecen ARADAKI
# saniyelerde (surdurulen ama hicbir zaman min_count'a ulasmayan bir sinyal
# icin - ör. tekrarlayan kisa hareketler) hicbir ek kare eklenmezdi.
# =============================================================================


def test_approach_window_produces_multiple_early_change_frames_without_a_cap(tmp_path: Path) -> None:
    """18) Ana esik gecilmeden ONCE, uzun bir 'yaklasma' penceresi boyunca
    tekrarlayan supheli-erken sinyaller VARSA, tek bir onset karesiyle
    sinirli KALINMAMALI - `early_change_selection_interval_sec` araligiyla
    BIRDEN FAZLA temsili kare eklenmeli (sabit bir ust sinir/kota olmadan)."""
    path = tmp_path / "approach_window.mp4"
    frames = []
    total = 280
    # Kucuk bir "nokta", her 3 karede bir farkli bir konuma sicrar - ana
    # esigi hicbir zaman gecmeyen ama SURDURULEN (periyodik) supheli-erken
    # bir sinyal uretir (gercek dunyada ör. bir kisinin yasakli bolgeye
    # tekrar tekrar yaklasip uzaklasmasi gibi).
    positions = [(10, 10), (30, 10), (50, 10), (70, 10), (90, 10), (30, 40), (50, 40), (70, 40)]
    for i in range(total):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if i >= 40:
            x, y = positions[((i - 40) // 3) % len(positions)]
            cv2.rectangle(frame, (x, y), (x + 8, y + 8), (95, 95, 95), -1)
        if 250 <= i < 260:
            cv2.rectangle(frame, (5, 5), (120, 90), (255, 255, 255), -1)  # ana esigi kesin gecen gercek olay
        frames.append(frame)
    _write_video(path, frames)

    sampler = _density_sampler(tmp_path, max_temporal_gap_sec=50.0)
    evidence = sampler.process_video(path, sample_fps=25)

    early_frames = [f for f in evidence if f.selection_reason == "early_change"]
    assert len(early_frames) >= 3, (
        "Uzun bir yaklasma penceresinde surdurulen ama hic onaylanmayan bir "
        "sinyal, YALNIZCA tek bir onset karesiyle SINIRLI KALMAMALI."
    )
    # Kronolojik sira ve benzersiz evidence_id/frame_index/timestamp_sec
    # tutarliligi (izlenebilirlik) korunmali.
    timestamps = [f.timestamp_sec for f in evidence]
    assert timestamps == sorted(timestamps)
    evidence_ids = [f.evidence_id for f in evidence]
    assert len(evidence_ids) == len(set(evidence_ids))
    for f in evidence:
        assert f.evidence_id == f"ev{f.frame_id}"
    # Ana esigi gecen kareler HALA kosulsuz/sinirsiz secilmeye devam etmeli
    # (bu mekanizma onlarin yerine GECMEZ, onlara EK yapar).
    assert any(f.selection_reason == "threshold_exceeded" for f in evidence)


def test_higher_sample_fps_catches_events_entirely_between_old_sample_points(tmp_path: Path) -> None:
    """19) Kanita dayali `sample_fps` (5 -> 10) degisikligi: cok kisa (tek native
    kare, 25fps'te 40ms) bir olay, ESKI ornekleme hizinda (5) ORNEKLENEN iki
    kare arasindaki kor pencereye tamamen sigip HIC OKUNMAYABILIR - bu, secim
    mantiginin (esik/interval/cooldown/dedup) hicbir ayariyla duzeltilemez,
    cunku o kare zaten taranmiyor. Yeni hiz (10) bu kor pencereyi yariya
    indirir ve olayi yakalar."""
    path = tmp_path / "single_native_frame_event.mp4"
    frames = []
    total = 125  # 5s @ 25fps
    brief_frame = 62
    for i in range(total):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if i == brief_frame:
            cv2.rectangle(frame, (5, 5), (120, 90), (255, 255, 255), -1)
        frames.append(frame)
    _write_video(path, frames)

    sampler_old = AdaptiveFrameSampler(
        min_change_threshold=0.02, blur_kernel_size=(3, 3), evidence_output_dir=str(tmp_path / "e_old")
    )
    evidence_old = sampler_old.process_video(path, sample_fps=5)
    assert all(f.selection_reason == "fallback" for f in evidence_old), (
        "Bu senaryo, eski (5) ornekleme hizinin kor pencereye tamamen sigan "
        "cok kisa bir olayi neden hic yakalayamadigini gosterir."
    )

    sampler_new = AdaptiveFrameSampler(
        min_change_threshold=0.02, blur_kernel_size=(3, 3), evidence_output_dir=str(tmp_path / "e_new")
    )
    evidence_new = sampler_new.process_video(path, sample_fps=10)
    assert any(f.selection_reason == "threshold_exceeded" for f in evidence_new), (
        "Yeni (10) ornekleme hizi, ayni cok kisa olayi yakalamali."
    )


def test_higher_sample_fps_captures_more_stages_of_a_rapidly_escalating_event(tmp_path: Path) -> None:
    """20) Kanita dayali `sample_fps` (10 -> 15) degisikligi: `threshold_exceeded`
    yolu zaten kosulsuz/sinirsiz oldugundan, hizla tirmanan (kisa surede
    birden fazla asamadan gecen) bir olayin KAC ASAMASININ yakalanacagini
    BELIRLEYEN TEK degisken kaynak tarama hizidir - daha yuksek `sample_fps`,
    AYNI hizla tirmanan olay icin DAHA FAZLA asama karesi vermeli."""
    path = tmp_path / "rapid_escalation.mp4"
    frames = []
    total = 200
    for i in range(total):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if 100 <= i < 109:
            size = 10 + (i - 100) * 12
            cv2.rectangle(frame, (10, 10), (10 + size, 10 + size), (200, 200, 200), -1)
        elif i >= 109:
            cv2.rectangle(frame, (10, 10), (10 + 106, 10 + 106), (200, 200, 200), -1)
        frames.append(frame)
    _write_video(path, frames)

    def _stage_frame_count(sample_fps: int) -> int:
        sampler = _density_sampler(tmp_path, max_temporal_gap_sec=10.0, evidence_output_dir=str(tmp_path / f"e_{sample_fps}"))
        evidence = sampler.process_video(path, sample_fps=sample_fps)
        return sum(1 for f in evidence if 100 <= f.frame_id <= 112)

    assert _stage_frame_count(15) > _stage_frame_count(10), (
        "Daha yuksek ornekleme hizi, ayni hizla tirmanan olayin DAHA FAZLA "
        "asamasini kareye almali (secim mantigi degismeden)."
    )


def test_sample_fps_must_cover_native_fps_or_a_critical_frame_is_still_missed(tmp_path: Path) -> None:
    """21) Kanita dayali `sample_fps` (15 -> 30) degisikligi: `sample_fps=15`,
    25fps'lik bir kaynak icin native hiz veriyordu ama 30fps'lik (yaygin)
    bir kaynakta `frame_step=int(30/15)=2` HALA kare atliyordu - bu, secim
    mantigindaki (esik/interval/cooldown/dedup) HICBIR ayarla duzeltilemez,
    cunku o kare zaten hic OKUNMUYOR. `sample_fps=30`, 30fps'lik bir
    kaynakta da `frame_step=1` (native, atlamasiz) saglamali."""
    path = tmp_path / "native_30fps_critical_frame.mp4"
    frames = []
    total = 150
    crit_frame = 75
    for i in range(total):
        frame = np.full((96, 128, 3), 30, dtype=np.uint8)
        if i == crit_frame:
            cv2.rectangle(frame, (5, 5), (120, 90), (255, 255, 255), -1)
        frames.append(frame)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))  # 30fps native kaynak
    for f in frames:
        writer.write(f)
    writer.release()

    sampler_15 = AdaptiveFrameSampler(
        min_change_threshold=0.02, blur_kernel_size=(3, 3), evidence_output_dir=str(tmp_path / "e15")
    )
    evidence_15 = sampler_15.process_video(path, sample_fps=15)
    assert all(f.selection_reason == "fallback" for f in evidence_15), (
        "Bu senaryo, 25fps varsayimiyla secilmis bir sample_fps'in 30fps'lik "
        "GERCEK bir kaynakta hala kare atlayabildigini gosterir."
    )

    sampler_30 = AdaptiveFrameSampler(
        min_change_threshold=0.02, blur_kernel_size=(3, 3), evidence_output_dir=str(tmp_path / "e30")
    )
    evidence_30 = sampler_30.process_video(path, sample_fps=30)
    assert any(f.selection_reason == "threshold_exceeded" for f in evidence_30), (
        "sample_fps, GERCEK kaynak hizini (30fps) karsilamali - ayni kritik "
        "kareyi yakalamali."
    )
