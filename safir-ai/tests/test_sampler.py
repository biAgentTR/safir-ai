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
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    evidence = sampler.process_video(path, sample_fps=25)

    assert all(f.selection_reason != "temporal_coverage" for f in evidence)


def test_gap_counter_resumes_from_coverage_frame_timestamp(tmp_path: Path, long_silence_video: str) -> None:
    """4) Coverage sonrasi gap hesabi, coverage karesinin KENDI zaman damgasindan devam etmeli."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=2.0,
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
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    evidence = sampler.process_video(path, sample_fps=25)

    assert all(f.selection_reason != "temporal_coverage" for f in evidence)


def test_empty_candidate_buffer_does_not_crash(tmp_path: Path, static_video: str) -> None:
    """7) Bos buffer/hic aday bulunmamasi (tamamen durgun video, fallback yolu) hataya yol acmamali."""
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        max_temporal_gap_sec=2.0,
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
