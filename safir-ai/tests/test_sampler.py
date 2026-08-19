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
        # En yuksek skorlu kare kalici bir 'peak' etiketi TASIMAZ; kimligi
        # `cluster.peak_frame.frame_id` ile eslesen kayittan bulunur.
        highest_score_entries = [
            rf for rf in cluster.representative_frames if rf.frame_index == cluster.peak_frame.frame_id
        ]
        assert len(highest_score_entries) == 1
        assert highest_score_entries[0].selection_reason == "highest_evidence_score"

        timestamps = [rf.timestamp_sec for rf in cluster.representative_frames]
        assert timestamps == sorted(timestamps)

        frame_indices = [rf.frame_index for rf in cluster.representative_frames]
        assert len(frame_indices) == len(set(frame_indices))


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
    frame_indices = [rf.frame_index for rf in rep_frames]
    assert len(frame_indices) == len(set(frame_indices))
    # Fallback tek kareyle kapanan bir olay -> tek temsili kare (cogaltma yok).
    assert len(rep_frames) == 1


def test_vlm_and_disk_output_share_identical_frame_identity(
    tmp_path: Path, long_multi_event_video: str
) -> None:
    """VLM'e giden (`representative_frames`) ve diske yazilan kareler AYNI frame_index/timestamp setini kullanmali."""
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
        disk_frame_indices = {f["frame_index"] for f in metadata["frames"]}
        vlm_frame_indices = {rf.frame_index for rf in cluster.representative_frames}
        assert disk_frame_indices == vlm_frame_indices

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


# =============================================================================
# Guclendirilmis clustering: yalnizca zaman degil, konumsal (bbox IoU) ve
# transitive-chaining/asiri-uzun-cluster onlemleri de dogrulanir. Testler,
# gercek video uretmek yerine dogrudan sentetik `EvidenceFrame` listeleri
# kullanir (yalnizca gercekten var olan sinyaller: timestamp, change_score,
# motion_bbox); track/nesne kimligi veya risk turu FABRIKE EDILMEZ (mevcut
# veri modelinde yoktur).
# =============================================================================


def _synthetic_evidence_frame(
    frame_id: int, timestamp_sec: float, motion_bbox=None, change_score: float = 1.0
) -> EvidenceFrame:
    minutes, seconds = divmod(int(timestamp_sec), 60)
    return EvidenceFrame(
        frame_id=frame_id,
        timestamp_sec=timestamp_sec,
        timestamp_str=f"{minutes:02d}:{seconds:02d}",
        change_score=change_score,
        image_bytes=b"\xff\xd8\xff",
        base64_image=f"data:image/jpeg;base64,F{frame_id}",
        image_shape=(48, 64, 3),
        motion_bbox=motion_bbox,
    )


def test_spatially_different_events_not_merged_despite_time_proximity(tmp_path: Path) -> None:
    """Zaman olarak yakin ama konumsal olarak COK farkli iki olay ayni cluster'a girmemeli."""
    sampler = AdaptiveFrameSampler(
        min_event_interval_sec=2.0,
        cluster_merge_gap_sec=20.0,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = [
        _synthetic_evidence_frame(0, 0.0, motion_bbox=(0, 0, 10, 10)),
        _synthetic_evidence_frame(1, 5.0, motion_bbox=(500, 500, 510, 510)),  # tamamen farkli bolge
    ]
    clusters = sampler.cluster_events(frames)

    assert len(clusters) == 2


def test_transitive_chaining_does_not_create_mega_cluster(tmp_path: Path) -> None:
    """A-B ve B-C bbox olarak ortussе bile, A-C ortusmuyorsa A,B,C TEK cluster'a zincirlenmemeli."""
    sampler = AdaptiveFrameSampler(
        min_event_interval_sec=2.0,
        cluster_merge_gap_sec=6.0,
        bbox_iou_merge_threshold=0.10,
        max_cluster_duration_sec=120.0,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    # A: x=[0,10], B: x=[6,16] (A ile ortusur), C: x=[12,22] (B ile ortusur, A ile ORTUSMEZ).
    frames = [
        _synthetic_evidence_frame(0, 0.0, motion_bbox=(0, 0, 10, 10)),
        _synthetic_evidence_frame(1, 5.0, motion_bbox=(6, 0, 16, 10)),
        _synthetic_evidence_frame(2, 10.0, motion_bbox=(12, 0, 22, 10)),
    ]
    clusters = sampler.cluster_events(frames)

    # Eski (yalnizca komsu-kontrollu) mantik A+B+C'yi TEK mega-cluster'da
    # birlestirirdi; yeni ankor kontrolu C'yi A'nin devami saymamali.
    assert len(clusters) == 2
    assert clusters[0].total_candidate_frames == 2  # A + B
    assert clusters[1].total_candidate_frames == 1  # C ayri


def test_max_cluster_duration_splits_overlong_continuous_event(tmp_path: Path) -> None:
    """Konumsal olarak surekli ayni olay bile, `max_cluster_duration_sec`i asarsa YENI bir cluster'a bolunmeli."""
    sampler = AdaptiveFrameSampler(
        min_event_interval_sec=2.0,
        cluster_merge_gap_sec=10.0,
        bbox_iou_merge_threshold=0.10,
        max_cluster_duration_sec=5.0,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    same_bbox = (0, 0, 10, 10)
    frames = [
        _synthetic_evidence_frame(0, 0.0, motion_bbox=same_bbox),
        _synthetic_evidence_frame(1, 4.0, motion_bbox=same_bbox),
        _synthetic_evidence_frame(2, 8.0, motion_bbox=same_bbox),
    ]
    clusters = sampler.cluster_events(frames)

    # 0.0 -> 4.0 (sure=4.0<=5.0) birlesir; 0.0 -> 8.0 (sure=8.0>5.0) BIRLESMEZ.
    assert len(clusters) == 2
    assert clusters[0].total_candidate_frames == 2
    assert clusters[1].total_candidate_frames == 1
    # Hicbir kare kaybolmadi: toplam aday kare sayisi girdiyle ayni.
    assert sum(c.total_candidate_frames for c in clusters) == len(frames)


def test_short_evidence_gap_within_same_event_does_not_split_unnecessarily(tmp_path: Path) -> None:
    """Ayni fiziksel olay icindeki KISA bir evidence kesintisi, olayi gereksiz yere bolmemeli."""
    sampler = AdaptiveFrameSampler(
        min_event_interval_sec=1.0,
        cluster_merge_gap_sec=10.0,
        bbox_iou_merge_threshold=0.10,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    same_bbox = (0, 0, 10, 10)
    frames = [
        _synthetic_evidence_frame(0, 0.0, motion_bbox=same_bbox),
        # 3s'lik kisa bir sessiz aralik (min_event_interval_sec'i asar, HAM
        # grubu boler) ama cluster_merge_gap_sec'in COK altinda kalir ve
        # bbox AYNI kaliyor -> nihai olay TEK cluster olarak kalmali.
        _synthetic_evidence_frame(1, 3.0, motion_bbox=same_bbox),
    ]
    clusters = sampler.cluster_events(frames)

    assert len(clusters) == 1
    assert clusters[0].total_candidate_frames == 2


# =============================================================================
# Evidence-esigi disiplini: yalnizca esigi gecmis kareler cluster/secim
# uyesi olabilir; representative_frames'teki her karenin change_score'u
# process_video tarafindan uygulanan esigi (min_change_threshold sonrasi
# net_change_score) yansitmalidir.
# =============================================================================


def test_representative_frames_only_come_from_evidence_threshold_passing_frames(
    tmp_path: Path, motion_video: str
) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001, evidence_output_dir=str(tmp_path / "evidence")
    )
    frames = sampler.process_video(motion_video, sample_fps=5)
    evidence_frame_ids = {f.frame_id for f in frames}
    clusters = sampler.cluster_events(frames)

    for cluster in clusters:
        for rf in cluster.representative_frames:
            # Her secilen kare, process_video'nun urettigi (esigi gecmis)
            # Kanit Kareleri kumesinden gelmis olmali.
            assert rf.frame_index in evidence_frame_ids


# =============================================================================
# Pre/peak/post konumsal bilgisi TAMAMEN kaldirilmis olmali: sampler ciktisinda,
# VLM payload'inda, disk metadata'sinda hicbir konumsal rol/etiket bulunmamali.
# =============================================================================


def test_sampler_output_has_no_pre_peak_post_role_labels(tmp_path: Path, long_multi_event_video: str) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        min_event_interval_sec=0.5,
        cluster_merge_gap_sec=1.0,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = sampler.process_video(long_multi_event_video, sample_fps=25)
    clusters = sampler.cluster_events(frames)

    for cluster in clusters:
        for rf in cluster.representative_frames:
            field_names = set(type(rf).model_fields.keys())
            assert "label" not in field_names
            assert "frame_role" not in field_names
            assert "frame_type" not in field_names
            assert "position_type" not in field_names
            reason_lower = rf.selection_reason.lower()
            assert "pre" not in reason_lower
            assert "post" not in reason_lower
            assert "peak" not in reason_lower


def test_vlm_payload_contains_no_pre_peak_post_tags(tmp_path: Path, long_multi_event_video: str) -> None:
    from src.sampler.payload_builder import VLMPayloadBuilder

    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        min_event_interval_sec=0.5,
        cluster_merge_gap_sec=1.0,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = sampler.process_video(long_multi_event_video, sample_fps=25)
    clusters = sampler.cluster_events(frames)

    content = VLMPayloadBuilder.build_content_blocks(clusters, prompt="test")
    text_blocks = " ".join(b["text"].lower() for b in content if b["type"] == "text")

    assert "pre-event" not in text_blocks
    assert "post-event" not in text_blocks
    assert "kare rolu" not in text_blocks
    assert "pre_context" not in text_blocks
    assert "post_context" not in text_blocks


def test_no_pre_peak_post_named_files_written_to_disk(tmp_path: Path, long_multi_event_video: str) -> None:
    sampler = AdaptiveFrameSampler(
        min_change_threshold=0.001,
        min_event_interval_sec=0.5,
        cluster_merge_gap_sec=1.0,
        evidence_output_dir=str(tmp_path / "evidence"),
    )
    frames = sampler.process_video(long_multi_event_video, sample_fps=25)
    clusters = sampler.cluster_events(frames, export_to_disk=True)
    assert clusters

    all_files = list((tmp_path / "evidence").rglob("*"))
    assert all_files
    for path in all_files:
        lowered = path.name.lower()
        assert "pre_peak" not in lowered
        assert "post_peak" not in lowered
        assert not lowered.startswith("peak.")
        assert not lowered.startswith("pre_peak.")
        assert not lowered.startswith("post_peak.")
