"""`FrameArchiver` (src/sampler/context/frame_archiver.py) icin birim testleri.

`FrameArchiver` kendi basina HICBIR kare secimi yapmaz; yalnizca zaten
`FrameSelector` tarafindan secilmis `EventCluster.representative_frames`i
diske yazan pasif bir persistence katmanidir. Bu testler, diske yazilan
kare kimliklerinin VLM'e giden kimliklerle BIREBIR AYNI oldugunu ve hicbir
'pre'/'peak'/'post' adlandirmasinin dosya adinda/metadata'da bulunmadigini
dogrular.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from src.sampler.context.frame_archiver import FrameArchiver
from src.sampler.schema import EventCluster, EvidenceFrame, RepresentativeFrame


def _evidence_frame(frame_id: int, timestamp_sec: float) -> EvidenceFrame:
    minutes, seconds = divmod(int(timestamp_sec), 60)
    return EvidenceFrame(
        frame_id=frame_id,
        timestamp_sec=timestamp_sec,
        timestamp_str=f"{minutes:02d}:{seconds:02d}",
        change_score=1.0,
        image_bytes=b"\xff\xd8\xff",
        base64_image="data:image/jpeg;base64," + base64.b64encode(b"fake-jpeg-bytes").decode("utf-8"),
        image_shape=(48, 64, 3),
    )


def _cluster_with_representative_frames(event_id: int, count: int) -> EventCluster:
    peak = _evidence_frame(0, 0.0)
    reps = [
        RepresentativeFrame(
            frame_index=i,
            event_id=event_id,
            timestamp_sec=float(i),
            timestamp_str=f"00:0{i}",
            evidence_score=float(count - i),
            selection_reason="highest_evidence_score" if i == 0 else "temporal_coverage",
            base64_image="data:image/jpeg;base64," + base64.b64encode(b"fake-jpeg-bytes").decode("utf-8"),
        )
        for i in range(count)
    ]
    return EventCluster(
        event_id=event_id,
        start_time=0.0,
        end_time=float(count),
        peak_frame=peak,
        total_candidate_frames=count,
        duration_sec=float(count),
        representative_frames=reps,
    )


def test_export_writes_one_jpeg_per_representative_frame(tmp_path: Path) -> None:
    cluster = _cluster_with_representative_frames(event_id=1, count=5)

    event_dirs = FrameArchiver.export([cluster], output_dir=str(tmp_path))

    assert len(event_dirs) == 1
    event_dir = Path(event_dirs[0])
    jpeg_files = sorted(event_dir.glob("*.jpg"))
    assert len(jpeg_files) == 5


def test_exported_filenames_use_generic_evidence_naming_not_pre_peak_post(tmp_path: Path) -> None:
    """Dosya adlari `evidence_NNN.jpg` seklinde olmali; 'pre'/'peak'/'post' ICERMEMELI."""
    cluster = _cluster_with_representative_frames(event_id=1, count=5)

    event_dirs = FrameArchiver.export([cluster], output_dir=str(tmp_path))
    event_dir = Path(event_dirs[0])
    filenames = sorted(p.name for p in event_dir.glob("*.jpg"))

    assert filenames == ["evidence_001.jpg", "evidence_002.jpg", "evidence_003.jpg", "evidence_004.jpg", "evidence_005.jpg"]
    for name in filenames:
        lowered = name.lower()
        assert "pre" not in lowered
        assert "post" not in lowered
        assert "peak" not in lowered


def test_exported_filenames_are_numbered_by_timestamp_order(tmp_path: Path) -> None:
    """Dosyalar timestamp sirasina gore numaralandirilmali (representative_frames zaten kronolojik)."""
    peak = _evidence_frame(0, 0.0)
    reps = [
        RepresentativeFrame(
            frame_index=i,
            event_id=1,
            timestamp_sec=float(i),
            timestamp_str=f"00:0{i}",
            evidence_score=1.0,
            selection_reason="temporal_coverage",
            base64_image="data:image/jpeg;base64," + base64.b64encode(f"frame{i}".encode()).decode("utf-8"),
        )
        for i in range(3)
    ]
    cluster = EventCluster(
        event_id=1, start_time=0.0, end_time=2.0, peak_frame=peak, total_candidate_frames=3,
        representative_frames=reps,
    )

    event_dirs = FrameArchiver.export([cluster], output_dir=str(tmp_path))
    metadata = json.loads((Path(event_dirs[0]) / "metadata.json").read_text(encoding="utf-8"))

    filenames_in_order = [f["filename"] for f in metadata["frames"]]
    assert filenames_in_order == ["evidence_001.jpg", "evidence_002.jpg", "evidence_003.jpg"]
    timestamps_in_order = [f["timestamp_sec"] for f in metadata["frames"]]
    assert timestamps_in_order == sorted(timestamps_in_order)


def test_export_metadata_frame_indices_match_representative_frames(tmp_path: Path) -> None:
    """Diske yazilan metadata'daki frame_index'ler, VLM'e giden representative_frames ile AYNI olmali."""
    cluster = _cluster_with_representative_frames(event_id=2, count=3)

    event_dirs = FrameArchiver.export([cluster], output_dir=str(tmp_path))
    metadata = json.loads((Path(event_dirs[0]) / "metadata.json").read_text(encoding="utf-8"))

    written_frame_indices = {f["frame_index"] for f in metadata["frames"]}
    vlm_frame_indices = {rf.frame_index for rf in cluster.representative_frames}
    assert written_frame_indices == vlm_frame_indices
    assert metadata["selected_frame_count"] == len(cluster.representative_frames)

    for f in metadata["frames"]:
        assert f["event_id"] == 2
        assert "evidence_score" in f
        assert "selection_reason" in f
        assert "label" not in f


def test_metadata_json_contains_no_pre_peak_post_keys_or_values(tmp_path: Path) -> None:
    """Disk metadata dosyasi hicbir yerde 'pre'/'peak'/'post' anahtar veya deger ICERMEMELI."""
    cluster = _cluster_with_representative_frames(event_id=4, count=5)

    event_dirs = FrameArchiver.export([cluster], output_dir=str(tmp_path))
    raw_text = (Path(event_dirs[0]) / "metadata.json").read_text(encoding="utf-8").lower()

    assert "\"pre" not in raw_text
    assert "\"post" not in raw_text
    assert "\"peak" not in raw_text
    assert "pre_peak" not in raw_text
    assert "post_peak" not in raw_text


def test_export_does_not_perform_independent_selection(tmp_path: Path) -> None:
    """FrameArchiver, kendisine verilenden FAZLA veya FARKLI kare uretmemeli (bagimsiz secim yok)."""
    cluster = _cluster_with_representative_frames(event_id=3, count=2)

    FrameArchiver.export([cluster], output_dir=str(tmp_path))

    event_dir = Path(str(tmp_path)) / "event_0003"
    jpeg_files = list(event_dir.glob("*.jpg"))
    # Girdi kadar (2) dosya var; extractor kendi basina ek kare uretmedi.
    assert len(jpeg_files) == 2


def test_export_empty_clusters_returns_empty_list(tmp_path: Path) -> None:
    assert FrameArchiver.export([], output_dir=str(tmp_path)) == []
