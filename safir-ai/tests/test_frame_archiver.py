"""`FrameArchiver` (src/sampler/context/frame_archiver.py) icin birim testleri.

`FrameArchiver` kendi basina HICBIR kare secimi yapmaz; yalnizca zaten
`FrameSelector` tarafindan secilmis `EventCluster.representative_frames`i
diske yazan pasif bir persistence katmanidir. Bu testler, diske yazilan
kare kimliklerinin (frame_id) VLM'e giden kimliklerle BIREBIR AYNI oldugunu
dogrular (ortak kaynak garantisi).
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
            label="peak" if i == 0 else "context",
            frame_id=i,
            timestamp_sec=float(i),
            timestamp_str=f"00:0{i}",
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
    jpeg_files = sorted(event_dir.glob("frame_*.jpg"))
    assert len(jpeg_files) == 5


def test_export_metadata_frame_ids_match_representative_frames(tmp_path: Path) -> None:
    """Diske yazilan metadata'daki frame_id'ler, VLM'e giden representative_frames ile AYNI olmali."""
    cluster = _cluster_with_representative_frames(event_id=2, count=3)

    event_dirs = FrameArchiver.export([cluster], output_dir=str(tmp_path))
    metadata = json.loads((Path(event_dirs[0]) / "metadata.json").read_text(encoding="utf-8"))

    written_frame_ids = {f["frame_id"] for f in metadata["frames"]}
    vlm_frame_ids = {rf.frame_id for rf in cluster.representative_frames}
    assert written_frame_ids == vlm_frame_ids
    assert metadata["selected_frame_count"] == len(cluster.representative_frames)


def test_export_does_not_perform_independent_selection(tmp_path: Path) -> None:
    """FrameArchiver, kendisine verilenden FAZLA veya FARKLI kare uretmemeli (bagimsiz secim yok)."""
    cluster = _cluster_with_representative_frames(event_id=3, count=2)

    FrameArchiver.export([cluster], output_dir=str(tmp_path))

    event_dir = Path(str(tmp_path)) / "event_0003"
    jpeg_files = list(event_dir.glob("frame_*.jpg"))
    # Girdi kadar (2) dosya var; extractor kendi basina ek kare uretmedi.
    assert len(jpeg_files) == 2


def test_export_empty_clusters_returns_empty_list(tmp_path: Path) -> None:
    assert FrameArchiver.export([], output_dir=str(tmp_path)) == []
