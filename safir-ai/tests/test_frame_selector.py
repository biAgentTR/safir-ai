"""`FrameSelector` (src/sampler/context/frame_selector.py) icin birim testleri.

Onceki iki bagimsiz mekanizmanin (RepresentativeFrameExtractor + PeakFrameExporter)
yerini alan TEK ortak kare secim mekanizmasini dogrular: her Olay Grubu icin
en fazla 5 benzersiz, kronolojik, zirve dahil kare secilir; video/dosya
sistemine HICBIR erisim olmadan, tamamen bellekteki `EvidenceFrame` nesneleri
uzerinden calisir.
"""

from __future__ import annotations

from src.sampler.context.frame_selector import TARGET_FRAME_COUNT, FrameSelector
from src.sampler.schema import EvidenceFrame


def _evidence_frame(frame_id: int, timestamp_sec: float, change_score: float = 0.5) -> EvidenceFrame:
    """Video/dosya erisimi olmadan, tamamen sentetik bir `EvidenceFrame` uretir."""
    minutes, seconds = divmod(int(timestamp_sec), 60)
    return EvidenceFrame(
        frame_id=frame_id,
        timestamp_sec=timestamp_sec,
        timestamp_str=f"{minutes:02d}:{seconds:02d}",
        change_score=change_score,
        image_bytes=b"\xff\xd8\xff",
        base64_image=f"data:image/jpeg;base64,FRAME{frame_id}",
        image_shape=(48, 64, 3),
    )


def test_selects_at_most_target_frame_count() -> None:
    """Uzun bir olayda (20 aday) en fazla TARGET_FRAME_COUNT (5) kare secilmeli."""
    candidates = [_evidence_frame(i, float(i)) for i in range(20)]
    peak = max(candidates, key=lambda f: f.change_score)

    result = FrameSelector.select(peak, candidates)

    assert len(result) == TARGET_FRAME_COUNT == 5


def test_peak_frame_always_included() -> None:
    candidates = [_evidence_frame(i, float(i)) for i in range(20)]
    peak = _evidence_frame(999, 7.5, change_score=10.0)
    candidates_with_peak = candidates + [peak]

    result = FrameSelector.select(peak, candidates_with_peak)

    peak_entries = [rf for rf in result if rf.frame_id == peak.frame_id]
    assert len(peak_entries) == 1
    assert peak_entries[0].label == "peak"
    assert peak_entries[0].base64_image == peak.base64_image


def test_selected_frames_are_chronologically_sorted() -> None:
    # Kasten karisik sirada veriliyor.
    candidates = [_evidence_frame(i, float(i)) for i in [5, 1, 9, 3, 7, 0, 8, 2, 6, 4]]
    peak = max(candidates, key=lambda f: f.timestamp_sec)

    result = FrameSelector.select(peak, candidates)

    timestamps = [rf.timestamp_sec for rf in result]
    assert timestamps == sorted(timestamps)


def test_short_event_does_not_duplicate_frames() -> None:
    """Aday havuzunda TARGET_FRAME_COUNT'tan az benzersiz kare varsa, kare COGALTILMAZ."""
    candidates = [_evidence_frame(1, 1.0), _evidence_frame(2, 1.2)]
    peak = candidates[1]

    result = FrameSelector.select(peak, candidates)

    assert len(result) == 2  # cogaltma yok, mevcut 2 benzersiz kare kullanildi
    frame_ids = [rf.frame_id for rf in result]
    assert len(frame_ids) == len(set(frame_ids))


def test_single_frame_event_returns_only_peak() -> None:
    """Tek karelik bir olayda (yalnizca zirve) cikti da tek kare olmali; kare uydurulmaz."""
    peak = _evidence_frame(42, 3.3)

    result = FrameSelector.select(peak, [peak])

    assert len(result) == 1
    assert result[0].label == "peak"
    assert result[0].frame_id == 42


def test_peak_not_in_candidates_is_still_included() -> None:
    """`peak_frame` candidate_frames listesinde olmasa bile secime dahil edilir."""
    candidates = [_evidence_frame(i, float(i)) for i in range(3)]
    peak = _evidence_frame(100, 50.0, change_score=99.0)

    result = FrameSelector.select(peak, candidates)

    assert any(rf.frame_id == 100 and rf.label == "peak" for rf in result)


def test_no_duplicate_frame_ids_in_large_pool() -> None:
    candidates = [_evidence_frame(i, float(i) * 0.5) for i in range(50)]
    peak = candidates[25]

    result = FrameSelector.select(peak, candidates)

    frame_ids = [rf.frame_id for rf in result]
    assert len(frame_ids) == len(set(frame_ids))
    assert len(result) <= TARGET_FRAME_COUNT


def test_representative_frame_reuses_existing_base64_no_reencode() -> None:
    """Secilen kare, kaynak EvidenceFrame'in base64'unu AYNEN tasimali (yeniden kodlama yok)."""
    candidates = [_evidence_frame(i, float(i)) for i in range(3)]
    peak = candidates[1]

    result = FrameSelector.select(peak, candidates)

    for rf in result:
        source = next(f for f in candidates if f.frame_id == rf.frame_id)
        assert rf.base64_image == source.base64_image
        assert rf.timestamp_str == source.timestamp_str
