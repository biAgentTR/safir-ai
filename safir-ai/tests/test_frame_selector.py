"""`FrameSelector` (src/sampler/context/frame_selector.py) icin birim testleri.

Onceki iki bagimsiz mekanizmanin (RepresentativeFrameExtractor + PeakFrameExporter)
yerini alan TEK ortak kare secim mekanizmasini dogrular: her Olay Grubu icin
en yuksek skorlu evidence karesi + zaman ekseninde dengeli dagitilmis en fazla
4 ek evidence karesi (toplam en fazla 5) secilir. HICBIR kare kalici bir
'pre'/'peak'/'post' konumsal rolle ETIKETLENMEZ; video/dosya sistemine hicbir
erisim olmadan, tamamen bellekteki `EvidenceFrame` nesneleri uzerinden calisir.
"""

from __future__ import annotations

from src.sampler.context.frame_selector import TARGET_FRAME_COUNT, FrameSelector
from src.sampler.schema import EvidenceFrame

_EVENT_ID = 7


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
    peak = candidates[10]
    peak.change_score = 10.0

    result = FrameSelector.select(peak, candidates, event_id=_EVENT_ID)

    assert len(result) == TARGET_FRAME_COUNT == 5


def test_highest_score_frame_always_included() -> None:
    candidates = [_evidence_frame(i, float(i)) for i in range(20)]
    highest = _evidence_frame(999, 7.5, change_score=10.0)
    candidates_with_highest = candidates + [highest]

    result = FrameSelector.select(highest, candidates_with_highest, event_id=_EVENT_ID)

    matches = [rf for rf in result if rf.frame_index == highest.frame_id]
    assert len(matches) == 1
    assert matches[0].base64_image == highest.base64_image
    assert matches[0].event_id == _EVENT_ID
    assert matches[0].selection_reason == "highest_evidence_score"


def test_selected_frames_are_chronologically_sorted() -> None:
    # Kasten karisik sirada veriliyor.
    candidates = [_evidence_frame(i, float(i)) for i in [5, 1, 9, 3, 7, 0, 8, 2, 6, 4]]
    peak = max(candidates, key=lambda f: f.timestamp_sec)
    peak.change_score = 10.0

    result = FrameSelector.select(peak, candidates, event_id=_EVENT_ID)

    timestamps = [rf.timestamp_sec for rf in result]
    assert timestamps == sorted(timestamps)


def test_short_event_does_not_duplicate_frames() -> None:
    """Aday havuzunda TARGET_FRAME_COUNT'tan az benzersiz kare varsa, kare COGALTILMAZ."""
    candidates = [_evidence_frame(1, 1.0), _evidence_frame(2, 1.2, change_score=5.0)]
    peak = candidates[1]

    result = FrameSelector.select(peak, candidates, event_id=_EVENT_ID)

    assert len(result) == 2  # cogaltma yok, mevcut 2 benzersiz kare kullanildi
    frame_indices = [rf.frame_index for rf in result]
    assert len(frame_indices) == len(set(frame_indices))


def test_single_frame_event_returns_only_highest_score_frame() -> None:
    """Tek karelik bir olayda cikti da tek kare olmali; kare uydurulmaz."""
    peak = _evidence_frame(42, 3.3)

    result = FrameSelector.select(peak, [peak], event_id=_EVENT_ID)

    assert len(result) == 1
    assert result[0].frame_index == 42
    assert result[0].selection_reason == "highest_evidence_score"


def test_highest_score_frame_not_in_candidates_is_still_included() -> None:
    """`peak_frame` candidate_frames listesinde olmasa bile secime dahil edilir."""
    candidates = [_evidence_frame(i, float(i)) for i in range(3)]
    highest = _evidence_frame(100, 50.0, change_score=99.0)

    result = FrameSelector.select(highest, candidates, event_id=_EVENT_ID)

    assert any(rf.frame_index == 100 for rf in result)


def test_no_duplicate_frame_indices_in_large_pool() -> None:
    candidates = [_evidence_frame(i, float(i) * 0.5) for i in range(50)]
    peak = candidates[25]
    peak.change_score = 10.0

    result = FrameSelector.select(peak, candidates, event_id=_EVENT_ID)

    frame_indices = [rf.frame_index for rf in result]
    assert len(frame_indices) == len(set(frame_indices))
    assert len(result) <= TARGET_FRAME_COUNT


def test_representative_frame_reuses_existing_base64_no_reencode() -> None:
    """Secilen kare, kaynak EvidenceFrame'in base64'unu AYNEN tasimali (yeniden kodlama yok)."""
    candidates = [_evidence_frame(i, float(i)) for i in range(3)]
    peak = candidates[1]
    peak.change_score = 10.0

    result = FrameSelector.select(peak, candidates, event_id=_EVENT_ID)

    for rf in result:
        source = next(f for f in candidates if f.frame_id == rf.frame_index)
        assert rf.base64_image == source.base64_image
        assert rf.timestamp_str == source.timestamp_str
        assert rf.evidence_score == source.change_score


# --------------------------- Evidence-esigi disi kareler asla secilmemeli ---------------------------


def test_only_frames_passed_in_pool_can_be_selected() -> None:
    """FrameSelector, kendisine verilen havuzun DISINDAN hicbir kare uretemez (evidence disi kare yok)."""
    candidates = [_evidence_frame(i, float(i)) for i in range(4)]
    peak = candidates[2]
    peak.change_score = 10.0
    pool_ids = {f.frame_id for f in candidates}

    result = FrameSelector.select(peak, candidates, event_id=_EVENT_ID)

    assert all(rf.frame_index in pool_ids for rf in result)


# --------------------------- En yuksek skorlu kare basta/sonda: butce dengelenir ---------------------------


def test_highest_score_at_start_still_reaches_target_count() -> None:
    """En yuksek skorlu kare olayin en basindaysa, kalan butce diger tarafa aktarilip yine 5 kareye ulasilmali."""
    candidates = [_evidence_frame(i, float(i)) for i in range(10)]
    peak = candidates[0]
    peak.change_score = 10.0

    result = FrameSelector.select(peak, candidates, event_id=_EVENT_ID)

    assert len(result) == TARGET_FRAME_COUNT
    assert any(rf.frame_index == 0 for rf in result)


def test_highest_score_at_end_still_reaches_target_count() -> None:
    candidates = [_evidence_frame(i, float(i)) for i in range(10)]
    peak = candidates[-1]
    peak.change_score = 10.0

    result = FrameSelector.select(peak, candidates, event_id=_EVENT_ID)

    assert len(result) == TARGET_FRAME_COUNT
    assert any(rf.frame_index == 9 for rf in result)


# --------------------------- Metadata: frame kimligi/timestamp/skor/event/gerekce ---------------------------


def test_representative_frame_carries_full_metadata() -> None:
    """Her secilen kare icin frame_index, timestamp, evidence skoru, event_id ve secilme nedeni korunmali."""
    candidates = [_evidence_frame(i, float(i), change_score=float(i)) for i in range(5)]
    peak = candidates[2]
    peak.change_score = 99.0

    result = FrameSelector.select(peak, candidates, event_id=_EVENT_ID)

    for rf in result:
        assert isinstance(rf.frame_index, int)
        assert rf.event_id == _EVENT_ID
        assert isinstance(rf.timestamp_sec, float)
        assert isinstance(rf.evidence_score, float)
        assert rf.selection_reason  # bos olmayan, insan-okur bir gerekce


# --------------------------- Pre/peak/post konumsal etiketi TAMAMEN kaldirilmis olmali ---------------------------


def test_representative_frame_model_has_no_positional_role_field() -> None:
    """Cikti modelinde artik 'label'/'frame_role'/'position_type' gibi konumsal bir alan bulunmamali."""
    candidates = [_evidence_frame(i, float(i)) for i in range(5)]
    peak = candidates[2]
    peak.change_score = 99.0

    result = FrameSelector.select(peak, candidates, event_id=_EVENT_ID)

    for rf in result:
        field_names = set(type(rf).model_fields.keys())
        assert "label" not in field_names
        assert "frame_role" not in field_names
        assert "frame_type" not in field_names
        assert "position_type" not in field_names


def test_no_pre_peak_post_strings_anywhere_in_selection_reasons() -> None:
    """`selection_reason` degerleri 'pre'/'peak'/'post' gibi konumsal kelimeler ICERMEMELI."""
    candidates = [_evidence_frame(i, float(i)) for i in range(20)]
    peak = candidates[10]
    peak.change_score = 10.0

    result = FrameSelector.select(peak, candidates, event_id=_EVENT_ID)

    for rf in result:
        reason_lower = rf.selection_reason.lower()
        assert "pre" not in reason_lower
        assert "post" not in reason_lower
        assert "peak" not in reason_lower
