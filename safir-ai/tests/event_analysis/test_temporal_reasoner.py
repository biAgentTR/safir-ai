"""T009 (src/event_analysis/temporal_reasoner.py) icin GPU/ag bagimliligi gerektirmeyen birim testleri.

`TemporalReasoner.reason()`'in ayni tipteki yakin zamanli olaylari birlestirme
(merge) ve birlesmis olaylari birbirine baglama (relate) mantigini
dogrular; hicbir dis bagimlilik gerektirmez.
"""

from __future__ import annotations

from src.event_analysis.schemas import DetectedEvent
from src.event_analysis.temporal_reasoner import TemporalReasoner


def _event(
    event_type: str,
    timestamp: float,
    confidence: float = 0.5,
    keywords: list | None = None,
    description: str = "test aciklamasi",
) -> DetectedEvent:
    return DetectedEvent(
        event_type=event_type,
        description=description,
        timestamp=timestamp,
        confidence=confidence,
        matched_keywords=keywords or [],
        source_model="test-vlm",
    )


def test_empty_input_returns_empty_list() -> None:
    assert TemporalReasoner().reason([]) == []


def test_single_event_becomes_single_temporal_event_with_zero_duration() -> None:
    events = [_event("dusme_riski", timestamp=10.0)]
    result = TemporalReasoner().reason(events)

    assert len(result) == 1
    assert result[0].occurrence_count == 1
    assert result[0].duration == 0.0
    assert result[0].start_timestamp == result[0].end_timestamp == 10.0
    assert result[0].related_events == []


def test_same_type_events_five_seconds_apart_merge_into_one() -> None:
    """Varsayilan esik: `merge_window_sec=10.0`. 5s ayni tip -> birlesmeli."""
    events = [
        _event("dusme_riski", timestamp=0.0, keywords=["iskele"]),
        _event("dusme_riski", timestamp=5.0, keywords=["dusme onleyici"]),
    ]
    result = TemporalReasoner().reason(events)

    assert len(result) == 1
    merged = result[0]
    assert merged.occurrence_count == 2
    assert merged.start_timestamp == 0.0
    assert merged.end_timestamp == 5.0
    assert merged.duration == 5.0
    assert set(merged.matched_keywords) == {"iskele", "dusme onleyici"}


def test_same_type_events_five_minutes_apart_stay_separate() -> None:
    """Varsayilan esik: `merge_window_sec=10.0`. 300s (5dk) ayni tip -> ayri kalmali."""
    events = [
        _event("dusme_riski", timestamp=0.0),
        _event("dusme_riski", timestamp=300.0),
    ]
    result = TemporalReasoner().reason(events)

    assert len(result) == 2
    assert all(e.occurrence_count == 1 for e in result)
    assert all(e.duration == 0.0 for e in result)


def test_different_type_events_close_in_time_do_not_merge() -> None:
    events = [
        _event("dusme_riski", timestamp=0.0),
        _event("kkd_ihlali", timestamp=2.0),
    ]
    result = TemporalReasoner().reason(events)

    assert len(result) == 2
    assert {e.event_type for e in result} == {"dusme_riski", "kkd_ihlali"}


def test_three_consecutive_same_type_events_merge_into_one_ongoing_event() -> None:
    events = [
        _event("dusme_riski", timestamp=0.0),
        _event("dusme_riski", timestamp=5.0),
        _event("dusme_riski", timestamp=9.0),
    ]
    result = TemporalReasoner().reason(events)

    assert len(result) == 1
    assert result[0].occurrence_count == 3
    assert result[0].duration == 9.0


def test_confidence_is_boosted_for_repeated_corroborated_occurrences() -> None:
    events = [
        _event("dusme_riski", timestamp=0.0, confidence=0.5),
        _event("dusme_riski", timestamp=5.0, confidence=0.5),
        _event("dusme_riski", timestamp=9.0, confidence=0.5),
    ]
    result = TemporalReasoner().reason(events)

    assert result[0].confidence == 0.6  # 0.5 + 0.05 * (3 - 1)


def test_confidence_boost_is_capped_at_one() -> None:
    events = [
        _event("dusme_riski", timestamp=0.0, confidence=0.95),
        _event("dusme_riski", timestamp=5.0, confidence=0.95),
        _event("dusme_riski", timestamp=9.0, confidence=1.0),
    ]
    result = TemporalReasoner().reason(events)

    assert result[0].confidence == 1.0


def test_description_reflects_latest_occurrence_in_merged_group() -> None:
    events = [
        _event("dusme_riski", timestamp=0.0, description="ilk gozlem"),
        _event("dusme_riski", timestamp=5.0, description="son gozlem"),
    ]
    result = TemporalReasoner().reason(events)

    assert result[0].description == "son gozlem"


def test_related_events_link_temporal_events_within_relation_window() -> None:
    """Varsayilan `relation_window_sec=30.0`. Farkli tip, 8s arayla -> iliskili olmali."""
    events = [
        _event("arac_yaya_yakinligi", timestamp=0.0),
        _event("kkd_ihlali", timestamp=8.0),
    ]
    result = TemporalReasoner().reason(events)

    assert len(result) == 2
    by_type = {e.event_type: e for e in result}
    assert by_type["kkd_ihlali"].event_id in by_type["arac_yaya_yakinligi"].related_events
    assert by_type["arac_yaya_yakinligi"].event_id in by_type["kkd_ihlali"].related_events


def test_related_events_stay_empty_when_outside_relation_window() -> None:
    events = [
        _event("arac_yaya_yakinligi", timestamp=0.0),
        _event("kkd_ihlali", timestamp=120.0),
    ]
    result = TemporalReasoner().reason(events)

    assert len(result) == 2
    for temporal_event in result:
        assert temporal_event.related_events == []


def test_unsorted_input_is_handled_defensively() -> None:
    events = [
        _event("dusme_riski", timestamp=5.0),
        _event("dusme_riski", timestamp=0.0),
    ]
    result = TemporalReasoner().reason(events)

    assert len(result) == 1
    assert result[0].start_timestamp == 0.0
    assert result[0].end_timestamp == 5.0


def test_custom_merge_window_can_be_tightened() -> None:
    reasoner = TemporalReasoner(merge_window_sec=3.0)
    events = [
        _event("dusme_riski", timestamp=0.0),
        _event("dusme_riski", timestamp=5.0),
    ]
    result = reasoner.reason(events)

    assert len(result) == 2


def test_custom_relation_window_can_be_widened() -> None:
    reasoner = TemporalReasoner(relation_window_sec=200.0)
    events = [
        _event("arac_yaya_yakinligi", timestamp=0.0),
        _event("kkd_ihlali", timestamp=120.0),
    ]
    result = reasoner.reason(events)

    assert all(temporal_event.related_events for temporal_event in result)


def test_event_ids_are_unique_and_stable_ordering() -> None:
    events = [
        _event("dusme_riski", timestamp=0.0),
        _event("kkd_ihlali", timestamp=50.0),
        _event("yangin_duman", timestamp=100.0),
    ]
    result = TemporalReasoner().reason(events)

    ids = [e.event_id for e in result]
    assert ids == ["evt_0", "evt_1", "evt_2"]
    assert len(set(ids)) == len(ids)


# --- T015: tip-bazli gruplama (araya farkli tip girmesi birlestirmeyi engellemez) ---


def test_same_type_events_merge_even_with_a_different_type_interleaved() -> None:
    """A(t=0) -> B(t=3) -> A(t=5): iki A, araya B girmis olsa da birlesmeli."""
    events = [
        _event("kkd_ihlali", timestamp=0.0, keywords=["baretsiz"]),
        _event("arac_yaya_yakinligi", timestamp=3.0, keywords=["forklift"]),
        _event("kkd_ihlali", timestamp=5.0, keywords=["korumasiz alan"]),
    ]
    result = TemporalReasoner().reason(events)

    assert len(result) == 2
    by_type = {e.event_type: e for e in result}

    merged_kkd = by_type["kkd_ihlali"]
    assert merged_kkd.occurrence_count == 2
    assert merged_kkd.start_timestamp == 0.0
    assert merged_kkd.end_timestamp == 5.0
    assert merged_kkd.duration == 5.0
    assert set(merged_kkd.matched_keywords) == {"baretsiz", "korumasiz alan"}

    standalone_arac_yaya = by_type["arac_yaya_yakinligi"]
    assert standalone_arac_yaya.occurrence_count == 1
    assert standalone_arac_yaya.duration == 0.0


def test_interleaved_same_type_events_stay_separate_when_gap_too_large() -> None:
    """A(t=0) -> B(t=3) -> A(t=50): araya girme, zaman penceresini gecersiz kilmaz - gap 50 > 10 -> ayri kalmali."""
    events = [
        _event("kkd_ihlali", timestamp=0.0),
        _event("arac_yaya_yakinligi", timestamp=3.0),
        _event("kkd_ihlali", timestamp=50.0),
    ]
    result = TemporalReasoner().reason(events)

    kkd_events = [e for e in result if e.event_type == "kkd_ihlali"]
    assert len(kkd_events) == 2
    assert all(e.occurrence_count == 1 for e in kkd_events)

    arac_yaya_events = [e for e in result if e.event_type == "arac_yaya_yakinligi"]
    assert len(arac_yaya_events) == 1
    assert arac_yaya_events[0].occurrence_count == 1


def test_three_way_interleaving_each_type_merges_independently() -> None:
    """A(0) -> B(2) -> A(4) -> B(6) -> A(8): her tip kendi ardisik olaylarini bagimsiz birlestirmeli."""
    events = [
        _event("kkd_ihlali", timestamp=0.0),
        _event("arac_yaya_yakinligi", timestamp=2.0),
        _event("kkd_ihlali", timestamp=4.0),
        _event("arac_yaya_yakinligi", timestamp=6.0),
        _event("kkd_ihlali", timestamp=8.0),
    ]
    result = TemporalReasoner().reason(events)

    assert len(result) == 2
    by_type = {e.event_type: e for e in result}

    merged_kkd = by_type["kkd_ihlali"]
    assert merged_kkd.occurrence_count == 3
    assert merged_kkd.start_timestamp == 0.0
    assert merged_kkd.end_timestamp == 8.0

    merged_arac_yaya = by_type["arac_yaya_yakinligi"]
    assert merged_arac_yaya.occurrence_count == 2
    assert merged_arac_yaya.start_timestamp == 2.0
    assert merged_arac_yaya.end_timestamp == 6.0


def test_closed_group_never_reopens_even_if_a_later_same_type_event_would_be_close_to_it() -> None:
    """A(0) -> A(15) (esik disi, yeni grup acilir) -> B(16) -> A(17): 3. A, KAPANMIS ilk gruba degil, EN SON acik A grubuna (t=15) eklenmeli."""
    events = [
        _event("kkd_ihlali", timestamp=0.0),
        _event("kkd_ihlali", timestamp=15.0),
        _event("arac_yaya_yakinligi", timestamp=16.0),
        _event("kkd_ihlali", timestamp=17.0),
    ]
    result = TemporalReasoner().reason(events)

    kkd_events = sorted(
        (e for e in result if e.event_type == "kkd_ihlali"), key=lambda e: e.start_timestamp
    )
    assert len(kkd_events) == 2
    assert kkd_events[0].occurrence_count == 1
    assert kkd_events[0].start_timestamp == 0.0
    assert kkd_events[1].occurrence_count == 2
    assert kkd_events[1].start_timestamp == 15.0
    assert kkd_events[1].end_timestamp == 17.0
