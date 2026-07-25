"""T012 (src/event_analysis/event_history.py) icin GPU/ag/gercek-DB bagimliligi gerektirmeyen birim testleri.

`EventHistory`, gercek `src.memory.event_store.EventStore`e degil, bu
dosyada tanimli in-memory bir sahte depoya (`_FakeEventStore`) karsi test
edilir; `EventStoreLike` Protocol'une uyar.
"""

from __future__ import annotations

import pytest

from src.event_analysis.event_history import EventHistory, EventStoreLike
from src.event_analysis.schemas import StructuredEvent


class _FakeEventStore:
    """`EventStoreLike` Protocol'une uyan, gercek SQLite'a bagli olmayan in-memory test cifti."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.feedback_calls: list[tuple[int, str]] = []

    def add_event(
        self,
        timestamp: float,
        description: str,
        risk_score: int | None = None,
        risk_level: str | None = None,
        source_model: str | None = None,
    ) -> int:
        event_id = len(self.rows)
        self.rows.append(
            {
                "id": event_id,
                "timestamp": timestamp,
                "description": description,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "source_model": source_model,
            }
        )
        return event_id

    def record_feedback(self, event_id: int, feedback: str) -> None:
        self.feedback_calls.append((event_id, feedback))


def _structured_event(
    timestamp: float = 10.0,
    description: str = "Personel baretsiz calisiyor.",
    event_type: str = "kkd_ihlali",
    temporal_event_id: str = "evt_0",
) -> StructuredEvent:
    return StructuredEvent(
        timestamp=timestamp,
        description=description,
        source_model="test-vlm",
        event_type=event_type,
        confidence=0.6,
        temporal_event_id=temporal_event_id,
    )


def test_fake_event_store_conforms_to_protocol() -> None:
    assert isinstance(_FakeEventStore(), EventStoreLike)


def test_record_without_risk_writes_none_none() -> None:
    store = _FakeEventStore()
    event = _structured_event()

    event_id = EventHistory(store).record(event)

    assert event_id == 0
    row = store.rows[0]
    assert row["risk_score"] is None
    assert row["risk_level"] is None
    assert row["timestamp"] == 10.0
    assert row["description"] == "Personel baretsiz calisiyor."
    assert row["source_model"] == "test-vlm"


def test_record_with_risk_overrides_none_defaults() -> None:
    store = _FakeEventStore()
    event = _structured_event()

    event_id = EventHistory(store).record(event, risk_score=75, risk_level="yuksek")

    row = store.rows[event_id]
    assert row["risk_score"] == 75
    assert row["risk_level"] == "yuksek"


def test_record_with_zero_risk_score_is_treated_as_a_real_override() -> None:
    """`risk_score=0` gecerli bir deger (dusuk risk sinirinda); `None` ile karistirilmamali."""
    store = _FakeEventStore()
    event = _structured_event()

    EventHistory(store).record(event, risk_score=0, risk_level="dusuk")

    assert store.rows[0]["risk_score"] == 0
    assert store.rows[0]["risk_level"] == "dusuk"


def test_record_partial_override_only_risk_score() -> None:
    store = _FakeEventStore()
    event = _structured_event()

    EventHistory(store).record(event, risk_score=30)

    assert store.rows[0]["risk_score"] == 30
    assert store.rows[0]["risk_level"] is None


def test_record_batch_writes_in_order_with_correct_count() -> None:
    store = _FakeEventStore()
    events = [_structured_event(timestamp=float(i), temporal_event_id=f"evt_{i}") for i in range(3)]

    event_ids = EventHistory(store).record_batch(events)

    assert event_ids == [0, 1, 2]
    assert [row["timestamp"] for row in store.rows] == [0.0, 1.0, 2.0]
    assert all(row["risk_score"] is None for row in store.rows)


def test_record_batch_applies_per_event_risk_overrides() -> None:
    store = _FakeEventStore()
    events = [_structured_event(temporal_event_id=f"evt_{i}") for i in range(3)]

    EventHistory(store).record_batch(
        events,
        risk_scores=[10, None, 90],
        risk_levels=["dusuk", None, "kritik"],
    )

    assert [row["risk_score"] for row in store.rows] == [10, None, 90]
    assert [row["risk_level"] for row in store.rows] == ["dusuk", None, "kritik"]


def test_record_batch_raises_on_mismatched_risk_scores_length() -> None:
    store = _FakeEventStore()
    events = [_structured_event(), _structured_event()]

    with pytest.raises(ValueError):
        EventHistory(store).record_batch(events, risk_scores=[10])


def test_record_batch_raises_on_mismatched_risk_levels_length() -> None:
    store = _FakeEventStore()
    events = [_structured_event(), _structured_event()]

    with pytest.raises(ValueError):
        EventHistory(store).record_batch(events, risk_levels=["dusuk"])


def test_record_batch_with_empty_events_returns_empty_list() -> None:
    store = _FakeEventStore()
    assert EventHistory(store).record_batch([]) == []


def test_mark_feedback_forwards_event_id_and_value_unchanged() -> None:
    store = _FakeEventStore()
    history = EventHistory(store)
    event_id = history.record(_structured_event())

    history.mark_feedback(event_id, "false_positive")

    assert store.feedback_calls == [(event_id, "false_positive")]


def test_mark_feedback_does_not_add_extra_validation_logic() -> None:
    """mark_feedback ince bir sarmalayici olmali: dogrulamayi EventStore tarafina birakir."""
    store = _FakeEventStore()
    history = EventHistory(store)

    history.mark_feedback(1, "gecersiz_bir_deger")

    assert store.feedback_calls == [(1, "gecersiz_bir_deger")]
