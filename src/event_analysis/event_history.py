
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

from src.event_analysis.schemas import StructuredEvent

if TYPE_CHECKING:
    from src.memory.event_store import EventStore

logger = logging.getLogger(__name__)


@runtime_checkable
class EventStoreLike(Protocol):

    def add_event(
        self,
        timestamp: float,
        description: str,
        risk_score: Optional[int] = None,
        risk_level: Optional[str] = None,
        source_model: Optional[str] = None,
    ) -> int:
        ...

    def record_feedback(self, event_id: int, feedback: str) -> None:
        ...


class EventHistory:

    def __init__(self, event_store: "EventStore | EventStoreLike") -> None:
        self._event_store = event_store

    def record(
        self,
        event: StructuredEvent,
        risk_score: Optional[int] = None,
        risk_level: Optional[str] = None,
    ) -> int:
        kwargs = event.to_event_store_kwargs()
        if risk_score is not None:
            kwargs["risk_score"] = risk_score
        if risk_level is not None:
            kwargs["risk_level"] = risk_level

        event_id = self._event_store.add_event(**kwargs)
        logger.debug(
            "EventHistory: olay kaydedildi id=%s event_type=%s risk_score=%s risk_level=%s",
            event_id,
            event.event_type,
            kwargs["risk_score"],
            kwargs["risk_level"],
        )
        return event_id

    def record_batch(
        self,
        events: List[StructuredEvent],
        risk_scores: Optional[List[Optional[int]]] = None,
        risk_levels: Optional[List[Optional[str]]] = None,
    ) -> List[int]:
        
        if risk_scores is not None and len(risk_scores) != len(events):
            raise ValueError("risk_scores uzunlugu events ile eslesmeli.")
        if risk_levels is not None and len(risk_levels) != len(events):
            raise ValueError("risk_levels uzunlugu events ile eslesmeli.")

        event_ids: List[int] = []
        for index, event in enumerate(events):
            score = risk_scores[index] if risk_scores is not None else None
            level = risk_levels[index] if risk_levels is not None else None
            event_ids.append(self.record(event, risk_score=score, risk_level=level))

        logger.debug("EventHistory: %d olay toplu kaydedildi.", len(event_ids))
        return event_ids

    def mark_feedback(self, event_id: int, feedback: str) -> None:
        self._event_store.record_feedback(event_id, feedback)


if __name__ == "__main__":
    import tempfile

    from src.memory.event_store import EventStore
    from src.utils.config_loader import SQLiteMemoryConfig

    logging.basicConfig(level=logging.INFO)

    demo_event = StructuredEvent(
        timestamp=12.4,
        description="Personel baretsiz calisiyor.",
        source_model="demo-vlm",
        event_type="kkd_ihlali",
        confidence=0.6,
        temporal_event_id="evt_0",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        demo_store = EventStore(SQLiteMemoryConfig(db_path=f"{tmp_dir}/demo_events.db"))
        demo_history = EventHistory(demo_store)

        early_id = demo_history.record(demo_event)
        print(f"Erken kayit (risk yok): id={early_id}")

        decided_id = demo_history.record(demo_event, risk_score=45, risk_level="orta")
        print(f"Karar sonrasi kayit: id={decided_id}")

        demo_history.mark_feedback(decided_id, "true_positive")
        print("Geri bildirim islendi.")

        demo_store.close()
