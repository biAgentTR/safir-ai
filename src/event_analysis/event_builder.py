
from __future__ import annotations

import logging
from typing import Dict, List

from src.event_analysis.schemas import RuleMatch, StructuredEvent, TemporalEvent

logger = logging.getLogger(__name__)


class EventBuilder:

    def build(self, temporal_event: TemporalEvent, rule_matches: List[RuleMatch]) -> StructuredEvent:
        description = self._compose_description(temporal_event, rule_matches)

        return StructuredEvent(
            timestamp=temporal_event.end_timestamp,
            description=description,
            risk_score=None,
            risk_level=None,
            source_model=temporal_event.source_model,
            event_type=temporal_event.event_type,
            confidence=temporal_event.confidence,
            temporal_event_id=temporal_event.event_id,
            related_rule_matches=list(rule_matches),
            occurrence_count=temporal_event.occurrence_count,
            duration=temporal_event.duration,
        )

    def build_batch(
        self, temporal_events: List[TemporalEvent], rule_matches: List[RuleMatch]
    ) -> List[StructuredEvent]:
        matches_by_event_id = self._group_rule_matches_by_event_id(rule_matches)
        structured_events = [
            self.build(temporal_event, matches_by_event_id.get(temporal_event.event_id, []))
            for temporal_event in temporal_events
        ]
        logger.debug(
            "EventBuilder: %d TemporalEvent + %d RuleMatch -> %d StructuredEvent",
            len(temporal_events),
            len(rule_matches),
            len(structured_events),
        )
        return structured_events

    @staticmethod
    def _group_rule_matches_by_event_id(rule_matches: List[RuleMatch]) -> Dict[str, List[RuleMatch]]:
        grouped: Dict[str, List[RuleMatch]] = {}
        for match in rule_matches:
            for event_id in [match.source_event_id, *match.related_event_ids]:
                grouped.setdefault(event_id, []).append(match)
        return grouped

    @staticmethod
    def _compose_description(temporal_event: TemporalEvent, rule_matches: List[RuleMatch]) -> str:
        parts = [temporal_event.description]

        if temporal_event.occurrence_count > 1:
            parts.append(
                f"(Bu durum {temporal_event.occurrence_count} ardisik gozlemde, "
                f"{temporal_event.duration:.1f}s boyunca tekrarlandi.)"
            )

        if rule_matches:
            rule_summary = " | ".join(
                f"{match.rule_id} ({match.severity}): {match.rule_description}" for match in rule_matches
            )
            parts.append(f"Tetiklenen kural(lar): {rule_summary}")

        return " ".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo_temporal_event = TemporalEvent(
        event_id="evt_0",
        event_type="kkd_ihlali",
        description="Personel baretsiz calisiyor.",
        start_timestamp=0.0,
        end_timestamp=5.0,
        duration=5.0,
        confidence=0.6,
        occurrence_count=2,
        matched_keywords=["baretsiz"],
        source_model="demo-vlm",
        related_events=[],
    )
    demo_rule_match = RuleMatch(
        rule_id="ISG-M24",
        rule_description="ISG Yonetmeligi Madde 24",
        event_type="kkd_ihlali",
        severity="orta",
        source_event_id="evt_0",
        related_event_ids=[],
    )

    demo_structured_event = EventBuilder().build(demo_temporal_event, [demo_rule_match])
    print(demo_structured_event.model_dump_json(indent=2))
    print("to_event_store_kwargs:", demo_structured_event.to_event_store_kwargs())
