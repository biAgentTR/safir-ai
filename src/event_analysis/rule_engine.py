
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

import yaml

from src.event_analysis.schemas import (
    EVENT_TYPE_REGULATION_MAP,
    EventType,
    RuleMatch,
    TemporalEvent,
)

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "rules" / "isg_rules.yaml"

_RULE_ID_BY_EVENT_TYPE: Dict[EventType, str] = {
    EventType.DUSME_RISKI: "ISG-M12",
    EventType.KKD_IHLALI: "ISG-M24",
    EventType.ARAC_YAYA_YAKINLIGI: "OK-07",
    EventType.SICAK_CALISMA_IHLALI: "ISG-M31",
    EventType.YANGIN_DUMAN: "YG-03",
    EventType.DAR_ALAN_IHLALI: "ISG-M45",
    EventType.ENERJI_KESME_IHLALI: "OK-15",
    EventType.AGIR_YUK_RISKI: "ISG-M52",
}

_DEFAULT_SEVERITY_BY_EVENT_TYPE: Dict[EventType, str] = {
    EventType.DUSME_RISKI: "yuksek",
    EventType.KKD_IHLALI: "orta",
    EventType.ARAC_YAYA_YAKINLIGI: "yuksek",
    EventType.SICAK_CALISMA_IHLALI: "yuksek",
    EventType.YANGIN_DUMAN: "kritik",
    EventType.DAR_ALAN_IHLALI: "yuksek",
    EventType.ENERJI_KESME_IHLALI: "kritik",
    EventType.AGIR_YUK_RISKI: "yuksek",
}


@runtime_checkable
class RegulationRetriever(Protocol):

    def run(self, question: str, top_k: int = 3) -> str:
        ...


def _safe_event_type(value: str) -> Optional[EventType]:
    try:
        return EventType(value)
    except ValueError:
        return None


class RuleEngine:

    def __init__(
        self,
        retriever: Optional[RegulationRetriever] = None,
        rules_path: str | Path = DEFAULT_RULES_PATH,
    ) -> None:
        self._retriever = retriever
        self._combination_rules = self._load_combination_rules(Path(rules_path))

    @staticmethod
    def _load_combination_rules(rules_path: Path) -> List[dict]:
        if not rules_path.exists():
            raise FileNotFoundError(f"Kombinasyon kurallari dosyasi bulunamadi: {rules_path}")

        with rules_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        return data.get("combination_rules", [])

    def evaluate(self, temporal_events: List[TemporalEvent]) -> List[RuleMatch]:
        if not temporal_events:
            return []

        sorted_events = sorted(temporal_events, key=lambda event: event.start_timestamp)

        matches: List[RuleMatch] = []
        for event in sorted_events:
            matches.extend(self._match_single_event_rule(event))
        matches.extend(self._match_combination_rules(sorted_events))

        logger.debug(
            "RuleEngine: %d TemporalEvent -> %d RuleMatch", len(temporal_events), len(matches)
        )
        return matches

    def _match_single_event_rule(self, event: TemporalEvent) -> List[RuleMatch]:
        event_type = _safe_event_type(event.event_type)
        if event_type is None:
            return []

        regulation_label = EVENT_TYPE_REGULATION_MAP.get(event_type)
        if regulation_label is None:
            return []

        return [
            RuleMatch(
                rule_id=_RULE_ID_BY_EVENT_TYPE.get(event_type, event_type.value),
                rule_description=self._describe_regulation(regulation_label),
                event_type=event.event_type,
                severity=_DEFAULT_SEVERITY_BY_EVENT_TYPE.get(event_type, "orta"),
                source_event_id=event.event_id,
                related_event_ids=[],
            )
        ]

    def _describe_regulation(self, regulation_label: str) -> str:
        if self._retriever is None:
            return regulation_label

        try:
            enriched = self._retriever.run(question=regulation_label, top_k=1)
        except Exception:
            logger.exception(
                "RuleEngine: retriever cagrisi basarisiz, kisa mevzuat etiketiyle devam ediliyor."
            )
            return regulation_label

        return enriched or regulation_label

    def _match_combination_rules(self, sorted_events: List[TemporalEvent]) -> List[RuleMatch]:
        events_by_id = {event.event_id: event for event in sorted_events}
        seen_matches: set = set()
        matches: List[RuleMatch] = []

        for anchor in sorted_events:
            cluster = [anchor] + [
                events_by_id[related_id]
                for related_id in anchor.related_events
                if related_id in events_by_id
            ]

            for rule in self._combination_rules:
                required_types = {
                    event_type
                    for raw in rule.get("required_event_types", [])
                    if (event_type := _safe_event_type(raw)) is not None
                }
                if not required_types:
                    continue

                participants_by_id = {
                    member.event_id: member
                    for member in cluster
                    if _safe_event_type(member.event_type) in required_types
                }
                participant_types = {
                    _safe_event_type(member.event_type) for member in participants_by_id.values()
                }
                if not required_types.issubset(participant_types):
                    continue

                participant_ids = tuple(sorted(participants_by_id.keys()))
                dedup_key = (rule["rule_id"], participant_ids)
                if dedup_key in seen_matches:
                    continue
                seen_matches.add(dedup_key)

                source_event_id = anchor.event_id if anchor.event_id in participant_ids else participant_ids[0]
                related_event_ids = [pid for pid in participant_ids if pid != source_event_id]
                combined_event_type = "+".join(
                    sorted(t.value for t in required_types)
                )

                matches.append(
                    RuleMatch(
                        rule_id=rule["rule_id"],
                        rule_description=rule["description"].strip(),
                        event_type=combined_event_type,
                        severity=rule["severity"],
                        source_event_id=source_event_id,
                        related_event_ids=related_event_ids,
                    )
                )

        return matches


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo_events = [
        TemporalEvent(
            event_id="evt_0",
            event_type="kkd_ihlali",
            description="Personel baretsiz calisiyor.",
            start_timestamp=0.0,
            end_timestamp=0.0,
            duration=0.0,
            confidence=0.6,
            occurrence_count=1,
            matched_keywords=["baretsiz"],
            source_model="demo-vlm",
            related_events=["evt_1"],
        ),
        TemporalEvent(
            event_id="evt_1",
            event_type="arac_yaya_yakinligi",
            description="Forklift yaya gecidine yaklasiyor.",
            start_timestamp=8.0,
            end_timestamp=8.0,
            duration=0.0,
            confidence=0.6,
            occurrence_count=1,
            matched_keywords=["forklift"],
            source_model="demo-vlm",
            related_events=["evt_0"],
        ),
    ]

    for demo_match in RuleEngine().evaluate(demo_events):
        print(
            f"[{demo_match.rule_id}] ({demo_match.severity}) {demo_match.event_type} "
            f"source={demo_match.source_event_id} related={demo_match.related_event_ids}"
        )
