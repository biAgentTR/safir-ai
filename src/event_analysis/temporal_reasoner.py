
from __future__ import annotations

import logging
from typing import Dict, List

from src.event_analysis.schemas import DetectedEvent, TemporalEvent

logger = logging.getLogger(__name__)

DEFAULT_MERGE_WINDOW_SEC = 10.0
"""Ayni `event_type`e sahip ardisik `DetectedEvent`lerin tek bir suregelen
`TemporalEvent`e birlesmesi icin izin verilen maksimum zaman farki (saniye)."""

DEFAULT_RELATION_WINDOW_SEC = 30.0
"""Farkli (veya birlesmemis ayni) `TemporalEvent`lerin birbirine
`related_events` olarak baglanmasi icin izin verilen maksimum zaman
bosluğu (saniye)."""

_CONFIDENCE_BOOST_PER_EXTRA_OCCURRENCE = 0.05


class TemporalReasoner:

    def __init__(
        self,
        merge_window_sec: float = DEFAULT_MERGE_WINDOW_SEC,
        relation_window_sec: float = DEFAULT_RELATION_WINDOW_SEC,
    ) -> None:
        self._merge_window_sec = merge_window_sec
        self._relation_window_sec = relation_window_sec

    def reason(self, detected_events: List[DetectedEvent]) -> List[TemporalEvent]:
        if not detected_events:
            return []

        sorted_events = sorted(detected_events, key=lambda event: event.timestamp)
        groups = self._group_by_type_and_proximity(sorted_events)

        temporal_events = [
            self._build_temporal_event(group, index) for index, group in enumerate(groups)
        ]
        self._link_related_events(temporal_events)

        logger.debug(
            "TemporalReasoner: %d DetectedEvent -> %d TemporalEvent",
            len(detected_events),
            len(temporal_events),
        )
        return temporal_events

    def _group_by_type_and_proximity(
        self, sorted_events: List[DetectedEvent]
    ) -> List[List[DetectedEvent]]:
        groups: List[List[DetectedEvent]] = []
        last_group_index_by_type: Dict[str, int] = {}

        for event in sorted_events:
            last_index = last_group_index_by_type.get(event.event_type)
            if last_index is not None:
                last_event = groups[last_index][-1]
                if (event.timestamp - last_event.timestamp) <= self._merge_window_sec:
                    groups[last_index].append(event)
                    continue

            groups.append([event])
            last_group_index_by_type[event.event_type] = len(groups) - 1

        return groups

    def _build_temporal_event(self, group: List[DetectedEvent], index: int) -> TemporalEvent:
        start_timestamp = group[0].timestamp
        end_timestamp = group[-1].timestamp
        latest = group[-1]

        max_confidence = max(event.confidence for event in group)
        occurrence_count = len(group)
        boosted_confidence = round(
            min(1.0, max_confidence + _CONFIDENCE_BOOST_PER_EXTRA_OCCURRENCE * (occurrence_count - 1)),
            2,
        )

        return TemporalEvent(
            event_id=f"evt_{index}",
            event_type=latest.event_type,
            description=latest.description,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            duration=round(end_timestamp - start_timestamp, 2),
            confidence=boosted_confidence,
            occurrence_count=occurrence_count,
            matched_keywords=self._union_keywords(group),
            source_model=latest.source_model,
            related_events=[],
        )

    @staticmethod
    def _union_keywords(group: List[DetectedEvent]) -> List[str]:
        seen: List[str] = []
        for event in group:
            for keyword in event.matched_keywords:
                if keyword not in seen:
                    seen.append(keyword)
        return seen

    def _link_related_events(self, temporal_events: List[TemporalEvent]) -> None:
        for i, current in enumerate(temporal_events):
            related_ids = [
                other.event_id
                for j, other in enumerate(temporal_events)
                if i != j and self._time_gap(current, other) <= self._relation_window_sec
            ]
            current.related_events = related_ids

    @staticmethod
    def _time_gap(a: TemporalEvent, b: TemporalEvent) -> float:
        if a.end_timestamp < b.start_timestamp:
            return b.start_timestamp - a.end_timestamp
        if b.end_timestamp < a.start_timestamp:
            return a.start_timestamp - b.end_timestamp
        return 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo_events = [
        DetectedEvent(
            event_type="dusme_riski",
            description="Personel korkuluksuz iskelede calisiyor.",
            timestamp=0.0,
            confidence=0.5,
            matched_keywords=["iskele"],
            source_model="demo-vlm",
        ),
        DetectedEvent(
            event_type="dusme_riski",
            description="Personel hala korkuluksuz iskelede.",
            timestamp=5.0,
            confidence=0.5,
            matched_keywords=["iskele"],
            source_model="demo-vlm",
        ),
        DetectedEvent(
            event_type="kkd_ihlali",
            description="Ayni bolgede baretsiz bir personel daha goruldu.",
            timestamp=12.0,
            confidence=0.5,
            matched_keywords=["baretsiz"],
            source_model="demo-vlm",
        ),
    ]

    for demo_temporal_event in TemporalReasoner().reason(demo_events):
        print(
            f"[{demo_temporal_event.event_id}] {demo_temporal_event.event_type} "
            f"occurrence_count={demo_temporal_event.occurrence_count} "
            f"duration={demo_temporal_event.duration} "
            f"related={demo_temporal_event.related_events}"
        )