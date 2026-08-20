"""T011 - Event Builder: `TemporalEvent` + ilgili `RuleMatch`lerden nihai `StructuredEvent` uretir.

Bu modul, T009 (`TemporalReasoner`) ve T010 (`RuleEngine`) ciktilarini
birlestirip `05 LangGraph Agentic Loop`a ve `Event Gecmisi`ne (T012,
`EventStore`) gidecek tek bir yapilandirilmis nesneye (`StructuredEvent`)
indirger. `description` alani, VLM'in ham gozlem metnini tetiklenen
kural(lar)in ozetiyle birlestirerek insan-okunur bir metne cevirir; bu metin
dogrudan `EventStore.add_event(description=...)`e yazilabilir.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from src.event_analysis.risk_resolver import resolve_deterministic_risk
from src.event_analysis.schemas import RuleMatch, StructuredEvent, TemporalEvent

logger = logging.getLogger(__name__)


class EventBuilder:
    """`TemporalEvent` + `RuleMatch` listesini `StructuredEvent`e indirgeyen katman."""

    def build(self, temporal_event: TemporalEvent, rule_matches: List[RuleMatch]) -> StructuredEvent:
        """Tek bir `TemporalEvent`i, ona ait `RuleMatch`lerle birlikte `StructuredEvent`e cevirir.

        `risk_score`/`risk_level`, VLM/LLM'den DEGIL, bu olaya ait
        `rule_matches`den `resolve_deterministic_risk` ile deterministik
        olarak turetilir (bkz. `src/event_analysis/risk_resolver.py`). Hicbir
        kural eslesmediyse (`rule_matches` bos veya tumu bilinmeyen siddette)
        `None` kalir - risk UYDURULMAZ; asagi akıştaki `05 LangGraph Ajani`
        (varsa) bu bosluğu doldurabilir.

        Args:
            temporal_event: `TemporalReasoner.reason(...)` ciktisindan tek bir olay.
            rule_matches: `RuleEngine.evaluate(...)` ciktisindan bu olaya ait
                (tekli ve/veya kombinasyon) kural eslesmeleri; bos olabilir.

        Returns:
            `EventStore.add_event(...)`e (bkz. `StructuredEvent.to_event_store_kwargs`)
            ve `05 LangGraph Ajani`na dogrudan verilebilecek `StructuredEvent`.
        """
        description = self._compose_description(temporal_event, rule_matches)
        risk_level, risk_score = resolve_deterministic_risk(rule_matches)

        return StructuredEvent(
            timestamp=temporal_event.end_timestamp,
            description=description,
            risk_score=risk_score,
            risk_level=risk_level,
            source_model=temporal_event.source_model,
            event_name=temporal_event.event_name,
            event_type=temporal_event.event_type,
            confidence=temporal_event.confidence,
            temporal_event_id=temporal_event.event_id,
            related_rule_matches=list(rule_matches),
            occurrence_count=temporal_event.occurrence_count,
            duration=temporal_event.duration,
            evidence_ids=list(temporal_event.evidence_ids),
            keywords=list(temporal_event.matched_keywords),
        )

    def build_batch(
        self, temporal_events: List[TemporalEvent], rule_matches: List[RuleMatch]
    ) -> List[StructuredEvent]:
        """Bir `TemporalEvent` listesini, ilgili `RuleMatch`leri dogru olaya eslestirerek toplu `StructuredEvent`e cevirir.

        Her `RuleMatch`, `source_event_id`sine VE `related_event_ids`
        listesindeki her kimlige (kombinasyon kurallarinda katilan diger
        olaylar) atanir; boylece bir kombinasyon eslesmesi, ilgili tum
        `TemporalEvent`lerin `StructuredEvent.related_rule_matches`
        alaninda gorunur.

        Args:
            temporal_events: `TemporalReasoner.reason(...)` ciktisi.
            rule_matches: `RuleEngine.evaluate(temporal_events)` ciktisi.

        Returns:
            `temporal_events` ile ayni sirada, her biri kendi ilgili
            `RuleMatch`leriyle zenginlestirilmis `StructuredEvent` listesi.
        """
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
        """`RuleMatch`leri, `source_event_id` ve `related_event_ids` uzerinden ilgili olduklari `event_id`lere gruplar.

        Args:
            rule_matches: `RuleEngine.evaluate(...)` ciktisi.

        Returns:
            `TemporalEvent.event_id` -> o olayi ilgilendiren `RuleMatch`
            listesi sozlugu. Bir kombinasyon eslesmesi, hem kaynak hem
            iliskili tum kimliklerin listesinde gorunur.
        """
        grouped: Dict[str, List[RuleMatch]] = {}
        for match in rule_matches:
            for event_id in [match.source_event_id, *match.related_event_ids]:
                grouped.setdefault(event_id, []).append(match)
        return grouped

    @staticmethod
    def _compose_description(temporal_event: TemporalEvent, rule_matches: List[RuleMatch]) -> str:
        """VLM gozlem metnini, tekrar bilgisini ve tetiklenen kural(lar)i tek bir insan-okunur metne birlestirir.

        Args:
            temporal_event: Temsil edilen olay.
            rule_matches: Bu olaya ait kural eslesmeleri (bos olabilir).

        Returns:
            `EventStore`in `description TEXT` alanina dogrudan yazilabilecek metin.
        """
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
    # T011'in bagimsiz calistirilabilirlik testi:
    #   python -m src.event_analysis.event_builder
    logging.basicConfig(level=logging.INFO)

    demo_temporal_event = TemporalEvent(
        event_id="evt_0",
        event_name="kkd_ihlali",
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
