"""Olay Analizi katmaninin (T008-T012) uctan uca manuel inceleme scripti.

`04 Context Builder` ile `05 LangGraph Agentic Loop` arasindaki `src/event_analysis/`
zincirini (EventEngine -> TemporalReasoner -> RuleEngine -> EventBuilder ->
EventHistory), gercekci, birden fazla zaman damgali TR VLM aciklama metniyle
uctan uca calistirir. `EventHistory`, gercek `EventStore`/SQLite yerine
in-memory bir sahte depoya (`_InMemoryEventStore`) yazar; bu yuzden GPU,
Docker veya gercek bir veritabani gerektirmez.

Kullanim (repo kokunden veya `safir-ai/` icinden):
    python scripts/manual_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from src.event_analysis.event_builder import EventBuilder  # noqa: E402
from src.event_analysis.event_engine import EventEngine  # noqa: E402
from src.event_analysis.event_history import EventHistory  # noqa: E402
from src.event_analysis.rule_engine import RuleEngine  # noqa: E402
from src.event_analysis.schemas import DetectedEvent, EventEngineInput  # noqa: E402
from src.event_analysis.temporal_reasoner import TemporalReasoner  # noqa: E402


class _InMemoryEventStore:
    """`EventStoreLike` Protocol'une uyan, gercek SQLite gerektirmeyen sahte depo.

    `src/memory/event_store.py::EventStore` ile ayni `add_event`/
    `record_feedback` imzalarina sahiptir; bu script bunun disinda
    `src/memory/`e hicbir bagimlilik eklemez.
    """

    def __init__(self) -> None:
        self.rows: List[dict] = []

    def add_event(
        self,
        timestamp: float,
        description: str,
        risk_score: Optional[int] = None,
        risk_level: Optional[str] = None,
        source_model: Optional[str] = None,
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
        for row in self.rows:
            if row["id"] == event_id:
                row["feedback"] = feedback
                return
        raise ValueError(f"Olay bulunamadi: id={event_id}")


# Gercekci senaryo: bir personel baretsiz tehlikeli bolgeye girip 4 saniye
# sonra da orada calismaya devam ediyor (ayni tip -> birlesmeli), ardindan
# forklift o bolgeye yaklasiyor (farkli tip ama yakin zamanli -> iliskili
# olmali, COMBO-01'i tetiklemeli). Ayrica saha aninda, ilkiyle alakasiz,
# gec bir zaman diliminde bir duman/yangin olayi da gozlemleniyor (30
# saniyelik iliski penceresinin disinda kalmali, iliskisiz kalmali).
SCENARIO: List[Tuple[float, str]] = [
    (0.0, "Personel baretsiz, korumasiz alanda hareket ediyor."),
    (4.0, "Ayni personel hala baretsiz sekilde alanda calismaya devam ediyor."),
    (9.0, "Forklift, personelin bulundugu yaya gecidine yaklasiyor."),
    (40.0, "Sahada duman gorulmeye basladi."),
    (45.0, "Duman yogunlasti, yangin riski olusuyor."),
]

# Gercek pipeline'da risk_score/risk_level `05 LangGraph Ajani`nin kararindan
# gelir (bkz. src/event_analysis/event_history.py docstring'i: EventHistory,
# EventStore'da UPDATE metodu olmadigi icin bu bilgiyi record() cagrisinda
# TEK SEFERDE alir). Bu script'te gercek bir ajan calistirilmadigi icin,
# yalnizca demo amacli, olay tipine gore sabit bir yer-tutucu risk atanir.
_PLACEHOLDER_RISK_BY_EVENT_TYPE: Dict[str, Tuple[int, str]] = {
    "kkd_ihlali": (35, "orta"),
    "arac_yaya_yakinligi": (70, "yuksek"),
    "yangin_duman": (85, "kritik"),
}


def main() -> None:
    print("=" * 78)
    print("SAFIR Event Analysis (T008-T012) - Manuel Uctan Uca Inceleme")
    print("=" * 78)

    event_engine = EventEngine()
    temporal_reasoner = TemporalReasoner()
    rule_engine = RuleEngine()  # retriever=None -> kisa mevzuat etiketleriyle calisir
    event_builder = EventBuilder()
    event_store = _InMemoryEventStore()
    event_history = EventHistory(event_store)

    # --- 1) Event Engine (T008): her VLM cikisini ayri ayri tara ---
    print("\n--- 1) Event Engine (T008): VLM aciklamalari -> DetectedEvent ---")
    detected_events: List[DetectedEvent] = []
    for timestamp, description in SCENARIO:
        engine_input = EventEngineInput(
            vlm_description=description,
            timestamp=timestamp,
            source_model="qwen2.5-vl-demo",
            frame_count=1,
        )
        events = event_engine.detect(engine_input)
        detected_events.extend(events)
        for event in events:
            print(
                f"  [t={timestamp:5.1f}s] {event.event_type:<24} "
                f"conf={event.confidence:.2f} kw={event.matched_keywords}"
            )

    # --- 2) Temporal Reasoning (T009): DetectedEvent -> TemporalEvent ---
    print("\n--- 2) Temporal Reasoning (T009): DetectedEvent -> TemporalEvent ---")
    temporal_events = temporal_reasoner.reason(detected_events)
    for te in temporal_events:
        print(
            f"  [{te.event_id}] {te.event_type:<24} "
            f"t=[{te.start_timestamp:5.1f} - {te.end_timestamp:5.1f}] "
            f"occurrence={te.occurrence_count} conf={te.confidence:.2f} "
            f"related={te.related_events}"
        )

    # --- 3) Rule Engine (T010): TemporalEvent -> RuleMatch (tekli + kombinasyon) ---
    print("\n--- 3) Rule Engine (T010): TemporalEvent -> RuleMatch ---")
    rule_matches = rule_engine.evaluate(temporal_events)
    for match in rule_matches:
        print(
            f"  [{match.rule_id}] ({match.severity}) event_type={match.event_type} "
            f"source={match.source_event_id} related={match.related_event_ids}"
        )
        print(f"      -> {match.rule_description}")

    # --- 4) Event Builder (T011): TemporalEvent + RuleMatch -> StructuredEvent ---
    print("\n--- 4) Event Builder (T011): TemporalEvent + RuleMatch -> StructuredEvent ---")
    structured_events = event_builder.build_batch(temporal_events, rule_matches)
    for se in structured_events:
        print(f"  [{se.temporal_event_id}] {se.event_type} (t={se.timestamp:.1f}s)")
        print(f"      description: {se.description}")

    # --- 5) Event History (T012): StructuredEvent -> EventStore (in-memory) ---
    print("\n--- 5) Event History (T012): StructuredEvent -> EventStore (in-memory) ---")
    event_ids: List[int] = []
    for se in structured_events:
        score, level = _PLACEHOLDER_RISK_BY_EVENT_TYPE.get(se.event_type, (None, None))
        event_id = event_history.record(se, risk_score=score, risk_level=level)
        event_ids.append(event_id)
        print(f"  kaydedildi: event_id={event_id} risk={score}({level})")

    if event_ids:
        event_history.mark_feedback(event_ids[0], "true_positive")
        print(f"\n  HITL geri bildirimi islendi: event_id={event_ids[0]} -> true_positive")

    print("\n--- SQLite'a yazilacak nihai satirlar (in-memory sahte depo) ---")
    pprint(event_store.rows)

    print("\n" + "=" * 78)
    print(
        f"Ozet: {len(SCENARIO)} VLM cikisi -> {len(detected_events)} DetectedEvent -> "
        f"{len(temporal_events)} TemporalEvent -> {len(rule_matches)} RuleMatch -> "
        f"{len(structured_events)} StructuredEvent -> {len(event_store.rows)} SQLite satiri"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
