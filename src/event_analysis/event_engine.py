
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.event_analysis.schemas import DetectedEvent, EventEngineInput, EventType

logger = logging.getLogger(__name__)

_KEYWORD_RULES: Dict[EventType, List[str]] = {
    EventType.DUSME_RISKI: [
        "dusme onleyici",
        "yukseklikte calisma",
        "guvenlik kemeri",
        "korkuluk yok",
        "iskele",
    ],
    EventType.KKD_IHLALI: [
        "baretsiz",
        "yeleksiz",
        "koruyucu ekipman eksik",
        "kkd eksik",
        "korumasiz alan",
    ],
    EventType.ARAC_YAYA_YAKINLIGI: [
        "forklift",
        "yaya gecidi",
        "arac yaklas",
        "carpisma riski",
        "yaya trafigi",
    ],
    EventType.SICAK_CALISMA_IHLALI: [
        "sicak calisma izni",
        "kaynak islemi",
        "kivilcim",
        "izinsiz ates",
    ],
    EventType.YANGIN_DUMAN: [
        "duman",
        "yangin",
        "alev",
        "yanik kokusu",
    ],
    EventType.DAR_ALAN_IHLALI: [
        "kapali alan",
        "dar alan",
        "gaz olcumu yapilmadan",
        "gozetmen olmadan",
    ],
    EventType.ENERJI_KESME_IHLALI: [
        "elektrik pano",
        "enerji kesme",
        "loto",
        "kilitleme etiketleme",
    ],
    EventType.AGIR_YUK_RISKI: [
        "vinc",
        "kren",
        "agir yuk kaldirma",
        "sinyalman olmadan",
    ],
    EventType.YETKISIZ_ERISIM: [
        "yetkisiz",
        "izinsiz giris",
        "yasakli alan",
        "guvenlik ihlali",
    ],
}

_BASE_CONFIDENCE = 0.5
_CONFIDENCE_STEP_PER_EXTRA_MATCH = 0.1

_NEGATION_CUES: List[str] = [
    "degil",
    "yok",
    "yoktur",
    "olmadi",
    "bulunmuyor",
    "gorulmedi",
    "gozlemlenmedi",
    "gozlenmedi",
    "tespit edilmedi",
    "rastlanmadi",
]

_NEGATION_WINDOW_WORDS = 5

_CLAUSE_SPLIT_PATTERN = re.compile(r"[.!?;]+")


def _split_into_clauses(text: str) -> List[str]:
    return [clause.strip() for clause in _CLAUSE_SPLIT_PATTERN.split(text) if clause.strip()]


def _is_negated(clause: str, keyword: str, window_words: int = _NEGATION_WINDOW_WORDS) -> bool:
    
    start_char = clause.find(keyword)
    if start_char == -1:
        return False

    words = clause.split()
    prefix_word_count = len(clause[:start_char].split())
    keyword_word_count = len(keyword.split())

    window_start = max(0, prefix_word_count - window_words)
    window_end = min(len(words), prefix_word_count + keyword_word_count + window_words)
    context = " ".join(words[window_start:window_end])

    return any(cue in context for cue in _NEGATION_CUES)


class EventEngine:

    def __init__(self, classifier: Optional[Any] = None, min_confidence: float = 0.0) -> None:
        self._classifier = classifier
        self._min_confidence = min_confidence

    def detect(self, engine_input: EventEngineInput) -> List[DetectedEvent]:
        text_lower = engine_input.vlm_description.lower()
        clauses = _split_into_clauses(text_lower)
        detections: List[DetectedEvent] = []

        for event_type, keywords in _KEYWORD_RULES.items():
            matched = self._match_keywords_excluding_negated(keywords, text_lower, clauses)
            if not matched:
                continue

            confidence = self._compute_confidence(matched, keywords)
            if confidence < self._min_confidence:
                continue

            detections.append(
                DetectedEvent(
                    event_type=event_type.value,
                    description=engine_input.vlm_description,
                    timestamp=engine_input.timestamp,
                    confidence=confidence,
                    matched_keywords=matched,
                    source_model=engine_input.source_model,
                )
            )

        if not detections:
            detections.append(
                DetectedEvent(
                    event_type=EventType.GENEL_GOZLEM.value,
                    description=engine_input.vlm_description,
                    timestamp=engine_input.timestamp,
                    confidence=0.0,
                    matched_keywords=[],
                    source_model=engine_input.source_model,
                )
            )

        detections.sort(key=lambda event: event.confidence, reverse=True)
        logger.debug(
            "EventEngine: t=%.2f -> %d olay tespit edildi (%s)",
            engine_input.timestamp,
            len(detections),
            ", ".join(d.event_type for d in detections),
        )
        return detections

    @staticmethod
    def _match_keywords_excluding_negated(
        keywords: List[str], text_lower: str, clauses: List[str]
    ) -> List[str]:
        matched: List[str] = []
        for keyword in keywords:
            if keyword not in text_lower:
                continue

            containing_clause = next((clause for clause in clauses if keyword in clause), None)
            if containing_clause is not None and _is_negated(containing_clause, keyword):
                continue

            matched.append(keyword)
        return matched

    @staticmethod
    def _compute_confidence(matched: List[str], keywords: List[str]) -> float:
        extra_matches = len(matched) - 1
        confidence = _BASE_CONFIDENCE + extra_matches * _CONFIDENCE_STEP_PER_EXTRA_MATCH
        return round(min(1.0, confidence), 2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo_input = EventEngineInput(
        vlm_description=(
            "Sahada bir personel korumasiz alanda, baretsiz calisiyor. Yakin "
            "cevrede forklift bir yaya gecidine yaklasiyor."
        ),
        timestamp=12.4,
        source_model="demo-vlm",
        frame_count=1,
    )
    demo_events = EventEngine().detect(demo_input)
    for demo_event in demo_events:
        print(
            f"[{demo_event.event_type}] conf={demo_event.confidence} "
            f"keywords={demo_event.matched_keywords}"
        )
