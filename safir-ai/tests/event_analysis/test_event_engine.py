"""T008 (src/event_analysis/event_engine.py) icin GPU/ag bagimliligi gerektirmeyen birim testleri.

Kural tabanli `EventEngine.detect()` metodunun deterministik davranisini ve
`EventType`/`EVENT_TYPE_REGULATION_MAP`in `DEFAULT_ISG_REGULATIONS` (8 madde)
ile hizasini dogrular; hicbir dis bagimlilik (VLM/LLM/veritabani) gerektirmez.
"""

from __future__ import annotations

import pytest

from src.event_analysis.event_engine import EventEngine
from src.event_analysis.schemas import (
    EVENT_TYPE_REGULATION_MAP,
    DetectedEvent,
    EventEngineInput,
    EventType,
)


def _input(description: str, timestamp: float = 10.0) -> EventEngineInput:
    return EventEngineInput(
        vlm_description=description,
        timestamp=timestamp,
        source_model="test-vlm",
        frame_count=1,
    )


def test_from_vlm_response_maps_fields_correctly() -> None:
    class _FakeVLMResponse:
        description = "Sahada bir personel korumasiz alanda hareket etti."
        model_name = "qwen2.5-vl"
        frame_count = 3

    engine_input = EventEngineInput.from_vlm_response(_FakeVLMResponse(), timestamp=42.5)

    assert engine_input.vlm_description == _FakeVLMResponse.description
    assert engine_input.timestamp == 42.5
    assert engine_input.source_model == "qwen2.5-vl"
    assert engine_input.frame_count == 3


def test_detects_single_category_with_one_keyword() -> None:
    events = EventEngine().detect(_input("Sahada duman gorunuyor."))

    assert len(events) == 1
    assert events[0].event_type == EventType.YANGIN_DUMAN.value
    assert events[0].matched_keywords == ["duman"]
    assert events[0].confidence == 0.5


def test_confidence_increases_with_additional_keyword_matches() -> None:
    events = EventEngine().detect(
        _input("Personel baretsiz calisiyor, korumasiz alanda bulunuyor.")
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.KKD_IHLALI.value
    assert set(events[0].matched_keywords) == {"baretsiz", "korumasiz alan"}
    assert events[0].confidence == 0.6


def test_detects_multiple_categories_and_sorts_by_confidence_descending() -> None:
    events = EventEngine().detect(
        _input(
            "Forklift yaya gecidine yaklasiyor; ayrica sahada duman gorunuyor."
        )
    )

    event_types = [e.event_type for e in events]
    assert EventType.ARAC_YAYA_YAKINLIGI.value in event_types
    assert EventType.YANGIN_DUMAN.value in event_types
    assert events == sorted(events, key=lambda e: e.confidence, reverse=True)


def test_falls_back_to_genel_gozlem_when_no_keyword_matches() -> None:
    events = EventEngine().detect(_input("Sahada normal calisma gozlemlendi."))

    assert len(events) == 1
    assert events[0].event_type == EventType.GENEL_GOZLEM.value
    assert events[0].confidence == 0.0
    assert events[0].matched_keywords == []


def test_all_detections_preserve_timestamp_and_source_model() -> None:
    events = EventEngine().detect(_input("Forklift yaya gecidine yaklasiyor.", timestamp=99.9))

    assert all(isinstance(e, DetectedEvent) for e in events)
    assert all(e.timestamp == 99.9 for e in events)
    assert all(e.source_model == "test-vlm" for e in events)


def test_min_confidence_filters_low_confidence_detections() -> None:
    engine = EventEngine(min_confidence=0.55)
    events = engine.detect(_input("Sahada duman gorunuyor."))

    assert len(events) == 1
    assert events[0].event_type == EventType.GENEL_GOZLEM.value


@pytest.mark.parametrize(
    ("description", "expected_type"),
    [
        ("Personel sicak calisma izni olmadan kaynak islemi yapiyor.", EventType.SICAK_CALISMA_IHLALI),
        ("Ekip gaz olcumu yapilmadan kapali alan icine giriyor.", EventType.DAR_ALAN_IHLALI),
        ("Teknisyen elektrik pano uzerinde enerji kesme yapmadan calisiyor.", EventType.ENERJI_KESME_IHLALI),
        ("Vinc, sinyalman olmadan agir yuk kaldirma yapiyor.", EventType.AGIR_YUK_RISKI),
    ],
)
def test_detects_new_regulation_aligned_categories(description: str, expected_type: EventType) -> None:
    events = EventEngine().detect(_input(description))

    assert events[0].event_type == expected_type.value
    assert events[0].confidence >= 0.5


def test_event_type_regulation_map_covers_all_eight_default_regulations() -> None:
    mapped_to_regulation = {
        event_type: label for event_type, label in EVENT_TYPE_REGULATION_MAP.items() if label is not None
    }

    assert len(mapped_to_regulation) == 8
    assert set(mapped_to_regulation.keys()) == {
        EventType.DUSME_RISKI,
        EventType.KKD_IHLALI,
        EventType.ARAC_YAYA_YAKINLIGI,
        EventType.SICAK_CALISMA_IHLALI,
        EventType.YANGIN_DUMAN,
        EventType.DAR_ALAN_IHLALI,
        EventType.ENERJI_KESME_IHLALI,
        EventType.AGIR_YUK_RISKI,
    }


def test_event_type_regulation_map_has_no_regulation_for_operational_only_categories() -> None:
    assert EVENT_TYPE_REGULATION_MAP[EventType.YETKISIZ_ERISIM] is None
    assert EVENT_TYPE_REGULATION_MAP[EventType.GENEL_GOZLEM] is None


def test_event_type_regulation_map_covers_every_enum_member() -> None:
    assert set(EVENT_TYPE_REGULATION_MAP.keys()) == set(EventType)
