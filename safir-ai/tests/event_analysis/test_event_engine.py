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


# --- T014: Olumsuzlama tespiti ---------------------------------------------


def test_negated_keyword_at_clause_end_falls_back_to_genel_gozlem() -> None:
    """Turkce SOV sozdiziminde olumsuzlama yuklemi (`tespit edilmedi`) klanuzun SONUNDA olabilir."""
    events = EventEngine().detect(_input("Sahada duman veya yangin belirtisi tespit edilmedi."))

    assert len(events) == 1
    assert events[0].event_type == EventType.GENEL_GOZLEM.value
    assert events[0].matched_keywords == []


def test_negation_only_suppresses_the_negated_clause_not_the_whole_description() -> None:
    """Bir klanuzdaki olumsuzlama, farkli bir klanuzdaki gercek pozitif tespiti bastirmamali."""
    events = EventEngine().detect(
        _input(
            "Yakin cevrede forklift trafiginin oldugu gozlemleniyor. "
            "Herhangi bir duman veya yangin belirtisi tespit edilmedi."
        )
    )

    assert len(events) == 1
    assert events[0].event_type == EventType.ARAC_YAYA_YAKINLIGI.value
    assert events[0].matched_keywords == ["forklift"]


def test_non_negated_keyword_still_detected() -> None:
    """Olumsuzlama kontrolu, olumsuzlanmamis gercek pozitifleri bastirmamali (regresyon guvencesi)."""
    events = EventEngine().detect(_input("Sahada duman gorunuyor."))

    assert len(events) == 1
    assert events[0].event_type == EventType.YANGIN_DUMAN.value
    assert events[0].matched_keywords == ["duman"]


@pytest.mark.parametrize(
    "description",
    [
        "Sahada duman yok.",
        "Yanginla ilgili bir belirti gorulmedi.",
        "Duman tespit edilmedi.",
        "Alev gozlemlenmedi.",
        "Yangin riski bulunmuyor.",
    ],
)
def test_various_negation_cues_suppress_the_match(description: str) -> None:
    events = EventEngine().detect(_input(description))

    assert len(events) == 1
    assert events[0].event_type == EventType.GENEL_GOZLEM.value


def test_negation_word_window_does_not_reach_across_clause_boundary() -> None:
    """Bir klanuzdaki olumsuzlama ifadesi, NOKTAYLA ayrilmis bir SONRAKI klanuzdaki eslesmeyi etkilememeli."""
    events = EventEngine().detect(
        _input("Baret eksikligi gorulmedi. Forklift yaya gecidine yaklasiyor.")
    )

    event_types = {e.event_type for e in events}
    assert EventType.ARAC_YAYA_YAKINLIGI.value in event_types
    arac_yaya_event = next(e for e in events if e.event_type == EventType.ARAC_YAYA_YAKINLIGI.value)
    assert "forklift" in arac_yaya_event.matched_keywords


def test_multiple_categories_one_negated_one_not() -> None:
    """Ayni metinde bir kategori olumsuzlanirken digeri gercek pozitif olarak kalabilmeli."""
    events = EventEngine().detect(
        _input("Personel baretsiz calisiyor. Forklift yaya gecidine yaklastigi gorulmedi.")
    )

    event_types = {e.event_type for e in events}
    assert EventType.KKD_IHLALI.value in event_types
    assert EventType.ARAC_YAYA_YAKINLIGI.value not in event_types


# --- T008b: model-tabanli (structured_events) tespit yolu + anahtar-kelime fallback ---


def _input_structured(structured, description="belirgin tehlike yok", timestamp=15.0) -> EventEngineInput:
    return EventEngineInput(
        vlm_description=description,
        timestamp=timestamp,
        source_model="test-vlm",
        frame_count=1,
        structured_events=structured,
    )


def test_structured_events_take_precedence_over_keywords() -> None:
    """VLM tipli olay urettiyse, olay TIPI icin anahtar-kelime taramasi yerine ONLAR kullanilir."""
    # Structured olay VAR (tip anahtar-kelime taramasiyla belirlenmiyor); ama
    # bu olayin KENDI aciklamasindan (evidence) canonical keyword'ler yine de
    # cikarilir (bkz. T016 - matched_keywords artik structured path'te de doluyor).
    engine = EventEngine()
    events = engine.detect(
        _input_structured(
            [{"type": "arac_yaya_yakinligi", "timestamp": 15, "confidence": 0.9, "evidence": "forklift yaklasti"}]
        )
    )
    assert len(events) == 1
    assert events[0].event_type == EventType.ARAC_YAYA_YAKINLIGI.value
    assert events[0].confidence == 0.9
    assert events[0].timestamp == 15.0
    assert events[0].matched_keywords == ["forklift"]


def test_structured_events_keyword_extraction_is_scoped_to_own_event_type() -> None:
    """Keyword cikarimi, olayin KENDI `event_type`inin canonical listesiyle sinirlidir (baska tiplerinkiyle degil)."""
    engine = EventEngine()
    events = engine.detect(
        _input_structured(
            [{"type": "kkd_ihlali", "timestamp": 15, "confidence": 0.9, "evidence": "forklift yaklasti"}]
        )
    )
    assert len(events) == 1
    # "forklift" arac_yaya_yakinligi kategorisinin kelimesidir, kkd_ihlali'nin degil.
    assert events[0].matched_keywords == []


def test_structured_invalid_type_falls_back_to_keywords() -> None:
    """Structured olaylarin tipi enum'da yoksa, anahtar-kelime fallback devreye girer."""
    engine = EventEngine()
    events = engine.detect(
        _input_structured(
            [{"type": "GECERSIZ_TIP", "confidence": 0.9}],
            description="forklift yaya gecidine yaklasti",
        )
    )
    types = {e.event_type for e in events}
    assert EventType.ARAC_YAYA_YAKINLIGI.value in types
    # Fallback yolu anahtar kelimeyle esler.
    assert any(e.matched_keywords for e in events)


def test_structured_confidence_clamped_and_timestamp_defaults() -> None:
    """Guven skoru 0-1'e kirpilir; timestamp yoksa cagrinin zaman damgasina duser."""
    engine = EventEngine()
    events = engine.detect(
        _input_structured(
            [{"type": "kkd_ihlali", "confidence": 5.0}],  # asiri deger + timestamp yok
            timestamp=42.0,
        )
    )
    assert len(events) == 1
    assert events[0].confidence == 1.0
    assert events[0].timestamp == 42.0


# --- T018: VLM-uretimi serbest bicimli keywords (EventType = kategori, keywords = serbest kanit) ---


def test_vlm_arbitrary_keywords_are_used_verbatim_not_restricted_to_fixed_taxonomy() -> None:
    """VLM'in `EVENTS_JSON.keywords`inde urettigi terimler, `_KEYWORD_RULES`
    sabit taksonomisiyle SINIRLANMADAN aynen kullanilmali - taksonomide
    OLMAYAN terimler bile (orn. 'yogun siyah duman', 'alevlenme') korunmali."""
    engine = EventEngine()
    events = engine.detect(
        _input_structured(
            [
                {
                    "event_id": "00:06yangin_duman",
                    "type": "yangin_duman",
                    "keywords": ["duman", "yogun siyah duman", "alev", "alevlenme"],
                    "confidence": 0.91,
                }
            ]
        )
    )

    assert len(events) == 1
    assert events[0].matched_keywords == ["duman", "yogun siyah duman", "alev", "alevlenme"]
    # 'yogun siyah duman' ve 'alevlenme' _KEYWORD_RULES'ta YOKTUR - yine de korunmali.
    assert "yogun siyah duman" in events[0].matched_keywords
    assert "alevlenme" in events[0].matched_keywords


def test_more_than_eight_keywords_are_preserved_no_artificial_cap() -> None:
    many_keywords = [f"kanit-terimi-{i}" for i in range(12)]
    assert len(many_keywords) > 8

    engine = EventEngine()
    events = engine.detect(
        _input_structured([{"type": "yangin_duman", "keywords": many_keywords, "confidence": 0.8}])
    )

    assert len(events) == 1
    assert events[0].matched_keywords == many_keywords
    assert len(events[0].matched_keywords) == 12


def test_same_event_type_can_have_different_keyword_sets() -> None:
    """Ayni `type`e sahip iki farkli structured olay, farkli `keywords` listeleri tasiyabilmeli."""
    engine = EventEngine()

    first = engine.detect(
        _input_structured(
            [{"type": "yangin_duman", "keywords": ["duman", "yogun duman"], "confidence": 0.9}], timestamp=6.0
        )
    )
    second = engine.detect(
        _input_structured(
            [{"type": "yangin_duman", "keywords": ["siyah duman", "yanma belirtisi"], "confidence": 0.9}],
            timestamp=40.0,
        )
    )

    assert first[0].matched_keywords == ["duman", "yogun duman"]
    assert second[0].matched_keywords == ["siyah duman", "yanma belirtisi"]
    assert first[0].matched_keywords != second[0].matched_keywords


def test_keywords_field_strips_whitespace_drops_empty_and_non_string_and_dedupes() -> None:
    engine = EventEngine()
    events = engine.detect(
        _input_structured(
            [
                {
                    "type": "kkd_ihlali",
                    "keywords": ["  baret yok  ", "", "  ", "baret yok", 42, None, "yelek yok"],
                    "confidence": 0.7,
                }
            ]
        )
    )

    assert events[0].matched_keywords == ["baret yok", "yelek yok"]


def test_missing_keywords_field_falls_back_to_fixed_taxonomy_extraction() -> None:
    """VLM `keywords` uretmediyse (eski/degrade cikti), guvenli taksonomi-fallback devreye girmeli - CRASH olmamali."""
    engine = EventEngine()
    events = engine.detect(
        _input_structured(
            [{"type": "arac_yaya_yakinligi", "description": "forklift yaya gecidine yaklasti", "confidence": 0.8}]
        )
    )

    assert len(events) == 1
    assert events[0].matched_keywords == ["forklift", "yaya gecidi"]


def test_non_list_keywords_value_does_not_crash_falls_back() -> None:
    engine = EventEngine()
    events = engine.detect(
        _input_structured(
            [
                {
                    "type": "arac_yaya_yakinligi",
                    "keywords": "forklift, yaya",  # liste degil, hatali bicim
                    "description": "forklift yaya gecidine yaklasti",
                    "confidence": 0.8,
                }
            ]
        )
    )

    assert len(events) == 1
    assert events[0].matched_keywords == ["forklift", "yaya gecidi"]


def test_structured_event_keywords_do_not_affect_confidence_or_type() -> None:
    """Keywords bir risk/karar sinyali DEGILDIR: farkli keyword sayisi/icerigi,
    tespitin confidence/event_type degerlerini ETKILEMEMELI (bunlar VLM'in
    kendi verdigi `confidence`/`type` alanlarindan gelir)."""
    engine = EventEngine()
    few = engine.detect(_input_structured([{"type": "yangin_duman", "keywords": ["duman"], "confidence": 0.5}]))
    many = engine.detect(
        _input_structured(
            [{"type": "yangin_duman", "keywords": [f"terim-{i}" for i in range(15)], "confidence": 0.5}]
        )
    )

    assert few[0].confidence == many[0].confidence == 0.5
    assert few[0].event_type == many[0].event_type == "yangin_duman"


def test_deliberately_non_taxonomy_keywords_survive_exactly_unchanged() -> None:
    """T019 regresyonu: kullanicinin verdigi TAM ornek liste - hicbiri
    `_KEYWORD_RULES` taksonomisinde YOKTUR (yalnizca 'duman' tek basina
    taksonomide var, ama burada HEP baska ifadelerin PARCASI). Sonuc,
    taksonomiye ("duman", "alev" gibi tek kelimelere) INDIRGENMEMELI."""
    deliberately_non_taxonomy_keywords = [
        "yoğun siyah duman",
        "alevlenme",
        "dumanın tavana doğru yükselmesi",
        "yanma belirtisi",
    ]
    engine = EventEngine()
    events = engine.detect(
        _input_structured(
            [{"type": "yangin_duman", "keywords": deliberately_non_taxonomy_keywords, "confidence": 0.91}]
        )
    )

    assert len(events) == 1
    assert events[0].matched_keywords == deliberately_non_taxonomy_keywords
    assert events[0].matched_keywords != ["duman", "alev", "yangin", "yanma"]
