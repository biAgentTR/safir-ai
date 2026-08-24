"""T011 (src/event_analysis/event_builder.py) icin GPU/ag bagimliligi gerektirmeyen birim testleri.

`EventBuilder.build()`/`build_batch()`'in `TemporalEvent` + `RuleMatch`
birlestirme, aciklama metni uretme ve `EventStore.add_event(...)` ile
uyumlu kwargs uretme davranisini dogrular; hicbir dis bagimlilik gerektirmez.
"""

from __future__ import annotations

from src.event_analysis.event_builder import EventBuilder
from src.event_analysis.schemas import RuleMatch, TemporalEvent


def _temporal_event(
    event_id: str,
    event_type: str = "kkd_ihlali",
    description: str = "Personel baretsiz calisiyor.",
    start: float = 0.0,
    end: float = 0.0,
    duration: float = 0.0,
    confidence: float = 0.6,
    occurrence_count: int = 1,
    source_model: str = "test-vlm",
    keywords: list | None = None,
) -> TemporalEvent:
    return TemporalEvent(
        event_id=event_id,
        event_name=event_type,
        event_type=event_type,
        description=description,
        start_timestamp=start,
        end_timestamp=end,
        duration=duration,
        confidence=confidence,
        occurrence_count=occurrence_count,
        matched_keywords=keywords or [],
        source_model=source_model,
        related_events=[],
    )


def _rule_match(
    rule_id: str,
    source_event_id: str,
    related_event_ids: list | None = None,
    event_type: str = "kkd_ihlali",
    severity: str = "orta",
    rule_description: str = "ISG Yonetmeligi Madde 24",
) -> RuleMatch:
    return RuleMatch(
        rule_id=rule_id,
        rule_description=rule_description,
        event_type=event_type,
        severity=severity,
        source_event_id=source_event_id,
        related_event_ids=related_event_ids or [],
    )


def test_build_without_rule_matches_uses_raw_vlm_description() -> None:
    event = _temporal_event("evt_0", end=5.0)
    structured = EventBuilder().build(event, [])

    assert structured.description == "Personel baretsiz calisiyor."
    assert structured.related_rule_matches == []


def test_build_maps_core_fields_from_temporal_event() -> None:
    event = _temporal_event("evt_0", event_type="dusme_riski", end=12.5, confidence=0.75, occurrence_count=1)
    structured = EventBuilder().build(event, [])

    assert structured.timestamp == 12.5
    assert structured.event_type == "dusme_riski"
    assert structured.confidence == 0.75
    assert structured.temporal_event_id == "evt_0"
    assert structured.source_model == "test-vlm"
    assert structured.risk_score is None
    assert structured.risk_level is None


def test_build_with_single_rule_match_appends_rule_summary_to_description() -> None:
    event = _temporal_event("evt_0")
    match = _rule_match("ISG-M24", source_event_id="evt_0")

    structured = EventBuilder().build(event, [match])

    assert "Personel baretsiz calisiyor." in structured.description
    assert "Tetiklenen kural(lar): ISG-M24 (orta): ISG Yonetmeligi Madde 24" in structured.description
    assert structured.related_rule_matches == [match]


def test_build_with_multiple_rule_matches_joins_all_summaries() -> None:
    event = _temporal_event("evt_0")
    matches = [
        _rule_match("ISG-M24", source_event_id="evt_0"),
        _rule_match("COMBO-01", source_event_id="evt_0", severity="kritik", rule_description="Bilesik ihlal."),
    ]

    structured = EventBuilder().build(event, matches)

    assert "ISG-M24 (orta): ISG Yonetmeligi Madde 24" in structured.description
    assert "COMBO-01 (kritik): Bilesik ihlal." in structured.description
    assert structured.description.count("Tetiklenen kural(lar):") == 1


def test_build_mentions_repetition_when_occurrence_count_greater_than_one() -> None:
    event = _temporal_event("evt_0", start=0.0, end=9.0, duration=9.0, occurrence_count=3)
    structured = EventBuilder().build(event, [])

    assert "3 ardisik gozlemde" in structured.description
    assert "9.0s" in structured.description
    assert structured.occurrence_count == 3
    assert structured.duration == 9.0


def test_build_does_not_mention_repetition_for_single_occurrence() -> None:
    event = _temporal_event("evt_0", occurrence_count=1)
    structured = EventBuilder().build(event, [])

    assert "ardisik gozlemde" not in structured.description


def test_to_event_store_kwargs_matches_event_store_add_event_signature() -> None:
    event = _temporal_event("evt_0", end=42.0)
    structured = EventBuilder().build(event, [])

    kwargs = structured.to_event_store_kwargs()

    assert set(kwargs.keys()) == {
        "timestamp",
        "description",
        "risk_score",
        "risk_level",
        "source_model",
        "temporal_event_id",
        "event_name",
        "event_type",
        "confidence",
        "occurrence_count",
        "duration",
        "evidence_ids",
        "keywords",
    }
    assert kwargs["timestamp"] == 42.0
    assert kwargs["description"] == structured.description
    assert kwargs["risk_score"] is None
    assert kwargs["risk_level"] is None
    assert kwargs["evidence_ids"] == structured.evidence_ids
    assert kwargs["temporal_event_id"] == structured.temporal_event_id
    assert kwargs["source_model"] == "test-vlm"


def test_build_batch_assigns_matches_via_source_event_id() -> None:
    events = [_temporal_event("evt_0"), _temporal_event("evt_1", event_type="arac_yaya_yakinligi")]
    matches = [_rule_match("ISG-M24", source_event_id="evt_0")]

    structured_events = EventBuilder().build_batch(events, matches)

    by_id = {s.temporal_event_id: s for s in structured_events}
    assert len(by_id["evt_0"].related_rule_matches) == 1
    assert by_id["evt_1"].related_rule_matches == []


def test_build_batch_assigns_combination_match_to_both_source_and_related_events() -> None:
    events = [
        _temporal_event("evt_0", event_type="kkd_ihlali"),
        _temporal_event("evt_1", event_type="arac_yaya_yakinligi"),
    ]
    combo_match = _rule_match(
        "COMBO-01",
        source_event_id="evt_0",
        related_event_ids=["evt_1"],
        event_type="arac_yaya_yakinligi+kkd_ihlali",
        severity="kritik",
        rule_description="Bilesik ihlal.",
    )

    structured_events = EventBuilder().build_batch(events, [combo_match])

    by_id = {s.temporal_event_id: s for s in structured_events}
    assert by_id["evt_0"].related_rule_matches == [combo_match]
    assert by_id["evt_1"].related_rule_matches == [combo_match]
    assert "COMBO-01" in by_id["evt_0"].description
    assert "COMBO-01" in by_id["evt_1"].description


def test_build_batch_preserves_temporal_event_order() -> None:
    events = [_temporal_event("evt_0", end=0.0), _temporal_event("evt_1", end=10.0), _temporal_event("evt_2", end=20.0)]

    structured_events = EventBuilder().build_batch(events, [])

    assert [s.temporal_event_id for s in structured_events] == ["evt_0", "evt_1", "evt_2"]


def test_build_batch_with_empty_events_returns_empty_list() -> None:
    assert EventBuilder().build_batch([], []) == []


def test_build_batch_with_no_matching_rule_matches_still_builds_events() -> None:
    events = [_temporal_event("evt_0")]
    unrelated_match = _rule_match("ISG-M12", source_event_id="evt_999")

    structured_events = EventBuilder().build_batch(events, [unrelated_match])

    assert len(structured_events) == 1
    assert structured_events[0].related_rule_matches == []


# --- T016: deterministic risk (rule engine -> StructuredEvent.risk_score/risk_level) -----


def test_build_with_rule_match_sets_deterministic_risk_from_severity() -> None:
    """VLM/LLM degil, RuleEngine'in siddeti (+ bu olayin KENDI temporal kaniti) nihai risk_score/risk_level'i belirler (RISK ENGINE V2: formul-tabanli, bkz. `risk_model.py`).

    Skor UYDURULMADI - dogrudan `resolve_deterministic_risk_with_provenance`
    (EventBuilder'in cagirdigi AYNI fonksiyon) ile hesaplanip karsilastirilir.
    """
    from src.event_analysis.risk_resolver import resolve_deterministic_risk_with_provenance

    event = _temporal_event("evt_0")
    match = _rule_match("ISG-M24", source_event_id="evt_0", severity="orta")
    expected = resolve_deterministic_risk_with_provenance([match], temporal_events=[event])

    structured = EventBuilder().build(event, [match])

    assert structured.risk_level == expected.risk_level
    assert structured.risk_score == expected.risk_score


def test_build_with_combination_rule_uses_the_highest_severity() -> None:
    from src.event_analysis.risk_resolver import resolve_deterministic_risk_with_provenance

    event = _temporal_event("evt_0")
    matches = [
        _rule_match("ISG-M24", source_event_id="evt_0", severity="orta"),
        _rule_match("COMBO-01", source_event_id="evt_0", severity="kritik", rule_description="Bilesik ihlal."),
    ]
    expected = resolve_deterministic_risk_with_provenance(matches, temporal_events=[event])

    structured = EventBuilder().build(event, matches)

    assert structured.risk_level == expected.risk_level == "orta"  # RuleEngine siddeti 'kritik', formul-skoru 'orta' bandinda
    assert structured.risk_score == expected.risk_score
    # RuleEngine'in KENDI (en yuksek) siddeti hala dogrulanabilir - COMBO-01 kazandi.
    assert expected.rule_ids == ["COMBO-01"]
    assert expected.rule_severities == ["kritik"]


def test_build_without_any_rule_match_leaves_risk_none_not_fabricated() -> None:
    event = _temporal_event("evt_0", event_type="genel_gozlem")

    structured = EventBuilder().build(event, [])

    assert structured.risk_level is None
    assert structured.risk_score is None


# --- T018: StructuredEvent artik VLM'in serbest-bicimli keywords'lerini tasiyabiliyor ---


def test_build_passes_through_free_form_keywords_to_structured_event() -> None:
    event = _temporal_event("evt_0", event_type="yangin_duman", keywords=["duman", "yogun siyah duman", "alev"])

    structured = EventBuilder().build(event, [])

    assert structured.keywords == ["duman", "yogun siyah duman", "alev"]


def test_build_with_more_than_eight_keywords_preserves_all_of_them() -> None:
    many_keywords = [f"terim-{i}" for i in range(11)]
    event = _temporal_event("evt_0", keywords=many_keywords)

    structured = EventBuilder().build(event, [])

    assert structured.keywords == many_keywords
    assert len(structured.keywords) == 11


def test_build_with_no_keywords_defaults_to_empty_list() -> None:
    event = _temporal_event("evt_0")

    structured = EventBuilder().build(event, [])

    assert structured.keywords == []
