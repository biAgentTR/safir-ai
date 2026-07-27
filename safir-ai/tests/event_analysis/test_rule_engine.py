"""T010 (src/event_analysis/rule_engine.py) icin GPU/ag bagimliligi gerektirmeyen birim testleri.

`RuleEngine.evaluate()`'in tekli mevzuat eslestirmesini, kombinasyon
(bilesik ihlal) kurallarini ve opsiyonel `RegulationRetriever` entegrasyonunu
dogrular. Gercek `EmbeddingRAGService`/`RetrieverTool`'a bagimlilik yoktur;
`RegulationRetriever` Protocol'une uyan basit bir mock kullanilir.
"""

from __future__ import annotations

from src.event_analysis.rule_engine import RuleEngine
from src.event_analysis.schemas import TemporalEvent


class _MockRetriever:
    """`RegulationRetriever` Protocol'une uyan, gercek RAG'a bagli olmayan test cifti."""

    def __init__(self, response: str = "[MOCK] Ilgili mevzuat metni.") -> None:
        self.response = response
        self.calls: list[tuple[str, int]] = []

    def run(self, question: str, top_k: int = 3) -> str:
        self.calls.append((question, top_k))
        return self.response


class _FailingRetriever:
    """Her cagride istisna firlatan retriever; RuleEngine'in hataya dayanikliligini test eder."""

    def run(self, question: str, top_k: int = 3) -> str:
        raise RuntimeError("retriever servisine erisilemedi")


def _temporal_event(
    event_id: str,
    event_type: str,
    timestamp: float = 0.0,
    related_events: list | None = None,
) -> TemporalEvent:
    return TemporalEvent(
        event_id=event_id,
        event_type=event_type,
        description="test aciklamasi",
        start_timestamp=timestamp,
        end_timestamp=timestamp,
        duration=0.0,
        confidence=0.6,
        occurrence_count=1,
        matched_keywords=[],
        source_model="test-vlm",
        related_events=related_events or [],
    )


def test_empty_input_returns_empty_list() -> None:
    assert RuleEngine().evaluate([]) == []


def test_single_event_maps_to_correct_regulation_rule_id_and_severity() -> None:
    events = [_temporal_event("evt_0", "dusme_riski")]
    matches = RuleEngine().evaluate(events)

    assert len(matches) == 1
    assert matches[0].rule_id == "ISG-M12"
    assert matches[0].event_type == "dusme_riski"
    assert matches[0].severity == "yuksek"
    assert matches[0].source_event_id == "evt_0"
    assert matches[0].related_event_ids == []


def test_all_eight_regulation_aligned_event_types_produce_a_single_rule_match() -> None:
    event_types = [
        "dusme_riski",
        "kkd_ihlali",
        "arac_yaya_yakinligi",
        "sicak_calisma_ihlali",
        "yangin_duman",
        "dar_alan_ihlali",
        "enerji_kesme_ihlali",
        "agir_yuk_riski",
    ]
    events = [_temporal_event(f"evt_{i}", et, timestamp=float(i) * 100) for i, et in enumerate(event_types)]

    matches = RuleEngine().evaluate(events)

    assert len(matches) == 8
    assert {m.event_type for m in matches} == set(event_types)
    assert all(m.rule_id.startswith(("ISG-", "OK-", "YG-")) for m in matches)


def test_categories_without_regulation_produce_no_single_event_match() -> None:
    events = [
        _temporal_event("evt_0", "yetkisiz_erisim"),
        _temporal_event("evt_1", "genel_gozlem"),
    ]
    matches = RuleEngine().evaluate(events)

    assert matches == []


def test_unknown_event_type_is_skipped_gracefully() -> None:
    events = [_temporal_event("evt_0", "bilinmeyen_kategori")]
    matches = RuleEngine().evaluate(events)

    assert matches == []


def test_without_retriever_rule_description_is_short_regulation_label() -> None:
    events = [_temporal_event("evt_0", "kkd_ihlali")]
    matches = RuleEngine(retriever=None).evaluate(events)

    assert matches[0].rule_description == "ISG Yonetmeligi Madde 24"


def test_with_retriever_rule_description_uses_enriched_text() -> None:
    retriever = _MockRetriever(response="[MOCK] KKD zorunlulugu detayli metin.")
    events = [_temporal_event("evt_0", "kkd_ihlali")]

    matches = RuleEngine(retriever=retriever).evaluate(events)

    assert matches[0].rule_description == "[MOCK] KKD zorunlulugu detayli metin."
    assert retriever.calls == [("ISG Yonetmeligi Madde 24", 1)]


def test_retriever_failure_falls_back_to_short_label_without_raising() -> None:
    events = [_temporal_event("evt_0", "kkd_ihlali")]
    matches = RuleEngine(retriever=_FailingRetriever()).evaluate(events)

    assert matches[0].rule_description == "ISG Yonetmeligi Madde 24"


def test_combination_rule_triggers_when_related_events_cover_required_types() -> None:
    events = [
        _temporal_event("evt_0", "kkd_ihlali", timestamp=0.0, related_events=["evt_1"]),
        _temporal_event("evt_1", "arac_yaya_yakinligi", timestamp=8.0, related_events=["evt_0"]),
    ]
    matches = RuleEngine().evaluate(events)

    combo_matches = [m for m in matches if m.rule_id == "COMBO-01"]
    assert len(combo_matches) == 1
    combo = combo_matches[0]
    assert combo.severity == "kritik"
    assert combo.event_type == "arac_yaya_yakinligi+kkd_ihlali"
    assert combo.source_event_id == "evt_0"
    assert combo.related_event_ids == ["evt_1"]

    # tekli eslesmeler de ayrica uretilmeye devam eder
    assert any(m.rule_id == "ISG-M24" for m in matches)
    assert any(m.rule_id == "OK-07" for m in matches)


def test_combination_rule_does_not_trigger_without_relation() -> None:
    events = [
        _temporal_event("evt_0", "kkd_ihlali", timestamp=0.0, related_events=[]),
        _temporal_event("evt_1", "arac_yaya_yakinligi", timestamp=8.0, related_events=[]),
    ]
    matches = RuleEngine().evaluate(events)

    assert not any(m.rule_id == "COMBO-01" for m in matches)


def test_combination_rule_is_not_duplicated_when_both_anchors_would_trigger_it() -> None:
    events = [
        _temporal_event("evt_0", "dusme_riski", timestamp=0.0, related_events=["evt_1"]),
        _temporal_event("evt_1", "kkd_ihlali", timestamp=5.0, related_events=["evt_0"]),
    ]
    matches = RuleEngine().evaluate(events)

    combo_matches = [m for m in matches if m.rule_id == "COMBO-02"]
    assert len(combo_matches) == 1


def test_multiple_combination_rules_can_trigger_independently() -> None:
    events = [
        _temporal_event("evt_0", "yangin_duman", timestamp=0.0, related_events=["evt_1"]),
        _temporal_event("evt_1", "dar_alan_ihlali", timestamp=4.0, related_events=["evt_0"]),
        _temporal_event("evt_2", "enerji_kesme_ihlali", timestamp=100.0, related_events=["evt_3"]),
        _temporal_event("evt_3", "sicak_calisma_ihlali", timestamp=104.0, related_events=["evt_2"]),
    ]
    matches = RuleEngine().evaluate(events)

    combo_ids = {m.rule_id for m in matches if m.rule_id.startswith("COMBO")}
    assert combo_ids == {"COMBO-03", "COMBO-04"}


def test_evaluate_handles_unsorted_input_defensively() -> None:
    events = [
        _temporal_event("evt_1", "arac_yaya_yakinligi", timestamp=8.0, related_events=["evt_0"]),
        _temporal_event("evt_0", "kkd_ihlali", timestamp=0.0, related_events=["evt_1"]),
    ]
    matches = RuleEngine().evaluate(events)

    combo = next(m for m in matches if m.rule_id == "COMBO-01")
    assert combo.source_event_id == "evt_0"


def test_custom_rules_path_can_override_default_combination_rules(tmp_path) -> None:
    custom_rules = tmp_path / "custom_rules.yaml"
    custom_rules.write_text(
        "combination_rules:\n"
        "  - rule_id: \"CUSTOM-01\"\n"
        "    description: \"Ozel test kurali.\"\n"
        "    required_event_types:\n"
        "      - kkd_ihlali\n"
        "      - yangin_duman\n"
        "    severity: \"orta\"\n",
        encoding="utf-8",
    )

    events = [
        _temporal_event("evt_0", "kkd_ihlali", timestamp=0.0, related_events=["evt_1"]),
        _temporal_event("evt_1", "yangin_duman", timestamp=2.0, related_events=["evt_0"]),
    ]
    matches = RuleEngine(rules_path=custom_rules).evaluate(events)

    combo_matches = [m for m in matches if m.rule_id == "CUSTOM-01"]
    assert len(combo_matches) == 1
    assert combo_matches[0].severity == "orta"


def test_missing_rules_path_raises_file_not_found() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        RuleEngine(rules_path="/nonexistent/path/isg_rules.yaml")
