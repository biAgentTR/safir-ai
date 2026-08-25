"""T010 (src/event_analysis/rule_engine.py) icin GPU/ag bagimliligi gerektirmeyen birim testleri.

`RuleEngine.evaluate()`'in tekli mevzuat eslestirmesini, kombinasyon
(bilesik ihlal) kurallarini ve opsiyonel `RegulationRetriever` entegrasyonunu
dogrular. Gercek `EmbeddingRAGService`e bagimlilik yoktur; `RegulationRetriever`
Protocol'une (`.query(question, top_k) -> List[...]`, `EmbeddingRAGService.query`
ile AYNI imza) uyan basit bir mock kullanilir.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.event_analysis.rule_engine import RuleEngine
from src.event_analysis.schemas import TemporalEvent


@dataclass
class _FakeDoc:
    """`EmbeddingRAGService.query()`nin dondurdugu `RetrievedDocument`nin, bu testler icin yeterli minimal (duck-typed) izdusumu."""

    text: str


class _MockRetriever:
    """`RegulationRetriever` Protocol'une uyan, gercek RAG'a bagli olmayan test cifti.

    `EmbeddingRAGService.query()` ZATEN yalnizca deterministik relevance
    esiginden GECMIS ("accepted") adaylari dondurur (bkz. `rule_engine.py`
    modul dokustringindeki 2026-08-25 duzeltme notu); bu yuzden bu mock,
    "iliskisiz sonuc" senaryosunu ARTIK yanlis metin dondurerek DEGIL, bos
    liste (`results=[]`) dondurerek temsil eder - gercek sistemde deterministik
    gate zaten boyle davranir.
    """

    def __init__(self, results: list[str] | None = ("[MOCK] Ilgili mevzuat metni.",)) -> None:
        self.results = list(results) if results else []
        self.calls: list[tuple[str, int]] = []

    def query(self, question: str, top_k: int = 1) -> list[_FakeDoc]:
        self.calls.append((question, top_k))
        return [_FakeDoc(text=r) for r in self.results]


class _FailingRetriever:
    """Her cagride istisna firlatan retriever; RuleEngine'in hataya dayanikliligini test eder."""

    def query(self, question: str, top_k: int = 1) -> list[_FakeDoc]:
        raise RuntimeError("retriever servisine erisilemedi")


def _temporal_event(
    event_id: str,
    event_type: str,
    timestamp: float = 0.0,
    related_events: list | None = None,
) -> TemporalEvent:
    return TemporalEvent(
        event_id=event_id,
        event_name=event_type,
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
    """Retriever GERCEK bir sonuc donduruyorsa (EmbeddingRAGService.query()'nin
    zaten deterministik relevance esiginden GECIRDIGI, "accepted" bir aday),
    bu metin kisa etiketin ARDINDAN eklenerek KULLANILIR."""
    retriever = _MockRetriever(results=["KKD zorunlulugu detayli metin (gercek KB chunk'i)."])
    events = [_temporal_event("evt_0", "kkd_ihlali")]

    matches = RuleEngine(retriever=retriever).evaluate(events)

    assert matches[0].rule_description == "ISG Yonetmeligi Madde 24: KKD zorunlulugu detayli metin (gercek KB chunk'i)."
    assert retriever.calls == [("ISG Yonetmeligi Madde 24", 1)]


def test_retriever_failure_falls_back_to_short_label_without_raising() -> None:
    events = [_temporal_event("evt_0", "kkd_ihlali")]
    matches = RuleEngine(retriever=_FailingRetriever()).evaluate(events)

    assert matches[0].rule_description == "ISG Yonetmeligi Madde 24"


# --- T017: mevzuat "getirme" (RAG lookup) != "uygulanabilirlik" karari -----
# 2026-08-25: alakasiz sonuclara karsi koruma artik TEK bir yerde -
# `EmbeddingRAGService.query()`nin deterministik relevance/score_threshold
# gate'inde (bkz. `deterministic_reranker.py` testleri) - yasiyor; `query()`
# esigin ALTINDA kalan/alakasiz adaylari zaten BOS LISTE ile eler. Bu yuzden
# RuleEngine tarafindaki regresyon testi de ayni sozlesmeyi (bos liste =
# "ilgili sonuc yok") dogrulayacak sekilde guncellendi.


def test_retriever_returning_no_accepted_results_falls_back_to_short_label() -> None:
    """En onemli regresyon: retriever (gercek sistemde `EmbeddingRAGService.query()`)
    sorgulanan kategoriyle ilgili HICBIR "accepted" sonuc bulamadiginda (deterministik
    relevance gate onlari zaten eledigi icin) BOS LISTE doner - RuleEngine bu durumda
    kisa (guvenli) etikete geri donmeli, uydurma/ilgisiz bir mevzuat metni ASLA rapora
    SIZMAMALI."""
    empty_retriever = _MockRetriever(results=[])
    events = [_temporal_event("evt_0", "kkd_ihlali")]

    matches = RuleEngine(retriever=empty_retriever).evaluate(events)

    assert matches[0].rule_description == "ISG Yonetmeligi Madde 24"


def test_retriever_returning_empty_string_falls_back_to_short_label() -> None:
    events = [_temporal_event("evt_0", "kkd_ihlali")]
    matches = RuleEngine(retriever=_MockRetriever(results=[""])).evaluate(events)

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


# --- T018: keywords, risk/mevzuat eslesmesini ETKILEMEZ (RuleEngine yalnizca event_type okur) ---


def test_rule_engine_output_is_identical_regardless_of_matched_keywords() -> None:
    """RuleEngine, `TemporalEvent.matched_keywords`i HICBIR ZAMAN okumaz - yalnizca
    `event_type`e bakar. Farkli (hatta cok sayida) keyword, ayni event_type icin
    AYNI RuleMatch sonucunu uretmeli - keywords bir risk KARARI DEGILDIR."""
    few_keywords_event = _temporal_event("evt_0", "yangin_duman")
    few_keywords_event.matched_keywords = ["duman"]

    many_keywords_event = _temporal_event("evt_0", "yangin_duman")
    many_keywords_event.matched_keywords = [f"terim-{i}" for i in range(20)]

    matches_few = RuleEngine().evaluate([few_keywords_event])
    matches_many = RuleEngine().evaluate([many_keywords_event])

    assert len(matches_few) == len(matches_many) == 1
    assert matches_few[0].rule_id == matches_many[0].rule_id
    assert matches_few[0].severity == matches_many[0].severity
    assert matches_few[0].rule_description == matches_many[0].rule_description
