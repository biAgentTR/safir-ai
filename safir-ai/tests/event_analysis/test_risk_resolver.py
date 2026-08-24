"""T016 (src/event_analysis/risk_resolver.py) icin birim testleri.

RISK ENGINE V2 (2026-08-24): eski sabit-bucket (dusuk->12, orta->38, yuksek->63,
kritik->88) mantigi TAMAMEN KALDIRILDI - skor artik `risk_model.compute_risk_score`
agirlikli-carpimsal modelinden gelir (bkz. o modulun izole testleri:
`tests/event_analysis/test_risk_model.py`). Bu dosyadaki testler, ESKI ozel
sayisal degerler yerine, formulun `risk_resolver.py`den GERCEKTEN cagirilip
cagirilmadigini ve severity-secim/provenance mantiginin (rule_ids/contributing_
event_ids/vb.) hala DOGRU calistigini dogrular - beklenen skorlar formulden
BAGIMSIZ elle UYDURULMADI, `resolve_deterministic_risk`in KENDISI cagirilarak
elde edildi (bkz. her testin yorumu).

ONEMLI (yeni davranis): `risk_level` ARTIK RuleEngine'in HAM `severity`sinin
BIREBIR YANSIMASI DEGILDIR - nihai skordan (`score_to_risk_level`) turer.
Bir 'kritik' siddetli RuleMatch, TEK BASINA (baska hicbir kanit/temporal
context olmadan) 'kritik' RISK LEVEL'e ULASMAYABILIR - bu KASITLIDIR (bkz.
`risk_model.py` modul dokustringi): severity, formulun GIRDILERINDEN
biridir, cikardaki tek belirleyici DEGILDIR. RuleEngine'in KENDI siddet
sinifi hala `rule_severities` alaninda AYRICA korunur.
"""

from __future__ import annotations

from src.event_analysis.risk_model import score_to_risk_level
from src.event_analysis.risk_resolver import (
    resolve_deterministic_risk,
    resolve_deterministic_risk_with_provenance,
)
from src.event_analysis.schemas import RuleMatch


def _match(rule_id: str, severity: str, source_event_id: str = "evt_0") -> RuleMatch:
    return RuleMatch(
        rule_id=rule_id,
        rule_description=f"{rule_id} aciklamasi",
        event_type="kkd_ihlali",
        severity=severity,
        source_event_id=source_event_id,
    )


def test_no_matches_returns_none_none_risk_is_not_fabricated() -> None:
    assert resolve_deterministic_risk([]) == (None, None)


def test_single_match_score_matches_the_formula_and_level_matches_the_score() -> None:
    """Skor UYDURULMADI - formulun KENDI ciktisiyla karsilastirilir (tutarlilik/regresyon testi)."""
    provenance = resolve_deterministic_risk_with_provenance([_match("ISG-M24", "orta")])
    risk_level, risk_score = resolve_deterministic_risk([_match("ISG-M24", "orta")])

    assert risk_score == provenance.risk_score
    assert risk_level == provenance.risk_level == score_to_risk_level(provenance.final_score)
    # RuleEngine'in KENDI siddeti hala AYRICA (rule_severities'te) korunur - kaybolmadi.
    assert provenance.rule_severities == ["orta"]


def test_multiple_matches_take_the_highest_rule_severity_for_feature_extraction() -> None:
    """`rule_ids`/`rule_severities`, EN YUKSEK siddetli eslesme(ler)i (formulun `S` girdisi) yansitmali."""
    matches = [_match("ISG-M24", "orta"), _match("COMBO-01", "kritik"), _match("ISG-M12", "yuksek")]

    provenance = resolve_deterministic_risk_with_provenance(matches)

    assert provenance.rule_severities == ["kritik"]
    assert provenance.rule_ids == ["COMBO-01"]
    assert provenance.features["severity"] == 1.0  # kritik -> (3+1)/4 = 1.0


def test_severity_feature_is_strictly_monotonic_with_rule_engine_severity() -> None:
    """Her siddet seviyesi, formulun `severity` (S) feature'inda KENDINE OZGU, artan bir deger URETMELI."""
    severities_and_scores = [
        (sev, resolve_deterministic_risk_with_provenance([_match("R", sev)]))
        for sev in ["dusuk", "orta", "yuksek", "kritik"]
    ]
    feature_values = [p.features["severity"] for _, p in severities_and_scores]

    assert feature_values == sorted(feature_values)
    assert len(set(feature_values)) == 4  # dort seviye, dort FARKLI deger


def test_unknown_severity_values_are_ignored_and_do_not_crash() -> None:
    matches = [
        RuleMatch(
            rule_id="WEIRD",
            rule_description="bilinmeyen siddet",
            event_type="kkd_ihlali",
            severity="belirsiz",
            source_event_id="evt_0",
        )
    ]

    assert resolve_deterministic_risk(matches) == (None, None)


def test_known_and_unknown_severity_mixed_uses_only_known_ones() -> None:
    matches = [
        RuleMatch(
            rule_id="WEIRD",
            rule_description="bilinmeyen siddet",
            event_type="kkd_ihlali",
            severity="belirsiz",
            source_event_id="evt_0",
        ),
        _match("ISG-M24", "orta"),
    ]

    provenance = resolve_deterministic_risk_with_provenance(matches)

    assert provenance.rule_severities == ["orta"]
    assert provenance.rule_ids == ["ISG-M24"]


# --- resolve_deterministic_risk_with_provenance: risk explainability (90 vs 88 root cause fix) ---


def test_provenance_matches_resolve_deterministic_risk_exactly() -> None:
    """Provenance fonksiyonu, 2-tuple donen fonksiyonla AYNI risk_level/risk_score'u uretmelidir (tek kaynak/tek hesaplama yolu)."""
    matches = [_match("ISG-M24", "orta"), _match("COMBO-01", "kritik"), _match("ISG-M12", "yuksek")]

    risk_level, risk_score = resolve_deterministic_risk(matches)
    provenance = resolve_deterministic_risk_with_provenance(matches)

    assert provenance.risk_level == risk_level
    assert provenance.risk_score == risk_score
    assert provenance.rule_severities == ["kritik"]  # RuleEngine'in secili siddeti hala 'kritik'


def test_provenance_lists_only_the_winning_severity_rule_ids() -> None:
    matches = [_match("ISG-M24", "orta"), _match("COMBO-01", "kritik")]

    provenance = resolve_deterministic_risk_with_provenance(matches)

    assert provenance.rule_ids == ["COMBO-01"]
    assert provenance.rule_severities == ["kritik"]
    assert "ISG-M24" not in provenance.rule_ids


def test_provenance_includes_all_rule_ids_tied_at_the_winning_severity() -> None:
    matches = [_match("YG-03", "kritik"), _match("ISG-M45", "kritik")]

    provenance = resolve_deterministic_risk_with_provenance(matches)

    assert set(provenance.rule_ids) == {"YG-03", "ISG-M45"}


def test_provenance_tracks_contributing_event_ids_including_combo_related_events() -> None:
    combo_match = RuleMatch(
        rule_id="COMBO-01",
        rule_description="KKD + arac-yaya yakinligi",
        event_type="kkd_ihlali+arac_yaya_yakinligi",
        severity="kritik",
        source_event_id="evt_0",
        related_event_ids=["evt_1"],
    )

    provenance = resolve_deterministic_risk_with_provenance([combo_match])

    assert provenance.contributing_event_ids == ["evt_0", "evt_1"]


def test_provenance_no_match_returns_none_risk_and_empty_lists_not_fabricated() -> None:
    provenance = resolve_deterministic_risk_with_provenance([])

    assert provenance.risk_level is None
    assert provenance.risk_score is None
    assert provenance.rule_ids == []
    assert provenance.contributing_event_ids == []
    assert provenance.final_score is None
    assert provenance.features is None
    assert "belirlenemedi" in provenance.explanation()


def test_provenance_explanation_is_deterministic_and_cites_the_rule_id_and_its_own_severity() -> None:
    provenance = resolve_deterministic_risk_with_provenance([_match("YG-03", "kritik")])

    explanation = provenance.explanation()

    assert "YG-03" in explanation
    assert "kritik" in explanation  # RuleEngine'in KENDI siddeti (rule_severities) hala goruluyor


# ---------------------------------------------------------------------------
# RISK ENGINE V2: yeni provenance alanlari (scoring_method/features/vb.)
# ---------------------------------------------------------------------------


def test_provenance_scoring_method_is_the_v2_evidence_weighted_model() -> None:
    provenance = resolve_deterministic_risk_with_provenance([_match("ISG-M24", "orta")])
    assert provenance.scoring_method == "safir_evidence_weighted_v2"


def test_provenance_feature_values_match_the_values_actually_used_in_calculation() -> None:
    """HEDEF 12 (gorev tanimi 14. bolum): provenance'taki feature degerleri, GERCEKTEN hesaplamada kullanilanlarla AYNI olmali."""
    from src.event_analysis.risk_model import compute_risk_score

    match = _match("ISG-M24", "kritik")
    provenance = resolve_deterministic_risk_with_provenance([match])
    breakdown = compute_risk_score("kritik", [match], ["ISG-M24"], ["evt_0"])

    assert provenance.features == breakdown.features.as_dict()
    assert provenance.feature_contributions == breakdown.as_contributions_dict()
    assert provenance.risk_score == round(breakdown.final_score)


def test_llm_proposed_score_is_recorded_but_never_determines_final_score() -> None:
    """HEDEF 11: LLM'in taslak skoru (99) NE KADAR YUKSEK/DUSUK olursa olsun, final_score'u DEGISTIRMEZ."""
    match = _match("ISG-M24", "dusuk")

    provenance_high_llm = resolve_deterministic_risk_with_provenance([match], llm_proposed_score=99)
    provenance_low_llm = resolve_deterministic_risk_with_provenance([match], llm_proposed_score=1)
    provenance_no_llm = resolve_deterministic_risk_with_provenance([match])

    assert provenance_high_llm.llm_proposed_score == 99
    assert provenance_low_llm.llm_proposed_score == 1
    # final_score/risk_score UCUNDE de AYNI - llm_proposed_score hesaplamayi ETKILEMEDI.
    assert provenance_high_llm.final_score == provenance_low_llm.final_score == provenance_no_llm.final_score
    assert provenance_high_llm.risk_score == provenance_low_llm.risk_score == provenance_no_llm.risk_score


def test_temporal_events_enrich_the_calculation_when_provided() -> None:
    from src.event_analysis.schemas import TemporalEvent

    match = _match("ISG-M24", "yuksek")
    te = TemporalEvent(
        event_id="evt_0",
        event_name="test",
        event_type="kkd_ihlali",
        description="d",
        start_timestamp=0.0,
        end_timestamp=20.0,
        duration=20.0,
        confidence=0.9,
        occurrence_count=3,
        matched_keywords=[],
        source_model="m",
        related_events=[],
    )

    without_temporal = resolve_deterministic_risk_with_provenance([match])
    with_temporal = resolve_deterministic_risk_with_provenance([match], temporal_events=[te])

    assert with_temporal.feature_sources["likelihood"] == "measured"
    assert without_temporal.feature_sources["likelihood"] == "unavailable_neutral"
    assert with_temporal.final_score != without_temporal.final_score


# ---------------------------------------------------------------------------
# 2026-08-24 kalibrasyon duzeltmesi: safety_floor_applied provenance'a KADAR akiyor
# ---------------------------------------------------------------------------


def test_safety_floor_applied_flag_propagates_through_provenance() -> None:
    from src.event_analysis.schemas import TemporalEvent

    fire_match = RuleMatch(
        rule_id="YG-03",
        rule_description="Yangin Guvenligi Talimati",
        event_type="yangin_duman",
        severity="kritik",
        source_event_id="evt_0",
    )
    te = TemporalEvent(
        event_id="evt_0",
        event_name="yangin_duman",
        event_type="yangin_duman",
        description="duman, alev, buyuyen, kontrolsuz",
        start_timestamp=0.0,
        end_timestamp=45.0,
        duration=45.0,
        confidence=0.9,
        occurrence_count=3,
        matched_keywords=["duman", "alev", "buyuyen", "kontrolsuz"],
        source_model="test-vlm",
        related_events=[],
    )

    provenance = resolve_deterministic_risk_with_provenance([fire_match], temporal_events=[te])

    assert provenance.safety_floor_applied is True
    assert provenance.risk_level == "kritik"
    assert provenance.risk_score >= 80
    # LLM'in taslak skoru degistirilse bile taban/final_score AYNI kalir.
    provenance_with_llm = resolve_deterministic_risk_with_provenance(
        [fire_match], temporal_events=[te], llm_proposed_score=10
    )
    assert provenance_with_llm.risk_score == provenance.risk_score
    assert provenance_with_llm.safety_floor_applied is True
