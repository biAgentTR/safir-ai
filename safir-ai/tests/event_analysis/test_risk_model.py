"""`src/event_analysis/risk_model.py` icin izole birim + monotoniklik testleri.

RISK ENGINE V2: eski sabit-bucket (12/38/63/88) mantigi TAMAMEN KALDIRILDI;
bu dosya, yerine gelen `compute_risk_score` agirlikli-carpimsal modelinin
determinizmini, sinirliligini ([0,100]) ve HER SEKIZ feature'da monotonik
azalmayan davranisini dogrular.
"""

from __future__ import annotations

import pytest

from src.event_analysis.risk_model import compute_risk_score, score_to_risk_level
from src.event_analysis.schemas import RuleMatch, TemporalEvent


def _match(rule_id: str, severity: str, event_type: str = "arac_yaya_yakinligi", source_event_id: str = "evt_0") -> RuleMatch:
    return RuleMatch(
        rule_id=rule_id,
        rule_description=f"{rule_id} aciklamasi",
        event_type=event_type,
        severity=severity,
        source_event_id=source_event_id,
    )


def _temporal_event(
    event_id: str = "evt_0",
    confidence: float = 0.7,
    duration: float = 5.0,
    occurrence_count: int = 1,
) -> TemporalEvent:
    return TemporalEvent(
        event_id=event_id,
        event_name="test_event",
        event_type="arac_yaya_yakinligi",
        description="test",
        start_timestamp=0.0,
        end_timestamp=duration,
        duration=duration,
        confidence=confidence,
        occurrence_count=occurrence_count,
        matched_keywords=[],
        source_model="test-vlm",
        related_events=[],
    )


class _FakeRagSource:
    def __init__(self, relevance_score: float, source_verified: bool = True):
        self.relevance_score = relevance_score
        self.source_verified = source_verified


# ---------------------------------------------------------------------------
# Determinizm
# ---------------------------------------------------------------------------


def test_same_input_produces_same_score_every_time() -> None:
    match = _match("OK-07", "kritik")
    te = _temporal_event()
    results = [
        compute_risk_score("kritik", [match], ["OK-07"], ["evt_0"], temporal_events=[te]).final_score
        for _ in range(5)
    ]
    assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# Sinirlilik ([0,100]) + seviye esikleri
# ---------------------------------------------------------------------------


def test_score_is_always_within_zero_and_hundred() -> None:
    for severity in ["dusuk", "orta", "yuksek", "kritik"]:
        match = _match("R", severity)
        te = _temporal_event(confidence=1.0, duration=1000.0, occurrence_count=100)
        rag = [_FakeRagSource(1.0)]
        breakdown = compute_risk_score(severity, [match, match], ["R", "R"], ["evt_0"], temporal_events=[te], semantic_rag_sources=rag)
        assert 0.0 <= breakdown.final_score <= 100.0


def test_maximum_evidence_at_kritik_severity_is_the_highest_achievable_score() -> None:
    """Formulun tasarim ozelligi: TUM OLCULEBILEN feature'lar maksimumdayken kritik siddet, ULASILABILIR tavana (bkz. modul dokustringi) cikmali.

    NOT: `exposure` bu surumde HER ZAMAN bilinmiyor (bkz. `test_exposure_is_always_unknown...`)
    - `exposure_factor` HER ZAMAN 1.0 (notr) kaldigi icin GERCEK ulasilabilir tavan, TEORIK
    `_RAW_SCORE_MAX`in (exposure_factor=1.20 varsayimiyla hesaplanmis) ALTINDADIR; bu, "eksik
    kanit icin uydurma deger uretme" ilkesinin DOGRUDAN, beklenen bir sonucudur.
    """
    combo_match = _match("COMBO-01", "kritik", event_type="kkd_ihlali+arac_yaya_yakinligi")
    te = _temporal_event(confidence=1.0, duration=1000.0, occurrence_count=100)
    rag = [_FakeRagSource(1.0)]

    breakdown = compute_risk_score(
        "kritik", [combo_match], ["COMBO-01"], ["evt_0"], temporal_events=[te], semantic_rag_sources=rag
    )

    assert breakdown.risk_level == "kritik"
    assert breakdown.exposure_factor == 1.0  # exposure notr - bkz. yukaridaki NOT
    # Ulasilabilir tavan: butun feature'lar 1.0 iken exposure_factor=1.0 sabitken hesaplanan deger.
    expected_max_given_unknown_exposure = 100.0 / 1.20  # _W_EXPOSURE=0.20 -> (1+0.20)=1.20 carpani EKSIK
    assert breakdown.final_score == pytest.approx(expected_max_given_unknown_exposure, abs=0.01)


def test_risk_level_matches_score_thresholds() -> None:
    assert score_to_risk_level(0.0) == "dusuk"
    assert score_to_risk_level(24.9) == "dusuk"
    assert score_to_risk_level(25.0) == "orta"
    assert score_to_risk_level(49.9) == "orta"
    assert score_to_risk_level(50.0) == "yuksek"
    assert score_to_risk_level(74.9) == "yuksek"
    assert score_to_risk_level(75.0) == "kritik"
    assert score_to_risk_level(100.0) == "kritik"


# ---------------------------------------------------------------------------
# Monotoniklik: her feature ARTARKEN skor ASLA DUSMEMELI (gorev tanimi 6. bolum)
# ---------------------------------------------------------------------------


def test_severity_monotonicity() -> None:
    scores = [
        compute_risk_score(sev, [_match("R", sev)], ["R"], ["evt_0"]).final_score
        for sev in ["dusuk", "orta", "yuksek", "kritik"]
    ]
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_likelihood_monotonicity() -> None:
    match = _match("R", "yuksek")
    scores = []
    for confidence in [0.0, 0.25, 0.5, 0.75, 1.0]:
        te = _temporal_event(confidence=confidence)
        breakdown = compute_risk_score("yuksek", [match], ["R"], ["evt_0"], temporal_events=[te])
        scores.append(breakdown.final_score)
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_duration_monotonicity() -> None:
    match = _match("R", "yuksek")
    scores = []
    for duration in [0.0, 5.0, 10.0, 20.0, 30.0, 60.0]:
        te = _temporal_event(duration=duration)
        breakdown = compute_risk_score("yuksek", [match], ["R"], ["evt_0"], temporal_events=[te])
        scores.append(breakdown.final_score)
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_recurrence_monotonicity() -> None:
    match = _match("R", "yuksek")
    scores = []
    for occurrence_count in [1, 2, 3, 4, 5, 10]:
        te = _temporal_event(occurrence_count=occurrence_count)
        breakdown = compute_risk_score("yuksek", [match], ["R"], ["evt_0"], temporal_events=[te])
        scores.append(breakdown.final_score)
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_protection_gap_monotonicity() -> None:
    """kkd_ihlali (KKD/koruma ihlali) VARLIGI, skoru ASLA DUSUREMEZ - yalnizca artirir/degistirmez."""
    without_gap = _match("R", "yuksek", event_type="arac_yaya_yakinligi")
    with_gap = _match("R", "yuksek", event_type="kkd_ihlali")

    score_without = compute_risk_score("yuksek", [without_gap], ["R"], ["evt_0"]).final_score
    score_with = compute_risk_score("yuksek", [with_gap], ["R"], ["evt_0"]).final_score

    assert score_with > score_without


def test_rule_support_monotonicity() -> None:
    single = compute_risk_score("kritik", [_match("R1", "kritik")], ["R1"], ["evt_0"]).final_score
    double = compute_risk_score(
        "kritik", [_match("R1", "kritik"), _match("R2", "kritik")], ["R1", "R2"], ["evt_0"]
    ).final_score
    combo = compute_risk_score(
        "kritik", [_match("COMBO-01", "kritik")], ["COMBO-01"], ["evt_0"]
    ).final_score

    assert single <= double <= combo
    assert single < combo


def test_regulatory_support_monotonicity() -> None:
    match = _match("R", "yuksek")
    scores = []
    for rag_score in [0.0, 0.25, 0.5, 0.75, 1.0]:
        breakdown = compute_risk_score(
            "yuksek", [match], ["R"], ["evt_0"], semantic_rag_sources=[_FakeRagSource(rag_score)]
        )
        scores.append(breakdown.final_score)
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_regulatory_support_cannot_independently_create_extreme_risk() -> None:
    """HEDEF 9: RAG relevance_score TEK BASINA (dusuk siddetli bir olayi) asiri yuksek risge SICRATAMAZ."""
    match = _match("R", "dusuk")
    no_rag = compute_risk_score("dusuk", [match], ["R"], ["evt_0"]).final_score
    max_rag = compute_risk_score(
        "dusuk", [match], ["R"], ["evt_0"], semantic_rag_sources=[_FakeRagSource(1.0)]
    ).final_score

    assert max_rag > no_rag  # katkisi VAR (monotonik)
    assert max_rag <= 25.0  # ama "dusuk" siddet tavanini (bkz. modul dokustringi) ASAMAZ
    assert score_to_risk_level(max_rag) == "dusuk"


def test_unverified_rag_source_does_not_contribute_to_regulatory_support() -> None:
    match = _match("R", "yuksek")
    verified = compute_risk_score(
        "yuksek", [match], ["R"], ["evt_0"], semantic_rag_sources=[_FakeRagSource(0.9, source_verified=True)]
    ).final_score
    unverified = compute_risk_score(
        "yuksek", [match], ["R"], ["evt_0"], semantic_rag_sources=[_FakeRagSource(0.9, source_verified=False)]
    ).final_score

    assert unverified < verified


# ---------------------------------------------------------------------------
# Eksik kanit -> notr katki, CRASH YOK, "unknown" ile "safe" KARISTIRILMAZ
# ---------------------------------------------------------------------------


def test_missing_temporal_events_does_not_crash_and_uses_neutral_defaults() -> None:
    breakdown = compute_risk_score("orta", [_match("R", "orta")], ["R"], ["evt_0"], temporal_events=None)

    assert breakdown.features.likelihood is None
    assert breakdown.features.duration is None
    assert breakdown.features.recurrence is None
    assert breakdown.feature_sources.likelihood == "unavailable_neutral"
    assert 0.0 <= breakdown.final_score <= 100.0


def test_missing_rag_sources_does_not_crash_and_uses_neutral_default() -> None:
    breakdown = compute_risk_score("orta", [_match("R", "orta")], ["R"], ["evt_0"], semantic_rag_sources=None)

    assert breakdown.features.regulatory_support is None
    assert breakdown.feature_sources.regulatory_support == "unavailable_neutral"
    assert 0.0 <= breakdown.final_score <= 100.0


def test_exposure_is_always_unknown_in_this_version_and_does_not_crash() -> None:
    """Gorev tanimi 4C/12. bolum: proximity olcumu YOK - FAKE deger UYDURULMAZ, notr (1.0 carpan) kullanilir."""
    breakdown = compute_risk_score("orta", [_match("R", "orta")], ["R"], ["evt_0"])

    assert breakdown.features.exposure is None
    assert breakdown.feature_sources.exposure == "unavailable_neutral"
    assert breakdown.exposure_factor == 1.0


def test_unknown_likelihood_is_not_treated_as_zero_or_full_confidence() -> None:
    """'Bilinmiyor' ile 'guvenli/sifir' VEYA 'tam guvenli' KARISTIRILMAZ - dengeli notr (L_FLOOR) kullanilir."""
    match = _match("R", "yuksek")
    unknown = compute_risk_score("yuksek", [match], ["R"], ["evt_0"], temporal_events=None).final_score
    zero_confidence = compute_risk_score(
        "yuksek", [match], ["R"], ["evt_0"], temporal_events=[_temporal_event(confidence=0.0)]
    ).final_score
    full_confidence = compute_risk_score(
        "yuksek", [match], ["R"], ["evt_0"], temporal_events=[_temporal_event(confidence=1.0)]
    ).final_score

    assert zero_confidence < unknown < full_confidence


# ---------------------------------------------------------------------------
# Feature contribution izlenebilirligi
# ---------------------------------------------------------------------------


def test_feature_contributions_are_traceable_and_reflect_actual_calculation() -> None:
    match = _match("R", "kritik")
    te = _temporal_event(confidence=0.8, duration=15.0, occurrence_count=3)
    breakdown = compute_risk_score("kritik", [match], ["R"], ["evt_0"], temporal_events=[te])

    contributions = breakdown.as_contributions_dict()
    assert contributions["base_risk"] == round(breakdown.base_risk, 4)
    assert contributions["boost_factor"] == round(breakdown.boost_factor, 4)
    # Nihai skor GERCEKTEN bu ara degerlerden hesaplanmis (uydurulmus DEGIL).
    assert breakdown.raw_score == breakdown.base_risk * breakdown.boost_factor


def test_explanation_mentions_scoring_method_and_key_features() -> None:
    match = _match("R", "kritik")
    breakdown = compute_risk_score("kritik", [match], ["R"], ["evt_0"])

    explanation = breakdown.explanation()

    assert "safir_evidence_weighted_v2" in explanation
    assert "severity=" in explanation
    assert f"{breakdown.final_score:.1f}" in explanation
