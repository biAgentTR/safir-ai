"""Dinamik takip sorusu onerileri (`src/assistant/suggestion_engine.py`) icin birim testleri.

Mentor eleştirisi ("inisiyatif alma ve dogru sorulari sorma"): oneri chip'leri
rapora OZGU olmali, sabit bir liste OLMAMALI. Bu testler, her sinyalin
(siniflandirilamayan olay, pending_review eskalasyonu, VLM metnindeki
belirsizlik dili, kisi-iceren olay kategorisi) dogru sekilde oneriye
donustugunu ve hicbir sinyal yoksa BOS liste (uydurma oneri YOK) dondugunu
dogrular.
"""

from __future__ import annotations

from src.assistant.suggestion_engine import build_dynamic_suggestions
from src.schemas.report import EventSummary, SafirReport


def _base_report(**overrides) -> SafirReport:
    defaults = dict(
        video_source="data/test.mp4",
        generated_at="2026-08-25T00:00:00Z",
        natural_language_summary="Sahada rutin faaliyet gozlenmistir.",
        summary="Rutin.",
        risk_score=10,
        risk_level="dusuk",
        recommended_action="izlemeye devam et",
    )
    defaults.update(overrides)
    return SafirReport(**defaults)


def test_no_signals_returns_empty_list() -> None:
    """Hicbir belirsizlik/eskalasyon/kisi-icerik sinyali yoksa oneri UYDURULMAZ - bos liste doner."""
    report = _base_report()
    assert build_dynamic_suggestions(report) == []


def test_unresolved_event_type_produces_reclassification_question() -> None:
    report = _base_report(
        events=[EventSummary(event_name="belirsiz_durum", event_type=None)],
    )
    suggestions = build_dynamic_suggestions(report)
    assert any("belirsiz_durum" in s for s in suggestions)


def test_llm_fallback_failure_marker_produces_reclassification_question() -> None:
    report = _base_report(
        events=[EventSummary(event_name="degerlendirme_yapilamadi", event_type=None)],
    )
    suggestions = build_dynamic_suggestions(report)
    assert any("degerlendirme_yapilamadi" in s for s in suggestions)


def test_pending_review_escalation_produces_uncertainty_question() -> None:
    report = _base_report(escalation_tier="pending_review")
    suggestions = build_dynamic_suggestions(report)
    assert any("insan incelemesine" in s for s in suggestions)


def test_unassessed_risk_status_produces_uncertainty_question() -> None:
    report = _base_report(risk_status="unknown", risk_score=None, risk_level="unknown")
    suggestions = build_dynamic_suggestions(report)
    assert any("insan incelemesine" in s for s in suggestions)


def test_hedge_language_in_summary_produces_clarification_question() -> None:
    report = _base_report(natural_language_summary="Personelin yüzü net değil, KKD durumu da net görünmüyor.")
    suggestions = build_dynamic_suggestions(report)
    assert any("net değil" in s for s in suggestions)


def test_person_involving_event_type_produces_mentor_example_question() -> None:
    """Mentörün KENDI ornegi ('İlgili personelin yüzü net mi?') - kisi-iceren bir
    olay kategorisi (ör. kkd_ihlali) tespit edildiyse GERCEKTEN onerilmeli."""
    report = _base_report(
        events=[EventSummary(event_name="baret_eksik", event_type="kkd_ihlali")],
    )
    suggestions = build_dynamic_suggestions(report)
    assert any("yüz" in s.lower() for s in suggestions)


def test_high_risk_level_produces_evidence_question() -> None:
    report = _base_report(risk_score=85, risk_level="kritik")
    suggestions = build_dynamic_suggestions(report)
    assert any("risk seviyesini" in s for s in suggestions)


def test_suggestions_are_capped_and_deduplicated() -> None:
    report = _base_report(
        risk_score=90,
        risk_level="kritik",
        risk_status="unknown",
        escalation_tier="pending_review",
        natural_language_summary="Yuz net degil, durum belirsiz.",
        events=[
            EventSummary(event_name="belirsiz_1", event_type=None),
            EventSummary(event_name="kkd_ihlali_olayi", event_type="kkd_ihlali"),
        ],
    )
    suggestions = build_dynamic_suggestions(report)
    assert 1 <= len(suggestions) <= 4
    assert len(suggestions) == len(set(suggestions))  # tekrar yok
