"""MockVLM ve SafirAgent ile Uçtan Uca Senaryo Doğrulama Testleri."""

from __future__ import annotations

import pytest
from src.agent.langgraph_agent import AgentDecision, SafirAgent
from src.utils.config_loader import load_config
from tests.mock_scenarios import MOCK_SCENARIOS


@pytest.fixture
def safir_agent():
    """Birim testler için GPU bağımsız mock SafirAgent üretir."""
    config = load_config()
    return SafirAgent(
        llm_config=config.llm,
        agent_config=config.agent,
        event_store=None,
        rag_service=None,
        use_mock_llm=True,
    )


def test_scenario_high_risk_critical(safir_agent):
    """Yüksek riskli senaryonun (Kaza/Hareketsiz Kişi) doğru skorlandığını doğrular."""
    sc = MOCK_SCENARIOS["high_risk_critical"]
    context_block = f"## Güncel Gözlem\n{sc['observation']}\n\n## İstem\n{sc['user_prompt']}"

    decision: AgentDecision = safir_agent.run(context_block)

    assert decision.risk_score is not None
    assert decision.risk_score >= sc["expected_min_score"]
    assert decision.risk_level in ("yuksek", "kritik")
    assert decision.summary
    assert decision.actions
    assert decision.recommended_action == decision.actions[0]


def test_scenario_low_risk_routine(safir_agent):
    """Düşük riskli rutin çalışma senaryosunu doğrular."""
    sc = MOCK_SCENARIOS["low_risk_routine"]
    context_block = f"## Güncel Gözlem\n{sc['observation']}\n\n## İstem\n{sc['user_prompt']}"

    decision: AgentDecision = safir_agent.run(context_block)

    assert decision.risk_score is not None
    assert decision.risk_score <= sc["expected_max_score"]
    assert decision.risk_level == "dusuk"
    assert decision.guardrail_triggered is False


def test_scenario_negation_guardrail(safir_agent):
    """Olumsuzlanmış Türkçe ifadeler içeren (görülmedi/tespit edilmedi) senaryoyu doğrular."""
    sc = MOCK_SCENARIOS["negation_safe"]
    context_block = f"## Güncel Gözlem\n{sc['observation']}\n\n## İstem\n{sc['user_prompt']}"

    decision: AgentDecision = safir_agent.run(context_block)

    assert decision.risk_score == 0
    assert decision.risk_level == "dusuk"


def test_scenario_medium_risk_uncertain(safir_agent):
    """Orta/Belirsiz riskli senaryoda risk skorunun tetiklenme eşiğini geçtiğini doğrular."""
    sc = MOCK_SCENARIOS["medium_risk_uncertain"]
    context_block = f"## Güncel Gözlem\n{sc['observation']}\n\n## İstem\n{sc['user_prompt']}"

    decision: AgentDecision = safir_agent.run(context_block)

    assert decision.risk_score is not None
    assert decision.risk_score >= sc["expected_min_score"]
    assert decision.risk_level in ("orta", "yuksek", "kritik")


def test_scenario_security_unauthorized_access(safir_agent):
    """Güvenlik senaryosu: İzinsiz alan girişi ve tel örgü sızmasını doğrular."""
    sc = MOCK_SCENARIOS["security_unauthorized_access"]
    context_block = f"## Güncel Gözlem\n{sc['observation']}\n\n## İstem\n{sc['user_prompt']}"

    decision: AgentDecision = safir_agent.run(context_block)

    assert decision.risk_score is not None
    assert decision.risk_score >= sc["expected_min_score"]
    assert decision.event_category == "security"
    assert decision.risk_level in ("yuksek", "kritik")


def test_scenario_security_suspicious_object(safir_agent):
    """Güvenlik senaryosu: Terk edilmiş şüpheli nesne / çanta tespitini doğrular."""
    sc = MOCK_SCENARIOS["security_suspicious_object"]
    context_block = f"## Güncel Gözlem\n{sc['observation']}\n\n## İstem\n{sc['user_prompt']}"

    decision: AgentDecision = safir_agent.run(context_block)

    assert decision.risk_score is not None
    assert decision.risk_score >= sc["expected_min_score"]
    assert decision.event_category == "security"


def test_scenario_security_negation_safe(safir_agent):
    """Güvenlik negasyon senaryosu: Olumsuzlanmış sızma ifadesini doğrular."""
    sc = MOCK_SCENARIOS["security_negation_safe"]
    context_block = f"## Güncel Gözlem\n{sc['observation']}\n\n## İstem\n{sc['user_prompt']}"

    decision: AgentDecision = safir_agent.run(context_block)

    assert decision.risk_score == 0
    assert decision.event_category == "security"
    assert decision.risk_level == "dusuk"


def test_scenario_ambiguous_dual_risk(safir_agent):
    """İkili/Belirsiz risk senaryosu: Hem Sağlık (Safety) hem Güvenlik (Security) unsurlarını doğrular."""
    sc = MOCK_SCENARIOS["ambiguous_dual_risk"]
    context_block = f"## Güncel Gözlem\n{sc['observation']}\n\n## İstem\n{sc['user_prompt']}"

    decision: AgentDecision = safir_agent.run(context_block)

    assert decision.risk_score is not None
    assert decision.risk_score >= sc["expected_min_score"]
    assert decision.event_category == "ambiguous"
    # Hem sağlık hem güvenlik aksiyonu içerdiğini doğrula
    actions_str = " ".join(decision.actions).lower()
    assert any(k in actions_str for k in ["sağlık", "saglik", "ilk yardım", "yardim"])
    assert any(k in actions_str for k in ["güvenlik", "guvenlik", "devriye", "erisim", "erişim"])
