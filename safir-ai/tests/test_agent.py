"""Modul 4 (src/agent/) icin GPU gerektirmeyen birim testleri.

`SafirAgent` her zaman `use_mock_llm=True` ile kurulur; boylece gercek bir
vLLM servisine hic baglanilmadan LangGraph durum makinesinin tamami
(reasoning -> decision) uctan uca calistirilir.
"""

from __future__ import annotations

from src.agent.agent_workflow import AgentDecision, SafirAgent


def _build_mock_agent(safir_config) -> SafirAgent:
    return SafirAgent(
        llm_config=safir_config.llm,
        agent_config=safir_config.agent,
        event_store=None,
        rag_service=None,
        use_mock_llm=True,
    )


def test_safir_agent_runs_end_to_end_with_mock_llm(safir_config) -> None:
    agent = _build_mock_agent(safir_config)

    decision = agent.run(
        "## Guncel Gozlem\nSahada bir personel baretsiz calisiyor.\n\n"
        "## Kullanici Istemi\nSahnede riskli bir durum var mi degerlendir.\n"
    )

    assert isinstance(decision, AgentDecision)
    assert 0 <= decision.risk_score <= 100
    assert decision.risk_level in ("dusuk", "orta", "yuksek", "kritik")
    assert decision.recommended_action
    assert "RISK_SKORU" in decision.raw_response


def test_safir_agent_model_name_reflects_mock_client(safir_config) -> None:
    agent = _build_mock_agent(safir_config)
    assert agent.model_name == "mock-llm"


def test_safir_agent_risk_level_matches_configured_thresholds(safir_config) -> None:
    agent = _build_mock_agent(safir_config)
    decision = agent.run("## Guncel Gozlem\nTest baglami.\n\n## Kullanici Istemi\nRisk var mi?\n")

    thresholds = safir_config.agent.risk_thresholds
    if decision.risk_score <= thresholds.low:
        expected = "dusuk"
    elif decision.risk_score <= thresholds.medium:
        expected = "orta"
    elif decision.risk_score <= thresholds.high:
        expected = "yuksek"
    else:
        expected = "kritik"

    assert decision.risk_level == expected
