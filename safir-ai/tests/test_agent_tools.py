"""Ajan arac yonetimi (config bayraklari + verification_tool) ve guided-JSON retry testleri."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.langgraph_agent import SafirAgent
from src.agent.tools import VerificationTool, build_tool_registry
from src.utils.config_loader import AgentToolsConfig


def _tools_config(**overrides) -> AgentToolsConfig:
    base = dict(
        sql_tool_enabled=True,
        rag_tool_enabled=True,
        retriever_tool_enabled=True,
        timeline_tool_enabled=True,
        verification_tool_enabled=True,
    )
    base.update(overrides)
    return AgentToolsConfig(**base)


def test_build_tool_registry_honors_flags() -> None:
    names = {t.name for t in build_tool_registry(None, None, _tools_config())}
    assert names == {"sql_tool", "retriever_tool", "timeline_tool", "verification_tool"}


def test_build_tool_registry_disables_flagged_off_tools() -> None:
    cfg = _tools_config(
        timeline_tool_enabled=False, verification_tool_enabled=False, rag_tool_enabled=False, retriever_tool_enabled=False
    )
    names = {t.name for t in build_tool_registry(None, None, cfg)}
    assert names == {"sql_tool"}


def test_build_tool_registry_backward_compatible_without_config() -> None:
    """tools_config verilmezse (geriye-uyum) tum araclar kurulur (verification dahil)."""
    names = {t.name for t in build_tool_registry(None, None)}
    assert "verification_tool" in names


def test_verification_tool_reports_regulation_support_and_precedent() -> None:
    out = VerificationTool().run("forklift yayaya cok yaklasti", "arac_yaya_yakinligi")
    assert "Mevzuat destegi: VAR" in out
    assert "Gecmis emsal" in out
    assert "arac_yaya_yakinligi" in out


def test_guided_json_retry_recovers_non_json_output(safir_config) -> None:
    """Serbest cikti gecerli JSON degilse, JSON-modu (invoke_json) ile kurtarilmali."""
    agent = SafirAgent(
        llm_config=safir_config.llm, agent_config=safir_config.agent, use_mock_llm=True
    )
    calls = {"json": 0}
    agent._llm.invoke = lambda messages: AIMessage(content="JSON vermeyen serbest metin.", tool_calls=[])

    def _fake_json(messages):
        calls["json"] += 1
        return AIMessage(
            content='{"summary":"s","events":[],"risk_score":80,"risk_level":"yuksek","actions":["acil mudahale"]}',
            tool_calls=[],
        )

    agent._llm.invoke_json = _fake_json
    decision = agent.run("## Gozlem\ntest")

    assert calls["json"] == 1
    assert decision.risk_score == 80
    assert decision.recommended_action == "acil mudahale"


def test_guided_json_retry_skipped_when_output_already_json(safir_config) -> None:
    """Cikti zaten gecerli JSON ise JSON-modu yeniden-denemesi CAGRILMAMALI."""
    agent = SafirAgent(
        llm_config=safir_config.llm, agent_config=safir_config.agent, use_mock_llm=True
    )
    calls = {"json": 0}

    def _fake_json(messages):
        calls["json"] += 1
        return AIMessage(content="{}", tool_calls=[])

    agent._llm.invoke_json = _fake_json
    # Mock LLM zaten gecerli JSON dondurur -> retry tetiklenmemeli.
    agent.run("## Gozlem\ntest")
    assert calls["json"] == 0
