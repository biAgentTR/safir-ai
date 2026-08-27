"""Ajan arac yonetimi (config bayraklari + verification_tool) ve guided-JSON retry testleri."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.langgraph_agent import SafirAgent
from src.agent.tools import MOCK_ACTION_TOOL_NAMES, VerificationTool, build_tool_registry
from src.utils.config_loader import AgentToolsConfig


def _tools_config(**overrides) -> AgentToolsConfig:
    base = dict(
        sql_tool_enabled=True,
        rag_tool_enabled=True,
        retriever_tool_enabled=True,
        timeline_tool_enabled=True,
        verification_tool_enabled=True,
        mock_action_tools_enabled=True,
    )
    base.update(overrides)
    return AgentToolsConfig(**base)


def test_build_tool_registry_honors_flags() -> None:
    names = {t.name for t in build_tool_registry(None, None, _tools_config())}
    assert names == {"sql_tool", "retriever_tool", "timeline_tool", "verification_tool"} | MOCK_ACTION_TOOL_NAMES


def test_build_tool_registry_disables_flagged_off_tools() -> None:
    cfg = _tools_config(
        timeline_tool_enabled=False,
        verification_tool_enabled=False,
        rag_tool_enabled=False,
        retriever_tool_enabled=False,
        mock_action_tools_enabled=False,
    )
    names = {t.name for t in build_tool_registry(None, None, cfg)}
    assert names == {"sql_tool"}


def test_build_tool_registry_backward_compatible_without_config() -> None:
    """tools_config verilmezse (geriye-uyum) tum araclar kurulur (verification + mock aksiyon araclari dahil)."""
    names = {t.name for t in build_tool_registry(None, None)}
    assert "verification_tool" in names
    assert MOCK_ACTION_TOOL_NAMES <= names


def test_build_tool_registry_mock_action_tools_can_be_disabled_alone() -> None:
    cfg = _tools_config(mock_action_tools_enabled=False)
    names = {t.name for t in build_tool_registry(None, None, cfg)}
    assert names == {"sql_tool", "retriever_tool", "timeline_tool", "verification_tool"}


def test_verification_tool_reports_regulation_support_and_precedent() -> None:
    out = VerificationTool().run("forklift yayaya cok yaklasti", "arac_yaya_yakinligi")
    assert "Mevzuat destegi: VAR" in out
    assert "Gecmis emsal" in out
    assert "arac_yaya_yakinligi" in out


def test_guided_json_retry_recovers_non_json_output(safir_config) -> None:
    """Serbest cikti gecerli JSON degilse, JSON-modu (invoke_json) ile kurtarilmali.

    NOT: `safir_config`teki (gercek config.yaml) `llm.decision_model` model
    hiyerarsisini ACIK tutar (bkz. `SafirAgent._decision_node`/modul
    dokustringi "Model hiyerarsisi") - bu durumda nihai mesaji fiilen
    `_decision_llm` uretir/retry de ONUNLA yapilir, `_llm` (hizli/arac-
    secim modeli) DEGIL. Bu yuzden burada `_decision_llm` sahtelenir.
    """
    agent = SafirAgent(
        llm_config=safir_config.llm, agent_config=safir_config.agent, use_mock_llm=True
    )
    assert agent._decision_llm is not None  # hiyerarsi safir_config'te aktif
    calls = {"json": 0}
    agent._decision_llm.invoke = lambda messages: AIMessage(content="JSON vermeyen serbest metin.", tool_calls=[])

    def _fake_json(messages):
        calls["json"] += 1
        return AIMessage(
            content='{"summary":"s","events":[],"risk_score":80,"risk_level":"yuksek","actions":["acil mudahale"]}',
            tool_calls=[],
        )

    agent._decision_llm.invoke_json = _fake_json
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
    if agent._decision_llm is not None:
        agent._decision_llm.invoke_json = _fake_json
    # Mock LLM zaten gecerli JSON dondurur -> retry tetiklenmemeli.
    agent.run("## Gozlem\ntest")
    assert calls["json"] == 0


def test_decision_model_hierarchy_routes_final_synthesis_to_decision_llm(safir_config) -> None:
    """`llm.decision_model` yapilandirilmissa, nihai JSON `_decision_llm`den gelmeli - `_llm`in
    (hizli/arac-secim) icerigi ONEMSIZDIR (mentor eleştirisi: model hiyerarsisi)."""
    agent = SafirAgent(
        llm_config=safir_config.llm, agent_config=safir_config.agent, use_mock_llm=True
    )
    assert agent._decision_llm is not None
    assert agent._decision_llm is not agent._llm

    # Hizli model (_llm) tamamen ILGISIZ bir metin dondurse bile, nihai karar
    # DECISION_LLM'in ciktisindan gelmelidir.
    agent._llm.invoke = lambda messages: AIMessage(content="tool cagrisi yok, ilgisiz metin", tool_calls=[])
    agent._decision_llm.invoke = lambda messages: AIMessage(
        content='{"summary":"buyuk model karari","events":[],"risk_score":42,"risk_level":"orta","actions":["izle"]}',
        tool_calls=[],
    )

    decision = agent.run("## Gozlem\ntest")
    assert decision.risk_score == 42
    assert decision.summary == "buyuk model karari"


def test_run_collects_triggered_mock_actions(safir_config) -> None:
    """Ajan bir mock aksiyon aracini (tool_call olarak) cagirirsa, bu `AgentDecision.
    triggered_mock_actions`e toplanmali - sql/retriever/timeline/verification gibi ic
    sorgu araclari BURAYA GIRMEMELI (sartname: 'mock fonksiyonlarin ajanin araclari
    olarak basariyla kullanilmasi')."""
    agent = SafirAgent(
        llm_config=safir_config.llm, agent_config=safir_config.agent, use_mock_llm=True
    )
    call_count = {"n": 0}

    def _fake_invoke(messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "notify_health_team_tool",
                        "args": {"event_id": "dusme_00:18", "urgency": "kritik"},
                        "id": "call_1",
                    },
                    {
                        "name": "sql_tool",
                        "args": {"query_type": "recent"},
                        "id": "call_2",
                    },
                ],
            )
        return AIMessage(
            content='{"summary":"s","events":[],"risk_score":90,"risk_level":"kritik","actions":["saglik ekibini cagir"]}',
            tool_calls=[],
        )

    agent._llm.invoke = _fake_invoke
    if agent._decision_llm is not None:
        agent._decision_llm.invoke = _fake_invoke

    decision = agent.run("## Gozlem\ntest")

    assert len(decision.triggered_mock_actions) == 1
    triggered = decision.triggered_mock_actions[0]
    assert triggered["tool"] == "notify_health_team_tool"
    assert triggered["args"] == {"event_id": "dusme_00:18", "urgency": "kritik"}
    assert "Saglik ekibi bilgilendirildi" in triggered["result"]


def test_decision_model_unset_keeps_single_model_behavior(safir_config) -> None:
    """`decision_model=None`/`active_model` ile ayniysa hiyerarsi devre disi - `_decision_llm is None`
    ve nihai karar dogrudan `_llm`in son yanitindan gelir (davranis ONCEKI haliyle AYNI)."""
    single_model_llm_config = safir_config.llm.model_copy(update={"decision_model": None})
    agent = SafirAgent(
        llm_config=single_model_llm_config, agent_config=safir_config.agent, use_mock_llm=True
    )
    assert agent._decision_llm is None

    decision = agent.run("## Gozlem\ntest")
    assert isinstance(decision.risk_score, int)
