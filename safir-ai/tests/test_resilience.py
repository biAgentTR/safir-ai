"""#4 Hata dayanikliligi testleri: VLM retry, ajan degraded karar.

VLM gecici ag hatalarinda yeniden dener; kalici hatada RuntimeError firlatir.
Ajan muhakemesi patlarsa is cokmez, guvenli bir degraded karar doner.
"""

from __future__ import annotations

import httpx
import pytest

import src.vlm.base_vlm as base_vlm
from src.sampler.schema import EventCluster, EvidenceFrame
from src.utils.config_loader import VLLMEndpointConfig
from src.vlm.qwen_vlm import QwenVLM


class _OkResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "Gozlem metni."}}]}


def _cluster() -> EventCluster:
    frame = EvidenceFrame(
        frame_id=0,
        timestamp_sec=1.0,
        timestamp_str="00:01",
        change_score=0.5,
        image_bytes=b"x",
        base64_image="data:image/jpeg;base64,AA==",
        image_shape=(1, 1, 3),
    )
    return EventCluster(event_id=1, start_time=1.0, end_time=1.0, peak_frame=frame, total_candidate_frames=1)


@pytest.fixture
def _no_sleep(monkeypatch):
    monkeypatch.setattr(base_vlm.time, "sleep", lambda _s: None)


def _vlm() -> QwenVLM:
    endpoint = VLLMEndpointConfig(
        model_name="m", vllm_host="h", vllm_port=1, max_new_tokens=10, temperature=0.1
    )
    return QwenVLM(endpoint)


def test_vlm_retries_transient_error_then_succeeds(monkeypatch, _no_sleep) -> None:
    calls = {"n": 0}

    def _post(url, json, headers, timeout):  # noqa: A002
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("gecici ag hatasi")
        return _OkResponse()

    monkeypatch.setattr(base_vlm.httpx, "post", _post)
    response = _vlm().describe_events([_cluster()], "test")

    assert calls["n"] == 3  # 2 basarisiz + 1 basarili
    assert response.description == "Gozlem metni."


def test_vlm_raises_after_exhausting_retries(monkeypatch, _no_sleep) -> None:
    def _post(url, json, headers, timeout):  # noqa: A002
        raise httpx.ConnectError("kalici ag hatasi")

    monkeypatch.setattr(base_vlm.httpx, "post", _post)
    with pytest.raises(RuntimeError, match="denemede basarisiz"):
        _vlm().describe_events([_cluster()], "test")


def test_agent_returns_degraded_decision_on_reasoning_failure(safir_config) -> None:
    """LLM muhakemesi patlarsa ajan cokmez; manuel incelemeye yonlendiren degraded karar doner."""
    from src.agent.langgraph_agent import SafirAgent

    agent = SafirAgent(llm_config=safir_config.llm, agent_config=safir_config.agent, use_mock_llm=True)

    def _boom(_messages):
        raise RuntimeError("LLM servisi coktu")

    agent._llm.invoke = _boom
    decision = agent.run("## Gozlem\ntest")

    # P0 fix: analiz basarisiz -> risk_score=None + risk_status="unknown",
    # ASLA 0/dusuk-risk'e sessizce dusmez (bkz. tests/agent/test_risk_status.py).
    assert decision.risk_score is None
    assert decision.risk_status == "unknown"
    assert decision.risk_level == "unknown"
    assert "manuel" in decision.recommended_action.lower()
    assert decision.actions
    assert "[HATA]" in decision.raw_response
