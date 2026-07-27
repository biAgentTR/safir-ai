"""Modul 2 (src/vlm/) icin GPU/aga bagimlilik gerektirmeyen birim testleri.

Yalnizca mock istemcileri (`MockVLMClient`, `MockLLMClient`) ve fabrika
fonksiyonlarinin (`get_vlm_client`, `get_llm_client`) mock secim mantigini
test eder. Gercek `VLMClient`/`LLMClient` yollari, tanim geregi bir GPU/vLLM
servisine ihtiyac duydugu icin (ve GPU'suz ortamda test edilemeyecegi icin)
bu dosyada egzersiz edilmez; bu, testin kapsam disi birakildigi bir eksiklik
degil, mock-mod garantisinin tam da hedefidir.
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage

from src.sampler.schema import EventCluster, EvidenceFrame
from src.vlm.factory import get_llm_client, get_vlm_client
from src.vlm.llm_client import MockLLMClient
from src.vlm.vlm_client import MockVLMClient


def _sample_cluster() -> EventCluster:
    frame = EvidenceFrame(
        frame_id=0,
        timestamp_sec=1.0,
        timestamp_str="00:01",
        change_score=0.5,
        image_bytes=b"",
        base64_image="data:image/jpeg;base64,AA==",
        image_shape=(1, 1, 3),
    )
    return EventCluster(event_id=1, start_time=1.0, end_time=1.0, peak_frame=frame, total_candidate_frames=1)


def test_mock_vlm_client_returns_turkish_description_quickly() -> None:
    client = MockVLMClient()

    started_at = time.perf_counter()
    response = client.describe_events([_sample_cluster()], prompt="Sahnede risk var mi?")
    elapsed = time.perf_counter() - started_at

    assert response.model_name == "mock-vlm"
    assert response.frame_count == 1
    assert isinstance(response.description, str) and len(response.description) > 0
    assert elapsed < 1.0
    assert client.health_check() is True


def test_mock_vlm_client_handles_empty_cluster_list_gracefully() -> None:
    """Gercek Qwen/Gemma implementasyonlarinin aksine, mock istemci bos listede de patlamaz."""
    client = MockVLMClient()
    response = client.describe_events([], prompt="bos")

    assert response.frame_count == 0
    assert response.model_name == "mock-vlm"


def test_get_vlm_client_returns_mock_when_flagged(safir_config) -> None:
    client = get_vlm_client(safir_config.vlm, use_mock=True)
    assert isinstance(client, MockVLMClient)


def test_mock_llm_client_produces_parseable_decision_with_no_tool_calls() -> None:
    client = MockLLMClient().bind_tools([])
    response = client.invoke([HumanMessage(content="Sahnede risk var mi?")])

    assert response.tool_calls == []
    assert "RISK_SKORU" in response.content
    assert "AKSIYON_ONERISI" in response.content


def test_mock_llm_client_bind_tools_returns_self() -> None:
    client = MockLLMClient()
    bound = client.bind_tools(["placeholder"])
    assert bound is client


def test_get_llm_client_returns_mock_when_flagged(safir_config) -> None:
    client = get_llm_client(safir_config.llm, use_mock=True)
    assert isinstance(client, MockLLMClient)
    assert client.model_name == "mock-llm"
