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

from src.sampler.schema import EvidenceFrame
from src.vlm.factory import get_llm_client, get_vlm_client
from src.vlm.llm_client import MockLLMClient
from src.vlm.vlm_client import MockVLMClient


def _sample_evidence_frame() -> EvidenceFrame:
    return EvidenceFrame(
        evidence_id="ev0",
        frame_id=0,
        timestamp_sec=1.0,
        timestamp_str="00:01",
        change_score=0.5,
        image_bytes=b"",
        base64_image="data:image/jpeg;base64,AA==",
        image_shape=(1, 1, 3),
    )


def test_mock_vlm_client_returns_turkish_description_quickly() -> None:
    client = MockVLMClient()

    started_at = time.perf_counter()
    response = client.analyze_evidence([_sample_evidence_frame()], prompt="Sahnede risk var mi?")
    elapsed = time.perf_counter() - started_at

    assert response.model_name == "mock-vlm"
    assert response.frame_count == 1
    assert isinstance(response.description, str) and len(response.description) > 0
    assert elapsed < 1.0
    assert client.health_check() is True


def test_mock_vlm_client_handles_empty_evidence_list_gracefully() -> None:
    """Gercek Qwen/Gemma implementasyonlarinin aksine, mock istemci bos listede de patlamaz."""
    client = MockVLMClient()
    response = client.analyze_evidence([], prompt="bos")

    assert response.frame_count == 0
    assert response.model_name == "mock-vlm"


def test_get_vlm_client_returns_mock_when_flagged(safir_config) -> None:
    client = get_vlm_client(safir_config.vlm, use_mock=True)
    assert isinstance(client, MockVLMClient)


def test_mock_llm_client_produces_parseable_decision_with_no_tool_calls() -> None:
    client = MockLLMClient().bind_tools([])
    response = client.invoke([HumanMessage(content="Sahnede risk var mi?")])

    assert response.tool_calls == []
    # Mock artik sartname-uyumlu JSON uretir; ayristirilabilir olmali.
    import json

    parsed = json.loads(response.content)
    assert parsed["risk_score"] == 35
    assert isinstance(parsed["actions"], list) and parsed["actions"]


def test_mock_llm_client_bind_tools_returns_self() -> None:
    client = MockLLMClient()
    bound = client.bind_tools(["placeholder"])
    assert bound is client


def test_get_llm_client_returns_mock_when_flagged(safir_config) -> None:
    client = get_llm_client(safir_config.llm, use_mock=True)
    assert isinstance(client, MockLLMClient)
    assert client.model_name == "mock-llm"


def test_parse_structured_events_extracts_and_cleans() -> None:
    """VLM ciktisindaki EVENTS_JSON blogu ayristirilmali ve insan-okur metinden ayrilmali."""
    from src.vlm.base_vlm import parse_structured_events

    content = (
        "Olay #1 (00:15):\n- Kisiler: 1 kisi yerde\n"
        'EVENTS_JSON: [{"type": "arac_yaya_yakinligi", "timestamp": 15, "confidence": 0.8}]'
    )
    desc, events = parse_structured_events(content)
    assert "EVENTS_JSON" not in desc
    assert desc.strip().endswith("yerde")
    assert events == [{"type": "arac_yaya_yakinligi", "timestamp": 15, "confidence": 0.8}]


def test_parse_structured_events_no_block_returns_empty() -> None:
    """EVENTS_JSON blogu yoksa metin degismeden kalmali, olay listesi bos donmeli."""
    from src.vlm.base_vlm import parse_structured_events

    desc, events = parse_structured_events("Sadece duz gozlem metni.")
    assert desc == "Sadece duz gozlem metni."
    assert events == []


def test_parse_structured_events_invalid_json_falls_back() -> None:
    """Gecersiz JSON blogu bos liste dondurmeli (EventEngine keyword fallback'ine duser)."""
    from src.vlm.base_vlm import parse_structured_events

    _, events = parse_structured_events("Gozlem.\nEVENTS_JSON: [not valid json}")
    assert events == []


def test_parse_structured_events_repairs_trailing_commas() -> None:
    """Kucuk modellerin yaygin hatasi (sondaki virgul) toleransli ayristirilmali."""
    from src.vlm.base_vlm import parse_structured_events

    _, events = parse_structured_events(
        'Gozlem.\nEVENTS_JSON: [{"type": "kkd_ihlali", "confidence": 0.7,},]'
    )
    assert events == [{"type": "kkd_ihlali", "confidence": 0.7}]


def test_parse_structured_events_preserves_arbitrary_free_form_keywords() -> None:
    """VLM'in `keywords` alaninda urettigi SERBEST BICIMLI (taksonomiyle sinirli
    OLMAYAN) terimler, EVENTS_JSON ayristirmasi sirasinda AYNEN korunmali -
    hicbiri atilmamali, hicbiri sabit bir listeye indirgenmemeli."""
    from src.vlm.base_vlm import parse_structured_events

    content = (
        "Olay #1 (00:06):\n- Duman gorunuyor.\n"
        'EVENTS_JSON: [{"event_id": "00:06yangin_duman", "type": "yangin_duman", '
        '"keywords": ["duman", "yogun duman", "alev", "alevlenme"], "confidence": 0.91}]'
    )
    _, events = parse_structured_events(content)

    assert len(events) == 1
    assert events[0]["keywords"] == ["duman", "yogun duman", "alev", "alevlenme"]


def test_parse_structured_events_keywords_field_is_optional() -> None:
    """`keywords` alani olmayan (eski/degrade) bir EVENTS_JSON girisi hala gecerli ayristirilmali."""
    from src.vlm.base_vlm import parse_structured_events

    _, events = parse_structured_events(
        'Gozlem.\nEVENTS_JSON: [{"type": "kkd_ihlali", "confidence": 0.7}]'
    )
    assert events == [{"type": "kkd_ihlali", "confidence": 0.7}]
    assert "keywords" not in events[0]


def test_endpoint_extra_body_merges_into_vlm_payload() -> None:
    """VLLMEndpointConfig.extra_body (vLLM guided decoding) istek govdesine eklenmeli, cekirdegi ezmemeli."""
    from src.sampler.schema import EvidenceFrame
    from src.utils.config_loader import VLLMEndpointConfig
    from src.vlm.qwen_vlm import QwenVLM

    endpoint = VLLMEndpointConfig(
        model_name="m", vllm_host="h", vllm_port=1, max_new_tokens=10, temperature=0.1,
        extra_body={"guided_json": {"type": "array"}, "model": "EZILMEMELI"},
    )
    frame = EvidenceFrame(
        evidence_id="ev0", frame_id=0, timestamp_sec=1.0, timestamp_str="00:01", change_score=0.5,
        image_bytes=b"x", base64_image="data:image/jpeg;base64,AA==", image_shape=(1, 1, 3),
    )

    payload = QwenVLM(endpoint)._build_chat_payload([frame], "test")
    assert payload["guided_json"] == {"type": "array"}
    assert payload["model"] == "m"  # cekirdek alan ezilmedi
