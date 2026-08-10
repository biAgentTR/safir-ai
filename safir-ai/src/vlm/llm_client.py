"""Modul 2 - LLM Client: LangGraph Ajani icin gercek/mock muhakeme istemcileri.

`LLMClient`, config'teki aktif LLM'e (Qwen3/Gemma3) baglanan bir `ChatOpenAI`
orneğini sarmalar; `SafirAgent` bunu `bind_tools(...)` ile aracli hale getirip
`invoke(messages)` ile cagirir — bugune kadarki davranisin birebir aynisidir,
yalnizca yeniden kullanilabilir bir sinifa tasinmistir. `MockLLMClient` ayni
`bind_tools`/`invoke` arayuzunu, hicbir arac cagirmadan, sabit bir risk
skoru/aksiyon onerisi ureten bir `AIMessage` ile karsilar; boylece
`use_mock_llm: true` oldugunda `SafirAgent`'in LangGraph durum makinesi
degistirilmeden dogrudan `decision` dugumune ilerler.
"""

from __future__ import annotations

import logging
import time
from typing import List, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from src.utils.config_loader import LLMConfig

logger = logging.getLogger(__name__)

_MOCK_DECISION_TEXT = (
    "{\n"
    '  "summary": "[MOCK] Sahada KKD eksikligi gozlendi; orta duzeyde risk mevcut.",\n'
    '  "events": [{"time": "00:12", "event": "Korumasiz alanda KKD eksik personel"}],\n'
    '  "risk_score": 35,\n'
    '  "risk_level": "orta",\n'
    '  "actions": ["Saha operatorunu bilgilendirin", "KKD kontrolu yapin"]\n'
    "}"
)


class LLMClient:
    """Config'teki aktif LLM'e baglanan gercek `ChatOpenAI` istemcisini sarmalar."""

    def __init__(self, llm_config: LLMConfig) -> None:
        """LLMClient'i config'teki aktif LLM uc noktasiyla baslatir.

        Args:
            llm_config: `configs/config.yaml` icindeki `llm` blogu.
        """
        endpoint = llm_config.active_endpoint()
        self._model_name = endpoint.model_name
        # Yerel vLLM icin base_url=http://host:port/v1 ve api_key="EMPTY";
        # provider="gemini" icin endpoint.base_url + GEMINI_API_KEY (api_key_env).
        self._base_chat = ChatOpenAI(
            model=endpoint.model_name,
            base_url=endpoint.resolved_base_url(),
            api_key=endpoint.resolved_api_key(),
            temperature=endpoint.temperature,
            max_tokens=endpoint.max_new_tokens,
        )
        # `bind_tools` bunu araci-baglanmis surumle degistirir; `_base_chat`
        # (araci baglanmamis) JSON-modu (invoke_json) icin saklanir.
        self._chat = self._base_chat

    @property
    def model_name(self) -> str:
        """Bu istemcinin baglandigi LLM'in model adini dondurur."""
        return self._model_name

    def bind_tools(self, tools: List[StructuredTool]) -> "LLMClient":
        """Verilen araclari alttaki `ChatOpenAI` orneğine baglar.

        Args:
            tools: LangGraph ajaninin `Dynamic Tool Router` araclari.

        Returns:
            Araclari baglanmis (kendisiyle ayni turden) istemci.
        """
        self._chat = self._chat.bind_tools(tools)
        return self

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Mesaj gecmisini gercek LLM'e gonderip yanitini dondurur.

        Args:
            messages: LangGraph durum makinesinin biriktirdigi mesaj gecmisi.

        Returns:
            LLM'in yanitini tasiyan `AIMessage`.
        """
        return self._chat.invoke(messages)

    def invoke_json(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Mesajlari JSON-modunda (araci baglanmamis) LLM'e gonderip gecerli JSON yaniti dondurur.

        `response_format={"type": "json_object"}` hem vLLM hem Gemini'nin
        OpenAI-uyumlu ucunda desteklenir ve modeli gecerli bir JSON nesnesi
        uretmeye zorlar; kucuk yerel modellerde bozuk JSON'u onler. Arac
        cagrilariyla catismamasi icin `_base_chat` (araci baglanmamis) kullanilir.

        Args:
            messages: Karar formatlama istemi dahil mesaj gecmisi.

        Returns:
            Gecerli JSON icerigi tasiyan `AIMessage`.
        """
        structured_chat = self._base_chat.bind(response_format={"type": "json_object"})
        return structured_chat.invoke(messages)


class MockLLMClient:
    """GPU'su olmayan gelistiriciler icin sahte LLM istemcisi (`use_mock_llm: true`).

    Gercek vLLM servisine hic baglanmadan, sabit formatli (RISK_SKORU/
    AKSIYON_ONERISI) bir `AIMessage` dondurur; hicbir arac cagirmadigi icin
    LangGraph durum makinesi bir sonraki adimda dogrudan `decision` dugumune
    ilerler.
    """

    def __init__(self) -> None:
        """MockLLMClient'i baslatir (arac listesi bos baslar)."""
        self._model_name = "mock-llm"
        self._tools: List[StructuredTool] = []

    @property
    def model_name(self) -> str:
        """Bu sahte istemcinin sabit model adini dondurur."""
        return self._model_name

    def bind_tools(self, tools: List[StructuredTool]) -> "MockLLMClient":
        """Araclari kaydeder (mock modda gercekten cagrilmaz) ve kendisini dondurur.

        Args:
            tools: LangGraph ajaninin `Dynamic Tool Router` araclari.

        Returns:
            Kendisi (arayuz uyumlulugu icin).
        """
        self._tools = tools
        return self

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Sahte gecikme sonrasi arac cagrisi icermeyen sabit bir karar dondurur.

        Args:
            messages: LangGraph durum makinesinin biriktirdigi mesaj gecmisi (yalnizca loglanir).

        Returns:
            `RISK_SKORU`/`AKSIYON_ONERISI` iceren, `tool_calls=[]` bir `AIMessage`.
        """
        time.sleep(0.1)
        logger.info("MockLLMClient: %d mesajlik gecmis icin sahte karar uretildi", len(messages))
        return AIMessage(content=_MOCK_DECISION_TEXT, tool_calls=[])

    def invoke_json(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Mock modda JSON-modu ile normal `invoke` ayni sabit JSON karari dondurur."""
        return self.invoke(messages)


if __name__ == "__main__":
    # Modul 2'nin bagimsiz calistirilabilirlik testi:
    #   python -m src.vlm.llm_client            -> mock istemciyi test eder
    #   python -m src.vlm.llm_client --real      -> config'teki gercek LLM'e istek atar
    import sys

    from langchain_core.messages import HumanMessage

    from src.utils.config_loader import load_config

    logging.basicConfig(level=logging.INFO)

    use_real = "--real" in sys.argv
    demo_llm = LLMClient(load_config().llm) if use_real else MockLLMClient()
    demo_llm = demo_llm.bind_tools([])

    demo_response = demo_llm.invoke([HumanMessage(content="Test istemi: sahnede risk var mi?")])
    print(f"content={demo_response.content}")
    print(f"tool_calls={demo_response.tool_calls}")
