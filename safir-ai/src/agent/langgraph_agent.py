"""05 - Ajan ve Muhakeme Katmani: LangGraph tabanli durum makinesi (Reasoning Agent).

VLM'den bagimsiz, saf bir LLM (Qwen3 / Gemma3) uzerinde calisan bu durum
makinesi; zenginlestirilmis baglami alir, gerektiginde `Dynamic Tool
Router` uzerinden sql_tool/retriever_tool/timeline_tool araclarini cagirir
ve sonunda 0-100 arasi bir risk skoru ile karar/aksiyon onerisi ureten
`AgentDecision` dondurur.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.agent.tools import build_tool_registry
from src.memory.embedding_rag_service import EmbeddingRAGService
from src.memory.event_store import EventStore
from src.prompts import AGENT_SYSTEM_PROMPT, build_agent_user_prompt
from src.utils.config_loader import AgentConfig, LLMConfig
from src.vlm.factory import get_llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT

# Yapisal JSON ayristirma basarisiz olursa kullanilan geriye-donuk regex'ler
# (eski RISK_SKORU/AKSIYON_ONERISI bicimini ve mock ciktilarini korur).
_RISK_LINE_PATTERN = re.compile(r"(?:RISK_SKORU|risk_score)\D{0,4}(\d{1,3})", re.IGNORECASE)
_ACTION_LINE_PATTERN = re.compile(r"AKSIYON_ONERISI:\s*(.+)", re.IGNORECASE)
# Serbest metin icine gomulu ilk JSON nesnesini yakalar (```json blogu dahil).
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class AgentState(TypedDict):
    """LangGraph durum makinesinin dugumler arasinda tasidigi paylasimli durum."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int


@dataclass
class AgentDecision:
    """Ajan durum makinesinin nihai muhakeme sonucu (sartname JSON'u ile hizali)."""

    risk_score: int
    risk_level: str
    recommended_action: str                       # geriye-uyum: actions[0]
    raw_response: str
    summary: str = ""                             # operatore yonelik Turkce durum ozeti
    actions: List[str] = field(default_factory=list)   # somut aksiyon onerileri listesi
    events: List[Dict[str, str]] = field(default_factory=list)  # [{"time": "MM:SS", "event": "..."}]


class SafirAgent:
    """LangGraph tabanli durum makinesi uzerinde calisan risk muhakeme ajani.

    Dugumler:
        reasoning  -> LLM'i (Qwen3/Gemma3) mevcut mesaj gecmisiyle cagirir.
        tools      -> LLM'in istedigi arac cagrilarini yurutur.
        decision   -> LLM'in son yanitindan risk skoru/seviyesi/aksiyonu cikarir.

    Kenarlar:
        reasoning -> tools     (arac cagrisi istendiginde)
        reasoning -> decision  (arac cagrisi istenmedigi veya iterasyon
                                 siniri asildiginda)
        tools     -> reasoning (arac sonucu ile muhakemeye devam)
        decision  -> END
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        agent_config: AgentConfig,
        event_store: Optional[EventStore] = None,
        rag_service: Optional[EmbeddingRAGService] = None,
        use_mock_llm: bool = False,
    ) -> None:
        """SafirAgent'i LLM/ajan konfigurasyonu ve bellek bagimliliklariyla kurar.

        Args:
            llm_config: `configs/config.yaml` icindeki `llm` blogu (aktif model + uc noktalar).
            agent_config: `configs/config.yaml` icindeki `agent` blogu (esikler, arac anahtarlari).
            event_store: SQL/Timeline araclarinin kullanacagi olay deposu.
            rag_service: `retriever_tool`'un kullanacagi Embedding & RAG servisi.
            use_mock_llm: `True` ise gercek vLLM'e hic baglanmadan `MockLLMClient`
                kullanilir (GPU'suz gelistirme icin, bkz. `configs/config.yaml`
                `app.use_mock_llm`).
        """
        self._agent_config = agent_config

        self._tools: List[StructuredTool] = build_tool_registry(event_store, rag_service)
        self._llm = get_llm_client(llm_config, use_mock=use_mock_llm).bind_tools(self._tools)

        self._tools_by_name = {tool.name: tool for tool in self._tools}
        self._graph = self._build_graph()

    @property
    def model_name(self) -> str:
        """Karari fiilen ureten LLM'in adini dondurur (mock modda `"mock-llm"`).

        Config'te tanimli model adini degil, `get_llm_client` tarafindan
        gercekten secilen istemcinin adini yansitir; boylece `SafirReport.llm_model`
        mock mod aktifken yaniltici bicimde gercek model adini gostermez.
        """
        return self._llm.model_name

    def _build_graph(self):
        """LangGraph `StateGraph`'ini dugum ve kenarlariyla insa eder ve derler.

        Returns:
            Calistirilmaya hazir, derlenmis LangGraph uygulamasi.
        """
        graph = StateGraph(AgentState)
        graph.add_node("reasoning", self._reasoning_node)
        graph.add_node("tools", self._tools_node)
        graph.add_node("decision", self._decision_node)

        graph.set_entry_point("reasoning")
        graph.add_conditional_edges(
            "reasoning",
            self._route_after_reasoning,
            {"tools": "tools", "decision": "decision"},
        )
        graph.add_edge("tools", "reasoning")
        graph.add_edge("decision", END)

        return graph.compile()

    def _reasoning_node(self, state: AgentState) -> AgentState:
        """LLM'i mevcut mesaj gecmisiyle cagirip yanitini duruma ekler.

        Args:
            state: Mevcut ajan durumu.

        Returns:
            LLM'in yanitiyla guncellenmis durum (mesaj ve iterasyon sayaci).
        """
        response: AIMessage = self._llm.invoke(state["messages"])
        return {"messages": [response], "iteration": state["iteration"] + 1}

    def _tools_node(self, state: AgentState) -> AgentState:
        """LLM'in son yanitindaki arac cagrilarini yurutur ve sonuclari duruma ekler.

        Args:
            state: Mevcut ajan durumu (son mesaj bir `AIMessage` ile arac cagrisi icermeli).

        Returns:
            Her arac cagrisi icin bir `ToolMessage` eklenmis durum.
        """
        last_message = state["messages"][-1]
        tool_messages: List[ToolMessage] = []

        for call in last_message.tool_calls:
            tool = self._tools_by_name.get(call["name"])
            if tool is None:
                content = f"Bilinmeyen arac: {call['name']}"
            else:
                try:
                    content = tool.invoke(call["args"])
                except Exception as exc:  # noqa: BLE001 - arac hatasi ajana geri bildirilir
                    content = f"Arac calistirma hatasi ({call['name']}): {exc}"
                    logger.exception("Arac calistirma hatasi: %s", call["name"])

            tool_messages.append(
                ToolMessage(content=str(content), tool_call_id=call["id"])
            )

        return {"messages": tool_messages, "iteration": state["iteration"]}

    def _decision_node(self, state: AgentState) -> AgentState:
        """Son LLM yanitindan risk skoru, seviyesi ve aksiyon onerisini cikarir.

        Args:
            state: Mevcut ajan durumu.

        Returns:
            Degisiklik yapilmamis durum (karar, `run` metodunda ayrica cozumlenir).
        """
        return state

    def _route_after_reasoning(self, state: AgentState) -> str:
        """Muhakeme dugumunden sonra arac mi yoksa karar mi calisacagini belirler.

        Args:
            state: Mevcut ajan durumu.

        Returns:
            "tools" (arac cagrisi varsa ve iterasyon siniri asilmadiysa) veya "decision".
        """
        last_message = state["messages"][-1]
        has_tool_calls = bool(getattr(last_message, "tool_calls", None))
        within_budget = state["iteration"] < self._agent_config.max_iterations

        if has_tool_calls and within_budget:
            return "tools"
        return "decision"

    def _resolve_risk_level(self, risk_score: int) -> str:
        """Sayisal risk skorunu config esiklerine gore risk seviyesi etiketine cevirir.

        Args:
            risk_score: 0-100 arasi risk skoru.

        Returns:
            "dusuk", "orta", "yuksek" veya "kritik".
        """
        thresholds = self._agent_config.risk_thresholds
        if risk_score <= thresholds.low:
            return "dusuk"
        if risk_score <= thresholds.medium:
            return "orta"
        if risk_score <= thresholds.high:
            return "yuksek"
        return "kritik"

    def run(self, context_block: str) -> AgentDecision:
        """Verilen zenginlestirilmis baglam uzerinde ajan durum makinesini calistirir.

        Args:
            context_block: `ContextBuilder.build(...).to_prompt_block()` ciktisi.

        Returns:
            Risk skoru, seviyesi, ozet, olaylar ve aksiyon onerilerini iceren `AgentDecision`.
        """
        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=build_agent_user_prompt(context_block)),
            ],
            "iteration": 0,
        }

        final_state = self._graph.invoke(initial_state)
        final_text = final_state["messages"][-1].content or ""
        return self._parse_decision(final_text)

    def _parse_decision(self, final_text: str) -> AgentDecision:
        """Ajanin son yanitini `AgentDecision`'a cevirir (once JSON, sonra regex fallback).

        Once yanit icindeki ilk JSON nesnesi ayristirilmaya calisilir (sartname
        semasi: summary/events/risk_score/actions). Bu basarisiz olursa eski
        `RISK_SKORU:`/`AKSIYON_ONERISI:` bicimi regex ile okunur; boylece hem
        yeni JSON-ureten modeller hem de eski/mock ciktilar desteklenir.

        Args:
            final_text: Ajanin son mesaj icerigi.

        Returns:
            Ayristirilmis `AgentDecision`. Risk seviyesi her zaman config
            esiklerinden yeniden hesaplanir (modelin iddia ettigi seviyeye guvenilmez).
        """
        parsed = self._extract_json(final_text)

        if parsed is not None:
            risk_score = self._coerce_risk_score(parsed.get("risk_score"))
            summary = str(parsed.get("summary", "")).strip()
            actions = self._coerce_actions(parsed.get("actions"))
            events = self._coerce_events(parsed.get("events"))
        else:
            logger.warning("Ajan yaniti JSON olarak ayristirilamadi, regex fallback kullaniliyor.")
            risk_match = _RISK_LINE_PATTERN.search(final_text)
            action_match = _ACTION_LINE_PATTERN.search(final_text)
            risk_score = self._coerce_risk_score(risk_match.group(1) if risk_match else None)
            summary = ""
            single_action = action_match.group(1).strip() if action_match else ""
            actions = [single_action] if single_action else []
            events = []

        recommended_action = actions[0] if actions else "Ek aksiyon onerisi uretilemedi."

        return AgentDecision(
            risk_score=risk_score,
            risk_level=self._resolve_risk_level(risk_score),
            recommended_action=recommended_action,
            raw_response=final_text,
            summary=summary,
            actions=actions,
            events=events,
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Serbest metin icindeki ilk JSON nesnesini ayristirir; bulunamazsa None."""
        match = _JSON_OBJECT_PATTERN.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _coerce_risk_score(value: Any) -> int:
        """Risk skorunu 0-100 araligina kirpilmis bir tam sayiya cevirir (gecersizse 0)."""
        try:
            score = int(float(value))
        except (TypeError, ValueError):
            return 0
        return max(0, min(100, score))

    @staticmethod
    def _coerce_actions(value: Any) -> List[str]:
        """`actions` alanini temiz bir string listesine cevirir (liste, string veya None)."""
        if isinstance(value, list):
            return [str(a).strip() for a in value if str(a).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _coerce_events(value: Any) -> List[Dict[str, str]]:
        """`events` alanini `{"time", "event"}` sozluklerinden olusan bir listeye cevirir."""
        if not isinstance(value, list):
            return []
        events: List[Dict[str, str]] = []
        for item in value:
            if isinstance(item, dict) and ("event" in item or "time" in item):
                events.append(
                    {"time": str(item.get("time", "")).strip(), "event": str(item.get("event", "")).strip()}
                )
        return events
