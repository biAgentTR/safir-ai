"""05 - Ajan ve Muhakeme Katmani: LangGraph tabanli durum makinesi (Reasoning Agent).

VLM'den bagimsiz, saf bir LLM (Qwen3 / Gemma3) uzerinde calisan bu durum
makinesi; zenginlestirilmis baglami alir, gerektiginde `Dynamic Tool
Router` uzerinden SQL/RAG/Timeline araclarini cagirir ve sonunda 0-100
arasi bir risk skoru ile karar/aksiyon onerisi ureten `AgentDecision`
dondurur.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Annotated, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.agent.tools import build_tool_registry
from src.memory.event_store import EventStore
from src.memory.semantic_memory import SemanticMemory
from src.utils.config_loader import AgentConfig, LLMConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Sen SAFIR sisteminin saha guvenligi muhakeme ajanisin. Sana verilen "
    "gozlem baglamini degerlendir; gerekirse sql_tool, rag_tool veya "
    "timeline_tool araclarini kullanarak gecmis olaylari ve ilgili mevzuati "
    "incele. Analizinin sonunda MUTLAKA su formatta bir sonuc satiri yaz:\n"
    "RISK_SKORU: <0-100 arasi tam sayi>\n"
    "AKSIYON_ONERISI: <operatore yonelik kisa, somut Turkce aksiyon onerisi>"
)

_RISK_LINE_PATTERN = re.compile(r"RISK_SKORU:\s*(\d{1,3})", re.IGNORECASE)
_ACTION_LINE_PATTERN = re.compile(r"AKSIYON_ONERISI:\s*(.+)", re.IGNORECASE)


class AgentState(TypedDict):
    """LangGraph durum makinesinin dugumler arasinda tasidigi paylasimli durum."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int


@dataclass
class AgentDecision:
    """Ajan durum makinesinin nihai muhakeme sonucu."""

    risk_score: int
    risk_level: str
    recommended_action: str
    raw_response: str


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
        semantic_memory: Optional[SemanticMemory] = None,
    ) -> None:
        """SafirAgent'i LLM/ajan konfigurasyonu ve bellek bagimliliklariyla kurar.

        Args:
            llm_config: `configs/config.yaml` icindeki `llm` blogu (aktif model + uc noktalar).
            agent_config: `configs/config.yaml` icindeki `agent` blogu (esikler, arac anahtarlari).
            event_store: SQL/Timeline araclarinin kullanacagi olay deposu.
            semantic_memory: RAG aracinin kullanacagi anlamsal bellek.
        """
        self._agent_config = agent_config
        endpoint = llm_config.active_endpoint()

        self._tools: List[StructuredTool] = build_tool_registry(event_store, semantic_memory)

        self._llm = ChatOpenAI(
            model=endpoint.model_name,
            base_url=f"http://{endpoint.vllm_host}:{endpoint.vllm_port}/v1",
            api_key="EMPTY",
            temperature=endpoint.temperature,
            max_tokens=endpoint.max_new_tokens,
        ).bind_tools(self._tools)

        self._tools_by_name = {tool.name: tool for tool in self._tools}
        self._graph = self._build_graph()

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
            Risk skoru, seviyesi ve aksiyon onerisini iceren `AgentDecision`.
        """
        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=context_block),
            ],
            "iteration": 0,
        }

        final_state = self._graph.invoke(initial_state)
        final_text = final_state["messages"][-1].content or ""

        risk_match = _RISK_LINE_PATTERN.search(final_text)
        action_match = _ACTION_LINE_PATTERN.search(final_text)

        risk_score = int(risk_match.group(1)) if risk_match else 0
        risk_score = max(0, min(100, risk_score))
        recommended_action = (
            action_match.group(1).strip() if action_match else "Ek aksiyon onerisi uretilemedi."
        )

        return AgentDecision(
            risk_score=risk_score,
            risk_level=self._resolve_risk_level(risk_score),
            recommended_action=recommended_action,
            raw_response=final_text,
        )
