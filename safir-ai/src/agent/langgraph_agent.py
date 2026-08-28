"""05 - Ajan ve Muhakeme Katmani: LangGraph tabanli durum makinesi (Reasoning Agent).

VLM'den bagimsiz, saf bir LLM (Qwen3 / Gemma3) uzerinde calisan bu durum
makinesi; zenginlestirilmis baglami alir, gerektiginde `Dynamic Tool
Router` uzerinden sql_tool/retriever_tool/timeline_tool araclarini cagirir
ve sonunda 0-100 arasi bir risk skoru ile karar/aksiyon onerisi ureten
`AgentDecision` dondurur.

Model hiyerarsisi (mentor eleştirisi: EVREN dokumantasyonu SS 6, "her gorev
icin buyuk modeli kullanmayin, hiyerarsi kurun")
----------------------------------------------------------------------------
`reasoning`/`tools` dugumleri (arac secimi/cagirma, JSON uretimi) HER ZAMAN
`llm.active_model`i ("hizli" model, `self._llm`, araclara BAGLI) kullanir -
EVREN dokumantasyonuna gore bu gorevlerde buyuk modelle olculebilir bir fark
yoktur (bkz. `configs/config.yaml` `llm.models.evren` yorumu). Dongu
bittiginde (arac cagrisi kalmadi veya iterasyon siniri asildi), `decision`
dugumu - eger `llm.decision_model` yapilandirilmissa - TEK bir ek cagriyla
"buyuk"/otonom-karar modeline (`self._decision_llm`, ARACSIZ - bu asamada
artik arac cagrisi beklenmez) gecer ve nihai JSON kararini SENTEZLER; bu,
"uzun akil yurutme zincirleri ve otonom karar anlari" icin daha guclu bir
modele ROUTE eden canli bir LangGraph dugumudur. `decision_model`
yapilandirilmamissa (varsayilan), `decision` dugumu ONCEKI gibi bir gecis
(no-op) kalir - davranis/cagri sayisi BIREBIR AYNI.
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

from src.agent.tools import MOCK_ACTION_TOOL_NAMES, build_tool_registry
from src.rag.embedding_rag_service import EmbeddingRAGService
from src.memory.event_store import EventStore
from src.prompts import AGENT_SYSTEM_PROMPT, DECISION_SYNTHESIS_INSTRUCTION, build_agent_user_prompt
from src.prompts.agent_prompts import AGENT_OUTPUT_SCHEMA_HINT
from src.utils.config_loader import AgentConfig, LLMConfig
from src.vlm.factory import get_llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT

# Ajanin ciktisi gecerli JSON degilse, JSON-modunda (invoke_json) verilecek
# yeniden-deneme talimati. 'JSON' kelimesi response_format=json_object icin gerekli.
_JSON_RETRY_INSTRUCTION = (
    "Onceki analizine dayanarak nihai kararini SADECE gecerli bir JSON nesnesi "
    "olarak ver (baska hicbir metin ekleme). Sema:\n" + AGENT_OUTPUT_SCHEMA_HINT
)

# Yapisal JSON ayristirma basarisiz olursa kullanilan geriye-donuk regex'ler
# (eski RISK_SKORU/AKSIYON_ONERISI bicimini ve mock ciktilarini korur).
_RISK_LINE_PATTERN = re.compile(r"(?:RISK_SKORU|risk_score)\D{0,4}(\d{1,3})", re.IGNORECASE)
_ACTION_LINE_PATTERN = re.compile(r"AKSIYON_ONERISI:\s*(.+)", re.IGNORECASE)
# Markdown ```json ... ``` (veya salt ``` ... ```) kod blogunun icerigini yakalar.
_MARKDOWN_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class AgentState(TypedDict):
    """LangGraph durum makinesinin dugumler arasinda tasidigi paylasimli durum."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int


@dataclass
class AgentDecision:
    """Ajan durum makinesinin nihai muhakeme sonucu (sartname JSON'u ile hizali).

    `risk_status`: "assessed" (risk_score gecerli bir 0-100 sayidir, GUVENILIR
    sekilde uretilmistir) | "unknown" (ajan muhakemesi basarisiz oldu VEYA
    modelin ciktisindan gecerli bir risk_score cikarilamadi — bu durumda
    `risk_score` HER ZAMAN `None`'dur). "unknown", ASLA "dusuk risk" ile
    KARISTIRILMAMALIDIR (bkz. `_degraded_decision`/`_coerce_risk_score`).
    """

    risk_score: Optional[int]
    risk_level: str
    recommended_action: str                       # geriye-uyum: actions[0]
    raw_response: str
    summary: str = ""                             # operatore yonelik Turkce durum ozeti
    actions: List[str] = field(default_factory=list)   # somut aksiyon onerileri listesi
    events: List[Dict[str, str]] = field(default_factory=list)  # [{"time": "MM:SS", "event": "..."}]
    risk_status: str = "assessed"                 # "assessed" | "unknown"
    onset_timestamp: Optional[str] = None
    safe_timestamps: List[str] = field(default_factory=list)
    incident_timestamps: List[str] = field(default_factory=list)
    triggered_mock_actions: List[Dict[str, Any]] = field(default_factory=list)
    """Ajanin bu calisma sirasinda GERCEKTEN cagirdigi mock aksiyon araclari
    (`notify_health_team_tool`/`dispatch_security_tool`/`trigger_area_lockdown_tool`
    - bkz. `src/agent/tools.py::MOCK_ACTION_TOOL_NAMES`). Her oge
    `{"tool": <arac adi>, "args": {...}, "result": <arac ciktisi>}` bicimindedir.
    Bu, sartnamenin "mock fonksiyonlarin ajanin araclari olarak BASARIYLA
    KULLANILMASI" gereksinimini somut/denetlenebilir kilar (bkz. `main.py::
    build_report`, `SafirReport.triggered_mock_actions`) - sql/retriever/
    timeline/verification gibi ic sorgu araclarindan farkli olarak burada
    ayrica izlenir, cunku bunlar disariya bir "eylem" temsil eder."""


class SafirAgent:
    """LangGraph tabanli durum makinesi uzerinde calisan risk muhakeme ajani.

    Dugumler:
        reasoning  -> "Hizli" LLM'i (`llm.active_model`) mevcut mesaj
                      gecmisiyle cagirir (arac secimi/JSON uretimi).
        tools      -> LLM'in istedigi arac cagrilarini yurutur.
        decision   -> `llm.decision_model` yapilandirilmissa "buyuk" modelle
                      TEK bir nihai karar-sentezi cagrisi yapar (bkz. modul
                      dokustringi "Model hiyerarsisi"); aksi halde no-op gecistir.

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
        self._use_json_mode = agent_config.guided_json

        self._tools: List[StructuredTool] = build_tool_registry(
            event_store, rag_service, agent_config.tools
        )
        self._llm = get_llm_client(llm_config, use_mock=use_mock_llm).bind_tools(self._tools)
        self._decision_llm = self._build_decision_llm(llm_config, use_mock_llm)

        self._tools_by_name = {tool.name: tool for tool in self._tools}
        self._graph = self._build_graph()

    @staticmethod
    def _build_decision_llm(llm_config: LLMConfig, use_mock_llm: bool):
        """`llm.decision_model` yapilandirilmissa, nihai karar sentezi icin ARACSIZ bir istemci kurar.

        Bkz. modul dokustringi "Model hiyerarsisi". `decision_model`
        tanimsiz/`active_model` ile ayniysa `None` doner - `_decision_node`
        bunu "hiyerarsi devre disi, no-op gecis" olarak yorumlar (davranis
        ONCEKI haliyle BIREBIR AYNI kalir, EK BIR API CAGRISI YAPILMAZ).

        Args:
            llm_config: `configs/config.yaml` icindeki `llm` blogu.
            use_mock_llm: `True` ise (GPU'suz mod) `decision_model` icin de
                gercek vLLM/EVREN'e baglanmadan `MockLLMClient` kurulur.

        Returns:
            Araclara BAGLANMAMIS bir LLM istemcisi veya `None`.
        """
        decision_model = llm_config.decision_model
        if not decision_model or decision_model == llm_config.active_model:
            return None
        if decision_model not in llm_config.models:
            logger.warning(
                "SafirAgent: llm.decision_model='%s' llm.models icinde tanimli degil; "
                "model hiyerarsisi devre disi, tek-model davranisina donuluyor.",
                decision_model,
            )
            return None
        decision_llm_config = llm_config.model_copy(update={"active_model": decision_model})
        return get_llm_client(decision_llm_config, use_mock=use_mock_llm)

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
        if not response.content and not getattr(response, "tool_calls", None):
            # Teshis: content VE tool_calls ikisi de bossa (ör. model butun
            # max_new_tokens butcesini gizli "dusunme" tokenlarina harcadiysa),
            # `finish_reason`/token kullanim bilgisi bunu dogrudan gosterir.
            logger.warning(
                "LLM yaniti bos (content ve tool_calls ikisi de bos). response_metadata=%s",
                getattr(response, "response_metadata", None),
            )
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
        """Hiyerarsi yapilandirilmissa, "buyuk" modelle TEK bir nihai karar-sentezi cagrisi yapar.

        `self._decision_llm is None` (varsayilan, hiyerarsi yapilandirilmamis)
        ise ONCEKI davranisla BIREBIR AYNI: hicbir sey yapmaz, son `reasoning`
        yanitinin icerigi `run()` tarafindan dogrudan nihai karar olarak
        ayristirilir. Yapilandirilmissa: dongu boyunca biriken TUM mesaj
        gecmisi (arac sonuclari dahil) + `DECISION_SYNTHESIS_INSTRUCTION`
        ile "buyuk" modele TEK bir ek (ARACSIZ) cagri yapilir; bu yanit
        `run()`in ayristiracagi nihai mesaj olur.

        Args:
            state: Mevcut ajan durumu (tool-routing dongusu tamamlanmis).

        Returns:
            `self._decision_llm is None` ise degisiklik yapilmamis durum;
            aksi halde "buyuk" modelin nihai karar mesajiyla guncellenmis durum.
        """
        if self._decision_llm is None:
            return state
        synthesis_prompt = HumanMessage(content=DECISION_SYNTHESIS_INSTRUCTION)
        response: AIMessage = self._decision_llm.invoke(list(state["messages"]) + [synthesis_prompt])
        return {"messages": [response], "iteration": state["iteration"]}

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

    def _resolve_risk_level(self, risk_score: Optional[int]) -> str:
        """Sayisal risk skorunu config esiklerine gore risk seviyesi etiketine cevirir.

        Args:
            risk_score: 0-100 arasi risk skoru; `None` ise (guvenilir bir skor
                uretilemedi) risk seviyesi de belirsizdir.

        Returns:
            `risk_score is None` ise `"unknown"`; aksi halde "dusuk", "orta",
            "yuksek" veya "kritik".
        """
        if risk_score is None:
            return "unknown"
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

        # Hata dayanikliligi: LLM/arac zinciri (retry'lardan sonra da) patlarsa,
        # is'i cokertmek yerine guvenli bir "degraded" karar dondurulur; operator
        # manuel incelemeye yonlendirilir (risk uydurulmaz).
        try:
            final_state = self._graph.invoke(initial_state)
            final_text = final_state["messages"][-1].content or ""
        except Exception as exc:  # noqa: BLE001 - degraded karara tasinir
            logger.exception("Ajan muhakemesi basarisiz; degraded karar donduruluyor.")
            return self._degraded_decision(str(exc))

        # Guided JSON: serbest cikti gecerli JSON degilse, JSON-modunda (vLLM/
        # Gemini response_format) TEK bir yeniden-deneme ile kurtarmayi dene.
        if self._use_json_mode and self._extract_json(final_text) is None:
            final_text = self._guided_json_retry(final_state["messages"], final_text)

        decision = self._parse_decision(final_text)
        decision.triggered_mock_actions = self._extract_triggered_mock_actions(final_state["messages"])
        return decision

    @staticmethod
    def _extract_triggered_mock_actions(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
        """Bu calistirmada GERCEKTEN cagirilan mock aksiyon araclarini (varsa) toplar.

        `_tools_node` her arac cagrisi icin bir `ToolMessage` uretir
        (`tool_call_id` ile eslesir) - burada yalnizca adi `MOCK_ACTION_TOOL_NAMES`
        icinde olan cagrilar (sql/retriever/timeline/verification gibi ic
        sorgu araclari HARIC) secilip ajanin cagirdigi arac adi/argumanlari ve
        arac ciktisiyla birlikte dondurulur (bkz. `AgentDecision.
        triggered_mock_actions` docstring'i).

        Args:
            messages: `self._graph.invoke(...)` sonrasi biriken tam mesaj gecmisi.

        Returns:
            `{"tool", "args", "result"}` sozlukleri listesi (cagri sirasiyla);
            hic mock aksiyon araci cagrilmadiysa bos liste.
        """
        tool_results: Dict[str, str] = {
            msg.tool_call_id: str(msg.content)
            for msg in messages
            if isinstance(msg, ToolMessage)
        }
        triggered: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            for call in getattr(msg, "tool_calls", None) or []:
                if call.get("name") not in MOCK_ACTION_TOOL_NAMES:
                    continue
                triggered.append(
                    {
                        "tool": call["name"],
                        "args": call.get("args", {}),
                        "result": tool_results.get(call.get("id"), ""),
                    }
                )
        return triggered

    def _degraded_decision(self, error: str) -> AgentDecision:
        """Muhakeme hatasinda operatoru manuel incelemeye yonlendiren guvenli karar.

        Risk skoru UYDURULMAZ: `risk_score=None`, `risk_status="unknown"` —
        (eskiden `0` idi; bu, "dusuk risk" ile "analiz basarisiz" ayrimini
        ORTADAN KALDIRIYORDU, bkz. P0 duzeltmesi). Ozet ve aksiyon, otomatik
        analizin BASARISIZ oldugunu ve insan mudahalesi gerektigini acikca belirtir.
        """
        return AgentDecision(
            risk_score=None,
            risk_level=self._resolve_risk_level(None),
            recommended_action="Otomatik muhakeme basarisiz; sahayi manuel inceleyin.",
            raw_response=f"[HATA] Ajan muhakemesi tamamlanamadi: {error}",
            summary="Otomatik risk muhakemesi bir hata nedeniyle tamamlanamadi; manuel inceleme gerekli.",
            actions=["Sahayi manuel inceleyin", "Sistemi/kayitlari kontrol edin"],
            events=[],
            risk_status="unknown",
        )

    def _guided_json_retry(self, messages: Sequence[BaseMessage], fallback_text: str) -> str:
        """Serbest cikti JSON degilse, JSON-modunda tek bir yeniden-deneme yapar.

        Ajanin biriktirdigi mesaj gecmisine bir formatlama talimati eklenip
        `LLMClient.invoke_json` (araci baglanmamis, `response_format=json_object`)
        ile cagrilir. Yeni cikti gecerli JSON ise o dondurulur; degilse veya
        cagri patlarsa orijinal `fallback_text` korunur (regex fallback devreye girer).

        Args:
            messages: LangGraph durum makinesinin son mesaj gecmisi.
            fallback_text: Yeniden-deneme basarisiz olursa korunacak orijinal cikti.

        Returns:
            Gecerli JSON iceren yeni metin veya `fallback_text`.
        """
        logger.info("Ajan ciktisi gecerli JSON degil; JSON-modu ile yeniden deneniyor.")
        try:
            retry_messages = list(messages) + [HumanMessage(content=_JSON_RETRY_INSTRUCTION)]
            # Nihai karari fiilen KIM uretmisse (hiyerarsi aktifse "buyuk"
            # decision_llm, degilse hizli self._llm) retry de AYNI modelle
            # yapilir - farkli bir modele sessizce gecmek TUTARSIZ olurdu.
            retry_llm = self._decision_llm or self._llm
            response = retry_llm.invoke_json(retry_messages)
            content = response.content or ""
            if self._extract_json(content) is not None:
                return content
            logger.warning(
                "JSON-modu yeniden denemesi de gecerli JSON uretmedi. Ham cikti: %s",
                self._truncate_for_log(content),
            )
        except Exception as exc:  # noqa: BLE001 - retry best-effort; hata fallback'e dusurulur
            logger.warning("JSON-modu yeniden denemesi basarisiz: %s", exc)
        return fallback_text

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
            esiklerinden yeniden hesaplanir (modelin iddia ettigi seviyeye
            guvenilmez). `risk_score` gecerli sekilde cikarilamazsa (JSON'da
            alan eksik/gecersiz VEYA regex hic eslesme bulamazsa) `None` doner
            ve `risk_status="unknown"` olur — ASLA `0`'a (dusuk risk) SESSIZCE
            DUSMEZ (bkz. P0 duzeltmesi).
        """
        parsed = self._extract_json(final_text)

        if parsed is not None:
            risk_score = self._coerce_risk_score(parsed.get("risk_score"))
            summary = str(parsed.get("summary", "")).strip()
            actions = self._coerce_actions(parsed.get("actions"))
            events = self._coerce_events(parsed.get("events"))
            onset_timestamp = parsed.get("onset_timestamp")
            safe_timestamps = parsed.get("safe_timestamps") or []
            incident_timestamps = parsed.get("incident_timestamps") or []
        else:
            logger.warning(
                "Ajan yaniti JSON olarak ayristirilamadi, regex fallback kullaniliyor. Ham cikti: %s",
                self._truncate_for_log(final_text),
            )
            risk_match = _RISK_LINE_PATTERN.search(final_text)
            action_match = _ACTION_LINE_PATTERN.search(final_text)
            risk_score = self._coerce_risk_score(risk_match.group(1) if risk_match else None)
            summary = ""
            single_action = action_match.group(1).strip() if action_match else ""
            actions = [single_action] if single_action else []
            events = []
            onset_timestamp = None
            safe_timestamps = []
            incident_timestamps = []

        recommended_action = actions[0] if actions else "Ek aksiyon onerisi uretilemedi."
        raw_status = parsed.get("risk_status") if parsed else None
        if raw_status in ("assessed", "unclassified", "unknown"):
            risk_status = raw_status
        else:
            risk_status = "assessed" if risk_score is not None else "unknown"

        if risk_status == "unclassified":
            risk_score = None

        return AgentDecision(
            risk_score=risk_score,
            risk_level=self._resolve_risk_level(risk_score),
            recommended_action=recommended_action,
            raw_response=final_text,
            summary=summary,
            actions=actions,
            events=events,
            risk_status=risk_status,
            onset_timestamp=onset_timestamp,
            safe_timestamps=safe_timestamps,
            incident_timestamps=incident_timestamps,
        )

    @staticmethod
    def _truncate_for_log(text: str, limit: int = 1000) -> str:
        """Ham LLM ciktisini teshis amacli loglamak icin guvenli/kirpilmis `repr` uretir.

        `repr` kullanmak, kacan/goze gorunmez karakterleri (ör. yanlislikla
        eklenmis `\\n`/kontrol karakteri, cikti kesilmesi) log satirinda
        AYIRT EDILEBILIR kilar - duz metin bunlari gizleyip teshisi zorlastirirdi.
        """
        if not text:
            return "(bos)"
        snippet = text if len(text) <= limit else text[:limit] + f"...(+{len(text) - limit} karakter)"
        return repr(snippet)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Serbest metin icindeki ilk gecerli JSON nesnesini ayristirir; bulunamazsa None.

        Sirasiyla dener:
        1. Dogrudan `json.loads` (model SADECE JSON dondurmusse - beklenen yol).
        2. Markdown ```json ... ``` kod blogunun icerigi.
        3. Metnin herhangi bir yerindeki (baslik/aciklama ile sarili olsa da)
           ilk DENGELI (balanced-brace) JSON nesnesi.

        Eskiden kullanilan `\\{.*\\}` (greedy) regex'i, metinde birden fazla
        `{`/`}` oldugunda (ör. kod blogu + sonrasinda ek aciklama, ic ice
        yapi) YANLIS sinirlar secip gecerli JSON'u bile bozuk hale
        getirebiliyordu; bu yuzden parantez derinligi + string/escape
        farkindaligiyla calisan `_extract_balanced_json_object` kullanilir.
        """
        if not text:
            return None

        direct = SafirAgent._try_parse_json_object(text.strip())
        if direct is not None:
            return direct

        fence_match = _MARKDOWN_JSON_FENCE_PATTERN.search(text)
        if fence_match:
            fenced_content = fence_match.group(1)
            fenced = SafirAgent._try_parse_json_object(fenced_content.strip())
            if fenced is not None:
                return fenced
            balanced_in_fence = SafirAgent._try_parse_json_object(
                SafirAgent._extract_balanced_json_object(fenced_content)
            )
            if balanced_in_fence is not None:
                return balanced_in_fence

        balanced = SafirAgent._try_parse_json_object(SafirAgent._extract_balanced_json_object(text))
        if balanced is not None:
            return balanced

        return None

    @staticmethod
    def _try_parse_json_object(candidate: Optional[str]) -> Optional[Dict[str, Any]]:
        """`candidate` gecerli bir JSON NESNESI ise sozluk olarak dondurur, aksi halde None."""
        if not candidate:
            return None
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _extract_balanced_json_object(text: str) -> Optional[str]:
        """Metindeki ilk `{`den baslayarak parantez derinligini sayip DENGELI JSON nesnesini bulur.

        String icindeki `{`/`}` karakterlerini (ve kacis/escape edilmis
        tirnaklari) yok sayar, boylece iceriginde suslu parantez GECEN bir
        alan degeri (ör. `"summary": "... {ozel} ..."`) nesnenin erken
        kapandigini varsaymaz. Bulunamazsa (dengelenmemis/kirik JSON) None doner.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @staticmethod
    def _coerce_risk_score(value: Any) -> Optional[int]:
        """Risk skorunu 0-100 araligina kirpilmis bir tam sayiya cevirir.

        Deger eksik/gecersizse (ör. `None`, sayisal olmayan bir metin) artik
        `0` DEGIL, `None` doner — cagiran taraf (`_parse_decision`) bunu
        `risk_status="unknown"`e cevirir. Onceki davranis (`0` varsayilani),
        "gecerli bir dusuk risk skoru" ile "skor hic uretilemedi" durumunu
        AYIRT EDEMIYORDU (bkz. P0 duzeltmesi).
        """
        try:
            score = int(float(value))
        except (TypeError, ValueError):
            return None
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
