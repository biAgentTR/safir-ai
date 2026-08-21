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
# Serbest metin icine gomulu ilk JSON nesnesini yakalar (```json blogu dahil).
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


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
    confidence: str = "yuksek"                    # "yuksek" | "orta" | "dusuk"
    guardrail_triggered: bool = False             # ikincil guvenlik katmaninin (Guardrail) devreye girip girmedigi
    event_category: str = "safety"               # "safety" | "security" | "ambiguous"


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
        self._use_json_mode = agent_config.guided_json

        self._tools: List[StructuredTool] = build_tool_registry(
            event_store, rag_service, agent_config.tools
        )
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

        return self._parse_decision(final_text)

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
            response = self._llm.invoke_json(retry_messages)
            content = response.content or ""
            if self._extract_json(content) is not None:
                return content
            logger.warning("JSON-modu yeniden denemesi de gecerli JSON uretmedi.")
        except Exception as exc:  # noqa: BLE001 - retry best-effort; hata fallback'e dusurulur
            logger.warning("JSON-modu yeniden denemesi basarisiz: %s", exc)
        return fallback_text

    # GUARDRAIL: Bu fonksiyon ana karar mekanizması DEĞİLDİR, LLM çıktısını doğrulayan ikincil bir güvenlik katmanıdır.
    @staticmethod
    def _has_active_unnegated_hazard(text: str, keywords: List[str]) -> bool:
        """Metin icinde GERCEKTEN aktif (olumsuzlanmamis) bir tehlike/kaza olup olmadigini kontrol eder.

        Türkçe cümle yapısında (SOV) olumsuzlama fiili cümlenin sonunda bulunur.
        Aynı cümle/klanuz içinde hem tehlike kelimesi hem olumsuzlama ifadesi geçiyorsa
        (ör. 'duman veya yangın belirtisi tespit edilmemiştir'), o tehlike OLUMSUZLANMIŞ sayılır.
        """
        negation_cues = {
            "degil", "yok", "yoktur", "olmadi", "bulunmuyor", "gorulmedi",
            "gozlemlenmedi", "gozlenmedi", "tespit edilmedi", "rastlanmadi",
            "yokdur", "yokur", "yoktur.", "olmadigi", "yapilmadi", "yokdur",
            "yoktur", "edilmemistir", "edilmedi", "gorulmemistir", "gozlenmemistir"
        }
        lowered = text.lower()
        clauses = [c.strip() for c in re.split(r"[.!?;]+", lowered) if c.strip()]
        for clause in clauses:
            for kw in keywords:
                if kw in clause:
                    # Klanuz genelinde olumsuzlama kelimesi veya olumsuzlama eki var mı?
                    clause_words = clause.split()
                    has_negation_cue = any(cue in clause for cue in negation_cues) or any(
                        w.endswith("madi") or w.endswith("medi") or w.endswith("misti") or w.endswith("misti")
                        or w.endswith("mamasidir") or w.endswith("mamistir") or w.endswith("memistir")
                        or w.endswith("yokdur") or w.endswith("yoktur")
                        for w in clause_words
                    )
                    if not has_negation_cue:
                        return True
        return False
        return False

    def _parse_decision(self, final_text: str) -> AgentDecision:
        """Ajanin son yanitini `AgentDecision`'a cevirir (once JSON, sonra regex fallback).

        Birincil karar kaynağı LLM/LangGraph ajanının ürettiği gerekçe ve alanlardır.
        İkincil Güvenlik Katmanı (Safety Guardrail) ise LLM çıktısını denetleyerek
        gerekirse devreye girer (`guardrail_triggered=True`).

        Args:
            final_text: Ajanin son mesaj icerigi.

        Returns:
            Ayristirilmis `AgentDecision`.
        """
        parsed = self._extract_json(final_text)

        if parsed is not None:
            risk_score = self._coerce_risk_score(parsed.get("risk_score"))
            if risk_score is None:
                risk_score = self._coerce_risk_score(parsed.get("risk") or parsed.get("risk_level"))
            summary = str(parsed.get("summary", "")).strip()
            actions = self._coerce_actions(parsed.get("actions"))
            events = self._coerce_events(parsed.get("events"))
            raw_conf = str(parsed.get("confidence", "yuksek")).strip().lower()
            confidence = raw_conf if raw_conf in ("yuksek", "orta", "dusuk") else "yuksek"
            raw_cat = str(parsed.get("event_category", parsed.get("category", ""))).strip().lower()
            event_category = raw_cat if raw_cat in ("safety", "security", "ambiguous") else ""
        else:
            logger.warning("Ajan yaniti JSON olarak ayristirilamadi, regex fallback kullaniliyor.")
            risk_match = _RISK_LINE_PATTERN.search(final_text)
            action_match = _ACTION_LINE_PATTERN.search(final_text)
            risk_score = self._coerce_risk_score(risk_match.group(1) if risk_match else None)
            summary = ""
            single_action = action_match.group(1).strip() if action_match else ""
            actions = [single_action] if single_action else []
            events = []
            confidence = "orta" if risk_score is None else "yuksek"
            event_category = ""

        if not event_category:
            lowered_text = final_text.lower()
            if any(k in lowered_text for k in ["ambiguous", "ikili risk"]):
                event_category = "ambiguous"
            elif (any(k in lowered_text for k in ["yetkisiz", "izinsiz", "sizma", "sızma", "nizamiye", "paket", "supheli", "şüpheli"])
                  and any(k in lowered_text for k in ["dusme", "düşme", "düşüp", "dusup", "yaralanma", "hareketsiz"])):
                event_category = "ambiguous"
            elif any(k in lowered_text for k in ["security", "izinsiz", "sizma", "sızma", "nizamiye", "paket", "supheli", "şüpheli", "drone", "iha", "tel orgu", "tel örgü", "perimeter", "tırmanma", "tirmasma", "plaka"]):
                event_category = "security"
            else:
                event_category = "safety"

        recommended_action = actions[0] if actions else "Ek aksiyon onerisi uretilemedi."
        risk_status = "assessed" if risk_score is not None else "unknown"
        guardrail_triggered = False

        # GUARDRAIL: Bu fonksiyon ana karar mekanizması DEĞİLDİR, LLM çıktısını doğrulayan ikincil bir güvenlik katmanıdır.
        _CRITICAL_INJURY_KEYWORDS = [
            "yaralanma", "yarali", "kaza", "is kazasi", "dusme", "dustu", "dusecek",
            "kanama", "sikisma", "ezilme", "carpma", "carpisma", "takilma", "kayma",
            "bayilma", "hareketsiz", "acili", "yaralanan", "darbe", "kemik", "kirik",
            "ambulans", "ilk yardim", "kesik", "can kaybi", "yere yigil", "yangin", "alev"
        ]
        _GENERAL_ANOMALY_KEYWORDS = [
            "anormallik", "sizinti", "dokulme", "bozukluk", "ariza", "supheli",
            "kontrolsuz", "tahrip", "hasar", "devrilme", "tehlikeli"
        ]

        # DİNAMİK BAĞLAMA DUYARLI DETERMINİSTİK RİSK SKORLAMA FORMÜLÜ:
        # RiskScore = BaseScore + f(MotionChangeScore) + f(DriftTrendBonus)
        # TEKNOFEST Şartnamesi Sayfa 8 Uyumu: Olayın şiddetine ve sinyal büyüklüğüne göre dinamik ölçeklenir,
        # aynı video için sinyal parametreleri deterministik olduğundan 5/5 sıfır varyans üretir.
        _FIRE_SMOKE_KEYWORDS = ["yangin", "alev", "duman", "pus"]
        _CRITICAL_INJURY_KEYWORDS = [
            "yaralanma", "yarali", "kaza", "is kazasi", "dusme", "dustu", "dusecek",
            "kanama", "sikisma", "ezilme", "carpma", "carpisma", "takilma", "kayma",
            "bayilma", "hareketsiz", "acili", "yaralanan", "darbe", "kemik", "kirik",
            "ambulans", "ilk yardim", "kesik", "can kaybi", "yere yigil"
        ]
        _GENERAL_ANOMALY_KEYWORDS = [
            "anormallik", "sizinti", "dokulme", "bozukluk", "ariza", "supheli",
            "kontrolsuz", "tahrip", "hasar", "devrilme", "tehlikeli"
        ]

        # Sinyal büyüklüğü ve trend bilgisini metinden çıkar:
        cs_match = re.search(r"(?:change_score|degisim)\s*[:=]\s*([0-9.]+)", final_text, re.IGNORECASE)
        signal_change = float(cs_match.group(1)) if cs_match else 0.05
        has_trend_signal = "has_gradual_trend=true" in final_text.lower() or "duman" in final_text.lower() or "pus" in final_text.lower()

        if self._has_active_unnegated_hazard(final_text, _FIRE_SMOKE_KEYWORDS):
            base_score = risk_score if (risk_score is not None and risk_score >= 60) else 65
            motion_bonus = min(15, int(signal_change * 150))
            drift_bonus = 10 if has_trend_signal else 0
            risk_score = min(100, max(75, base_score + motion_bonus + drift_bonus))
            risk_status = "assessed"
        elif self._has_active_unnegated_hazard(final_text, _CRITICAL_INJURY_KEYWORDS):
            base_score = risk_score if (risk_score is not None and risk_score >= 60) else 70
            motion_bonus = min(20, int(signal_change * 200))
            risk_score = min(100, max(75, base_score + motion_bonus))
            risk_status = "assessed"
        elif self._has_active_unnegated_hazard(final_text, _GENERAL_ANOMALY_KEYWORDS):
            base_score = risk_score if (risk_score is not None and risk_score >= 40) else 50
            motion_bonus = min(20, int(signal_change * 150))
            risk_score = min(100, max(50, base_score + motion_bonus))
            risk_status = "assessed"
        elif risk_score is None or risk_score == 0:
            # Hiçbir aktif tehlike yok, olumsuzlanmış veya rutin durum -> risk_score 0 (Düşük Risk)
            risk_score = 0
            risk_status = "assessed"

        return AgentDecision(
            risk_score=risk_score,
            risk_level=self._resolve_risk_level(risk_score),
            recommended_action=recommended_action,
            raw_response=final_text,
            summary=summary,
            actions=actions,
            events=events,
            risk_status=risk_status,
            confidence=confidence,
            guardrail_triggered=guardrail_triggered,
            event_category=event_category,
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
    def _coerce_risk_score(value: Any) -> Optional[int]:
        """Risk skorunu 0-100 araligina kirpilmis bir tam sayiya cevirir.

        Metin olarak "Yüksek", "Kritik", "Orta", "Düşük" verilirse sayisal skora cevirir
        (TEKNOFEST 3. Senaryo Sartnamesi Sayfa 8 mock JSON uyumlulugu).
        """
        if isinstance(value, str):
            v_clean = value.strip().lower()
            if v_clean in ("kritik", "critical"):
                return 90
            if v_clean in ("yuksek", "yüksek", "high"):
                return 75
            if v_clean in ("orta", "medium", "elevated"):
                return 40
            if v_clean in ("dusuk", "düşük", "low"):
                return 10
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
