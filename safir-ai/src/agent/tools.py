"""05 - Dynamic Tool Router: ajanin dinamik olarak cagirabilecegi araclar.

Bu modul, `Reasoning Agent`in baglama gore yonlendirdigi araclari tanimlar:
gecmis olay sorgulama (SQL Tool), zamansal cizelgeleme (Timeline Tool),
ISG/operasyonel mevzuat aramasi (`retriever_tool`, Embedding & RAG Katmani
uzerinden) ve bir risk iddiasini capraz-dogrulayan `verification_tool`.
Bunlarin hepsi ic/salt-okuma araclardir - ajanin baglamini zenginlestirir
ama disariya hicbir eylem tetiklemez.

Ayrica, sartnamenin acikca istedigi "mock fonksiyonlarin ajanin araclari
olarak kullanilmasi" gereksinimini karsilayan UC MOCK AKSIYON ARACI da
burada tanimlanir (bkz. `MOCK_ACTION_TOOL_NAMES`): `notify_health_team_tool`,
`dispatch_security_tool`, `trigger_area_lockdown_tool`. Bunlar, yukaridaki
sorgu araclarindan farkli olarak, ajanin SAHADA bir eylemi (saglik ekibi
bilgilendirme, guvenlik bildirimi, alan tahliye/kilitleme) SIMULE ETMEK icin
kendi karariyla cagirdigi araclardir - gercek bir dis sisteme baglanmazlar
(adlarindan da anlasilacagi gibi mock'tur), yalnizca loglayip bir onay/kayit
metni dondururler. Bilincli olarak `EscalationPolicy`den (`src/decision/
escalation.py`) AYRIDIR: EscalationPolicy, risk skoruna gore OTOMATIK ve
deterministik olarak saha alarmini tetikler (ajanin bir karari degildir);
bu mock araclar ise ajanin KENDI muhakemesiyle, bir aksiyon onerisini somut
bir arac-cagrisi olarak da disa vurmasini saglar (bkz. `SafirAgent`'in sistem
istemindeki "Mock Aksiyon Araclari" politikasi).

Araclar, gercek `EventStore`/`EmbeddingRAGService` bagimliliklari verilmezse
bile calisabilen mock veriye duser; boylece iskelet bagimsiz test edilebilir.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.rag.embedding_rag_service import EmbeddingRAGService
from src.memory.event_store import EventStore

logger = logging.getLogger(__name__)

_MOCK_RECENT_EVENTS = [
    {"timestamp": 12.4, "description": "Sahada bir personel korumasiz alanda hareket etti.", "risk_level": "orta"},
    {"timestamp": 45.1, "description": "Forklift, yaya gecidine yaklasti.", "risk_level": "yuksek"},
]

_MOCK_REGULATIONS = [
    "ISG Yonetmeligi Madde 12: Yuksekte calisma alanlarinda dusme onleyici ekipman zorunludur.",
    "Operasyonel Kural OK-07: Forklift trafiginde yaya gecitleri her zaman acik tutulmalidir.",
]


class SqlToolInput(BaseModel):
    """SQL Tool icin girdi semasi."""

    query_type: str = Field(
        description="Sorgu turu: 'recent' (son olaylar) veya 'risk_level' (risk seviyesine gore)."
    )
    risk_level: Optional[str] = Field(
        default=None, description="query_type='risk_level' ise filtrelenecek seviye (dusuk/orta/yuksek/kritik)."
    )
    limit: int = Field(default=5, description="Dondurulecek maksimum kayit sayisi.")


class SqlTool:
    """Gecmis olaylari yapilandirilmis olay bellegi (SQLite) uzerinden sorgulayan arac.

    `event_store` verilmezse mock veri uzerinden calisir; bu, gelistirme ve
    testte harici bagimlilik olmadan ajan dongusunu dogrulamayi saglar.
    """

    def __init__(self, event_store: Optional[EventStore] = None) -> None:
        """SqlTool'u opsiyonel bir `EventStore` bagimliligiyla baslatir.

        Args:
            event_store: Gercek SQLite tabanli olay deposu; `None` ise mock veri kullanilir.
        """
        self._event_store = event_store

    def run(self, query_type: str, risk_level: Optional[str] = None, limit: int = 5) -> str:
        """SQL Tool'u calistirir ve sonucu okunabilir metin olarak dondurur.

        Args:
            query_type: "recent" veya "risk_level".
            risk_level: `query_type="risk_level"` icin filtre degeri.
            limit: Maksimum sonuc sayisi.

        Returns:
            Sorgu sonucunu ozetleyen dogal dil metni.
        """
        if self._event_store is None:
            logger.warning("SqlTool: EventStore baglanmadi, mock veri donduruluyor.")
            events = _MOCK_RECENT_EVENTS[:limit]
        elif query_type == "risk_level" and risk_level:
            events = self._event_store.query_by_risk_level(risk_level, limit=limit)
        else:
            events = self._event_store.query_recent(limit=limit)

        if not events:
            return "Sorguya uygun gecmis olay bulunamadi."

        lines = [f"- [{e['timestamp']:.1f}s] {e['description']} (risk: {e.get('risk_level', 'bilinmiyor')})" for e in events]
        return "Gecmis olaylar:\n" + "\n".join(lines)

    def as_langchain_tool(self) -> StructuredTool:
        """Bu araci LangGraph/LangChain ajanina baglanabilecek `StructuredTool`'a cevirir."""
        return StructuredTool.from_function(
            func=self.run,
            name="sql_tool",
            description="Gecmis olaylari zaman veya risk seviyesine gore sorgular.",
            args_schema=SqlToolInput,
        )


class RetrieverToolInput(BaseModel):
    """`retriever_tool` icin girdi semasi."""

    question: str = Field(description="ISG/operasyonel mevzuat ile ilgili dogal dil sorusu.")
    top_k: int = Field(default=3, description="Getirilecek maksimum mevzuat maddesi sayisi.")


class RetrieverTool:
    """ISG mevzuati ve operasyonel kurallari Embedding & RAG Katmani (`EmbeddingRAGService`) uzerinden arayan arac."""

    def __init__(self, rag_service: Optional[EmbeddingRAGService] = None) -> None:
        """RetrieverTool'u opsiyonel bir `EmbeddingRAGService` bagimliligiyla baslatir.

        Args:
            rag_service: Gercek embedding+FAISS tabanli RAG servisi; `None` ise mock veri kullanilir.
        """
        self._rag_service = rag_service

    def run(self, question: str, top_k: int = 3) -> str:
        """`retriever_tool`'u calistirir ve ilgili mevzuat maddelerini kaynak metadata'siyla birlikte dondurur.

        Args:
            question: Dogal dil sorgusu.
            top_k: Maksimum sonuc sayisi.

        Returns:
            Her sonuc icin kaynak basligi/madde/sayfa/skor/URL + metni iceren,
            okunabilir metin. NOT: bu metin risk_score hesaplamasina ASLA
            girmez - yalnizca ajanin muhakeme baglamini zenginlestirir.
        """
        if self._rag_service is None:
            logger.warning("RetrieverTool: EmbeddingRAGService baglanmadi, mock veri donduruluyor.")
            regulations = _MOCK_REGULATIONS[:top_k]
            if not regulations:
                return "Sorguyla ilgili mevzuat maddesi bulunamadi."
            return "Ilgili mevzuat:\n" + "\n".join(f"- {r}" for r in regulations)

        results = self._rag_service.query(question, top_k=top_k)
        if not results:
            return "Sorguyla ilgili mevzuat maddesi bulunamadi."

        blocks = []
        for doc in results:
            # `getattr(..., None)` KASITLI: eski/duck-typed (yalnizca text/score
            # tasiyan) sahte RAG nesneleri icin de GUVENLI CALISIR - bkz.
            # `context_builder.py::_format_semantic_chunks` ile ayni gerekce.
            relevance_score = getattr(doc, "relevance_score", None)
            embedding_score = getattr(doc, "embedding_score", None)
            if relevance_score is not None:
                score_line = f"Relevance score: {relevance_score:.3f}"
            elif embedding_score is not None:
                score_line = f"Embedding score: {embedding_score:.3f}"
            else:
                score_line = f"Score: {getattr(doc, 'score', 0.0):.3f}"

            document_title = getattr(doc, "document_title", None)
            document_id = getattr(doc, "document_id", None)
            lines = [f"Kaynak: {document_title or document_id or '(bilinmeyen kaynak)'}"]
            article_number = getattr(doc, "article_number", None)
            if article_number:
                lines.append(f"Madde: {article_number}")
            page_start = getattr(doc, "page_start", None)
            if page_start:
                page_end = getattr(doc, "page_end", None)
                page = f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"
                lines.append(f"Sayfa: {page}")
            lines.append(score_line)
            source_url = getattr(doc, "source_url", None)
            if source_url:
                lines.append(f"Kaynak URL: {source_url}")
            lines.append(f"Metin:\n{doc.text}")
            blocks.append("\n".join(lines))

        return "Ilgili mevzuat:\n\n" + "\n\n".join(blocks)

    def as_langchain_tool(self) -> StructuredTool:
        """Bu araci LangGraph/LangChain ajanina baglanabilecek `StructuredTool`'a cevirir."""
        return StructuredTool.from_function(
            func=self.run,
            name="retriever_tool",
            description=(
                "ISG mevzuati ve operasyonel kurallari, embedding tabanli anlamsal "
                "arama (Embedding & RAG Katmani) ile arar."
            ),
            args_schema=RetrieverToolInput,
        )


class TimelineToolInput(BaseModel):
    """Timeline Tool icin girdi semasi."""

    start_ts: float = Field(description="Zaman araligi baslangici (saniye).")
    end_ts: float = Field(description="Zaman araligi bitisi (saniye).")


class VerificationToolInput(BaseModel):
    """`verification_tool` icin girdi semasi."""

    claim: str = Field(description="Dogrulanacak risk/gozlem iddiasi (dogal dil).")
    event_type: Optional[str] = Field(
        default=None, description="Ilgili olay kategorisi (varsa; orn. 'arac_yaya_yakinligi')."
    )


class VerificationTool:
    """Ajanin bir risk iddiasini, ISG mevzuati ve gecmis olay emsaliyle capraz-dogrulayan arac.

    Ajan, ozellikle yuksek/kritik bir risk skorunu sonuclandirmadan once bu
    araci cagirarak iddiasini iki kaynaga karsi test edebilir: (1) ilgili
    mevzuat maddesi var mi (RAG), (2) benzer olaylar gecmiste kaydedilmis mi
    (SQLite). Bu, kararin kanit ve mevzuatla temellenmesini saglar
    (sartname: cok adimli karar zinciri / dogrulama). `event_store`/`rag_service`
    verilmezse mock veriye duser; boylece bagimsiz test edilebilir.
    """

    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        rag_service: Optional[EmbeddingRAGService] = None,
    ) -> None:
        """VerificationTool'u opsiyonel olay deposu ve RAG servisiyle baslatir."""
        self._event_store = event_store
        self._rag_service = rag_service

    def run(self, claim: str, event_type: Optional[str] = None) -> str:
        """Iddiayi mevzuat destegi ve gecmis emsal acisindan dogrular.

        Args:
            claim: Dogrulanacak dogal dil iddiasi.
            event_type: Ilgili olay kategorisi (varsa).

        Returns:
            Mevzuat destegi (var/yok) ve gecmis emsal sayisini iceren dogrulama ozeti.
        """
        if self._rag_service is None:
            regulations = _MOCK_REGULATIONS[:2]
        else:
            regulations = [doc.text for doc in self._rag_service.query(claim, top_k=2)]

        if self._event_store is None:
            recent_events = _MOCK_RECENT_EVENTS
        else:
            recent_events = self._event_store.query_recent(limit=5)

        support = "VAR" if regulations else "YOK"
        lines = [
            f"Dogrulanan iddia: {claim}",
            f"Ilgili kategori: {event_type or 'belirtilmedi'}",
            f"Mevzuat destegi: {support}",
        ]
        lines.extend(f"  - {r}" for r in regulations)
        lines.append(f"Gecmis emsal (son kayitlar): {len(recent_events)} olay")
        return "\n".join(lines)

    def as_langchain_tool(self) -> StructuredTool:
        """Bu araci LangGraph/LangChain ajanina baglanabilecek `StructuredTool`'a cevirir."""
        return StructuredTool.from_function(
            func=self.run,
            name="verification_tool",
            description=(
                "Bir risk iddiasini ISG mevzuati (RAG) ve gecmis olay emsali (SQLite) ile "
                "capraz-dogrular. Yuksek/kritik risk karari vermeden once kullan."
            ),
            args_schema=VerificationToolInput,
        )


class TimelineTool:
    """Belirli bir zaman araligindaki olaylari kronolojik olarak cizelgeleyen arac."""

    def __init__(self, event_store: Optional[EventStore] = None) -> None:
        """TimelineTool'u opsiyonel bir `EventStore` bagimliligiyla baslatir.

        Args:
            event_store: Gercek SQLite tabanli olay deposu; `None` ise mock veri kullanilir.
        """
        self._event_store = event_store

    def run(self, start_ts: float, end_ts: float) -> str:
        """Timeline Tool'u calistirir ve zaman araligindaki olaylari dondurur.

        Args:
            start_ts: Aralik baslangici (saniye).
            end_ts: Aralik bitisi (saniye).

        Returns:
            Kronolojik siraya gore olay listesini iceren metin.
        """
        if self._event_store is None:
            logger.warning("TimelineTool: EventStore baglanmadi, mock veri donduruluyor.")
            events = [e for e in _MOCK_RECENT_EVENTS if start_ts <= e["timestamp"] <= end_ts]
        else:
            events = self._event_store.get_timeline(start_ts, end_ts)

        if not events:
            return f"[{start_ts:.1f}s - {end_ts:.1f}s] araliginda olay bulunamadi."

        lines = [f"- [{e['timestamp']:.1f}s] {e['description']}" for e in events]
        return f"[{start_ts:.1f}s - {end_ts:.1f}s] zaman cizelgesi:\n" + "\n".join(lines)

    def as_langchain_tool(self) -> StructuredTool:
        """Bu araci LangGraph/LangChain ajanina baglanabilecek `StructuredTool`'a cevirir."""
        return StructuredTool.from_function(
            func=self.run,
            name="timeline_tool",
            description="Belirli bir zaman araligindaki olaylari kronolojik olarak listeler.",
            args_schema=TimelineToolInput,
        )


class NotifyHealthTeamToolInput(BaseModel):
    """`notify_health_team_tool` icin girdi semasi."""

    event_id: str = Field(description="Ilgili olayin kimligi veya kisa tanimi (orn. 'dusme_00:18').")
    urgency: str = Field(description="Aciliyet seviyesi: 'dusuk' | 'orta' | 'yuksek' | 'kritik'.")
    note: Optional[str] = Field(default=None, description="Saglik ekibine iletilecek kisa ek not (varsa).")


class NotifyHealthTeamTool:
    """Saglik ekibini olay hakkinda bilgilendiren MOCK aksiyon araci.

    Gercek bir dispatch/SMS/telsiz sistemine baglanmaz - sartnamenin istedigi
    "mock fonksiyon" desenini karsilar: ajan, yaralanma/dusme/bilinc kaybi gibi
    bir gozlemde bu araci KENDI karariyla cagirabilir; sonuc, ajanin mesaj
    gecmisine bir `ToolMessage` olarak doner ve `SafirAgent.run()` tarafindan
    `AgentDecision.triggered_mock_actions`e toplanip rapora tasinir (bkz.
    `main.py::build_report`, `SafirReport.triggered_mock_actions`).
    """

    def run(self, event_id: str, urgency: str, note: Optional[str] = None) -> str:
        """Aracin mock cagrisini yapar ve dogal dil bir onay/kayit metni dondurur."""
        logger.warning(
            "MOCK AKSIYON: saglik ekibi bilgilendirildi (event_id=%s, urgency=%s, note=%s)",
            event_id,
            urgency,
            note or "-",
        )
        return (
            f"[MOCK] Saglik ekibi bilgilendirildi. Olay: {event_id}, aciliyet: {urgency}."
            + (f" Not: {note}" if note else "")
        )

    def as_langchain_tool(self) -> StructuredTool:
        """Bu araci LangGraph/LangChain ajanina baglanabilecek `StructuredTool`'a cevirir."""
        return StructuredTool.from_function(
            func=self.run,
            name="notify_health_team_tool",
            description=(
                "MOCK arac: yaralanma, dusme, bilinc kaybi gibi saglikla ilgili acil bir gozlem "
                "oldugunda saglik ekibini bilgilendirir. Gercek bir dis sisteme baglanmaz "
                "(simule edilmis saha aksiyonu); cagrilmasi rapora islenir."
            ),
            args_schema=NotifyHealthTeamToolInput,
        )


class DispatchSecurityToolInput(BaseModel):
    """`dispatch_security_tool` icin girdi semasi."""

    zone: str = Field(description="Guvenlik ekibinin yonlendirilecegi saha/bolge tanimi.")
    reason: str = Field(description="Yonlendirme gerekcesi (dogal dil, orn. 'yetkisiz erisim').")


class DispatchSecurityTool:
    """Guvenlik ekibini bir bolgeye yonlendiren MOCK aksiyon araci.

    Bkz. `NotifyHealthTeamTool` docstring'i - ayni mock desen: gercek bir dis
    sisteme baglanmaz, ajanin kendi karariyla (orn. yetkisiz erisim, ekipman
    kaybi, guvenlik ihlali gozlemlerinde) cagirabilecegi simule edilmis bir
    saha eylemidir.
    """

    def run(self, zone: str, reason: str) -> str:
        """Aracin mock cagrisini yapar ve dogal dil bir onay/kayit metni dondurur."""
        logger.warning("MOCK AKSIYON: guvenlik ekibi yonlendirildi (zone=%s, reason=%s)", zone, reason)
        return f"[MOCK] Guvenlik ekibi '{zone}' bolgesine yonlendirildi. Gerekce: {reason}"

    def as_langchain_tool(self) -> StructuredTool:
        """Bu araci LangGraph/LangChain ajanina baglanabilecek `StructuredTool`'a cevirir."""
        return StructuredTool.from_function(
            func=self.run,
            name="dispatch_security_tool",
            description=(
                "MOCK arac: yetkisiz erisim, guvenlik ihlali veya fiziksel mudahale gerektiren bir "
                "gozlem oldugunda guvenlik ekibini belirtilen bolgeye yonlendirir. Gercek bir dis "
                "sisteme baglanmaz (simule edilmis saha aksiyonu); cagrilmasi rapora islenir."
            ),
            args_schema=DispatchSecurityToolInput,
        )


class TriggerAreaLockdownToolInput(BaseModel):
    """`trigger_area_lockdown_tool` icin girdi semasi."""

    zone: str = Field(description="Tahliye edilecek/kilitlenecek saha/bolge tanimi.")
    reason: str = Field(description="Kilitleme gerekcesi (dogal dil, orn. 'yangin/patlama riski').")


class TriggerAreaLockdownTool:
    """Bir bolgeyi tahliye/kilitleme sinyali veren MOCK aksiyon araci (en agir kademe).

    Bkz. `NotifyHealthTeamTool` docstring'i - ayni mock desen. Bu, uc mock
    aksiyon aracinin en agir olanidir; ajan yalnizca aktif yangin/patlama/gaz
    kacagi gibi HAYATI tehlike gozlemlerinde bu araci cagirmalidir (bkz.
    `AGENT_SYSTEM_PROMPT`daki "Mock Aksiyon Araclari" politikasi).
    """

    def run(self, zone: str, reason: str) -> str:
        """Aracin mock cagrisini yapar ve dogal dil bir onay/kayit metni dondurur."""
        logger.warning("MOCK AKSIYON: alan kilitleme/tahliye tetiklendi (zone=%s, reason=%s)", zone, reason)
        return f"[MOCK] '{zone}' bolgesi icin tahliye/kilitleme sinyali verildi. Gerekce: {reason}"

    def as_langchain_tool(self) -> StructuredTool:
        """Bu araci LangGraph/LangChain ajanina baglanabilecek `StructuredTool`'a cevirir."""
        return StructuredTool.from_function(
            func=self.run,
            name="trigger_area_lockdown_tool",
            description=(
                "MOCK arac: aktif yangin, patlama, gaz kacagi gibi HAYATI tehlike icin bir "
                "bolgenin tahliyesini/kilitlenmesini tetikler (en agir mock aksiyon kademesi). "
                "Gercek bir dis sisteme baglanmaz (simule edilmis saha aksiyonu); cagrilmasi rapora islenir."
            ),
            args_schema=TriggerAreaLockdownToolInput,
        )


# Yukaridaki uc mock aksiyon aracinin adlari - `SafirAgent.run()` bu kumeyi
# kullanarak, hangi arac cagrilarinin (sql/retriever/timeline/verification
# gibi ic sorgu araclarindan farkli olarak) `AgentDecision.triggered_mock_actions`e
# toplanacagini belirler.
MOCK_ACTION_TOOL_NAMES = frozenset(
    {"notify_health_team_tool", "dispatch_security_tool", "trigger_area_lockdown_tool"}
)


def build_tool_registry(
    event_store: Optional[EventStore] = None,
    rag_service: Optional[EmbeddingRAGService] = None,
    tools_config: Optional[Any] = None,
) -> List[StructuredTool]:
    """Ajanin `Dynamic Tool Router`ina baglanacak araclarin listesini config bayraklarina gore uretir.

    Args:
        event_store: SQL/Timeline/Verification araclarinin kullanacagi olay deposu.
        rag_service: retriever/verification araclarinin kullanacagi Embedding & RAG servisi.
        tools_config: `configs/config.yaml` icindeki `agent.tools` blogu
            (`AgentToolsConfig`). `None` verilirse geriye-uyum icin TUM araclar
            (verification + mock aksiyon araclari dahil) kurulur; verilirse
            yalnizca `*_enabled=true` olanlar eklenir. `rag_tool_enabled` ile
            `retriever_tool_enabled` ayni RAG aracina isaret eder (herhangi
            biri true ise eklenir).

    Returns:
        LangGraph ajanina dogrudan baglanabilecek `StructuredTool` listesi.
    """

    def _enabled(attr: str) -> bool:
        return getattr(tools_config, attr, True) if tools_config is not None else True

    tools: List[StructuredTool] = []
    if _enabled("sql_tool_enabled"):
        tools.append(SqlTool(event_store).as_langchain_tool())
    if _enabled("retriever_tool_enabled") or _enabled("rag_tool_enabled"):
        tools.append(RetrieverTool(rag_service).as_langchain_tool())
    if _enabled("timeline_tool_enabled"):
        tools.append(TimelineTool(event_store).as_langchain_tool())
    if _enabled("verification_tool_enabled"):
        tools.append(VerificationTool(event_store, rag_service).as_langchain_tool())
    if _enabled("mock_action_tools_enabled"):
        tools.append(NotifyHealthTeamTool().as_langchain_tool())
        tools.append(DispatchSecurityTool().as_langchain_tool())
        tools.append(TriggerAreaLockdownTool().as_langchain_tool())
    return tools
