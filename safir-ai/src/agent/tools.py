"""05 - Dynamic Tool Router: ajanin dinamik olarak cagirabilecegi araclar.

Bu modul, `Reasoning Agent`in baglama gore yonlendirdigi uc mock/taslak araci
tanimlar: gecmis olay sorgulama (SQL Tool), zamansal cizelgeleme (Timeline
Tool) ve ISG/operasyonel mevzuat aramasi (RAG Tool). Araclar, gercek
`EventStore`/`SemanticMemory` bagimliliklari verilmezse bile calisabilen mock
veriye duser; boylece iskelet bagimsiz test edilebilir.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.memory.event_store import EventStore
from src.memory.semantic_memory import SemanticMemory

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


class RagToolInput(BaseModel):
    """RAG Tool icin girdi semasi."""

    question: str = Field(description="ISG/operasyonel mevzuat ile ilgili dogal dil sorusu.")
    top_k: int = Field(default=3, description="Getirilecek maksimum mevzuat maddesi sayisi.")


class RagTool:
    """ISG mevzuati ve operasyonel kurallari anlamsal bellek (FAISS) uzerinden arayan arac."""

    def __init__(self, semantic_memory: Optional[SemanticMemory] = None) -> None:
        """RagTool'u opsiyonel bir `SemanticMemory` bagimliligiyla baslatir.

        Args:
            semantic_memory: Gercek FAISS tabanli anlamsal bellek; `None` ise mock veri kullanilir.
        """
        self._semantic_memory = semantic_memory

    def run(self, question: str, top_k: int = 3) -> str:
        """RAG Tool'u calistirir ve ilgili mevzuat maddelerini dondurur.

        Args:
            question: Dogal dil sorgusu.
            top_k: Maksimum sonuc sayisi.

        Returns:
            Ilgili mevzuat maddelerini iceren metin.
        """
        if self._semantic_memory is None:
            logger.warning("RagTool: SemanticMemory baglanmadi, mock veri donduruluyor.")
            regulations = _MOCK_REGULATIONS[:top_k]
        else:
            regulations = [doc.text for doc in self._semantic_memory.query(question, top_k=top_k)]

        if not regulations:
            return "Sorguyla ilgili mevzuat maddesi bulunamadi."

        return "Ilgili mevzuat:\n" + "\n".join(f"- {r}" for r in regulations)

    def as_langchain_tool(self) -> StructuredTool:
        """Bu araci LangGraph/LangChain ajanina baglanabilecek `StructuredTool`'a cevirir."""
        return StructuredTool.from_function(
            func=self.run,
            name="rag_tool",
            description="ISG mevzuati ve operasyonel kurallari anlamsal olarak arar.",
            args_schema=RagToolInput,
        )


class TimelineToolInput(BaseModel):
    """Timeline Tool icin girdi semasi."""

    start_ts: float = Field(description="Zaman araligi baslangici (saniye).")
    end_ts: float = Field(description="Zaman araligi bitisi (saniye).")


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


def build_tool_registry(
    event_store: Optional[EventStore] = None,
    semantic_memory: Optional[SemanticMemory] = None,
) -> List[StructuredTool]:
    """Ajanin `Dynamic Tool Router`ina baglanacak tum araclarin listesini uretir.

    Args:
        event_store: SQL ve Timeline araclarinin kullanacagi olay deposu.
        semantic_memory: RAG aracinin kullanacagi anlamsal bellek.

    Returns:
        LangGraph ajanina dogrudan baglanabilecek `StructuredTool` listesi.
    """
    return [
        SqlTool(event_store).as_langchain_tool(),
        RagTool(semantic_memory).as_langchain_tool(),
        TimelineTool(event_store).as_langchain_tool(),
    ]
