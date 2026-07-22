"""04 - Context Builder: VLM ciktisini, istemleri ve zaman damgalarini birlestirir."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.memory.event_store import EventStore
from src.memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


@dataclass
class EnrichedContext:
    """Ajan/Muhakeme Katmanina iletilecek zenginlestirilmis baglam paketi."""

    vlm_description: str
    user_prompt: str
    timestamp: float
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    relevant_regulations: List[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Bu baglami LangGraph ajanina verilecek tek bir metin bloguna cevirir.

        Returns:
            Ajanin sistem/kullanici istemine dogrudan eklenebilecek metin.
        """
        recent = "\n".join(
            f"- [{event.get('timestamp'):.1f}s] {event.get('description')}"
            for event in self.recent_events
        ) or "(gecmis olay bulunamadi)"

        regulations = "\n".join(f"- {reg}" for reg in self.relevant_regulations) or (
            "(ilgili mevzuat bulunamadi)"
        )

        return (
            f"## Guncel Gozlem (t={self.timestamp:.1f}s)\n{self.vlm_description}\n\n"
            f"## Kullanici Istemi\n{self.user_prompt}\n\n"
            f"## Yakin Gecmis Olaylar\n{recent}\n\n"
            f"## Ilgili Operasyonel Mevzuat\n{regulations}"
        )


class ContextBuilder:
    """VLM metnini, kullanici istemini ve gecmis bellek erisimini birlestiren katman.

    Yapilandirilmis olay bellegi (`EventStore`) ve anlamsal bellek
    (`SemanticMemory`) birlesimini kullanarak ajan katmani icin
    zenginlestirilmis bir baglam uretir.
    """

    def __init__(self, event_store: EventStore, semantic_memory: SemanticMemory) -> None:
        """ContextBuilder'i olay ve anlamsal bellek bagimliliklariyla baslatir.

        Args:
            event_store: Gecmis olaylari sorgulamak icin SQLite tabanli depo.
            semantic_memory: Operasyonel kurallari sorgulamak icin FAISS tabanli depo.
        """
        self._event_store = event_store
        self._semantic_memory = semantic_memory

    def build(
        self,
        vlm_description: str,
        user_prompt: str,
        timestamp: float,
        recent_event_limit: int = 5,
        regulation_top_k: Optional[int] = None,
    ) -> EnrichedContext:
        """VLM ciktisini gecmis olaylar ve ilgili mevzuatla zenginlestirir.

        Args:
            vlm_description: VLM katmanindan gelen dogal dil olay aciklamasi.
            user_prompt: Operatorden veya sistemden gelen kullanici istemi.
            timestamp: Mevcut gozlemin zaman damgasi.
            recent_event_limit: Getirilecek yakin gecmis olay sayisi.
            regulation_top_k: Getirilecek mevzuat sonucu sayisi (varsayilan config).

        Returns:
            Ajan katmanina iletilmeye hazir `EnrichedContext` nesnesi.
        """
        recent_events = self._event_store.query_recent(limit=recent_event_limit)
        regulations = [
            doc.text
            for doc in self._semantic_memory.query(vlm_description, top_k=regulation_top_k)
        ]

        context = EnrichedContext(
            vlm_description=vlm_description,
            user_prompt=user_prompt,
            timestamp=timestamp,
            recent_events=recent_events,
            relevant_regulations=regulations,
        )
        logger.debug(
            "Baglam olusturuldu: %d gecmis olay, %d mevzuat sonucu",
            len(recent_events),
            len(regulations),
        )
        return context
