"""04 - Hibrit Bellek ve Baglam Olusturucu Katmani.

RAG (embedding/FAISS/rerank) alt paketi `src/rag/`e tasindi (bkz. o paketin
docstring'i) - bu paket artik yalnizca SQLite-tabanli depolari (`EventStore`,
`AnalysisStore`, `ConversationStore`) ve bunlari birlestiren `ContextBuilder`i
icerir; `qdrant-client`/`openai` gibi agir bagimliliklar burada YOKTUR.

Alt moduller PEP 562 (`__getattr__`) ile tembel (lazy) yuklenir.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.memory.context_builder import ContextBuilder, EnrichedContext
    from src.memory.event_store import EventStore, SQLiteEventStore

__all__ = [
    "ContextBuilder",
    "EnrichedContext",
    "EventStore",
    "SQLiteEventStore",
]

_CONTEXT_BUILDER_NAMES = {"ContextBuilder", "EnrichedContext"}
_EVENT_STORE_NAMES = {"EventStore", "SQLiteEventStore"}


def __getattr__(name: str) -> Any:
    """Istenen alt modulu yalnizca gercekten erisildiginde ice aktarir (PEP 562)."""
    if name in _CONTEXT_BUILDER_NAMES:
        from src.memory import context_builder

        return getattr(context_builder, name)
    if name in _EVENT_STORE_NAMES:
        from src.memory import event_store

        return getattr(event_store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
