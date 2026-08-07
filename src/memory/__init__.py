
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.memory.context_builder import ContextBuilder, EnrichedContext
    from src.memory.embedding_rag_service import (
        EmbeddingRAGService,
        FAISSRagService,
        RetrievedDocument,
    )
    from src.memory.event_store import EventStore, SQLiteEventStore

__all__ = [
    "ContextBuilder",
    "EnrichedContext",
    "EmbeddingRAGService",
    "FAISSRagService",
    "RetrievedDocument",
    "EventStore",
    "SQLiteEventStore",
]

_CONTEXT_BUILDER_NAMES = {"ContextBuilder", "EnrichedContext"}
_RAG_NAMES = {"EmbeddingRAGService", "FAISSRagService", "RetrievedDocument"}
_EVENT_STORE_NAMES = {"EventStore", "SQLiteEventStore"}


def __getattr__(name: str) -> Any:
    """Istenen alt modulu yalnizca gercekten erisildiginde ice aktarir (PEP 562)."""
    if name in _CONTEXT_BUILDER_NAMES:
        from src.memory import context_builder

        return getattr(context_builder, name)
    if name in _RAG_NAMES:
        from src.memory import embedding_rag_service

        return getattr(embedding_rag_service, name)
    if name in _EVENT_STORE_NAMES:
        from src.memory import event_store

        return getattr(event_store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")