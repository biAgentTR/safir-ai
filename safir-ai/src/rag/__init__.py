"""04b - RAG (Retrieval-Augmented Generation) Katmani.

ISG mevzuati/operasyonel kural corpus'unu Gemini embedding + FAISS ile
indeksleyip Gemini/Groq ile yeniden siralayan (rerank) alt paket:

- `embedding_rag_service.py` - `EmbeddingRAGService` (FAISS index, query,
  persist/load, `RagQueryTelemetry`).
- `embedding_providers.py` - embedding saglayici soyutlamasi (Gemini).
- `reranker.py` - iki-asamali retrieval'in rerank adimi (Gemini/Groq).
- `build_knowledge_index.py` - `python -m src.rag.build_knowledge_index`
  REBUILD CLI'i (gercek 748 mevzuat chunk'ini embed edip diske persist eder).

Alt moduller PEP 562 (`__getattr__`) ile tembel (lazy) yuklenir - bu paketin
ust duzeyi, `EmbeddingRAGService`in agir bagimliliklarini (faiss, google-genai)
yalnizca gercekten erisildiginde ice aktarir.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.rag.embedding_rag_service import (
        EmbeddingRAGService,
        FAISSRagService,
        RetrievedDocument,
    )

__all__ = [
    "EmbeddingRAGService",
    "FAISSRagService",
    "RetrievedDocument",
]

_RAG_NAMES = {"EmbeddingRAGService", "FAISSRagService", "RetrievedDocument"}


def __getattr__(name: str) -> Any:
    """Istenen alt modulu yalnizca gercekten erisildiginde ice aktarir (PEP 562)."""
    if name in _RAG_NAMES:
        from src.rag import embedding_rag_service

        return getattr(embedding_rag_service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
