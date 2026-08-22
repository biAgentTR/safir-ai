"""04 - Embedding & RAG Katmani: Knowledge Base index REBUILD CLI'i.

`data/knowledge_base/chunks/*.json` icindeki TUM (su an 748) madde-bazli
chunk'i `sources.yaml` ile join edip Gemini Embedding API ile embed eder,
bir FAISS `IndexFlatIP` insa eder ve `data/knowledge_base/index/` altina
KALICI olarak yazar (`faiss.index`, `documents.json`, `index_meta.json`).

Bu script YALNIZCA knowledge base ICERIGI DEGISTIGINDE (yeni doküman,
chunk stratejisi degisikligi, embedding modeli degisikligi) calistirilmasi
gereken, EXPLICIT bir komuttur - `SafirPipeline` HER BASLANGICTA bunu
OTOMATIK cagirmaz (bkz. `embedding_rag_service.py::seed_default_regulations`,
`_try_load_persisted_index`); pipeline yalnizca bu scriptin urettigi
persisted index'i YUKLER.

Kullanim:
    python -m src.memory.build_knowledge_index

GEMINI_API_KEY ortam degiskeni tanimli olmalidir (748 dokuman embed etmek
gercek API cagrisi ve maliyet gerektirir - bu YALNIZCA bilinçli, tek seferlik
bir komutla tetiklenir).
"""

from __future__ import annotations

import logging
import sys
import time

from src.memory.embedding_providers import ConfigurationError
from src.memory.embedding_rag_service import EmbeddingRAGService, _KB_CHUNKS_DIR, _KB_INDEX_DIR
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config()
    print("=" * 72)
    print("SAFIR Knowledge Base Index Rebuild")
    print("=" * 72)
    print(f"chunks kaynagi : {_KB_CHUNKS_DIR}")
    print(f"index hedefi   : {_KB_INDEX_DIR}")
    print(f"embedding      : {config.memory.embedding.provider} / {config.memory.embedding.model_name} "
          f"(dim={config.memory.embedding.output_dimensionality})")
    print("=" * 72)

    service = EmbeddingRAGService(config.memory.embedding, config.memory.faiss, config.memory.reranker)

    started = time.perf_counter()
    try:
        count = service.build_index_from_chunks()
    except ConfigurationError as exc:
        print(f"\n[HATA] Konfigurasyon eksik: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"\n[HATA] {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    print(f"\n[OK] {count} chunk embed edildi ({elapsed:.1f}s, {count / max(elapsed, 1e-6):.1f} chunk/s).")

    service.persist()
    print(f"[OK] Index persisted: {_KB_INDEX_DIR}")
    print(f"     - faiss.index    ({service.document_count()} vektor, dim={service.dimension})")
    print("     - documents.json (yapilandirilmis metadata + text)")
    print("     - index_meta.json (model_name/dimension/kb_hash/chunk_count/created_at)")
    print("\nPipeline artik bu index'i, embedding API'sini TEKRAR CAGIRMADAN yukleyecek.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
