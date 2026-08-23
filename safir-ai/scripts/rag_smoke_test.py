"""SAFIR RAG (LOKAL embedding + FAISS + opsiyonel Gemini/Groq rerank) icin GERCEK smoke test.

Embedding artik TAMAMEN LOKAL'dir (`sentence-transformers`, CPU, API anahtari
GEREKMEZ). Reranker OPSIYONELDIR; config'teki `memory.reranker.provider`e gore
Gemini VEYA Groq (`memory.reranker.api_key_env`) kullanabilir - gerekli ortam
degiskeni TANIMLI DEGILSE, bu script bunu ACIKCA belirtip mock'a SESSIZCE
DUSMEDEN cikar.

Kullanim:
    export GROQ_API_KEY=...   # reranker.provider="groq" ise gerekir (varsayilan)
    python scripts/rag_smoke_test.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

_QUERIES = [
    "fire_detected smoke_detected uncontrolled_open_flame",
    "yangın ve duman",
    "baret kullanılmaması kişisel koruyucu donanım",
    "forklift yaya çarpışma riski",
    "elektrik panosu ve elektrik teması",
    "kimyasal madde dökülmesi",
]


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from src.rag.embedding_rag_service import EmbeddingRAGService, _INDEX_FILE
    from src.utils.config_loader import load_config

    config = load_config()
    if not config.memory.reranker.enabled:
        print("[SKIPPED] memory.reranker devre disi - config.yaml'i kontrol edin.")
        return 0
    if config.memory.reranker.provider not in ("gemini", "groq"):
        print(f"[SKIPPED] memory.reranker.provider desteklenmiyor: '{config.memory.reranker.provider}'.")
        return 0
    if not os.environ.get(config.memory.reranker.api_key_env, "").strip():
        print(f"[SKIPPED] {config.memory.reranker.api_key_env} is not set (reranker icin gerekli)")
        return 0

    service = EmbeddingRAGService(config.memory.embedding, config.memory.faiss, config.memory.reranker)

    if not _INDEX_FILE.exists():
        print(f"[BILGI] Persisted index yok ({_INDEX_FILE}); simdi 748 chunk LOKAL modelle embed edilecek.")
        print("        (Bu, yalnizca bir kez calisir; sonrasi icin 'python -m src.rag.build_knowledge_index' tercih edin.)")
        count = service.build_index_from_chunks()
        print(f"[OK] {count} chunk embed edildi.\n")
    else:
        service.seed_default_regulations()
        print(f"[OK] Persisted index yuklendi ({service.document_count()} dokuman).\n")

    print("=" * 72)
    print(f"GERCEK RAG SMOKE TEST (lokal embedding + {config.memory.reranker.provider} rerank, GERCEK API)")
    print("=" * 72)

    for query in _QUERIES:
        candidate_k = min(config.memory.faiss.candidate_k, service.document_count())
        results = service.query(query)

        print(f"\nQUERY: {query!r}")
        print(f"CANDIDATES: {candidate_k}")
        print(f"RERANKED: {candidate_k}")
        print(f"FINAL: {len(results)}")

        if not results:
            print("  (esik-uzeri sonuc yok - GECERLI bir sonuc, rastgele top-k UYDURULMADI)")
            continue

        for i, doc in enumerate(results, start=1):
            score = doc.rerank_score if doc.rerank_score is not None else doc.embedding_score
            print(f"\n{i}. score={score:.2f}")
            print(f"   document={doc.document_title or doc.document_id or '(bilinmeyen kaynak)'}")
            print(f"   article={doc.article_number or '-'}")
            print(f"   source={doc.source_url or '-'}")

    print("\n" + "=" * 72)
    print("Not: sonuclarin semantik kalitesi burada IDDIA EDILMEMEKTEDIR - bu")
    print("script yalnizca lokal embedding + gercek rerank API'sinin uctan uca")
    print("CALISTIGINI dogrular. API anahtarlari hicbir yerde YAZDIRILMADI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
