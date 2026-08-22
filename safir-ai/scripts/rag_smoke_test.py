"""SAFIR RAG (Gemini embedding + FAISS + Cohere rerank) icin GERCEK smoke test.

GEMINI_API_KEY ve COHERE_API_KEY ortam degiskenleri TANIMLIYSA, gercek 748
mevzuat chunk'ina karsi gercek API cagrilariyla birkaç sorgu calistirir ve
sonuclari yapilandirilmis sekilde yazdirir. Anahtarlardan biri EKSIKSE,
bunu ACIKCA belirtip (0 haric bir kodla) cikar - SESSIZCE mock'a DUSMEZ ve
semantik kalite IDDIA ETMEZ (bkz. gorev tanimi 14. bolum, madde 6).

Kullanim:
    export GEMINI_API_KEY=...
    export COHERE_API_KEY=...
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
    "yangın ve duman",
    "baret kullanılmaması kişisel koruyucu donanım",
    "forklift yaya çarpışma riski",
    "elektrik panosu ve elektrik teması",
    "kimyasal madde dökülmesi",
]


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print("[ATLANDI] GEMINI_API_KEY tanimli degil - gercek RAG smoke test calistirilamiyor.")
        print("          (Bu, mock'a SESSIZCE dusmez - test'in kendisi kosulmadi.)")
        return 0
    if not os.environ.get("COHERE_API_KEY", "").strip():
        print("[ATLANDI] COHERE_API_KEY tanimli degil - gercek RAG smoke test calistirilamiyor.")
        return 0

    from src.memory.embedding_rag_service import EmbeddingRAGService, _INDEX_FILE
    from src.utils.config_loader import load_config

    config = load_config()
    service = EmbeddingRAGService(config.memory.embedding, config.memory.faiss, config.memory.reranker)

    if not _INDEX_FILE.exists():
        print(f"[BILGI] Persisted index yok ({_INDEX_FILE}); simdi 748 chunk GERCEK Gemini API ile embed edilecek.")
        print("        (Bu, yalnizca bir kez calisir; sonrasi icin 'python -m src.memory.build_knowledge_index' tercih edin.)")
        count = service.build_index_from_chunks()
        print(f"[OK] {count} chunk embed edildi.\n")
    else:
        service.seed_default_regulations()
        print(f"[OK] Persisted index yuklendi ({service.document_count()} dokuman).\n")

    print("=" * 72)
    print("GERCEK RAG SMOKE TEST (Gemini embedding + Cohere rerank, GERCEK API)")
    print("=" * 72)

    for query in _QUERIES:
        results = service.query(query)
        print(f"\nQUERY: {query!r}")
        print(f"final_count: {len(results)}")
        for i, doc in enumerate(results[:3], start=1):
            title = doc.document_title or doc.document_id or "(bilinmeyen kaynak)"
            article = doc.article_number or "-"
            score = doc.rerank_score if doc.rerank_score is not None else doc.embedding_score
            short_text = doc.text[:150].replace("\n", " ")
            print(f"  [{i}] document_title={title!r} article_number={article!r} rerank_score={score:.4f}")
            print(f"      text: {short_text}...")
        if not results:
            print("  (esik-uzeri sonuc yok - GECERLI bir sonuc)")

    print("\n" + "=" * 72)
    print("Not: sonuclarin semantik kalitesi burada IDDIA EDILMEMEKTEDIR - bu")
    print("script yalnizca gercek API'lerin uctan uca CALISTIGINI dogrular.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
