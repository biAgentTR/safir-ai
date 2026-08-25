"""SAFIR RAG (LOKAL E5 embedding + FAISS + deterministik relevance + opsiyonel LOKAL Cross-Encoder) icin GERCEK smoke test.

2026-08-24 (RAG+RISK PRODUCTION KAPANIS turu, terminology audit duzeltmesi):
bu script ONCEDEN eski Gemini/Groq LLM-as-judge reranker donemine aitti ve
artik var olmayan `doc.rerank_score` alanina (bkz. `RetrievedDocument` -
GERCEK alanlar `embedding_score`/`relevance_score`/`cross_encoder_score`dir)
referans veriyordu - hicbir zaman calismadan `[SKIPPED]` basip cikiyordu
(reranker.provider artik "gemini"/"groq" OLAMAZ, bkz. `RerankerConfig`
dokustringi). HICBIR AG/LLM/API CAGRISI icermez - hem deterministik relevance
hem (varsa) lokal Cross-Encoder TAMAMEN CPU'da calisir.

Kullanim:
    python scripts/rag_smoke_test.py
"""

from __future__ import annotations

import logging
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
    from src.rag.local_cross_encoder_reranker import DEFAULT_LOCAL_CROSS_ENCODER_MODEL, LocalCrossEncoderReranker
    from src.utils.config_loader import load_config

    config = load_config()
    if not config.memory.reranker.enabled:
        print("[BILGI] memory.reranker devre disi - yalnizca embedding_score siralamasi kullanilacak.")

    # Production'daki AYNI varsayilan: LOKAL Cross-Encoder (bkz. `src/main.py::SafirPipeline.__init__`).
    # HICBIR AG cagrisi API ANAHTARI GEREKTIRMEZ - model agirligi lazy yuklenir; bu ortamda
    # yuklenemezse (paket/internet yok) `query()` KONTROLLU sekilde deterministic relevance'a duser.
    service = EmbeddingRAGService(
        config.memory.embedding,
        config.memory.qdrant,
        config.memory.reranker,
        cross_encoder=LocalCrossEncoderReranker(DEFAULT_LOCAL_CROSS_ENCODER_MODEL),
    )

    if not _INDEX_FILE.exists():
        print(f"[BILGI] Persisted index yok ({_INDEX_FILE}); simdi 748 chunk LOKAL modelle embed edilecek.")
        print("        (Bu, yalnizca bir kez calisir; sonrasi icin 'python -m src.rag.build_knowledge_index' tercih edin.)")
        count = service.build_index_from_chunks()
        print(f"[OK] {count} chunk embed edildi.\n")
    else:
        service.seed_default_regulations()
        print(f"[OK] Persisted index yuklendi ({service.document_count()} dokuman).\n")

    print("=" * 72)
    print("GERCEK RAG SMOKE TEST (lokal E5 embedding + deterministik relevance + lokal Cross-Encoder)")
    print("=" * 72)

    for query in _QUERIES:
        candidate_k = min(config.memory.qdrant.candidate_k, service.document_count())
        results = service.query(query)
        telemetry = service.get_last_query_telemetry()

        print(f"\nQUERY: {query!r}")
        print(f"CANDIDATES: {candidate_k}")
        print(f"FINAL: {len(results)}")
        print(f"CROSS_ENCODER_STATUS: {telemetry.cross_encoder_status if telemetry else 'n/a'}")

        if not results:
            print("  (esik-uzeri sonuc yok - GECERLI bir sonuc, rastgele top-k UYDURULMADI)")
            continue

        for i, doc in enumerate(results, start=1):
            print(f"\n{i}. embedding_score={doc.embedding_score:.3f}", end="")
            print(f" relevance_score={doc.relevance_score:.3f}" if doc.relevance_score is not None else " relevance_score=yok", end="")
            print(f" cross_encoder_score={doc.cross_encoder_score:.3f}" if doc.cross_encoder_score is not None else " cross_encoder_score=yok")
            print(f"   document={doc.document_title or doc.document_id or '(bilinmeyen kaynak)'}")
            print(f"   article={doc.article_number or '-'}")
            print(f"   source={doc.source_url or '-'}")

    print("\n" + "=" * 72)
    print("Not: sonuclarin semantik kalitesi burada IDDIA EDILMEMEKTEDIR - bu script")
    print("yalnizca lokal embedding + deterministik relevance + (varsa) lokal")
    print("Cross-Encoder'in uctan uca CALISTIGINI dogrular. Hicbir API anahtari")
    print("gerekmez, hicbir agdan/harici serviste veri okunmaz/yazilmaz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
