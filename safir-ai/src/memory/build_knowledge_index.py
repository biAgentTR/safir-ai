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
from src.memory.embedding_rag_service import (
    EmbeddingRAGService,
    _KB_CHUNKS_DIR,
    _KB_INDEX_DIR,
    _load_kb_chunk_records,
)
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Gercek corpus icerigine gore secilmis, UYDURULMAMIS smoke-test sorgulari
# (bkz. gorev tanimi HEDEF 2 - her biri chunk metninde gercekten gecen bir
# konuya karsilik gelir: forklift/is_ekipmanlari, elektrik kilitleme, bykhy
# kapali alan, kimyasal etiketleme).
_SMOKE_TEST_QUERIES = [
    "forklift yaya güvenliği",
    "elektrik kilitleme etiketleme",
    "kapalı alan girişi",
    "kimyasal madde etiketleme",
]


def _run_load_verification_and_smoke_test(config) -> bool:
    """`persist()` sonrasi index'i TAZE bir servis orneginde yeniden yukleyip gercek sorgularla dogrular.

    Bu, `build_index_from_chunks()`in bellek-ici sonucunu DEGIL, DISKE
    YAZILMIS dosyalarin (`faiss.index`/`documents.json`/`index_meta.json`)
    gercekten `_try_load_persisted_index()` ile okunabildigini kanitlar -
    "persist edildi" ile "pipeline bunu gercekten yukleyebiliyor" ayni sey
    degildir.

    Returns:
        Yukleme+en az bir sorgunun sonuc dondurmesi basariliysa `True`.
    """
    print("\n" + "=" * 72)
    print("LOAD VERIFICATION")
    print("=" * 72)
    verify_service = EmbeddingRAGService(config.memory.embedding, config.memory.faiss, config.memory.reranker)
    loaded = verify_service._try_load_persisted_index()
    print(f"persisted_index = {'TRUE' if loaded else 'FALSE'}")
    if not loaded:
        print("[HATA] Persist edilen index taze bir instance'ta yuklenemedi.", file=sys.stderr)
        return False
    print(f"corpus_source   = {verify_service.corpus_source}")
    print(f"document_count  = {verify_service.document_count()}")

    print("\n" + "=" * 72)
    print("RETRIEVAL SMOKE TEST")
    print("=" * 72)
    any_result = False
    for query in _SMOKE_TEST_QUERIES:
        results = verify_service.query(query)
        telemetry = verify_service.get_last_query_telemetry()
        print(f"\nquery: {query!r}")
        print(
            f"  corpus_source={telemetry.corpus_source} candidate_count={telemetry.candidate_count} "
            f"final_count={telemetry.final_count} retrieval_status={telemetry.retrieval_status}"
        )
        if not results:
            print("  (esik-uzeri sonuc yok)")
            continue
        any_result = True
        for doc in results:
            score = f"rerank={doc.rerank_score:.3f}" if doc.rerank_score is not None else f"embedding={doc.embedding_score:.3f}"
            print(
                f"  -> chunk_id={doc.chunk_id} document_id={doc.document_id} "
                f"article={doc.article_number} {score}"
            )
    return any_result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config()
    records = _load_kb_chunk_records()
    document_ids = sorted({r.get("document_id") for r in records if r.get("document_id")})

    print("=" * 72)
    print("SAFIR Knowledge Base Index Rebuild")
    print("=" * 72)
    print("CORPUS:")
    print(f"  {len(document_ids)} documents: {', '.join(document_ids)}")
    print(f"  N chunks: {len(records)}")
    print(f"chunks kaynagi : {_KB_CHUNKS_DIR}")
    print(f"index hedefi   : {_KB_INDEX_DIR}")
    print("EMBEDDING:")
    print(f"  {config.memory.embedding.provider} / {config.memory.embedding.model_name} "
          f"({config.memory.embedding.output_dimensionality} dimensions)")
    print("=" * 72)

    if not records:
        print(f"\n[HATA] {_KB_CHUNKS_DIR} altinda hicbir chunk bulunamadi.", file=sys.stderr)
        return 1

    service = EmbeddingRAGService(config.memory.embedding, config.memory.faiss, config.memory.reranker)

    started = time.perf_counter()
    try:
        count = service.build_index_from_chunks()
    except ConfigurationError as exc:
        print(f"\n[HATA] Konfigurasyon eksik (orn. GEMINI_API_KEY): {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"\n[HATA] {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    print(f"\n[OK] {count} chunk embed edildi ({elapsed:.1f}s, {count / max(elapsed, 1e-6):.1f} chunk/s).")

    service.persist()
    print(f"\nINDEX: {_KB_INDEX_DIR}")
    print(f"  - faiss.index    ({service.document_count()} vektor, dim={service.dimension})")
    print("  - documents.json (yapilandirilmis metadata + text)")
    print("  - index_meta.json (model_name/dimension/kb_hash/chunk_count/created_at)")

    if not _run_load_verification_and_smoke_test(config):
        print(
            "\n[HATA] Index persist edildi ama yeniden-yukleme/smoke-test dogrulamasi basarisiz oldu.",
            file=sys.stderr,
        )
        return 1

    print("\n[OK] Pipeline artik bu index'i, embedding API'sini TEKRAR CAGIRMADAN yukleyecek.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
