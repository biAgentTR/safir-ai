"""RAG retrieval/reranking benchmark: golden query dataset + A/B/C pipeline comparison.

Bu script, `data/knowledge_base/index/` altindaki PERSISTED 748-chunk index'i
(hic yeniden olusturmadan) kullanarak uc pipeline'i karsilastirir:

    A: E5 -> FAISS (embedding_score'a gore siralama, rerank YOK)
    B: E5 -> FAISS -> deterministic relevance (mevcut production yolu)
    C: E5 -> FAISS -> deterministic relevance -> LOCAL Cross-Encoder (opsiyonel, --cross-encoder ile)

Ground truth, corpus'taki GERCEK chunk_id'lerden elle (grep/inceleme ile
dogrulanmis) derlenmistir - UYDURULMAMISTIR (bkz. `GOLDEN_QUERIES`daki
yorumlar, hangi metin/terimle dogrulandigini belirtir).

Kullanim:
    python scripts/rag_benchmark.py                  # A ve B
    python scripts/rag_benchmark.py --cross-encoder MODEL_NAME   # A, B, C
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.deterministic_reranker import RelevanceWeights, score_candidate  # noqa: E402
from src.rag.embedding_providers import LocalEmbeddingProvider  # noqa: E402
from src.rag.embedding_rag_service import EmbeddingRAGService  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402


@dataclass
class GoldenQuery:
    query: str
    exact_relevant: List[str] = field(default_factory=list)
    """Sorguyu DOGRUDAN, en isabetli sekilde cevaplayan chunk_id'ler."""
    relevant: List[str] = field(default_factory=list)
    """Ayni konuyu/dokumani ele alan ama exact kadar dogrudan olmayan chunk_id'ler."""
    partial: List[str] = field(default_factory=list)
    """Konuyla kismen ilgili (komsu/genel) chunk_id'ler - Recall/nDCG'de DUSUK agirlikla sayilir."""

    def graded_relevance(self) -> Dict[str, int]:
        """chunk_id -> relevance grade (2=exact, 1=relevant/partial, 0=digerleri) - nDCG icin."""
        grades: Dict[str, int] = {}
        for cid in self.exact_relevant:
            grades[cid] = 2
        for cid in self.relevant:
            grades.setdefault(cid, 1)
        for cid in self.partial:
            grades.setdefault(cid, 1)
        return grades

    def all_relevant_ids(self) -> List[str]:
        return list(dict.fromkeys(self.exact_relevant + self.relevant + self.partial))


# ---------------------------------------------------------------------------
# GOLDEN QUERY DATASET
# Ground truth, `data/knowledge_base/index/documents.json` (748 GERCEK chunk)
# uzerinde DOGRUDAN metin aramasiyla (bkz. her query'nin yorumundaki terimler)
# DOGRULANMISTIR - kafadan uydurulmamistir.
# ---------------------------------------------------------------------------

GOLDEN_QUERIES: List[GoldenQuery] = [
    GoldenQuery(
        query="forklift yaya güvenliği",
        # "forklift" GERCEKTEN gecen tek 2 chunk (is_ekipmanlari_yonetmeligi eki).
        exact_relevant=["is_ekipmanlari_yonetmeligi__ek_alt_madde_I.3.1"],
        relevant=["is_ekipmanlari_yonetmeligi__ek_alt_madde_III.2.2"],
        partial=["yapi_isleri_isg_yonetmeligi__madde_13", "yapi_isleri_isg_yonetmeligi__madde_14"],
    ),
    GoldenQuery(
        query="elektrik kilitleme etiketleme",
        # "kilitle"/"etiketle" terimlerinin GERCEKTEN birlikte gectigi elektrik maddeleri.
        exact_relevant=["elektrik_kuvvetli_akim_yonetmeligi__madde_35", "elektrik_kuvvetli_akim_yonetmeligi__madde_36"],
        relevant=["elektrik_ic_tesisleri_yonetmeligi__madde_49", "elektrik_ic_tesisleri_yonetmeligi__madde_59"],
        partial=["elektrik_ic_tesisleri_yonetmeligi__madde_60", "elektrik_ic_tesisleri_yonetmeligi__madde_72"],
    ),
    GoldenQuery(
        query="kapalı alan girişi",
        # "kapalı alan"/"dar alan" GERCEKTEN gecen TEK dokuman (bykhy).
        exact_relevant=["bykhy__madde_24"],
        relevant=["bykhy__madde_75", "bykhy__madde_165"],
        partial=[],
    ),
    GoldenQuery(
        query="kimyasal madde etiketleme",
        exact_relevant=["kimyasal_maddeler_yonetmeligi__madde_9"],
        relevant=["kimyasal_maddeler_yonetmeligi__madde_6", "kimyasal_maddeler_yonetmeligi__madde_4"],
        partial=["elektrik_kuvvetli_akim_yonetmeligi__madde_35"],
    ),
    GoldenQuery(
        query="yangın duman alev",
        exact_relevant=["bykhy__madde_2", "bykhy__madde_4"],
        relevant=["bykhy__madde_6", "bykhy__madde_7", "acil_durumlar_yonetmeligi__madde_12"],
        partial=["bykhy__madde_20", "bykhy__madde_24", "bykhy__madde_25"],
    ),
    GoldenQuery(
        query="acil çıkış yangın",
        exact_relevant=["acil_durumlar_yonetmeligi__madde_5", "bykhy__madde_38"],
        relevant=["bykhy__madde_62", "bykhy__madde_73", "acil_durumlar_yonetmeligi__madde_12"],
        partial=["bykhy__madde_149", "bykhy__madde_150"],
    ),
    GoldenQuery(
        query="patlayıcı ortam",
        exact_relevant=["patlayici_ortamlar_yonetmeligi__madde_1", "patlayici_ortamlar_yonetmeligi__madde_2"],
        relevant=["patlayici_ortamlar_yonetmeligi__madde_4", "patlayici_ortamlar_yonetmeligi__madde_5"],
        partial=["bykhy__madde_110", "bykhy__madde_111", "kimyasal_maddeler_yonetmeligi__madde_7"],
    ),
    GoldenQuery(
        query="kişisel koruyucu donanım",
        exact_relevant=["6331_isg_kanunu__madde_19", "yapi_isleri_isg_yonetmeligi__madde_5"],
        relevant=["6331_isg_kanunu__madde_10", "6331_isg_kanunu__madde_18", "6331_isg_kanunu__madde_26"],
        partial=["patlayici_ortamlar_yonetmeligi__ek_alt_madde_2.2.3", "kimyasal_maddeler_yonetmeligi__madde_8"],
    ),
    GoldenQuery(
        query="iş ekipmanı güvenli kullanım",
        exact_relevant=["is_ekipmanlari_yonetmeligi__madde_1", "is_ekipmanlari_yonetmeligi__madde_4"],
        relevant=["is_ekipmanlari_yonetmeligi__madde_5", "is_ekipmanlari_yonetmeligi__madde_6", "is_ekipmanlari_yonetmeligi__madde_7"],
        partial=["yapi_isleri_isg_yonetmeligi__madde_5", "yapi_isleri_isg_yonetmeligi__madde_7"],
    ),
    GoldenQuery(
        query="düşme riski",
        exact_relevant=["is_ekipmanlari_yonetmeligi__ek_alt_madde_II.4.3", "is_ekipmanlari_yonetmeligi__ek_alt_madde_II.4.4"],
        relevant=["yapi_isleri_isg_yonetmeligi__ek_liste_4.1", "yapi_isleri_isg_yonetmeligi__ek_liste_4.2"],
        partial=["bykhy__madde_33", "bykhy__madde_41", "bykhy__madde_44"],
    ),
    GoldenQuery(
        query="elektriksel tehlike",
        exact_relevant=["yapi_isleri_isg_yonetmeligi__ek_liste_4.13"],  # "elektrik çarpması" GERCEKTEN gecer
        relevant=["elektrik_ic_tesisleri_yonetmeligi__madde_38", "bykhy__madde_92"],
        partial=["elektrik_kuvvetli_akim_yonetmeligi__madde_51", "bykhy__madde_33"],
    ),
    GoldenQuery(
        query="sıcak çalışma",
        # "kıvılcım"/"alevle"/"yanıcı madde" GERCEKTEN gecen bykhy maddeleri (sıcak çalışma/kaynak-kesme baglami).
        exact_relevant=["bykhy__madde_7", "bykhy__madde_60"],
        relevant=["bykhy__madde_88", "bykhy__madde_103", "bykhy__madde_106"],
        partial=["bykhy__madde_4", "bykhy__madde_26"],
    ),
]


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    hit = sum(1 for r in relevant_ids if r in top_k)
    return hit / len(relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hit = sum(1 for r in top_k if r in relevant_ids)
    return hit / len(top_k)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_ids: Sequence[str], grades: Dict[str, int], k: int) -> float:
    import math

    total = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        grade = grades.get(cid, 0)
        if grade > 0:
            total += (2**grade - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(ranked_ids: Sequence[str], grades: Dict[str, int], k: int) -> float:
    ideal_order = sorted(grades.values(), reverse=True)[:k]
    import math

    ideal_dcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal_order, start=1))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, grades, k) / ideal_dcg


@dataclass
class QueryResult:
    query: str
    ranked_ids: List[str]
    latency_ms: float


@dataclass
class PipelineMetrics:
    name: str
    recall_at: Dict[int, float]
    mrr: float
    ndcg_at: Dict[int, float]
    precision_at_5: float
    avg_latency_ms: float

    def as_row(self) -> Dict[str, float]:
        return {
            "pipeline": self.name,
            "Recall@1": round(self.recall_at[1], 3),
            "Recall@3": round(self.recall_at[3], 3),
            "Recall@5": round(self.recall_at[5], 3),
            "Recall@10": round(self.recall_at[10], 3),
            "MRR": round(self.mrr, 3),
            "nDCG@5": round(self.ndcg_at[5], 3),
            "nDCG@10": round(self.ndcg_at[10], 3),
            "Precision@5": round(self.precision_at_5, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


def aggregate_metrics(name: str, per_query: List[Tuple[GoldenQuery, QueryResult]]) -> PipelineMetrics:
    recalls = {k: [] for k in (1, 3, 5, 10)}
    ndcgs = {k: [] for k in (5, 10)}
    mrrs = []
    precisions = []
    latencies = []

    for gq, qr in per_query:
        relevant_ids = gq.all_relevant_ids()
        grades = gq.graded_relevance()
        for k in recalls:
            recalls[k].append(recall_at_k(qr.ranked_ids, relevant_ids, k))
        for k in ndcgs:
            ndcgs[k].append(ndcg_at_k(qr.ranked_ids, grades, k))
        mrrs.append(reciprocal_rank(qr.ranked_ids, relevant_ids))
        precisions.append(precision_at_k(qr.ranked_ids, relevant_ids, 5))
        latencies.append(qr.latency_ms)

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return PipelineMetrics(
        name=name,
        recall_at={k: avg(v) for k, v in recalls.items()},
        mrr=avg(mrrs),
        ndcg_at={k: avg(v) for k, v in ndcgs.items()},
        precision_at_5=avg(precisions),
        avg_latency_ms=avg(latencies),
    )


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------


def run_pipeline_a_faiss_only(service: EmbeddingRAGService, queries: List[GoldenQuery], k: int = 20) -> List[Tuple[GoldenQuery, QueryResult]]:
    """A: E5 -> FAISS. Rerank/deterministic relevance YOK - saf embedding_score sirasi."""
    results = []
    for gq in queries:
        started = time.perf_counter()
        query_vector = service._provider.embed_query(gq.query)  # noqa: SLF001 - benchmark-only dogrudan erisim
        import numpy as np

        scores, indices = service._index.search(np.expand_dims(query_vector, axis=0), k)  # noqa: SLF001
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranked_ids = [service._documents[idx]["chunk_id"] for idx in indices[0] if idx != -1]  # noqa: SLF001
        results.append((gq, QueryResult(query=gq.query, ranked_ids=ranked_ids, latency_ms=latency_ms)))
    return results


def run_pipeline_b_deterministic(service: EmbeddingRAGService, queries: List[GoldenQuery], candidate_k: int = 20) -> List[Tuple[GoldenQuery, QueryResult]]:
    """B: E5 -> FAISS -> deterministic relevance (mevcut production yolu, threshold KAPALI - siralama icin tum adaylar tutulur)."""
    results = []
    for gq in queries:
        started = time.perf_counter()
        docs = service.query(gq.query, top_k=candidate_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranked_ids = [d.chunk_id for d in docs]
        results.append((gq, QueryResult(query=gq.query, ranked_ids=ranked_ids, latency_ms=latency_ms)))
    return results


def run_pipeline_c_cross_encoder(
    service: EmbeddingRAGService, queries: List[GoldenQuery], cross_encoder_model: str, candidate_k: int = 20, rerank_top_n: int = 10
) -> List[Tuple[GoldenQuery, QueryResult]]:
    """C: E5 -> FAISS -> deterministic relevance (adaylari sirala) -> LOCAL Cross-Encoder (top-N'i yeniden sirala)."""
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(cross_encoder_model, max_length=512)
    results = []
    for gq in queries:
        started = time.perf_counter()
        docs = service.query(gq.query, top_k=candidate_k)
        rerank_pool = docs[:rerank_top_n]
        if rerank_pool:
            pairs = [(gq.query, d.text) for d in rerank_pool]
            ce_scores = ce.predict(pairs)
            reordered = [d for d, _ in sorted(zip(rerank_pool, ce_scores), key=lambda x: x[1], reverse=True)]
        else:
            reordered = []
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranked_ids = [d.chunk_id for d in reordered] + [d.chunk_id for d in docs[rerank_top_n:]]
        results.append((gq, QueryResult(query=gq.query, ranked_ids=ranked_ids, latency_ms=latency_ms)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-encoder", type=str, default=None, help="HF Cross-Encoder model adi - verilirse C pipeline'i da calisir.")
    parser.add_argument("--json-out", type=str, default=None, help="Sonuclari JSON olarak da yaz.")
    args = parser.parse_args()

    config = load_config()
    service = EmbeddingRAGService(config.memory.embedding, config.memory.faiss, config.memory.reranker)
    service.seed_default_regulations()
    print(f"Corpus: {service.document_count()} chunk (corpus_source={service.corpus_source})\n")

    rows = []
    per_query_a = run_pipeline_a_faiss_only(service, GOLDEN_QUERIES)
    metrics_a = aggregate_metrics("A: E5+FAISS", per_query_a)
    rows.append(metrics_a.as_row())

    per_query_b = run_pipeline_b_deterministic(service, GOLDEN_QUERIES)
    metrics_b = aggregate_metrics("B: E5+FAISS+deterministic", per_query_b)
    rows.append(metrics_b.as_row())

    metrics_c = None
    if args.cross_encoder:
        per_query_c = run_pipeline_c_cross_encoder(service, GOLDEN_QUERIES, args.cross_encoder)
        metrics_c = aggregate_metrics(f"C: +CrossEncoder({args.cross_encoder})", per_query_c)
        rows.append(metrics_c.as_row())

    headers = list(rows[0].keys())
    widths = [max(len(str(r[h])) for r in rows + [dict(zip(headers, headers))]) for h in headers]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(w) for h, w in zip(headers, widths)))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON yazildi: {args.json_out}")


if __name__ == "__main__":
    main()
