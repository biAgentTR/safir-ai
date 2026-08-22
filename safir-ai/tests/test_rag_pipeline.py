"""Iki-asamali RAG retrieval (FAISS candidate_k -> Gemini rerank -> score_threshold) icin agsiz birim testleri.

Hem embedding hem rerank sahte/deterministik nesnelerle degistirilir - bu
dosyadaki hicbir test GERCEK bir API cagrisi yapmaz veya semantik kalite
IDDIA ETMEZ; yalnizca iki-asamali retrieval'in MEKANIGINI (candidate_k,
metadata korunumu, threshold filtreleme, reranker hatasi -> bos sonuc)
dogrular. Gercek API'lere karsi calisan smoke test icin bkz.
`scripts/rag_smoke_test.py`.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.memory import embedding_rag_service as rag_module
from src.memory.embedding_rag_service import EmbeddingRAGService, _load_kb_chunk_records
from src.memory.reranker import RerankerUnavailableError
from src.utils.config_loader import EmbeddingConfig, FaissMemoryConfig, RerankerConfig


class _FakeEmbeddingProvider:
    _DIMENSION = 16

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def _vector_for(self, text: str) -> np.ndarray:
        vector = np.zeros(self._DIMENSION, dtype="float32")
        for token in text.lower().split():
            token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vector[token_hash % self._DIMENSION] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def embed_documents(self, texts):
        return np.array([self._vector_for(t) for t in texts], dtype="float32")

    def embed_query(self, text):
        return self._vector_for(text)


class _FakeReranker:
    """`substring -> relevance_score` haritasina gore deterministik skor donduren sahte reranker."""

    def __init__(self, score_by_substring=None, default_score: float = 0.5, fail: bool = False):
        self._score_by_substring = score_by_substring or {}
        self._default_score = default_score
        self._fail = fail
        self.calls = []

    def rerank(self, query, candidates):
        self.calls.append((query, list(candidates)))
        if self._fail:
            raise RerankerUnavailableError("sahte reranker hatasi")
        scored = []
        for i, text in enumerate(candidates):
            score = self._default_score
            for substring, s in self._score_by_substring.items():
                if substring in text:
                    score = s
                    break
            scored.append((i, score))
        return sorted(scored, key=lambda t: t[1], reverse=True)


@pytest.fixture(autouse=True)
def _patch_embedding_provider(monkeypatch):
    monkeypatch.setattr(rag_module, "build_embedding_provider", lambda **kwargs: _FakeEmbeddingProvider())


def _make_service(tmp_path, candidate_k=20, top_k=5, reranker=None, score_threshold=0.10) -> EmbeddingRAGService:
    embedding_config = EmbeddingConfig(provider="gemini", model_name="fake-model", output_dimensionality=16)
    faiss_config = FaissMemoryConfig(
        index_path=str(tmp_path / "index.faiss"),
        embedding_model="fake-model",
        top_k=top_k,
        candidate_k=candidate_k,
    )
    reranker_config = RerankerConfig(enabled=reranker is not None, score_threshold=score_threshold, top_k=top_k)
    service = EmbeddingRAGService(embedding_config, faiss_config, reranker_config)
    if reranker is not None:
        service._reranker = reranker  # test-only: sahte reranker'i dogrudan enjekte et
    return service


# ---------------------------------------------------------------------------
# 2. FAISS retrieval: candidate_k + metadata korunumu
# ---------------------------------------------------------------------------


def test_candidate_k_limits_faiss_search_breadth(tmp_path) -> None:
    service = _make_service(tmp_path, candidate_k=3, top_k=3, reranker=None)
    service.add_documents([f"belge {i} icerik metni" for i in range(10)])

    results = service.query("belge sorgusu")

    # Reranker devre disi: final sonuc, FAISS'in dondurdugu EN FAZLA
    # candidate_k adayla sinirlidir (top_k=candidate_k=3 oldugu icin ayrica kesilmez).
    assert len(results) <= 3


def test_metadata_fields_are_preserved_through_query(tmp_path) -> None:
    service = _make_service(tmp_path, candidate_k=5, top_k=5, reranker=None)
    service._add_structured_documents(
        [
            {
                "chunk_id": "doc_a__madde_1",
                "document_id": "doc_a",
                "document_title": "Test Yönetmeliği",
                "level": "madde",
                "article_number": "1",
                "article_title": None,
                "is_annex": False,
                "page_start": 3,
                "page_end": 3,
                "source_url": "https://example.gov.tr/doc_a",
                "institution": "Test Bakanlığı",
                "publication_date": "2020-01-01",
                "text": "yangın ve duman tespiti hakkında hükümler",
            }
        ]
    )

    results = service.query("yangın duman")

    assert len(results) == 1
    doc = results[0]
    assert doc.document_id == "doc_a"
    assert doc.document_title == "Test Yönetmeliği"
    assert doc.article_number == "1"
    assert doc.page_start == 3 and doc.page_end == 3
    assert doc.source_url == "https://example.gov.tr/doc_a"
    assert doc.institution == "Test Bakanlığı"
    assert doc.publication_date == "2020-01-01"
    assert doc.rerank_score is None  # reranker devre disi
    assert doc.embedding_score == doc.score


def test_legacy_plain_string_documents_have_none_metadata(tmp_path) -> None:
    """Eski (metadata'siz) `add_documents([str, ...])` yolu, eksik alanlari acikca `None` tasimali - UYDURMAMALI."""
    service = _make_service(tmp_path, candidate_k=5, top_k=5, reranker=None)
    service.add_documents(["metadata'siz duz metin"])

    results = service.query("metadata")

    assert len(results) == 1
    assert results[0].document_id is None
    assert results[0].document_title is None
    assert results[0].source_url is None


# ---------------------------------------------------------------------------
# 4. Threshold davranisi
# ---------------------------------------------------------------------------


def test_below_threshold_results_are_excluded(tmp_path) -> None:
    reranker = _FakeReranker(score_by_substring={"ALAKALI": 0.9, "ALAKASIZ": 0.02})
    service = _make_service(tmp_path, candidate_k=5, top_k=5, reranker=reranker, score_threshold=0.10)
    service.add_documents(["Bu metin ALAKALI bir mevzuat.", "Bu metin ALAKASIZ bir konu."])

    results = service.query("sorgu")

    assert len(results) == 1
    assert "ALAKALI" in results[0].text
    assert results[0].rerank_score == pytest.approx(0.9)


def test_all_below_threshold_returns_empty_list_not_random_topk(tmp_path) -> None:
    reranker = _FakeReranker(default_score=0.01)
    service = _make_service(tmp_path, candidate_k=5, top_k=5, reranker=reranker, score_threshold=0.10)
    service.add_documents(["konu A hakkinda metin", "konu B hakkinda metin", "konu C hakkinda metin"])

    results = service.query("alakasiz bir sorgu")

    assert results == []  # 0 sonuc GECERLIDIR - rastgele top-k UYDURULMAZ


def test_reranker_failure_returns_empty_not_embedding_topk(tmp_path) -> None:
    """Reranker basarisiz olursa, embedding top-k'nin SESSIZCE final sonuc gibi DONMEMESI gerekir (bkz. gorev tanimi 7. bolum)."""
    reranker = _FakeReranker(fail=True)
    service = _make_service(tmp_path, candidate_k=5, top_k=5, reranker=reranker)
    service.add_documents(["kesinlikle alakali bir mevzuat metni"])

    results = service.query("sorgu")

    assert results == []


# ---------------------------------------------------------------------------
# 5. Uctan uca RAG (MOCK embedding + MOCK reranker - semantik kalite IDDIA EDILMEZ)
# ---------------------------------------------------------------------------


def test_end_to_end_query_against_real_748_chunks_with_mocks(tmp_path) -> None:
    """Gercek 748 KB chunk'inin ustunde, SAHTE embedding+rerank ile uctan uca akisi dogrular.

    NOT: `_FakeEmbeddingProvider` hash-tabanlidir, GERCEK semantik benzerlik
    OLCMEZ - bu test yalnizca akisin (candidate_k -> rerank -> threshold ->
    yapilandirilmis RetrievedDocument) uctan uca CALISTIGINI ve gercek
    chunk metadata'sinin (document_title/article_number/source_url) dogru
    tasindigini dogrular; sonuclarin semantik ALAKASINI degil.
    """
    records = _load_kb_chunk_records()
    if not records:
        pytest.skip("data/knowledge_base/chunks/ bos - bu test gercek KB corpus'una bagimlidir.")

    reranker = _FakeReranker(score_by_substring={"yangın": 0.8, "duman": 0.7}, default_score=0.05)
    service = _make_service(tmp_path, candidate_k=20, top_k=5, reranker=reranker, score_threshold=0.10)
    service._add_structured_documents(records[:200])  # tum 748'i embed etmek testte gereksiz yavas olur

    results = service.query("yangın duman kontrolsüz açık alev")

    assert isinstance(results, list)
    for doc in results:
        assert doc.rerank_score is not None
        assert doc.rerank_score >= 0.10
        # gercek KB'den geldigi icin en az document_id dolu olmali (UYDURULMAMIS metadata)
        assert doc.document_id is not None
