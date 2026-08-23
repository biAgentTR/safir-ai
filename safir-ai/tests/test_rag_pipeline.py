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
# Dashboard telemetrisi: get_last_query_telemetry()
# ---------------------------------------------------------------------------


def test_query_telemetry_captures_real_candidate_final_and_scores(tmp_path) -> None:
    reranker = _FakeReranker(score_by_substring={"ALAKALI": 0.9})
    service = _make_service(tmp_path, candidate_k=5, top_k=5, reranker=reranker, score_threshold=0.10)
    service.add_documents(["Bu metin ALAKALI bir mevzuat."])

    service.query("sorgu")
    telemetry = service.get_last_query_telemetry()

    assert telemetry is not None
    assert telemetry.candidate_count == 1
    assert telemetry.final_count == 1
    assert telemetry.zero_result is False
    assert telemetry.retrieval_status == "reranked"
    assert telemetry.avg_rerank_score == pytest.approx(0.9)
    assert telemetry.total_latency_ms >= 0.0
    assert telemetry.results[0].selected is True


def test_query_telemetry_on_zero_result_reports_zero_not_fabricated(tmp_path) -> None:
    reranker = _FakeReranker(default_score=0.01)
    service = _make_service(tmp_path, candidate_k=5, top_k=5, reranker=reranker, score_threshold=0.10)
    service.add_documents(["konu A hakkinda metin"])

    service.query("alakasiz bir sorgu")
    telemetry = service.get_last_query_telemetry()

    assert telemetry.zero_result is True
    assert telemetry.final_count == 0
    assert telemetry.avg_rerank_score is None  # 0 secilen sonuc -> ortalama YOK (0.0 DEGIL)
    assert all(r.selected is False for r in telemetry.results)


def test_query_telemetry_on_reranker_failure_marks_reranker_unavailable(tmp_path) -> None:
    reranker = _FakeReranker(fail=True)
    service = _make_service(tmp_path, candidate_k=5, top_k=5, reranker=reranker)
    service.add_documents(["kesinlikle alakali bir mevzuat metni"])

    service.query("sorgu")
    telemetry = service.get_last_query_telemetry()

    assert telemetry.retrieval_status == "reranker_unavailable"
    assert telemetry.zero_result is True
    assert telemetry.rerank_latency_ms is not None


def test_query_telemetry_is_none_before_any_query(tmp_path) -> None:
    service = _make_service(tmp_path, candidate_k=5, top_k=5, reranker=None)
    assert service.get_last_query_telemetry() is None


def test_query_telemetry_on_empty_index_reports_empty_index_status(tmp_path) -> None:
    service = _make_service(tmp_path, candidate_k=5, top_k=5, reranker=None)
    results = service.query("hicbir dokuman yokken sorgu")
    telemetry = service.get_last_query_telemetry()

    assert results == []
    assert telemetry.retrieval_status == "empty_index"
    assert telemetry.zero_result is True
    assert telemetry.candidate_count == 0


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


# ---------------------------------------------------------------------------
# 6. Icerik-tabanli retrieval dogrulugu (RAG P0: "chunks var ama retrieval
#    dogru sonuc veriyor mu?" sorusunun deterministik cevabi)
# ---------------------------------------------------------------------------


class _VocabEmbeddingProvider:
    """Corpus'a ozel, COLLISION-SIZ (hash yok, gercek kelime->boyut haritasi) sahte embedding sağlayıcı.

    `_FakeEmbeddingProvider` (16 boyut, hash-mod-tabanli) mevcut testler icin
    yeterlidir (yalnizca akis mekanigini dogrularlar), ama kisa cumlelerde
    hash CARPISMASI riski tasir - bir icerik-dogrulugu testi icin (bu dogru
    dokumani mi getiriyor?) bu risk kabul edilemez (yanlis-negatif/pozitif
    testi anlamsizlastirir). Bu siniftaki her benzersiz kelime KENDI boyutuna
    sahiptir - carpisma MATEMATIKSEL OLARAK IMKANSIZDIR, hala GERCEK Gemini
    API'sine bagimli DEGILDIR (bkz. gorev tanimi 8. bolum: "unit/in-memory
    deterministic alternatif kullan").
    """

    def __init__(self, vocabulary: list[str]) -> None:
        self._index = {word: i for i, word in enumerate(vocabulary)}

    @property
    def dimension(self) -> int:
        return len(self._index)

    def _vector_for(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self._index), dtype="float32")
        for token in text.lower().split():
            if token in self._index:
                vector[self._index[token]] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def embed_documents(self, texts):
        return np.array([self._vector_for(t) for t in texts], dtype="float32")

    def embed_query(self, text):
        return self._vector_for(text)


def test_content_based_retrieval_ranks_the_relevant_document_first(tmp_path, monkeypatch) -> None:
    """Kucuk, deterministik iki-dokumanli corpus: dogru sorgu dogru dokumani getirmeli (RAG P0).

    Zinciri (document -> chunk -> embed -> FAISS index -> query embed ->
    retrieval) GERCEK implementasyon kodunun (`add_documents`/`query`)
    UZERINDEN, reranker'i DEVRE DISI birakarak (`reranker=None`) SAF
    embedding-tabanli ayirt ediciligi dogrular - RAG mekanizmasinin
    "chunk'lar var ama alakasiz sonuc donuyor" seklinde sessizce bozuk
    OLMADIGINI kanitlar. `_VocabEmbeddingProvider` carpisma-siz oldugu icin
    bu, "test corpus'u kucuk oldugu icin tesadufen gecti" degil, gercekten
    dogru dokumanin secildiginin kaniti.
    """
    document_a = "Forklift çalışma alanında yaya bulunması yasaktır."
    document_b = "Yangın söndürücülerin aylık kontrolü yapılmalıdır."
    query_a = "Forklift yakınında yaya bulunması"
    query_b = "Yangın söndürücü kontrolü"

    vocabulary = sorted(
        {tok for text in (document_a, document_b, query_a, query_b) for tok in text.lower().split()}
    )
    monkeypatch.setattr(
        rag_module, "build_embedding_provider", lambda **kwargs: _VocabEmbeddingProvider(vocabulary)
    )

    embedding_config = EmbeddingConfig(provider="gemini", model_name="fake-model", output_dimensionality=len(vocabulary))
    faiss_config = FaissMemoryConfig(
        index_path=str(tmp_path / "index.faiss"), embedding_model="fake-model", top_k=2, candidate_k=5
    )
    service = EmbeddingRAGService(embedding_config, faiss_config, RerankerConfig(enabled=False))
    service.add_documents([document_a, document_b])

    forklift_results = service.query(query_a)
    assert forklift_results
    assert forklift_results[0].text == document_a

    yangin_results = service.query(query_b)
    assert yangin_results
    assert yangin_results[0].text == document_b


# ---------------------------------------------------------------------------
# 7. Persisted index round-trip + corpus_source (RAG P0: "index/ diskte yok,
#    sistem SESSIZCE placeholder'a duser" bulgusunun dogrulama testleri)
# ---------------------------------------------------------------------------


def test_persisted_index_round_trip_sets_corpus_source_and_survives_reload(tmp_path, monkeypatch) -> None:
    """`persist()` ile yazilan bir index, TAZE bir `EmbeddingRAGService` tarafindan gercekten geri yuklenebilmeli.

    Gercek `data/knowledge_base/index/` dizinine DOKUNMAMAK icin modul-
    seviyesi path sabitleri (`_KB_INDEX_DIR`/`_INDEX_FILE`/`_DOCUMENTS_FILE`/
    `_INDEX_META_FILE`) `tmp_path`e yonlendirilir - `persist()`/
    `_try_load_persisted_index()`in KENDISI (gercek production kodu)
    degistirilmeden cagrilir.
    """
    monkeypatch.setattr(rag_module, "_KB_INDEX_DIR", tmp_path)
    monkeypatch.setattr(rag_module, "_INDEX_FILE", tmp_path / "faiss.index")
    monkeypatch.setattr(rag_module, "_DOCUMENTS_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(rag_module, "_INDEX_META_FILE", tmp_path / "index_meta.json")

    builder = _make_service(tmp_path, candidate_k=5, top_k=3, reranker=None)
    builder.add_documents(["Forklift çalışma alanında yaya bulunması yasaktır."])
    builder.persist()

    assert (tmp_path / "faiss.index").exists()
    assert (tmp_path / "documents.json").exists()
    assert (tmp_path / "index_meta.json").exists()

    fresh = _make_service(tmp_path, candidate_k=5, top_k=3, reranker=None)
    assert fresh.corpus_source == "unseeded"

    loaded = fresh._try_load_persisted_index()

    assert loaded is True
    assert fresh.corpus_source == "persisted_index"
    assert fresh.document_count() == 1

    results = fresh.query("Forklift yakınında yaya")
    telemetry = fresh.get_last_query_telemetry()

    assert telemetry.corpus_source == "persisted_index"
    assert results
    assert results[0].text == "Forklift çalışma alanında yaya bulunması yasaktır."


def test_fallback_placeholder_corpus_source_when_no_persisted_index(tmp_path, monkeypatch) -> None:
    """Persisted index dosyalari yoksa `seed_default_regulations()` acikca `fallback_placeholder`e duser."""
    monkeypatch.setattr(rag_module, "_KB_INDEX_DIR", tmp_path / "does_not_exist")
    monkeypatch.setattr(rag_module, "_INDEX_FILE", tmp_path / "does_not_exist" / "faiss.index")
    monkeypatch.setattr(rag_module, "_DOCUMENTS_FILE", tmp_path / "does_not_exist" / "documents.json")
    monkeypatch.setattr(rag_module, "_INDEX_META_FILE", tmp_path / "does_not_exist" / "index_meta.json")

    service = _make_service(tmp_path, candidate_k=5, top_k=3, reranker=None)
    service.seed_default_regulations()

    assert service.corpus_source == "fallback_placeholder"
    assert service.document_count() == len(rag_module.DEFAULT_ISG_REGULATIONS)


def test_retrieval_result_carries_chunk_id_document_id_and_scores_end_to_end(tmp_path) -> None:
    """HEDEF 4/6: her retrieval sonucunda chunk_id + document_id + skor GORULEBILIR olmali (UYDURULMAMIS metadata)."""
    records = _load_kb_chunk_records()
    if not records:
        pytest.skip("data/knowledge_base/chunks/ bos - bu test gercek KB corpus'una bagimlidir.")

    reranker = _FakeReranker(score_by_substring={"forklift": 0.9}, default_score=0.02)
    service = _make_service(tmp_path, candidate_k=20, top_k=3, reranker=reranker, score_threshold=0.10)
    service._add_structured_documents(records[:200])
    service._corpus_source = "chunks_rebuild"

    results = service.query("forklift yaya güvenliği")
    telemetry = service.get_last_query_telemetry()

    assert telemetry.corpus_source == "chunks_rebuild"
    for result_telemetry in telemetry.results:
        # her aday icin chunk_id/document_id UYDURULMADAN tasinmis olmali
        # (gercek chunk kaydindan geliyorsa None olamaz).
        assert result_telemetry.chunk_id is not None
        assert result_telemetry.document_id is not None
    for doc in results:
        assert doc.chunk_id is not None
        assert doc.rerank_score is not None
