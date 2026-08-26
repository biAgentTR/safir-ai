"""Iki-asamali RAG retrieval (FAISS candidate_k -> deterministik relevance skorlama -> score_threshold) icin birim testleri.

2026-08-24 (RAG RERANKER DETERMINIZATION): ikinci asama artik bir LLM'e
SORULMUYOR - `src/rag/deterministic_reranker.py`nin TAMAMEN yerel,
agirlikli-toplam algoritmasidir; bu dosyadaki testler GERCEK relevance
skorlamayi (yalnizca embedding'i SAHTE, hash-tabanli bir saglayiciyla)
calistirir - HICBIR AG/LLM cagrisi yapilmaz. `deterministic_reranker`nin
saf-fonksiyon seviyesindeki izole testleri icin bkz.
`tests/test_deterministic_reranker.py`.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from src.rag import embedding_rag_service as rag_module
from src.rag.embedding_rag_service import EmbeddingRAGService, RetrievedDocument, _load_kb_chunk_records
from src.utils.config_loader import EmbeddingConfig, QdrantMemoryConfig, RerankerConfig, SQLiteMemoryConfig


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


@pytest.fixture(autouse=True)
def _patch_embedding_provider(monkeypatch):
    monkeypatch.setattr(rag_module, "build_embedding_provider", lambda **kwargs: _FakeEmbeddingProvider())


def _make_service(tmp_path, candidate_k=20, top_k=5, enable_relevance=False, score_threshold=0.10) -> EmbeddingRAGService:
    """RAG RERANKER DETERMINIZATION: `enable_relevance=True`, ARTIK bir LLM mock enjekte ETMEZ -

    ikinci-asama relevance skorlama, servisin KENDI deterministik
    `deterministic_reranker.score_candidate()` cagrisidir (bkz. `EmbeddingRAGService.query()`).
    """
    embedding_config = EmbeddingConfig(provider="local", model_name="fake-model", output_dimensionality=16)
    qdrant_config = QdrantMemoryConfig(url=":memory:", top_k=top_k, candidate_k=candidate_k)
    reranker_config = RerankerConfig(enabled=enable_relevance, score_threshold=score_threshold, top_k=top_k)
    return EmbeddingRAGService(embedding_config, qdrant_config, reranker_config)


class _FakeCrossEncoder:
    """Deterministik, ag/model-yuklemesi GEREKTIRMEYEN sahte Cross-Encoder - yalnizca `score()` sozlesmesini test eder.

    Skoru, `texts[i]`nin uzunlugundan turetir (deterministik, tekrar-uretilebilir) -
    GERCEK bir cross-encoder modelinin KENDISINI test ETMEZ (bu, `LocalCrossEncoderReranker`
    seviyesinde AYRI, ag erisimi olan bir ortamda dogrulanmalidir); yalnizca
    `EmbeddingRAGService.query()`nin Cross-Encoder ADIMINI (yeniden siralama,
    gate bypass etmeme, provenance) DOGRU cagirip cagirmadigini test eder.
    """

    def __init__(self):
        self.call_count = 0
        self.last_query = None
        self.last_texts = None

    def score(self, query, texts):
        self.call_count += 1
        self.last_query = query
        self.last_texts = list(texts)
        return [float(len(t)) for t in texts]


def _make_service_with_cross_encoder(
    tmp_path, cross_encoder, candidate_k=20, top_k=5, enable_relevance=True, score_threshold=0.0
) -> EmbeddingRAGService:
    embedding_config = EmbeddingConfig(provider="local", model_name="fake-model", output_dimensionality=16)
    qdrant_config = QdrantMemoryConfig(url=":memory:", top_k=top_k, candidate_k=candidate_k)
    reranker_config = RerankerConfig(enabled=enable_relevance, score_threshold=score_threshold, top_k=top_k)
    return EmbeddingRAGService(embedding_config, qdrant_config, reranker_config, cross_encoder=cross_encoder)


# ---------------------------------------------------------------------------
# 2. FAISS retrieval: candidate_k + metadata korunumu
# ---------------------------------------------------------------------------


def test_candidate_k_limits_faiss_search_breadth(tmp_path) -> None:
    service = _make_service(tmp_path, candidate_k=3, top_k=3)
    service.add_documents([f"belge {i} icerik metni" for i in range(10)])

    results = service.query("belge sorgusu")

    # Reranker devre disi: final sonuc, FAISS'in dondurdugu EN FAZLA
    # candidate_k adayla sinirlidir (top_k=candidate_k=3 oldugu icin ayrica kesilmez).
    assert len(results) <= 3


def test_metadata_fields_are_preserved_through_query(tmp_path) -> None:
    service = _make_service(tmp_path, candidate_k=5, top_k=5)
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
    assert doc.relevance_score is None  # reranker devre disi
    assert doc.embedding_score == doc.score
    assert doc.semantic_score is None  # component skorlari da devre-disiyken UYDURULMAZ


def test_legacy_plain_string_documents_have_none_metadata(tmp_path) -> None:
    """Eski (metadata'siz) `add_documents([str, ...])` yolu, eksik alanlari acikca `None` tasimali - UYDURMAMALI."""
    service = _make_service(tmp_path, candidate_k=5, top_k=5)
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
    """HEDEF 6: gercek deterministik skorlama - sorguyla lexical/semantic olarak orusen chunk ACCEPTED, ORUSMEYEN REJECTED olur."""
    service = _make_service(tmp_path, candidate_k=5, top_k=5, enable_relevance=True, score_threshold=0.10)
    service.add_documents(["Bu metin ALAKALI bir mevzuat.", "Bu metin ALAKASIZ bir konu."])

    results = service.query("ALAKALI mevzuat sorgusu")

    assert len(results) == 1
    assert "ALAKALI" in results[0].text
    assert results[0].relevance_score is not None
    assert results[0].relevance_score >= 0.10


def test_relevance_status_and_reason_are_stamped_deterministically_on_every_candidate(tmp_path) -> None:
    """HEDEF 1/7/15: 'neden secildi/elendi?' sorusu, AYRI bir yapi uretmeden dogrudan RetrievedDocument uzerinde cevaplanabilmeli; ayni girdi ile karar HER ZAMAN AYNI (deterministik)."""
    service = _make_service(tmp_path, candidate_k=5, top_k=5, enable_relevance=True, score_threshold=0.10)
    service.add_documents(["Bu metin ALAKALI bir mevzuat.", "Bu metin ALAKASIZ bir konu."])

    telemetry_accepted_reasons = []
    telemetry_rejected_reasons = []
    for _ in range(3):  # ayni sorgu 3 kez - karar HER SEFERINDE ayni olmali
        results = service.query("ALAKALI mevzuat sorgusu")
        telemetry = service.get_last_query_telemetry()
        assert len(results) == 1
        assert results[0].relevance_status == "accepted"
        assert "threshold" in results[0].relevance_reason
        rejected = [r for r in telemetry.results if r.relevance_status == "rejected"]
        assert len(rejected) == 1
        assert "threshold" in rejected[0].relevance_reason
        telemetry_accepted_reasons.append(results[0].relevance_reason)
        telemetry_rejected_reasons.append(rejected[0].relevance_reason)

    assert len(set(telemetry_accepted_reasons)) == 1  # 3 cagri, TEK bir gerekce metni -> deterministik
    assert len(set(telemetry_rejected_reasons)) == 1


def test_embedding_score_is_not_relabeled_as_confidence(tmp_path) -> None:
    """HEDEF 13: `embedding_score`, kalibre edilmis bir olasilik/confidence DEGIL, FAISS cosine benzerligidir - hicbir alan/telemetri onu 'confidence' olarak ADLANDIRMAZ."""
    service = _make_service(tmp_path, candidate_k=5, top_k=5)
    service.add_documents(["bir mevzuat metni"])

    results = service.query("sorgu")
    telemetry = service.get_last_query_telemetry()

    assert hasattr(results[0], "embedding_score")
    assert not hasattr(results[0], "confidence")
    assert not hasattr(telemetry.results[0], "confidence")
    import dataclasses

    assert "confidence" not in {f.name for f in dataclasses.fields(results[0])}
    assert "confidence" not in {f.name for f in dataclasses.fields(telemetry.results[0])}


def test_relevance_scoring_makes_no_network_or_llm_call(tmp_path) -> None:
    """HEDEF 7/8/9: relevance skorlama artik bir LLM/API'ye BAGIMLI DEGIL - `socket`/`httpx` hic tetiklenmeden calisir; Groq/Gemini API anahtari OLMADAN da RAG reranking calisiyor."""
    import os

    for env_var in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        os.environ.pop(env_var, None)  # bilinclli olarak API anahtari TANIMSIZ birakiliyor

    service = _make_service(tmp_path, candidate_k=5, top_k=5, enable_relevance=True, score_threshold=0.10)
    service.add_documents(["forklift yaya güvenlik mesafesi hakkında hükümler"])

    results = service.query("forklift yaya güvenliği")  # API anahtari yok, yine de CALISMALI

    assert results  # 429/400 gibi bir hataya DUSMEDEN basariyla sonuc uretti


def test_all_below_threshold_returns_empty_list_not_random_topk(tmp_path, monkeypatch) -> None:
    """`_FakeEmbeddingProvider` (16 boyut, hash-tabanli) TAMAMEN alakasiz metinler arasinda bile
    tesadufi hash CARPISMASI uretebilir (kucuk vektor uzayi) - bu test icin carpisma-siz
    `_VocabEmbeddingProvider` kullanilir (bkz. `test_content_based_retrieval_...`)."""
    docs = ["forklift yaya güvenlik mesafesi", "elektrik pano kilitleme etiketleme", "kimyasal madde depolama etiketleme"]
    query = "gezegen yıldız teleskop astronomi"
    vocabulary = sorted({tok for text in docs + [query] for tok in text.lower().split()})
    monkeypatch.setattr(rag_module, "build_embedding_provider", lambda **kwargs: _VocabEmbeddingProvider(vocabulary))

    embedding_config = EmbeddingConfig(provider="local", model_name="fake-model", output_dimensionality=len(vocabulary))
    qdrant_config = QdrantMemoryConfig(url=":memory:", top_k=5, candidate_k=5)
    service = EmbeddingRAGService(embedding_config, qdrant_config, RerankerConfig(enabled=True, score_threshold=0.10, top_k=5))
    service.add_documents(docs)

    results = service.query(query)

    assert results == []  # 0 sonuc GECERLIDIR - rastgele top-k UYDURULMAZ


# ---------------------------------------------------------------------------
# Dashboard telemetrisi: get_last_query_telemetry()
# ---------------------------------------------------------------------------


def test_query_telemetry_captures_real_candidate_final_and_scores(tmp_path) -> None:
    service = _make_service(tmp_path, candidate_k=5, top_k=5, enable_relevance=True, score_threshold=0.10)
    service.add_documents(["Bu metin ALAKALI bir mevzuat."])

    service.query("ALAKALI mevzuat sorgusu")
    telemetry = service.get_last_query_telemetry()

    assert telemetry is not None
    assert telemetry.candidate_count == 1
    assert telemetry.final_count == 1
    assert telemetry.zero_result is False
    assert telemetry.retrieval_status == "relevance_scored"
    assert telemetry.avg_relevance_score is not None
    assert telemetry.total_latency_ms >= 0.0
    assert telemetry.results[0].selected is True
    assert telemetry.results[0].rank == 1
    assert telemetry.results[0].relevance_status == "accepted"


def test_query_telemetry_on_zero_result_reports_zero_not_fabricated(tmp_path) -> None:
    service = _make_service(tmp_path, candidate_k=5, top_k=5, enable_relevance=True, score_threshold=0.10)
    service.add_documents(["forklift yaya güvenlik mesafesi"])

    service.query("gezegen yıldız teleskop astronomi")
    telemetry = service.get_last_query_telemetry()

    assert telemetry.zero_result is True
    assert telemetry.final_count == 0
    assert telemetry.retrieval_status == "insufficient_evidence"
    assert telemetry.avg_relevance_score is None  # 0 secilen sonuc -> ortalama YOK (0.0 DEGIL)
    assert all(r.selected is False for r in telemetry.results)


def test_query_telemetry_is_none_before_any_query(tmp_path) -> None:
    service = _make_service(tmp_path, candidate_k=5, top_k=5)
    assert service.get_last_query_telemetry() is None


def test_query_telemetry_on_empty_index_reports_empty_index_status(tmp_path) -> None:
    service = _make_service(tmp_path, candidate_k=5, top_k=5)
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
    """Gercek 748 KB chunk'inin ustunde, SAHTE (hash-tabanli) embedding + GERCEK deterministik relevance skorlama ile uctan uca akisi dogrular.

    NOT: `_FakeEmbeddingProvider` hash-tabanlidir, GERCEK semantik benzerlik
    OLCMEZ - ama `deterministic_reranker`nin lexical/phrase sinyalleri GERCEK
    metin uzerinde calisir (sorgu "yangın duman..." GERCEK chunk'lardaki bu
    kelimelerle orusur). Bu test akisin (candidate_k -> relevance skorlama ->
    threshold -> yapilandirilmis RetrievedDocument) uctan uca CALISTIGINI ve
    gercek chunk metadata'sinin (document_title/article_number/source_url)
    dogru tasindigini dogrular.
    """
    records = _load_kb_chunk_records()
    if not records:
        pytest.skip("data/knowledge_base/chunks/ bos - bu test gercek KB corpus'una bagimlidir.")

    service = _make_service(tmp_path, candidate_k=20, top_k=5, enable_relevance=True, score_threshold=0.10)
    service._add_structured_documents(records[:200])  # tum 748'i embed etmek testte gereksiz yavas olur

    results = service.query("yangın duman kontrolsüz açık alev")

    assert isinstance(results, list)
    for doc in results:
        assert doc.relevance_score is not None
        assert doc.relevance_score >= 0.10
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

    embedding_config = EmbeddingConfig(provider="local", model_name="fake-model", output_dimensionality=len(vocabulary))
    qdrant_config = QdrantMemoryConfig(url=":memory:", top_k=2, candidate_k=5)
    service = EmbeddingRAGService(embedding_config, qdrant_config, RerankerConfig(enabled=False))
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


def test_persisted_index_round_trip_sets_corpus_source_and_survives_reload(tmp_path) -> None:
    """`persist()` ile yazilan bir koleksiyon, TAZE bir `EmbeddingRAGService` tarafindan gercekten geri yuklenebilmeli.

    Gercek EVREN Qdrant orneğine DOKUNMAMAK icin YEREL DISK-tabanli bir
    Qdrant orneği kullanilir (`tmp_path` altinda - bkz.
    `_build_qdrant_client` dokustringi); `persist()`/
    `_try_load_qdrant_collection()`in KENDISI (gercek production kodu)
    degistirilmeden cagrilir.
    """
    qdrant_path = str(tmp_path / "qdrant_local")
    embedding_config = EmbeddingConfig(provider="local", model_name="fake-model", output_dimensionality=16)
    qdrant_config = QdrantMemoryConfig(url=qdrant_path, top_k=3, candidate_k=5)

    builder = EmbeddingRAGService(embedding_config, qdrant_config, RerankerConfig(enabled=False))
    builder.add_documents(["Forklift çalışma alanında yaya bulunması yasaktır."])
    builder.persist()
    builder._qdrant.close()  # noqa: SLF001 - yerel disk-tabanli Qdrant dosya kilidini serbest birak (tek-yazar)

    fresh = EmbeddingRAGService(embedding_config, qdrant_config, RerankerConfig(enabled=False))
    assert fresh.corpus_source == "unseeded"

    loaded = fresh._try_load_qdrant_collection()

    assert loaded is True
    assert fresh.corpus_source == "qdrant_collection"
    assert fresh.document_count() == 1

    results = fresh.query("Forklift yakınında yaya")
    telemetry = fresh.get_last_query_telemetry()

    assert telemetry.corpus_source == "qdrant_collection"
    assert results
    assert results[0].text == "Forklift çalışma alanında yaya bulunması yasaktır."


def test_seed_default_regulations_fails_fast_when_no_persisted_index(tmp_path) -> None:
    """Gorev tanimi 8. madde: doldurulmus bir Qdrant koleksiyonu yoksa DEFAULT_ISG_REGULATIONS gibi bir placeholder'a SESSIZ FALLBACK YOK - acik hata.

    `_make_service` (bkz. yukarida) her cagrida taze/izole bir `:memory:`
    Qdrant orneği kurar - bu testin sonucu, baska bir testin/repo'nun
    doldurdugu bir koleksiyondan BAGIMSIZ, deterministik olmalidir.
    """
    service = _make_service(tmp_path, candidate_k=5, top_k=3)

    with pytest.raises(rag_module.KnowledgeBaseNotBuiltError, match="build_knowledge_index"):
        service.seed_default_regulations()

    assert service.document_count() == 0
    assert service.corpus_source == "unseeded"


def test_retrieval_result_carries_chunk_id_document_id_and_scores_end_to_end(tmp_path) -> None:
    """HEDEF 4/6: her retrieval sonucunda chunk_id + document_id + skor GORULEBILIR olmali (UYDURULMAMIS metadata)."""
    records = _load_kb_chunk_records()
    if not records:
        pytest.skip("data/knowledge_base/chunks/ bos - bu test gercek KB corpus'una bagimlidir.")

    service = _make_service(tmp_path, candidate_k=20, top_k=3, enable_relevance=True, score_threshold=0.10)
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
        assert doc.relevance_score is not None


# ---------------------------------------------------------------------------
# RAG entegrasyon dogrulama turu (2026-08-24): GERCEK KB chunk metni + provenance'inin
# ContextBuilder -> agent prompt zincirinden kayipsiz gectigini dogrular
# (embedding/Qdrant'a BAGIMLI DEGIL - bkz. asagidaki test dokustringi).
# ---------------------------------------------------------------------------


def test_real_chunk_text_and_provenance_reach_context_builder_prompt(tmp_path) -> None:
    """HEDEF 1/4/5: GERCEK bir KB chunk'inin TAM METNI + provenance'i, ContextBuilder.to_prompt_block()'a (Agent'in GORDUGU metin) kayipsiz ulasiyor mu?

    2026-08-25 EVREN MIGRASYONU: bu test artik Qdrant'a/EVREN'e HIC
    BAGIMLI DEGIL - `_load_kb_chunk_records()` ile AYNI kaynak JSON
    chunk'lari (`data/knowledge_base/chunks/*.json`) dogrudan okunur (eskiden
    bu chunk'lar `documents.json` FAISS yan-dosyasi UZERINDEN okunuyordu -
    ama kaynak ICERIK AYNIYDI, yalnizca ERISIM yolu FAISS'e bagimliydi).
    """
    records = _load_kb_chunk_records()
    if not records:
        pytest.skip("data/knowledge_base/chunks/ bos - bu test gercek KB corpus'una bagimlidir.")

    from src.memory.context_builder import ContextBuilder
    from src.memory.event_store import EventStore

    # Gorev tanimindaki somut ornek: "Is Ekipmanlari ... Ek I, I.3.1" - GERCEK corpus'ta var mi dogrula.
    real_chunk = next(
        (d for d in records if d.get("document_id") == "is_ekipmanlari_yonetmeligi" and d.get("article_number") == "I.3.1"),
        None,
    )
    assert real_chunk is not None, "GERCEK corpus'ta is_ekipmanlari_yonetmeligi Ek I.3.1 chunk'i bulunamadi."
    assert real_chunk["chunk_id"] == "is_ekipmanlari_yonetmeligi__ek_alt_madde_I.3.1"

    retrieved = RetrievedDocument(
        text=real_chunk["text"],
        embedding_score=0.87,
        chunk_id=real_chunk["chunk_id"],
        document_id=real_chunk["document_id"],
        document_title=real_chunk["document_title"],
        article_number=real_chunk["article_number"],
        source_url=real_chunk["source_url"],
    )

    event_store = EventStore(SQLiteMemoryConfig(db_path=str(tmp_path / "events.db")))
    builder = ContextBuilder(event_store, guard=None)
    context = builder.build(
        vlm_description="Operator KKD olmadan is ekipmani kullaniyor.",
        user_prompt="Risk durumu nedir?",
        timestamp=10.0,
        semantically_related_chunks=[retrieved],
    )
    prompt_block = context.to_prompt_block()

    # Agent'in GORDUGU nihai metinde GERCEK chunk metni + TAM provenance (RAG PIPELINE
    # RECONSTRUCTION, gorev tanimi 12. bolum: "[RAG EVIDENCE N]" bloklari, kirpilmamis
    # metin, chunk_id/source_url dahil - artik yalnizca ilk 200 karakter DEGIL, TAMAMI).
    assert real_chunk["text"] in prompt_block
    assert "[RAG EVIDENCE 1]" in prompt_block
    assert f"document: {real_chunk['document_title']}" in prompt_block
    assert f"article: {real_chunk['article_number']}" in prompt_block
    assert f"chunk_id: {real_chunk['chunk_id']}" in prompt_block
    assert f"source_url: {real_chunk['source_url']}" in prompt_block
    assert context.semantically_related_chunks[0].chunk_id == real_chunk["chunk_id"]


# ---------------------------------------------------------------------------
# 2026-08-25 EVREN MIGRASYONU: GERCEK EVREN embedding (`bge-m3-embed`) ile
# calisan semantik retrieval testleri (eskiden GERCEK persisted FAISS index +
# lokal E5 modeli kullanirdi - artik Qdrant EVREN'e tahsis edilmis UZAK bir
# servis oldugu icin repo'ya PUSHLANAMAZ). `EVREN_API_KEY` ortam degiskeni
# tanimli DEGILSE (bu sandbox/CI ortaminda beklenen durum - evren-llmapi.
# ssyz.org.tr'ye ag erisimi YOK) TEMIZ bir sekilde SKIP edilir; gercek bir
# EVREN anahtariyla calistirildiginda GERCEK semantik kaliteyi dogrular.
# Qdrant TARAFI icin GERCEK EVREN sunucusuna ihtiyac YOKTUR - yerel disk-
# tabanli bir Qdrant orneği yeterlidir (semantik kalite YALNIZCA embedding
# modelinden gelir, vektor deposundan DEGIL).
# ---------------------------------------------------------------------------


@pytest.fixture
def _real_evren_rag_service(tmp_path):
    """GERCEK `EvrenEmbeddingProvider` (+ yerel Qdrant) ile kurulmus servis; EVREN_API_KEY yoksa/baglanti basarisizsa SKIP eder."""
    if not os.environ.get("EVREN_API_KEY", "").strip():
        pytest.skip("EVREN_API_KEY tanimli degil - gercek EVREN embedding testi atlandi.")

    records = _load_kb_chunk_records()
    if not records:
        pytest.skip("data/knowledge_base/chunks/ bos - bu test gercek KB corpus'una bagimlidir.")

    embedding_config = EmbeddingConfig(
        provider="evren",
        model_name="bge-m3-embed",
        output_dimensionality=1024,
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY",
    )
    qdrant_config = QdrantMemoryConfig(url=str(tmp_path / "qdrant_local"), top_k=5, candidate_k=20)
    service = EmbeddingRAGService(embedding_config, qdrant_config, RerankerConfig(enabled=False))
    try:
        service._add_structured_documents(records[:200])
        service._corpus_source = "chunks_rebuild"
        service.query("baglanti testi")  # gercek EVREN cagrisi burada olur
    except Exception as exc:  # noqa: BLE001 - ag erisimi yok/gecersiz anahtar, TEMIZ skip
        pytest.skip(f"Gercek EVREN embedding servisine baglanilamadi: {exc}")
    return service


@pytest.mark.parametrize(
    "query",
    [
        "forklift yaya güvenliği",
        "elektrik kilitleme etiketleme",
        "kimyasal madde etiketleme",
        "kapalı alan girişi",
    ],
)
def test_real_semantic_queries_against_real_corpus_return_relevant_documents(
    _real_evren_rag_service, query: str
) -> None:
    """HEDEF 15/1-4: gercek KB chunk'lari uzerinde, gercek EVREN embedding ile bu 4 sorgu GERCEKTEN sonuc doner."""
    results = _real_evren_rag_service.query(query)
    telemetry = _real_evren_rag_service.get_last_query_telemetry()

    assert telemetry.corpus_source == "chunks_rebuild"
    assert telemetry.candidate_count > 0
    assert results, f"{query!r} icin hic sonuc donmedi (candidate_count={telemetry.candidate_count})"
    for doc in results:
        assert doc.document_id is not None
        assert doc.retrieval_rank is not None
        assert doc.source_verified is True


def test_real_semantic_query_completely_outside_corpus_scope_does_not_fabricate_relevance(
    _real_evren_rag_service,
) -> None:
    """HEDEF 15/5 (corpus disi mevzuat): Qdrant her zaman "en az kotu" adaylari dondurur (cosine benzerligi hicbir zaman "sonuc yok" demez);

    bu test, corpus'la ILGISIZ bir sorgunun candidate_count > 0 uretse bile
    (bu, retrieval'in TEK BASINA "corpus disi" durumu ELEYEMEDIGINI kanitlar -
    reranker/relevance_threshold'un NEDEN gerekli oldugunu gosterir) skorlarin
    ISG mevzuatiyla eslesen sorgulardan (yukaridaki 4 test) ACIKCA daha dusuk
    oldugunu dogrular - "ayni yuksek guvenle" sunulmadigini kanitlar.
    """
    off_topic_results = _real_evren_rag_service.query("Roma imparatorluğu tarihi ve gladyatör dövüşleri")
    off_topic_telemetry = _real_evren_rag_service.get_last_query_telemetry()

    on_topic_results = _real_evren_rag_service.query("forklift yaya güvenliği")
    on_topic_telemetry = _real_evren_rag_service.get_last_query_telemetry()

    assert off_topic_telemetry.candidate_count > 0  # Qdrant HER ZAMAN bir seyler dondurur
    if off_topic_results and on_topic_results:
        assert off_topic_telemetry.avg_embedding_score < on_topic_telemetry.avg_embedding_score, (
            "corpus-disi sorgunun ortalama embedding skoru, corpus-ici bir sorgudan DUSUK olmali "
            "- degilse relevance_threshold/reranker GERCEKTEN ayirt edici degildir."
        )


# ---------------------------------------------------------------------------
# 2026-08-24 RAG retrieval/reranking benchmark incelemesi (bkz. scripts/rag_benchmark.py):
# canli E5/Cross-Encoder benchmark BU oturumda ag kisiti (huggingface.co
# erisimi org-policy ile engelli) yuzunden CALISTIRILAMADI - karar bu yuzden
# BILEREK KEEP_DETERMINISTIC_ONLY (bkz. rapor) - production'a hicbir
# Cross-Encoder BAGLANMADI. Bu test, o kararin KOD SEVIYESINDE hala gecerli
# oldugunu (birileri "sadece dene" diye sessizce bir CrossEncoder import'u
# EKLEMEDIYSE) guvence altina alir - GELECEKTE gercek bir benchmark
# Cross-Encoder eklemeyi HAKLI CIKARIRSA, bu guard testi o degisiklikle
# BIRLIKTE BILINCLI olarak guncellenmelidir.
# ---------------------------------------------------------------------------


def test_embedding_rag_service_cross_encoder_defaults_to_none_for_isolated_use() -> None:
    """`EmbeddingRAGService`in KENDI varsayilani (`cross_encoder=None`) NOTRDUR - test/izole kullanimda Cross-Encoder'i ZORLAMAZ.

    Production'da AKTIF olmasi `src/main.py::SafirPipeline`in BILEREK bir
    `LocalCrossEncoderReranker` GECMESINDEN gelir (bkz.
    `test_production_pipeline_instantiates_a_local_cross_encoder_by_default`) -
    servisin KENDISI bunu VARSAYMAZ, boylece `RerankerConfig(enabled=False)`
    gibi izole/test kurulumlari Cross-Encoder'a ISTEMEDEN BAGLANMAZ.
    """
    import inspect

    sig = inspect.signature(EmbeddingRAGService.__init__)
    assert sig.parameters["cross_encoder"].default is None

    embedding_config = EmbeddingConfig(provider="local", model_name="m", output_dimensionality=16)
    qdrant_config = QdrantMemoryConfig(url=":memory:", top_k=5, candidate_k=20)
    service = EmbeddingRAGService(embedding_config, qdrant_config, RerankerConfig(enabled=False))
    assert service._cross_encoder is None  # noqa: SLF001 - varsayilan devre-disi durumu dogrudan dogrular


def test_production_pipeline_does_not_instantiate_a_reranker_by_default() -> None:
    """CROSS_ENCODER_STATUS = KALDIRILDI - `SafirPipeline.__init__`, `EmbeddingRAGService`i `cross_encoder` GECMEDEN kurar.

    2026-08-26 (RERANK KALDIRILDI): EVREN dokumantasyonu (SS 10), saf yogun
    getirmenin (R@1=0.95) HER reranking varyantindan (dedike rerank ucu DAHIL,
    LLM-as-judge DAHIL - Sekil 3 "Yogun + yeniden siralama" R@1=0.55) daha iyi
    performans gosterdigini olcumle kanitlamaktadir; bu yuzden onceki
    `EvrenReranker` (LLM-as-judge, bkz. `src/rag/reranker.py`) wiring'i
    `SafirPipeline.__init__`den KALDIRILDI - `EvrenReranker` sinifi SILINMEDI
    (standalone/benchmark icin durur), yalnizca production wiring'i DEGISTI.
    """
    import inspect

    import src.main as main_module

    source = inspect.getsource(main_module)
    assert "cross_encoder=EvrenReranker(" not in source
    assert "from src.rag.reranker import EvrenReranker" not in source
    # eski Gemini/Groq LLM-as-judge reranker'lar KALDIRILDI - hicbir yerde cagrilmiyor.
    assert "GeminiReranker" not in source
    assert "GroqReranker" not in source


def test_gemini_and_groq_rerankers_are_not_in_the_production_import_chain() -> None:
    """Eski `GeminiReranker`/`GroqReranker` (LLM-as-judge) TAMAMEN KALDIRILDI - production path'te (`src/main.py`/`embedding_rag_service.py`) hicbir referans KALMAMALI."""
    import inspect

    import src.main as main_module

    main_source = inspect.getsource(main_module)
    rag_service_source = open(rag_module.__file__, encoding="utf-8").read()
    reranker_source = open(rag_module.__file__.replace("embedding_rag_service.py", "reranker.py"), encoding="utf-8").read()

    for source_text in (main_source, rag_service_source, reranker_source):
        # gercek kod cagrisi/sinif tanimi (docstring/yorum ICINDEKI tarihsel bahisler DEGIL):
        assert "class GeminiReranker" not in source_text
        assert "class GroqReranker" not in source_text
        assert "GeminiReranker(" not in source_text
        assert "GroqReranker(" not in source_text


# ---------------------------------------------------------------------------
# 2026-08-24 RAG+RISK PRODUCTION KAPANIS: LOKAL Cross-Encoder gercekten baglandi.
# ---------------------------------------------------------------------------


def _seed_two_docs(service: EmbeddingRAGService) -> None:
    service._add_structured_documents(
        [
            {
                "chunk_id": "doc_a__madde_1",
                "document_id": "doc_a",
                "document_title": "Yangin Yonetmeligi",
                "level": "madde",
                "article_number": "1",
                "article_title": None,
                "is_annex": False,
                "page_start": 1,
                "page_end": 1,
                "source_url": "https://example.gov.tr/doc_a",
                "institution": "Test Bakanligi",
                "publication_date": "2020-01-01",
                "text": "yangin ve duman tespiti halinde tahliye prosedurleri kisa metin",
            },
            {
                "chunk_id": "doc_b__madde_2",
                "document_id": "doc_b",
                "document_title": "Genel ISG Yonetmeligi",
                "level": "madde",
                "article_number": "2",
                "article_title": None,
                "is_annex": False,
                "page_start": 1,
                "page_end": 1,
                "source_url": "https://example.gov.tr/doc_b",
                "institution": "Test Bakanligi",
                "publication_date": "2020-01-01",
                "text": (
                    "yangin guvenligi ve duman algilama sistemleri hakkinda cok daha uzun ve detayli "
                    "aciklamalar iceren, ek onlemleri de kapsayan genisletilmis bir madde metni"
                ),
            },
        ]
    )


def test_cross_encoder_reranks_candidates_that_passed_the_deterministic_gate(tmp_path) -> None:
    """Cross-Encoder, deterministic relevance gate'ten GECMIS adaylari YENIDEN siralar (bkz. gorev tanimi 4. bolum)."""
    fake_ce = _FakeCrossEncoder()
    service = _make_service_with_cross_encoder(tmp_path, fake_ce)
    _seed_two_docs(service)

    results = service.query("yangin duman")

    assert fake_ce.call_count == 1
    assert len(results) == 2
    # sahte CE, UZUN metni daha yuksek puanlar - final sira buna gore degismis olmali.
    assert results[0].chunk_id == "doc_b__madde_2"
    assert results[0].cross_encoder_score is not None
    assert results[0].cross_encoder_score == pytest.approx(float(len(results[0].text)))


def test_same_query_and_candidates_produce_deterministic_cross_encoder_ranking(tmp_path) -> None:
    """Ayni query + ayni aday kumesi -> Cross-Encoder DAHIL, HER ZAMAN ayni final siralama (gorev tanimi 13.4)."""
    service = _make_service_with_cross_encoder(tmp_path, _FakeCrossEncoder())
    _seed_two_docs(service)

    first = [d.chunk_id for d in service.query("yangin duman")]
    second = [d.chunk_id for d in service.query("yangin duman")]

    assert first == second


def test_cross_encoder_score_never_becomes_risk_score_or_confidence(tmp_path) -> None:
    """`cross_encoder_score`, ASLA `risk_score`/`confidence`/`probability` alanina yazilmaz - ayri, kendi alaninda kalir."""
    service = _make_service_with_cross_encoder(tmp_path, _FakeCrossEncoder())
    _seed_two_docs(service)

    results = service.query("yangin duman")

    for doc in results:
        assert not hasattr(doc, "risk_score")
        assert not hasattr(doc, "confidence")
        assert not hasattr(doc, "probability")
        assert not hasattr(doc, "model_confidence")
    # Field'in KENDI adi acikca "cross_encoder_score" - "confidence"/"probability" DEGIL.
    assert hasattr(results[0], "cross_encoder_score")


def test_cross_encoder_provenance_fields_survive_query(tmp_path) -> None:
    """document_id/article_number/chunk_id/source_url/source_verified/retrieval_rank/embedding_score/relevance_score/relevance_status/cross_encoder_score/final_rank TUMU korunur (gorev tanimi 6. bolum)."""
    service = _make_service_with_cross_encoder(tmp_path, _FakeCrossEncoder())
    _seed_two_docs(service)

    results = service.query("yangin duman")

    for i, doc in enumerate(results, start=1):
        assert doc.document_id is not None
        assert doc.article_number is not None
        assert doc.chunk_id is not None
        assert doc.source_url is not None
        assert doc.source_verified is True
        assert doc.retrieval_rank is not None
        assert doc.embedding_score is not None
        assert doc.relevance_score is not None
        assert doc.relevance_status == "accepted"
        assert doc.cross_encoder_score is not None
        assert doc.final_rank == i


def test_cross_encoder_cannot_bypass_the_deterministic_relevance_gate(tmp_path) -> None:
    """Deterministic relevance gate tarafindan REDDEDILMIS (esik-alti) bir aday, Cross-Encoder tarafindan tekrar 'accepted' YAPILAMAZ (gorev tanimi 5. bolum)."""

    class _CrossEncoderThatLovesEverything:
        """Reddedilen adaya bile EN YUKSEK puani veren dusman bir sahte - gate'i atlatmaya CALISIR."""

        def score(self, query, texts):
            return [999.0 for _ in texts]

    service = _make_service_with_cross_encoder(
        tmp_path, _CrossEncoderThatLovesEverything(), enable_relevance=True, score_threshold=0.99
    )
    _seed_two_docs(service)

    results = service.query("tamamen alakasiz bir sorgu - forklift bakim kilavuzu XYZ123")

    # threshold=0.99 cok yuksek oldugu icin HICBIR aday gate'i GECEMEZ - Cross-Encoder'in
    # "999.0" puani, gate'ten GECMEMIS bir adayi ASLA final sonuca SOKAMAZ.
    assert results == []


def test_cross_encoder_unavailable_falls_back_to_deterministic_relevance_not_an_external_api(tmp_path) -> None:
    """Lokal Cross-Encoder model agirligi yuklenemezse (paket/model yok), KONTROLLU degradasyon olur - harici bir API'ye SESSIZCE DUSULMEZ, pipeline COKMEZ."""
    from src.rag.local_cross_encoder_reranker import CrossEncoderUnavailableError

    class _BrokenCrossEncoder:
        def score(self, query, texts):
            raise CrossEncoderUnavailableError("model agirligi bulunamadi (simulasyon)")

    service = _make_service_with_cross_encoder(tmp_path, _BrokenCrossEncoder())
    _seed_two_docs(service)

    results = service.query("yangin duman")

    assert len(results) == 2  # deterministic relevance siralamasina GUVENLI DUSTU, COKMEDI
    assert all(d.cross_encoder_score is None for d in results)
    telemetry = service.get_last_query_telemetry()
    assert telemetry.cross_encoder_status == "unavailable"


def test_cross_encoder_status_is_disabled_when_no_cross_encoder_given(tmp_path) -> None:
    service = _make_service(tmp_path, candidate_k=5, top_k=5, enable_relevance=True, score_threshold=0.0)
    _seed_two_docs(service)

    service.query("yangin duman")

    telemetry = service.get_last_query_telemetry()
    assert telemetry.cross_encoder_status == "disabled"


def test_cross_encoder_status_is_used_when_reranking_actually_runs(tmp_path) -> None:
    service = _make_service_with_cross_encoder(tmp_path, _FakeCrossEncoder())
    _seed_two_docs(service)

    service.query("yangin duman")

    telemetry = service.get_last_query_telemetry()
    assert telemetry.cross_encoder_status == "used"


def test_local_cross_encoder_reranker_construction_does_not_load_any_model() -> None:
    """`LocalCrossEncoderReranker(...)` OLUSTURULMASI (constructor), model agirligi YUKLEMEZ (lazy) - import/instantiation ASLA ag erisimi TETIKLEMEZ."""
    from src.rag.local_cross_encoder_reranker import LocalCrossEncoderReranker

    reranker = LocalCrossEncoderReranker("some/model-name")
    assert reranker._model is None  # noqa: SLF001 - lazy-loading garantisini dogrudan dogrular


def test_local_cross_encoder_reranker_is_not_forced_into_the_old_llm_reranker_interface() -> None:
    """`LocalCrossEncoderReranker`, eski `LLMReranker`-ailesi (`reranker.py`) sinif hiyerarsisine BAGLI DEGIL - kendi, ayri, kucuk sozlesmesi var."""
    from src.rag.local_cross_encoder_reranker import CrossEncoderReranker, LocalCrossEncoderReranker

    assert issubclass(LocalCrossEncoderReranker, CrossEncoderReranker)
    reranker_module_path = LocalCrossEncoderReranker.__module__
    assert reranker_module_path == "src.rag.local_cross_encoder_reranker"
    assert reranker_module_path != "src.rag.reranker"


# ---------------------------------------------------------------------------
# 2026-08-24 RAG scoring explainability: deterministic relevance component
# skorlari (semantic/lexical/keyword/metadata/phrase) `RetrievedDocument`
# uzerinde CANLI koddan (score_candidate) tasiniyor mu?
# ---------------------------------------------------------------------------


def test_relevance_component_scores_are_carried_on_retrieved_document(tmp_path) -> None:
    """Relevance skorlama AKTIFKEN, `score_candidate()`in urettigi BES bilesen de (yalnizca toplam relevance_score DEGIL) `RetrievedDocument` uzerinde GERCEKTEN tasinmali."""
    from src.rag.deterministic_reranker import RelevanceWeights, score_candidate

    service = _make_service(tmp_path, candidate_k=5, top_k=5, enable_relevance=True, score_threshold=0.0)
    service._add_structured_documents(
        [
            {
                "chunk_id": "doc_a__madde_1",
                "document_id": "doc_a",
                "document_title": "Yangin Yonetmeligi",
                "article_number": "1",
                "text": "yangin ve duman tespiti halinde tahliye prosedurleri",
            }
        ]
    )

    results = service.query("yangin duman")
    assert len(results) == 1
    doc = results[0]

    # Component skorlarinin HICBIRI None DEGIL (relevance skorlama aktif oldugu icin).
    assert doc.semantic_score is not None
    assert doc.lexical_score is not None
    assert doc.keyword_score is not None
    assert doc.metadata_score is not None
    assert doc.phrase_score is not None

    # Bagimsiz olarak AYNI fonksiyonu (score_candidate) DOGRUDAN cagirip KARSILASTIR -
    # RetrievedDocument'taki degerler UYDURULMAMIS, GERCEKTEN AYNI hesaplamadan gelmis olmali.
    expected = score_candidate(
        query="yangin duman",
        chunk_text=doc.text,
        embedding_score=doc.embedding_score,
        document_title=doc.document_title,
        article_number=doc.article_number,
        keywords=None,
        weights=RelevanceWeights(),
    )
    assert doc.semantic_score == expected.semantic_score
    assert doc.lexical_score == expected.lexical_score
    assert doc.keyword_score == expected.keyword_score
    assert doc.metadata_score == expected.metadata_score
    assert doc.phrase_score == expected.phrase_score
    assert doc.relevance_score == expected.relevance_score


def test_relevance_component_weights_match_the_service_configured_weights(tmp_path) -> None:
    """`EmbeddingRAGService.relevance_weights` (explainability icin acilan property), servisin `score_candidate()`e GERCEKTEN gecirdigi agirliklarla AYNI olmali - varsayilan DEGERLERI uydurmaz."""
    from src.utils.config_loader import RelevanceWeightsConfig, RerankerConfig

    custom_weights = RelevanceWeightsConfig(semantic=0.5, lexical=0.2, keyword=0.2, metadata=0.05, phrase=0.05)
    embedding_config = EmbeddingConfig(provider="local", model_name="fake-model", output_dimensionality=16)
    qdrant_config = QdrantMemoryConfig(url=":memory:", top_k=5, candidate_k=5)
    reranker_config = RerankerConfig(enabled=True, score_threshold=0.0, top_k=5, weights=custom_weights)

    service = EmbeddingRAGService(embedding_config, qdrant_config, reranker_config)

    assert service.relevance_weights.semantic == 0.5
    assert service.relevance_weights.lexical == 0.2
    assert service.relevance_weights.keyword == 0.2
    assert service.relevance_weights.metadata == 0.05
    assert service.relevance_weights.phrase == 0.05


def test_relevance_score_equals_sum_of_component_contributions(tmp_path) -> None:
    """`relevance_score`, TASINAN bes bilesenin (agirlik x skor) toplamiyla TUTARLI olmali - UI'nin gosterecegi breakdown, gercek toplamla EslesMELI."""
    service = _make_service(tmp_path, candidate_k=5, top_k=5, enable_relevance=True, score_threshold=0.0)
    service._add_structured_documents(
        [
            {
                "chunk_id": "doc_a__madde_1",
                "document_id": "doc_a",
                "document_title": "Yangin Yonetmeligi",
                "article_number": "1",
                "text": "yangin ve duman tespiti halinde tahliye prosedurleri",
            }
        ]
    )

    results = service.query("yangin duman")
    doc = results[0]
    weights = service.relevance_weights

    reconstructed = (
        doc.semantic_score * weights.semantic
        + doc.lexical_score * weights.lexical
        + doc.keyword_score * weights.keyword
        + doc.metadata_score * weights.metadata
        + doc.phrase_score * weights.phrase
    )
    assert reconstructed == pytest.approx(doc.relevance_score, abs=1e-9)
