"""`src/rag/embedding_providers.py` icin agsiz birim testleri.

`openai.OpenAI` istemcisi gercekten kurulmaz/ag cagrisi yapmaz - monkeypatch
ile sahte, deterministik bir istemciyle degistirilir.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.rag.embedding_providers import (
    ConfigurationError,
    DocumentTooLargeError,
    EvrenEmbeddingProvider,
    _build_safe_batches,
    _estimate_tokens,
    _MAX_CONTEXT_TOKENS,
    build_embedding_provider,
    split_oversized_text,
)


class _FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, data):
        self.data = data


class _FakeEmbeddingsResource:
    """`client.embeddings.create(model=..., input=[...])` yerine gecen, metne bagli deterministik vektor ureten sahte kaynak."""

    def __init__(self, dimension: int = 8):
        self._dimension = dimension
        self.create_calls = []

    def create(self, *, model, input):  # noqa: A002 - OpenAI SDK imzasiyla uyumlu
        self.create_calls.append({"model": model, "input": list(input)})
        vectors = [[float((hash(t) >> i) % 7) for i in range(self._dimension)] for t in input]
        return _FakeEmbeddingResponse([_FakeEmbeddingItem(v) for v in vectors])


class _FakeOpenAIClient:
    def __init__(self, base_url=None, api_key=None, dimension: int = 8):
        self.base_url = base_url
        self.api_key = api_key
        self.embeddings = _FakeEmbeddingsResource(dimension=dimension)


def _install_fake_openai(monkeypatch, dimension: int = 8):
    """`openai` modulunu sahte bir surumle degistirir; olusan istemcileri yakalar."""
    created_clients = []

    fake_module = types.ModuleType("openai")

    def _factory(base_url=None, api_key=None):
        client = _FakeOpenAIClient(base_url=base_url, api_key=api_key, dimension=dimension)
        created_clients.append(client)
        return client

    fake_module.OpenAI = _factory
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return created_clients


def test_provider_construction_never_touches_network(monkeypatch) -> None:
    """Yapicinin (constructor) HICBIR istemci kurulumu/ag cagrisi tetiklememesi gerekir - lazy client."""
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    provider = EvrenEmbeddingProvider(
        model_name="bge-m3-embed",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST_UNSET",
        output_dimensionality=1024,
    )
    assert provider.dimension == 1024  # config'ten geliyor, istemci KURULMADAN


def test_missing_api_key_raises_configuration_error(monkeypatch) -> None:
    monkeypatch.delenv("EVREN_API_KEY_TEST_UNSET", raising=False)
    provider = EvrenEmbeddingProvider(
        model_name="bge-m3-embed",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST_UNSET",
        output_dimensionality=8,
    )
    with pytest.raises(ConfigurationError, match="EVREN_API_KEY_TEST_UNSET"):
        provider.embed_query("test")


def test_embed_query_calls_embeddings_api(monkeypatch) -> None:
    monkeypatch.setenv("EVREN_API_KEY_TEST", "sk-test")
    clients = _install_fake_openai(monkeypatch, dimension=8)

    provider = EvrenEmbeddingProvider(
        model_name="bge-m3-embed",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST",
        output_dimensionality=8,
    )
    vector = provider.embed_query("yangın riski")

    assert vector.shape == (8,)
    assert clients[0].embeddings.create_calls[0]["input"] == ["yangın riski"]
    assert clients[0].embeddings.create_calls[0]["model"] == "bge-m3-embed"


def test_embed_documents_single_call(monkeypatch) -> None:
    monkeypatch.setenv("EVREN_API_KEY_TEST", "sk-test")
    clients = _install_fake_openai(monkeypatch, dimension=8)

    provider = EvrenEmbeddingProvider(
        model_name="bge-m3-embed",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST",
        output_dimensionality=8,
    )
    texts = ["madde 1", "madde 2", "madde 3"]
    vectors = provider.embed_documents(texts)

    assert vectors.shape == (3, 8)
    assert len(clients[0].embeddings.create_calls) == 1
    assert clients[0].embeddings.create_calls[0]["input"] == texts


def test_embed_documents_returns_normalized_vectors(monkeypatch) -> None:
    monkeypatch.setenv("EVREN_API_KEY_TEST", "sk-test")
    _install_fake_openai(monkeypatch, dimension=8)

    provider = EvrenEmbeddingProvider(
        model_name="bge-m3-embed",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST",
        output_dimensionality=8,
    )
    vectors = provider.embed_documents(["bir metin"])

    norm = np.linalg.norm(vectors[0])
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_document_and_query_embedding_share_same_client_instance(monkeypatch) -> None:
    """Dokuman ve sorgu embedding'i AYNI (lazy, tek kez kurulan) istemciyi kullanmali."""
    monkeypatch.setenv("EVREN_API_KEY_TEST", "sk-test")
    clients = _install_fake_openai(monkeypatch, dimension=8)

    provider = EvrenEmbeddingProvider(
        model_name="bge-m3-embed",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST",
        output_dimensionality=8,
    )
    provider.embed_documents(["madde 1"])
    provider.embed_query("soru")

    assert len(clients) == 1


def test_empty_dimension_falls_back_to_documented_default(monkeypatch) -> None:
    """`output_dimensionality` verilmezse `bge-m3-embed`in dokumante edilen boyutu (1024) kullanilir."""
    provider = EvrenEmbeddingProvider(
        model_name="bge-m3-embed",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST_UNSET",
    )
    assert provider.dimension == 1024


def test_build_embedding_provider_rejects_unsupported_provider() -> None:
    with pytest.raises(ConfigurationError, match="evren"):
        build_embedding_provider(provider="unknown-provider", model_name="x", output_dimensionality=8)


def test_build_embedding_provider_rejects_local() -> None:
    """Lokal (sentence-transformers) embedding TAMAMEN KALDIRILDI - 'local' provider'i acikca REDDEDILIR."""
    with pytest.raises(ConfigurationError, match="evren"):
        build_embedding_provider(provider="local", model_name="intfloat/multilingual-e5-small", output_dimensionality=384)


def test_build_embedding_provider_requires_base_url_and_api_key_env() -> None:
    with pytest.raises(ConfigurationError):
        build_embedding_provider(provider="evren", model_name="bge-m3-embed", output_dimensionality=1024)


def test_build_embedding_provider_returns_evren_provider() -> None:
    provider = build_embedding_provider(
        provider="evren",
        model_name="bge-m3-embed",
        output_dimensionality=1024,
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY",
    )
    assert isinstance(provider, EvrenEmbeddingProvider)
    assert provider.dimension == 1024


# ---------------------------------------------------------------------------
# Guvenli batching (EVREN bge-m3-embed 8192-token istek siniri) - bkz.
# `src/rag/embedding_providers.py::_build_safe_batches`/`_estimate_tokens`.
# ---------------------------------------------------------------------------


def test_estimate_tokens_is_conservative_and_never_zero_for_nonempty_text() -> None:
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("a") >= 1
    # dusuk chars_per_token orani -> UZUN metinler icin token tahmini, naif
    # "4 char/token" varsayimindan daha YUKSEK (fazla-tahmin) olmali.
    long_text = "x" * 12000
    assert _estimate_tokens(long_text) > len(long_text) / 4


def test_build_safe_batches_splits_748_like_input_into_multiple_batches() -> None:
    """(a) 748 KB chunk'ina benzer boyutta girdi, TEK batch'e SIGMAYIP birden fazla istege bolunmeli."""
    # Gercek KB chunk'lari tipik olarak birkac yuz karakter uzunlugundadir;
    # 748 x ~600 karakter, varsayilan 7000 token butcesine gore rahatlikla
    # tek batch'e sigmayacak toplam bir tahmini token yuku uretir.
    texts = [f"Madde {i}: " + ("ISG mevzuati hukmu metni. " * 20) for i in range(748)]

    batches = _build_safe_batches(texts, safe_token_budget=7000)

    assert len(batches) > 1
    # Batch'lerin birlesimi TUM orijinal indeksleri, TEKRARSIZ ve SIRALI kapsamali.
    flattened = [i for batch in batches for i in batch]
    assert flattened == list(range(len(texts)))


def test_build_safe_batches_no_batch_exceeds_configured_budget() -> None:
    """(b) Hicbir batch'in TAHMINI toplam token'i, yapilandirilmis guvenli butceyi asmamali."""
    texts = [f"chunk-{i} " + ("kelime " * (i % 50 + 1)) for i in range(200)]
    budget = 500

    batches = _build_safe_batches(texts, safe_token_budget=budget)

    for batch in batches:
        total = sum(_estimate_tokens(texts[i]) for i in batch)
        assert total <= budget, f"batch {batch} tahmini {total} token > butce {budget}"


def test_build_safe_batches_preserves_order_within_and_across_batches() -> None:
    texts = [f"metin-{i}" for i in range(50)]
    batches = _build_safe_batches(texts, safe_token_budget=15)

    flattened = [i for batch in batches for i in batch]
    assert flattened == list(range(len(texts)))
    # her batch da KENDI ICINDE artan sirali olmali (ardisik dilimleme).
    for batch in batches:
        assert batch == sorted(batch)


def test_build_safe_batches_raises_document_too_large_error_for_single_oversized_document() -> None:
    """(d) Tek basina butceyi asan bir dokuman SESSIZCE kirpilmaz - acik hata firlatilir."""
    oversized = "x" * 100_000  # tahmini token sayisi, kucuk butceyi acikca asar
    texts = ["kisa metin", oversized, "baska kisa metin"]

    with pytest.raises(DocumentTooLargeError, match=r"#1"):
        _build_safe_batches(texts, safe_token_budget=100)


def test_build_safe_batches_default_budget_stays_within_evren_max_context() -> None:
    """Varsayilan guvenlik payi, EVREN'in gercek 8192-token sinirinin (ACIKCA) altinda kalmali."""
    from src.rag.embedding_providers import _DEFAULT_SAFE_TOKEN_BUDGET

    assert _DEFAULT_SAFE_TOKEN_BUDGET < _MAX_CONTEXT_TOKENS
    assert 7000 <= _DEFAULT_SAFE_TOKEN_BUDGET <= 7500


def test_embed_documents_splits_into_multiple_batches_when_over_budget(monkeypatch) -> None:
    """(a) `EvrenEmbeddingProvider.embed_documents()` gercekten birden fazla `embeddings.create()` cagrisi yapiyor mu?"""
    monkeypatch.setenv("EVREN_API_KEY_TEST", "sk-test")
    clients = _install_fake_openai(monkeypatch, dimension=8)

    provider = EvrenEmbeddingProvider(
        model_name="bge-m3-embed",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST",
        output_dimensionality=8,
        max_batch_tokens=20,  # kucuk butce - kolayca birden fazla batch'e zorlar
    )
    texts = [f"belge-{i} icerik metni burada" for i in range(30)]

    vectors = provider.embed_documents(texts)

    assert vectors.shape == (30, 8)
    assert len(clients[0].embeddings.create_calls) > 1


def test_embed_documents_no_request_exceeds_configured_token_budget(monkeypatch) -> None:
    """(b) Provider seviyesinde de: hicbir GERCEK istegin `input`i, yapilandirilmis butceyi (tahminen) asmamali."""
    monkeypatch.setenv("EVREN_API_KEY_TEST", "sk-test")
    clients = _install_fake_openai(monkeypatch, dimension=8)

    budget = 50
    provider = EvrenEmbeddingProvider(
        model_name="bge-m3-embed",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST",
        output_dimensionality=8,
        max_batch_tokens=budget,
    )
    texts = [f"belge-{i} " + ("kelime " * (i % 10)) for i in range(40)]

    provider.embed_documents(texts)

    for call in clients[0].embeddings.create_calls:
        total = sum(_estimate_tokens(t) for t in call["input"])
        assert total <= budget


def test_embed_documents_preserves_output_order_across_batches(monkeypatch) -> None:
    """(c) Batch'lere bolunmus olsa bile, donen vektor sirasi GIRDI sirasiyla BIREBIR eslesmeli."""
    monkeypatch.setenv("EVREN_API_KEY_TEST", "sk-test")
    _install_fake_openai(monkeypatch, dimension=8)

    provider = EvrenEmbeddingProvider(
        model_name="bge-m3-embed",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST",
        output_dimensionality=8,
        max_batch_tokens=15,
    )
    texts = [f"benzersiz-metin-{i}" for i in range(25)]

    batched_vectors = provider.embed_documents(texts)
    # Sahte istemci, ciktiyi METNIN KENDISINDEN (hash) turetiyor - bu yuzden
    # her metnin batch'lenmis SONUCU, ayni metnin TEK BASINA embed edilmis
    # sonucuYLA (batching'den bagimsiz, deterministik) birebir AYNI olmalidir.
    for i, text in enumerate(texts):
        expected = provider.embed_query(text)
        assert np.array_equal(batched_vectors[i], expected), f"index {i} icin sira/eslesme bozuldu"


def test_embed_documents_single_oversized_document_fails_clearly(monkeypatch) -> None:
    """(d) Provider seviyesinde: tek basina cok buyuk bir dokuman, sessizce kirpilmadan acikca REDDEDILIR."""
    monkeypatch.setenv("EVREN_API_KEY_TEST", "sk-test")
    _install_fake_openai(monkeypatch, dimension=8)

    provider = EvrenEmbeddingProvider(
        model_name="bge-m3-embed",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST",
        output_dimensionality=8,
        max_batch_tokens=100,
    )
    texts = ["normal boyutlu bir dokuman metni", "x" * 100_000]

    with pytest.raises(DocumentTooLargeError, match=r"#1"):
        provider.embed_documents(texts)


def test_max_batch_tokens_is_configurable_via_build_embedding_provider(monkeypatch) -> None:
    """`memory.embedding.max_batch_tokens` config alani, gercekten provider'a ULASIYOR mu?"""
    monkeypatch.setenv("EVREN_API_KEY_TEST", "sk-test")
    clients = _install_fake_openai(monkeypatch, dimension=8)

    provider = build_embedding_provider(
        provider="evren",
        model_name="bge-m3-embed",
        output_dimensionality=8,
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST",
        max_batch_tokens=10,
    )
    texts = [f"metin-{i}" for i in range(20)]

    provider.embed_documents(texts)

    # 10 tokenlik kucuk butce ile 20 kisa metin, kesinlikle birden fazla batch'e bolunmus olmali.
    assert len(clients[0].embeddings.create_calls) > 1


# ---------------------------------------------------------------------------
# `split_oversized_text` - tek bir dokumani, semantik/metin sinirlarina gore,
# VERI KAYBI OLMADAN guvenli parcalara bolen fonksiyon (bkz. gorev tanimi:
# "Sessiz truncation yapma", "semantik/metin sinirlarina gore bol").
# ---------------------------------------------------------------------------


def test_split_oversized_text_is_noop_when_within_budget() -> None:
    text = "Kisa bir madde metni."
    assert split_oversized_text(text, safe_token_budget=1000) == [text]


def test_split_oversized_text_no_piece_exceeds_budget() -> None:
    text = "Bu bir cumledir. " * 500  # tekrarlanan cumleler - gercekci uzun madde metnine benzer
    budget = 200

    pieces = split_oversized_text(text, safe_token_budget=budget)

    assert len(pieces) > 1
    for piece in pieces:
        assert _estimate_tokens(piece) <= budget


def test_split_oversized_text_preserves_all_characters_no_data_loss() -> None:
    """Parcalarin sirali birlestirilmesi ORIJINAL metni BIREBIR yeniden uretmeli - hicbir karakter kaybolmaz/kirpilmaz."""
    text = "Madde 1.\n\nBirinci fikra. Ikinci cumle! Ucuncu mu?\n\nMadde 2, ek liste; virgullu, ifadeler, burada."
    pieces = split_oversized_text(text, safe_token_budget=10)

    assert len(pieces) > 1
    assert "".join(pieces) == text


def test_split_oversized_text_prefers_paragraph_boundaries_when_they_fit() -> None:
    """Her paragraf TEK BASINA butceye SIGIYORSA (ama ikisi birlikte SIGMIYORSA), bolme TAM paragraf sinirinda olmali - cumle/kelime ortasinda KESILMEMELI."""
    paragraph_a = "Birinci paragraf burada baslar ve devam eder ve biraz daha uzar."
    paragraph_b = "Ikinci paragraf tamamen farkli bir konudan bahseder ve o da uzar."
    text = f"{paragraph_a}\n\n{paragraph_b}"
    # Her paragraf ~22-23 tahmini token; ikisi birlikte ~45 - butce (25) TEK
    # paragrafa sigar ama IKISINE BIRDEN sigmaz -> paragraf sinirinda bolme beklenir.
    budget = 25

    pieces = split_oversized_text(text, safe_token_budget=budget)

    assert "".join(pieces) == text
    assert len(pieces) == 2
    assert pieces[0] == paragraph_a + "\n\n"
    assert pieces[1] == paragraph_b


def test_split_oversized_text_handles_pathological_no_separator_text_without_dropping_data() -> None:
    """Bosluksuz, ayiricisiz tek bir uzun dizi bile VERI KAYBI olmadan (yalnizca sert karakter kesimiyle) bolunmeli."""
    text = "x" * 5000
    pieces = split_oversized_text(text, safe_token_budget=100)

    assert len(pieces) > 1
    assert "".join(pieces) == text
    for piece in pieces:
        assert _estimate_tokens(piece) <= 100


def test_split_oversized_text_raises_nothing_and_never_truncates_real_oversized_chunk() -> None:
    """Gercek KB corpus'unda tespit edilen tipte (cok uzun, tek bir 'GECICI MADDE' metni) bir dokuman icin: sessiz kirpilma YOK, tum icerik korunur."""
    long_legal_text = (
        "GEÇİCİ MADDE 1 - Bu Yönetmeliğin yayımı tarihinden önce yapı ruhsatı verilen yapılar ile "
        "ilgili hükümler saklıdır. " * 400
    )
    budget = 500

    pieces = split_oversized_text(long_legal_text, safe_token_budget=budget)

    assert len(pieces) > 1
    assert "".join(pieces) == long_legal_text
    assert sum(_estimate_tokens(p) for p in pieces) >= _estimate_tokens(long_legal_text)
    for piece in pieces:
        assert _estimate_tokens(piece) <= budget


# ---------------------------------------------------------------------------
# `_split_oversized_records` - `split_oversized_text`i KB kayit semasina
# (metadata/chunk_id/document_id iliskileri) uygulayan katman - bkz.
# `src/rag/embedding_rag_service.py`.
# ---------------------------------------------------------------------------


def test_split_oversized_records_preserves_metadata_and_expands_chunk_id() -> None:
    from src.rag.embedding_rag_service import _split_oversized_records

    oversized_text = "Uzun bir madde metni cumlesi. " * 400
    record = {
        "chunk_id": "test_doc__madde_5",
        "document_id": "test_doc",
        "document_title": "Test Yönetmeliği",
        "level": "madde",
        "article_number": "5",
        "article_title": None,
        "is_annex": False,
        "page_start": 3,
        "page_end": 4,
        "source_url": "https://example.gov.tr/test_doc",
        "institution": "Test Bakanlığı",
        "publication_date": "2020-01-01",
        "text": oversized_text,
    }

    expanded = _split_oversized_records([record], safe_token_budget=200)

    assert len(expanded) > 1
    # TUM metadata alanlari (text/chunk_id HARIC) HER alt-kayitta AYNEN korunmali.
    for sub in expanded:
        assert sub["document_id"] == "test_doc"
        assert sub["document_title"] == "Test Yönetmeliği"
        assert sub["article_number"] == "5"
        assert sub["source_url"] == "https://example.gov.tr/test_doc"
        assert sub["institution"] == "Test Bakanlığı"
        assert sub["publication_date"] == "2020-01-01"
        assert sub["chunk_id"].startswith("test_doc__madde_5__part")
    # Parcalarin metni birlestirilince ORIJINAL metne esit olmali (veri kaybi yok).
    assert "".join(sub["text"] for sub in expanded) == oversized_text


def test_split_oversized_records_leaves_normal_sized_records_untouched() -> None:
    from src.rag.embedding_rag_service import _split_oversized_records

    record = {"chunk_id": "small__1", "document_id": "small", "text": "Kisa bir metin."}
    expanded = _split_oversized_records([record], safe_token_budget=1000)

    assert expanded == [record]
