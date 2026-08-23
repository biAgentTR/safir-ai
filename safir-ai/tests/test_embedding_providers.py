"""`src/rag/embedding_providers.py` icin agsiz birim testleri.

`google.genai` gercekten cagrilmaz - `genai.Client`/`types.EmbedContentConfig`
monkeypatch ile sahte, deterministik nesnelerle degistirilir.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.rag.embedding_providers import (
    ConfigurationError,
    GeminiEmbeddingProvider,
    build_embedding_provider,
)


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbedContentResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeModels:
    """`client.models.embed_content(...)` cagrisini kaydeden sahte istemci parcasi."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.calls = []

    def embed_content(self, *, model, contents, config):
        # `_embed()` artik HER ZAMAN TEK bir string gonderir (liste degil) -
        # bkz. embedding_providers.py::_embed docstring'i (kok neden aciklamasi).
        assert isinstance(contents, str), "contents TEK bir string olmali, liste degil (batch endpoint kullanilmamali)"
        self.calls.append({"model": model, "contents": contents, "config": config})
        embedding = _FakeEmbedding([float((hash(contents) >> i) % 7) for i in range(self.dimension)])
        return _FakeEmbedContentResponse([embedding])


class _FakeClient:
    def __init__(self, dimension: int):
        self.models = _FakeModels(dimension)


def _install_fake_genai(monkeypatch, dimension: int = 8) -> _FakeClient:
    """`google.genai`/`google.genai.types` modullerini sahte surumlerle degistirir ve olusan istemciyi dondurur."""
    fake_client = _FakeClient(dimension)

    fake_genai_module = types.ModuleType("google.genai")
    fake_genai_module.Client = lambda api_key: fake_client  # noqa: ARG005

    fake_types_module = types.ModuleType("google.genai.types")

    class _FakeEmbedContentConfig:
        def __init__(self, task_type, output_dimensionality):
            self.task_type = task_type
            self.output_dimensionality = output_dimensionality

    fake_types_module.EmbedContentConfig = _FakeEmbedContentConfig

    fake_google_module = sys.modules.get("google") or types.ModuleType("google")
    fake_google_module.genai = fake_genai_module

    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types_module)
    return fake_client


def test_missing_api_key_raises_configuration_error(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiEmbeddingProvider(model_name="gemini-embedding-001", output_dimensionality=8)

    with pytest.raises(ConfigurationError):
        provider.embed_query("test sorgusu")


def test_provider_construction_never_touches_network(monkeypatch) -> None:
    """Yapicinin (constructor) API anahtari olmasa bile PATLAMAMASI gerekir - lazy client."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiEmbeddingProvider(model_name="gemini-embedding-001", output_dimensionality=8)
    assert provider.dimension == 8


def test_embed_query_uses_retrieval_query_task_type(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    fake_client = _install_fake_genai(monkeypatch, dimension=8)

    provider = GeminiEmbeddingProvider(model_name="gemini-embedding-001", output_dimensionality=8)
    vector = provider.embed_query("yangın riski")

    assert vector.shape == (8,)
    assert fake_client.models.calls[0]["config"].task_type == "RETRIEVAL_QUERY"
    assert fake_client.models.calls[0]["contents"] == "yangın riski"


def test_embed_documents_uses_retrieval_document_task_type_and_sends_one_request_per_text(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    fake_client = _install_fake_genai(monkeypatch, dimension=8)
    sleep_calls = []
    monkeypatch.setattr("src.rag.embedding_providers.time.sleep", lambda s: sleep_calls.append(s))

    provider = GeminiEmbeddingProvider(model_name="gemini-embedding-001", output_dimensionality=8)
    texts = ["madde 1", "madde 2", "madde 3"]
    vectors = provider.embed_documents(texts)

    assert vectors.shape == (3, 8)
    # Batch endpoint KESINLIKLE kullanilmaz - her metin kendi tekil istegini alir.
    assert len(fake_client.models.calls) == 3
    assert [call["contents"] for call in fake_client.models.calls] == texts
    assert fake_client.models.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    # RPM yukunu azaltmak icin istekler arasinda (ama SONUNCUDAN SONRA DEGIL) beklenmeli.
    assert sleep_calls == [3.0, 3.0]


def test_embed_documents_returns_normalized_vectors(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    _install_fake_genai(monkeypatch, dimension=8)

    provider = GeminiEmbeddingProvider(model_name="gemini-embedding-001", output_dimensionality=8)
    vectors = provider.embed_documents(["bir metin"])

    norm = np.linalg.norm(vectors[0])
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_build_embedding_provider_rejects_unsupported_provider() -> None:
    with pytest.raises(ConfigurationError):
        build_embedding_provider(provider="unknown-provider", model_name="x", output_dimensionality=8)


def test_build_embedding_provider_requires_output_dimensionality() -> None:
    with pytest.raises(ConfigurationError):
        build_embedding_provider(provider="gemini", model_name="gemini-embedding-001", output_dimensionality=None)


class _FakeRateLimitError(Exception):
    def __init__(self, code: int = 429, status: str = "RESOURCE_EXHAUSTED"):
        super().__init__(f"{code} {status}")
        self.code = code
        self.status = status


class _FlakyThenOkModels(_FakeModels):
    """Ilk `fail_times` cagride 429 firlatir, sonra normal davranir."""

    def __init__(self, dimension: int, fail_times: int):
        super().__init__(dimension)
        self._fail_times = fail_times
        self._attempts = 0

    def embed_content(self, *, model, contents, config):
        self._attempts += 1
        if self._attempts <= self._fail_times:
            raise _FakeRateLimitError()
        return super().embed_content(model=model, contents=contents, config=config)


def test_rate_limit_429_is_retried_with_backoff_and_eventually_succeeds(monkeypatch) -> None:
    fake_client = _install_fake_genai(monkeypatch)
    fake_client.models = _FlakyThenOkModels(dimension=8, fail_times=2)
    sleep_calls = []
    monkeypatch.setattr("src.rag.embedding_providers.time.sleep", lambda s: sleep_calls.append(s))

    provider = GeminiEmbeddingProvider(model_name="gemini-embedding-001", output_dimensionality=8)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    result = provider.embed_documents(["metin"])

    assert result.shape == (1, 8)
    assert fake_client.models._attempts == 3  # 2 basarisiz + 1 basarili
    assert len(sleep_calls) == 2
    assert sleep_calls[1] > sleep_calls[0]  # katlanarak artan bekleme


def test_rate_limit_429_gives_up_after_max_retries(monkeypatch) -> None:
    fake_client = _install_fake_genai(monkeypatch)
    fake_client.models = _FlakyThenOkModels(dimension=8, fail_times=999)
    monkeypatch.setattr("src.rag.embedding_providers.time.sleep", lambda s: None)

    provider = GeminiEmbeddingProvider(model_name="gemini-embedding-001", output_dimensionality=8)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    with pytest.raises(_FakeRateLimitError):
        provider.embed_documents(["metin"])


def test_non_rate_limit_error_is_not_retried(monkeypatch) -> None:
    class _AuthErrorModels(_FakeModels):
        def embed_content(self, *, model, contents, config):
            raise RuntimeError("401 unauthorized")

    fake_client = _install_fake_genai(monkeypatch)
    fake_client.models = _AuthErrorModels(dimension=8)
    sleep_calls = []
    monkeypatch.setattr("src.rag.embedding_providers.time.sleep", lambda s: sleep_calls.append(s))

    provider = GeminiEmbeddingProvider(model_name="gemini-embedding-001", output_dimensionality=8)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    with pytest.raises(RuntimeError, match="401 unauthorized"):
        provider.embed_documents(["metin"])
    assert sleep_calls == []  # 429 disi hata icin HIC bekleme yapilmaz


def test_build_embedding_provider_returns_gemini_provider() -> None:
    provider = build_embedding_provider(provider="gemini", model_name="gemini-embedding-001", output_dimensionality=768)
    assert isinstance(provider, GeminiEmbeddingProvider)
    assert provider.dimension == 768
