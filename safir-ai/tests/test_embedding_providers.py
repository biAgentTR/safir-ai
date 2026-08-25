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
    EvrenEmbeddingProvider,
    build_embedding_provider,
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
