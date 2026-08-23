"""`src/rag/embedding_providers.py` icin agsiz birim testleri.

`sentence_transformers.SentenceTransformer` gercekten yuklenmez/indirilmez -
monkeypatch ile sahte, deterministik bir modelle degistirilir.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.rag.embedding_providers import (
    ConfigurationError,
    LocalEmbeddingProvider,
    build_embedding_provider,
)


class _FakeSentenceTransformer:
    """`SentenceTransformer(model_name, device=...)` yerine gecen, metne bagli deterministik vektor ureten sahte model."""

    def __init__(self, model_name: str, device: str = "cpu", dimension: int = 8):
        self.model_name = model_name
        self.device = device
        self._dimension = dimension
        self.encode_calls = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts, *, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings):
        self.encode_calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return np.array(
            [[float((hash(t) >> i) % 7) for i in range(self._dimension)] for t in texts],
            dtype="float32",
        )


def _install_fake_sentence_transformers(monkeypatch, dimension: int = 8):
    """`sentence_transformers` modulunu sahte bir surumle degistirir; olusan model orneklerini yakalar."""
    created_models = []

    fake_module = types.ModuleType("sentence_transformers")

    def _factory(model_name, device="cpu"):
        model = _FakeSentenceTransformer(model_name, device=device, dimension=dimension)
        created_models.append(model)
        return model

    fake_module.SentenceTransformer = _factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return created_models


def test_provider_construction_never_touches_network(monkeypatch) -> None:
    """Yapicinin (constructor) HICBIR model yuklemesi/agsi tetiklememesi gerekir - lazy model."""
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    provider = LocalEmbeddingProvider(model_name="intfloat/multilingual-e5-small", output_dimensionality=384)
    assert provider.dimension == 384  # config'ten geliyor, model YUKLENMEDEN


def test_missing_sentence_transformers_package_raises_configuration_error(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _raising_import_for("sentence_transformers"),
    )
    provider = LocalEmbeddingProvider(model_name="fake-model", output_dimensionality=8)
    with pytest.raises(ConfigurationError, match="sentence-transformers"):
        provider.embed_query("test")


def _raising_import_for(blocked_module: str):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == blocked_module or name.startswith(blocked_module + "."):
            raise ImportError(f"'{blocked_module}' kurulu degil (test simulasyonu)")
        return real_import(name, *args, **kwargs)

    return _fake_import


def test_embed_query_uses_query_prefix(monkeypatch) -> None:
    models = _install_fake_sentence_transformers(monkeypatch, dimension=8)

    provider = LocalEmbeddingProvider(model_name="intfloat/multilingual-e5-small", output_dimensionality=8)
    vector = provider.embed_query("yangın riski")

    assert vector.shape == (8,)
    assert models[0].encode_calls[0]["texts"] == ["query: yangın riski"]


def test_embed_documents_uses_passage_prefix_and_single_call(monkeypatch) -> None:
    models = _install_fake_sentence_transformers(monkeypatch, dimension=8)

    provider = LocalEmbeddingProvider(model_name="intfloat/multilingual-e5-small", output_dimensionality=8)
    texts = ["madde 1", "madde 2", "madde 3"]
    vectors = provider.embed_documents(texts)

    assert vectors.shape == (3, 8)
    # Lokal model - agirlik/kota kisitlamasi yok, TUM metinler TEK `encode()` cagrisinda gider.
    assert len(models[0].encode_calls) == 1
    assert models[0].encode_calls[0]["texts"] == [f"passage: {t}" for t in texts]


def test_embed_documents_returns_normalized_vectors(monkeypatch) -> None:
    _install_fake_sentence_transformers(monkeypatch, dimension=8)

    provider = LocalEmbeddingProvider(model_name="intfloat/multilingual-e5-small", output_dimensionality=8)
    vectors = provider.embed_documents(["bir metin"])

    norm = np.linalg.norm(vectors[0])
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_document_and_query_embedding_share_same_model_instance(monkeypatch) -> None:
    """Gorev tanimi 4. madde: dokuman ve sorgu embedding'i AYNI model/dimension/normalization kullanmali."""
    models = _install_fake_sentence_transformers(monkeypatch, dimension=8)

    provider = LocalEmbeddingProvider(model_name="intfloat/multilingual-e5-small", output_dimensionality=8)
    provider.embed_documents(["madde 1"])
    provider.embed_query("soru")

    # Lazy client TEK bir kez olusturulur - hem dokuman hem sorgu embedding'i AYNI model orneğini kullanir.
    assert len(models) == 1


def test_dimension_mismatch_is_not_silently_accepted(monkeypatch) -> None:
    """Gorev tanimi 7. madde: model/dimension uyumsuzlugu SESSIZCE KABUL EDILMEZ."""
    _install_fake_sentence_transformers(monkeypatch, dimension=8)

    # Config 16 boyut bekliyor ama sahte model 8 boyut uretiyor.
    provider = LocalEmbeddingProvider(model_name="intfloat/multilingual-e5-small", output_dimensionality=16)

    with pytest.raises(ConfigurationError, match="boyut"):
        provider.embed_query("test")


def test_build_embedding_provider_rejects_unsupported_provider() -> None:
    with pytest.raises(ConfigurationError):
        build_embedding_provider(provider="unknown-provider", model_name="x", output_dimensionality=8)


def test_build_embedding_provider_rejects_gemini() -> None:
    """Gorev tanimi 2. madde: Gemini embedding'e SESSIZ FALLBACK YOK - 'gemini' provider'i acikca REDDEDILIR."""
    with pytest.raises(ConfigurationError, match="local"):
        build_embedding_provider(provider="gemini", model_name="gemini-embedding-001", output_dimensionality=768)


def test_build_embedding_provider_requires_output_dimensionality() -> None:
    with pytest.raises(ConfigurationError):
        build_embedding_provider(provider="local", model_name="intfloat/multilingual-e5-small", output_dimensionality=None)


def test_build_embedding_provider_returns_local_provider() -> None:
    provider = build_embedding_provider(
        provider="local", model_name="intfloat/multilingual-e5-small", output_dimensionality=384
    )
    assert isinstance(provider, LocalEmbeddingProvider)
    assert provider.dimension == 384
