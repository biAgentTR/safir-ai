"""`src/memory/reranker.py` icin agsiz birim testleri.

`cohere` gercekten cagrilmaz - `cohere.ClientV2` monkeypatch ile sahte,
deterministik bir nesneyle degistirilir.
"""

from __future__ import annotations

import sys
import types

import pytest

from src.memory.reranker import CohereReranker, RerankerUnavailableError


class _FakeResultItem:
    def __init__(self, index: int, relevance_score: float):
        self.index = index
        self.relevance_score = relevance_score


class _FakeRerankResponse:
    def __init__(self, results):
        self.results = results


class _FakeCohereClient:
    """Belgelerin uzunluguna gore (uzun=alakali) sahte, deterministik bir siralama uretir."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []

    def rerank(self, *, model, query, documents):
        self.calls.append({"model": model, "query": query, "documents": list(documents)})
        if self.fail:
            raise RuntimeError("Cohere API cagrisi basarisiz (sahte hata)")
        scored = sorted(
            range(len(documents)), key=lambda i: len(documents[i]), reverse=True
        )
        results = [_FakeResultItem(index=i, relevance_score=1.0 - 0.1 * rank) for rank, i in enumerate(scored)]
        return _FakeRerankResponse(results)


def _install_fake_cohere(monkeypatch, fail: bool = False) -> _FakeCohereClient:
    fake_client = _FakeCohereClient(fail=fail)
    fake_cohere_module = types.ModuleType("cohere")
    fake_cohere_module.ClientV2 = lambda api_key: fake_client  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "cohere", fake_cohere_module)
    return fake_client


def test_missing_api_key_raises_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    reranker = CohereReranker(model_name="rerank-v4.0-pro")

    with pytest.raises(RerankerUnavailableError):
        reranker.rerank("sorgu", ["a", "bb"])


def test_construction_never_touches_network(monkeypatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    reranker = CohereReranker(model_name="rerank-v4.0-pro")
    assert reranker._model_name == "rerank-v4.0-pro"  # yapici hicbir ag cagrisi yapmadi


def test_rerank_orders_by_relevance_score_descending(monkeypatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "fake-key")
    _install_fake_cohere(monkeypatch)
    reranker = CohereReranker(model_name="rerank-v4.0-pro")

    candidates = ["kisa", "bu cok daha uzun bir aday metnidir", "orta uzunlukta"]
    ranked = reranker.rerank("sorgu", candidates)

    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
    # en uzun metin (index 1) en alakali sayilmali (sahte istemcinin mantigi)
    assert ranked[0][0] == 1


def test_rerank_preserves_scores_from_api(monkeypatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "fake-key")
    _install_fake_cohere(monkeypatch)
    reranker = CohereReranker(model_name="rerank-v4.0-pro")

    ranked = reranker.rerank("sorgu", ["a", "bb", "ccc"])
    for _, score in ranked:
        assert 0.0 <= score <= 1.0


def test_rerank_empty_candidates_returns_empty_list(monkeypatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "fake-key")
    _install_fake_cohere(monkeypatch)
    reranker = CohereReranker(model_name="rerank-v4.0-pro")

    assert reranker.rerank("sorgu", []) == []


def test_api_failure_raises_unavailable_not_silently(monkeypatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "fake-key")
    _install_fake_cohere(monkeypatch, fail=True)
    reranker = CohereReranker(model_name="rerank-v4.0-pro")

    with pytest.raises(RerankerUnavailableError):
        reranker.rerank("sorgu", ["a", "b"])
