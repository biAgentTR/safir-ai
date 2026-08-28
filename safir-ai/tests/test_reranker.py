"""`src/rag/reranker.py` (EvrenReranker) icin agsiz birim testleri.

`openai` gercekten cagrilmaz - istemcinin `chat.completions.create` metodu
monkeypatch ile sahte, deterministik nesnelerle degistirilir. Gemini/Groq
TAMAMEN KALDIRILDI - tek saglayici EVREN'in OpenAI-uyumlu LLM ucudur (bkz.
`test_no_gemini_or_groq_reranker_references_remain_in_source`).
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from src.rag.local_cross_encoder_reranker import CrossEncoderUnavailableError
from src.rag.reranker import EvrenReranker

_REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = types.SimpleNamespace(content=content)


class _FakeChatCompletionsResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeEvrenCompletions:
    def __init__(self, response_text: str | None = None, raise_error: bool = False):
        self._response_text = response_text
        self._raise_error = raise_error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_error:
            raise RuntimeError("EVREN API cagrisi basarisiz (sahte hata)")
        return _FakeChatCompletionsResponse(self._response_text)


class _FakeEvrenClient:
    def __init__(self, response_text: str | None = None, raise_error: bool = False):
        self.chat = types.SimpleNamespace(
            completions=_FakeEvrenCompletions(response_text=response_text, raise_error=raise_error)
        )


def _evren_reranker_with_fake_client(response_text: str | None = None, raise_error: bool = False) -> EvrenReranker:
    reranker = EvrenReranker(
        model_name="llm-fast",
        base_url="https://evren-llmapi.ssyz.org.tr/v1",
        api_key_env="EVREN_API_KEY_TEST",
    )
    reranker._client = _FakeEvrenClient(response_text=response_text, raise_error=raise_error)
    return reranker


def test_missing_api_key_raises_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("EVREN_API_KEY_TEST_UNSET", raising=False)
    reranker = EvrenReranker(
        model_name="llm-fast", base_url="https://evren-llmapi.ssyz.org.tr/v1", api_key_env="EVREN_API_KEY_TEST_UNSET"
    )

    with pytest.raises(CrossEncoderUnavailableError):
        reranker.score("sorgu", ["a", "bb"])


def test_construction_never_touches_network(monkeypatch) -> None:
    monkeypatch.delenv("EVREN_API_KEY_TEST_UNSET", raising=False)
    reranker = EvrenReranker(
        model_name="llm-fast", base_url="https://evren-llmapi.ssyz.org.tr/v1", api_key_env="EVREN_API_KEY_TEST_UNSET"
    )
    assert reranker._model_name == "llm-fast"  # yapici hicbir ag cagrisi yapmadi


def test_rerank_parses_structured_json_and_preserves_input_order() -> None:
    response = json.dumps({"results": [{"index": 0, "score": 0.2}, {"index": 1, "score": 0.9}, {"index": 2, "score": 0.5}]})
    reranker = _evren_reranker_with_fake_client(response_text=response)

    scores = reranker.score("sorgu", ["a", "b", "c"])

    # score() sozlesmesi: `texts` ile AYNI SIRADA skor listesi (rerank() gibi
    # skora gore SIRALANMAZ - siralamayi cagiran, `EmbeddingRAGService.query()` yapar).
    assert scores == [0.2, 0.9, 0.5]
    call = reranker._client.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["model"] == "llm-fast"


def test_rerank_ignores_surrounding_text_around_json() -> None:
    """Model bazen JSON'un etrafina istemeden metin ekleyebilir; yalnizca JSON blogu ayristirilmali."""
    response = "Iste sonuc:\n" + json.dumps({"results": [{"index": 0, "score": 0.7}]}) + "\nBu kadar."
    reranker = _evren_reranker_with_fake_client(response_text=response)

    scores = reranker.score("sorgu", ["tek aday"])

    assert scores == [0.7]


@pytest.mark.parametrize(
    "bad_response",
    [
        "bu JSON degil",
        json.dumps({"not_results": []}),
        json.dumps({"results": [{"index": 0}]}),  # score eksik
        json.dumps({"results": [{"index": 5, "score": 0.5}]}),  # araligin disinda index (candidate_count=2)
        json.dumps({"results": [{"index": 0, "score": 0.5}, {"index": 0, "score": 0.9}]}),  # tekrar eden index
        json.dumps({"results": [{"index": 0, "score": 1.5}]}),  # araligin disinda score
        json.dumps({"results": [{"index": "0", "score": 0.5}]}),  # index int degil
        json.dumps({"results": [{"index": 0, "score": 0.5}]}),  # EKSIK aday (candidate_count=2 ama 1 sonuc)
    ],
)
def test_invalid_or_malformed_json_raises_unavailable_not_silent_fallback(bad_response) -> None:
    reranker = _evren_reranker_with_fake_client(response_text=bad_response)

    with pytest.raises(CrossEncoderUnavailableError):
        reranker.score("sorgu", ["a", "b"])


def test_rerank_empty_candidates_returns_empty_list() -> None:
    reranker = _evren_reranker_with_fake_client(response_text=json.dumps({"results": []}))
    assert reranker.score("sorgu", []) == []


def test_api_failure_raises_unavailable_not_silently() -> None:
    reranker = _evren_reranker_with_fake_client(raise_error=True)

    with pytest.raises(CrossEncoderUnavailableError):
        reranker.score("sorgu", ["a", "b"])


def test_429_rate_limit_raises_unavailable_not_silent_fallback() -> None:
    """HEDEF 11: gercekte gozlenen '429 Too Many Requests' hatasi, SESSIZCE deterministic siralamaya DUSMEZ - `CrossEncoderUnavailableError` firlatir."""

    class _RateLimitError(Exception):
        def __init__(self) -> None:
            super().__init__("429 Too Many Requests: rate limit exceeded")

    reranker = EvrenReranker(
        model_name="llm-fast", base_url="https://evren-llmapi.ssyz.org.tr/v1", api_key_env="EVREN_API_KEY_TEST"
    )
    reranker._client = _FakeEvrenClient(raise_error=False)
    reranker._client.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(_RateLimitError())

    with pytest.raises(CrossEncoderUnavailableError, match="429"):
        reranker.score("sorgu", ["a", "b"])


def test_400_json_validate_failed_raises_unavailable_not_silent_fallback() -> None:
    """HEDEF 12: gercekte gozlenen '400 json_validate_failed' hatasi (model semaya uymayan JSON dondurdu) - `CrossEncoderUnavailableError` firlatir, deterministic siralamaya DUSULMEZ."""
    reranker = _evren_reranker_with_fake_client(response_text="Uzgunum, bu sorguyu degerlendiremiyorum.")

    with pytest.raises(CrossEncoderUnavailableError):
        reranker.score("sorgu", ["a", "b"])


def test_no_gemini_or_groq_reranker_references_remain_in_source() -> None:
    """Repo genelinde (bu test dosyasi haric) Gemini/Groq/ucuncu-taraf rerank saglayicisina referans KALMAMALI."""
    checked_paths = [
        _REPO_ROOT / "src" / "rag" / "reranker.py",
        _REPO_ROOT / "src" / "rag" / "embedding_rag_service.py",
        _REPO_ROOT / "src" / "main.py",
        _REPO_ROOT / "src" / "utils" / "config_loader.py",
        _REPO_ROOT / "configs" / "config.yaml",
    ]
    for path in checked_paths:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        content_lower = content.lower()
        assert "cohere" not in content_lower, f"Beklenmedik uçuncu-taraf reranker referansi: {path}"
        # gercek kod tanimi/cagrisi (docstring/yorum ICINDEKI tarihsel bahisler DEGIL):
        assert "class GeminiReranker" not in content, f"Beklenmedik Gemini reranker sinif tanimi: {path}"
        assert "class GroqReranker" not in content, f"Beklenmedik Groq reranker sinif tanimi: {path}"
        assert "GeminiReranker(" not in content, f"Beklenmedik Gemini reranker cagrisi: {path}"
        assert "GroqReranker(" not in content, f"Beklenmedik Groq reranker cagrisi: {path}"
