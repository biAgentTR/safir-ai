"""`src/rag/reranker.py` (GeminiReranker + GroqReranker) icin agsiz birim testleri.

`google.genai`/`openai` gercekten cagrilmaz - istemcilerin `generate_content`/
`chat.completions.create` metotlari monkeypatch ile sahte, deterministik
nesnelerle degistirilir. Desteklenen saglayicilar: Gemini ve Groq (AYNI
prompt/JSON semasi/dogrulama kurallari, bkz. `_build_rerank_prompt`/
`_parse_rerank_response`) - bkz. `test_no_third_party_reranker_references_remain_in_source`.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from src.rag.reranker import GeminiReranker, GroqReranker, RerankerUnavailableError

_REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, response_text: str | None = None, raise_error: bool = False):
        self._response_text = response_text
        self._raise_error = raise_error
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._raise_error:
            raise RuntimeError("Gemini API cagrisi basarisiz (sahte hata)")
        return _FakeResponse(self._response_text)


class _FakeClient:
    def __init__(self, response_text: str | None = None, raise_error: bool = False):
        self.models = _FakeModels(response_text=response_text, raise_error=raise_error)


def _install_fake_genai(monkeypatch, response_text: str | None = None, raise_error: bool = False) -> _FakeClient:
    fake_client = _FakeClient(response_text=response_text, raise_error=raise_error)

    fake_genai_module = types.ModuleType("google.genai")
    fake_genai_module.Client = lambda api_key: fake_client  # noqa: ARG005

    fake_types_module = types.ModuleType("google.genai.types")

    class _FakeGenerateContentConfig:
        def __init__(self, response_mime_type=None, temperature=None):
            self.response_mime_type = response_mime_type
            self.temperature = temperature

    fake_types_module.GenerateContentConfig = _FakeGenerateContentConfig

    fake_google_module = sys.modules.get("google") or types.ModuleType("google")
    fake_google_module.genai = fake_genai_module

    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types_module)
    return fake_client


def test_missing_api_key_raises_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    reranker = GeminiReranker(model_name="gemini-3.5-flash-lite")

    with pytest.raises(RerankerUnavailableError):
        reranker.rerank("sorgu", ["a", "bb"])


def test_construction_never_touches_network(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    reranker = GeminiReranker(model_name="gemini-3.5-flash-lite")
    assert reranker._model_name == "gemini-3.5-flash-lite"  # yapici hicbir ag cagrisi yapmadi


def test_rerank_parses_structured_json_and_orders_by_score(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    response = json.dumps({"results": [{"index": 0, "score": 0.2}, {"index": 1, "score": 0.9}, {"index": 2, "score": 0.5}]})
    fake_client = _install_fake_genai(monkeypatch, response_text=response)
    reranker = GeminiReranker(model_name="gemini-3.5-flash-lite")

    ranked = reranker.rerank("sorgu", ["a", "b", "c"])

    assert ranked == [(1, 0.9), (2, 0.5), (0, 0.2)]
    assert fake_client.models.calls[0]["config"].response_mime_type == "application/json"


def test_rerank_ignores_surrounding_text_around_json(monkeypatch) -> None:
    """Model bazen JSON'un etrafina istemeden metin ekleyebilir; yalnizca JSON blogu ayristirilmali."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    response = "Iste sonuc:\n" + json.dumps({"results": [{"index": 0, "score": 0.7}]}) + "\nBu kadar."
    _install_fake_genai(monkeypatch, response_text=response)
    reranker = GeminiReranker(model_name="gemini-3.5-flash-lite")

    ranked = reranker.rerank("sorgu", ["tek aday"])

    assert ranked == [(0, 0.7)]


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
    ],
)
def test_invalid_or_malformed_json_raises_unavailable_not_silent_fallback(monkeypatch, bad_response) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    _install_fake_genai(monkeypatch, response_text=bad_response)
    reranker = GeminiReranker(model_name="gemini-3.5-flash-lite")

    with pytest.raises(RerankerUnavailableError):
        reranker.rerank("sorgu", ["a", "b"])


def test_rerank_empty_candidates_returns_empty_list(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    _install_fake_genai(monkeypatch, response_text=json.dumps({"results": []}))
    reranker = GeminiReranker(model_name="gemini-3.5-flash-lite")

    assert reranker.rerank("sorgu", []) == []


def test_api_failure_raises_unavailable_not_silently(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    _install_fake_genai(monkeypatch, raise_error=True)
    reranker = GeminiReranker(model_name="gemini-3.5-flash-lite")

    with pytest.raises(RerankerUnavailableError):
        reranker.rerank("sorgu", ["a", "b"])


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = types.SimpleNamespace(content=content)


class _FakeChatCompletionsResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeGroqCompletions:
    def __init__(self, response_text: str | None = None, raise_error: bool = False):
        self._response_text = response_text
        self._raise_error = raise_error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_error:
            raise RuntimeError("Groq API cagrisi basarisiz (sahte hata)")
        return _FakeChatCompletionsResponse(self._response_text)


class _FakeGroqClient:
    def __init__(self, response_text: str | None = None, raise_error: bool = False):
        self.chat = types.SimpleNamespace(
            completions=_FakeGroqCompletions(response_text=response_text, raise_error=raise_error)
        )


def _groq_reranker_with_fake_client(response_text: str | None = None, raise_error: bool = False) -> GroqReranker:
    reranker = GroqReranker(model_name="openai/gpt-oss-20b")
    reranker._client = _FakeGroqClient(response_text=response_text, raise_error=raise_error)
    return reranker


def test_groq_missing_api_key_raises_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    reranker = GroqReranker(model_name="openai/gpt-oss-20b")

    with pytest.raises(RerankerUnavailableError):
        reranker.rerank("sorgu", ["a", "bb"])


def test_groq_construction_never_touches_network(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    reranker = GroqReranker(model_name="openai/gpt-oss-20b")
    assert reranker._model_name == "openai/gpt-oss-20b"  # yapici hicbir ag cagrisi yapmadi


def test_groq_rerank_parses_structured_json_and_orders_by_score() -> None:
    response = json.dumps({"results": [{"index": 0, "score": 0.2}, {"index": 1, "score": 0.9}, {"index": 2, "score": 0.5}]})
    reranker = _groq_reranker_with_fake_client(response_text=response)

    ranked = reranker.rerank("sorgu", ["a", "b", "c"])

    assert ranked == [(1, 0.9), (2, 0.5), (0, 0.2)]
    call = reranker._client.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["model"] == "openai/gpt-oss-20b"


def test_groq_rerank_uses_same_validation_rules_as_gemini() -> None:
    """Ayni gecersiz-JSON kurallari GroqReranker icin de gecerli olmali (paylasilan _parse_rerank_response)."""
    reranker = _groq_reranker_with_fake_client(response_text=json.dumps({"results": [{"index": 9, "score": 0.5}]}))

    with pytest.raises(RerankerUnavailableError):
        reranker.rerank("sorgu", ["a", "b"])


def test_groq_api_failure_raises_unavailable_not_silently() -> None:
    reranker = _groq_reranker_with_fake_client(raise_error=True)

    with pytest.raises(RerankerUnavailableError):
        reranker.rerank("sorgu", ["a", "b"])


def test_groq_rerank_empty_candidates_returns_empty_list() -> None:
    reranker = _groq_reranker_with_fake_client(response_text=json.dumps({"results": []}))
    assert reranker.rerank("sorgu", []) == []


def test_no_third_party_reranker_references_remain_in_source() -> None:
    """Repo genelinde (bu test dosyasi haric) baska hicbir rerank saglayicisina (orn. eski Cohere entegrasyonu) referans KALMAMALI."""
    checked_paths = [
        _REPO_ROOT / "src" / "memory" / "reranker.py",
        _REPO_ROOT / "src" / "memory" / "embedding_rag_service.py",
        _REPO_ROOT / "src" / "utils" / "config_loader.py",
        _REPO_ROOT / "configs" / "config.yaml",
        _REPO_ROOT / "requirements.txt",
        _REPO_ROOT / "requirements-gemini.txt",
        _REPO_ROOT / "scripts" / "rag_smoke_test.py",
    ]
    for path in checked_paths:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8").lower()
        assert "cohere" not in content, f"Beklenmedik uçuncu-taraf reranker referansi: {path}"
