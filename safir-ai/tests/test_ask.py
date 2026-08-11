"""ADIM 5B: Ask SAFIR (/ask) testleri.

Mevcut RAG + LLM soyutlamalari (fake RAG + MockLLMClient) ile; pipeline/history
davranisi degismeden, soru -> (opsiyonel analiz baglami) + RAG -> LLM -> dayanakli
cevap + gercek kaynaklar zincirini ve sizinti-yoklugunu dogrular.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.main as main
from src.main import app
from src.assistant.ask_service import AskService
from src.memory.analysis_store import AnalysisStore
from src.vlm.llm_client import MockLLMClient


# --------------------------- fakes / fixtures ---------------------------


class _RagDoc:
    def __init__(self, text: str, score: float) -> None:
        self.text = text
        self.score = score


class _FakeAskRAG:
    """Ask icin sabit iki ISG maddesi (gercek text+score sozlesmesi)."""

    def query(self, question: str, top_k=None):
        return [
            _RagDoc("ISG Yonetmeligi Madde 12: Yuksekte calismada KKD zorunludur.", 0.82),
            _RagDoc("Operasyonel Kural OK-07: Forklift-yaya ayrimi saglanmalidir.", 0.71),
        ]


class _FakeDoc:
    def __init__(self, text: str) -> None:
        self.text = text
        self.score = 1.0


class _FakePipelineRAG:
    def seed_default_regulations(self):
        return None

    def query(self, question, top_k=None):
        return [_FakeDoc(f"[FAKE] {question}")]


@pytest.fixture
def ask_env(monkeypatch, tmp_path):
    """Mock pipeline + izole AnalysisStore + fake-RAG/mock-LLM AskService."""
    monkeypatch.setattr(main, "EmbeddingRAGService", lambda *a, **k: _FakePipelineRAG())
    base = main.load_config()
    cfg = base.model_copy(update={"app": base.app.model_copy(update={"use_mock_vlm": True, "use_mock_llm": True})})
    monkeypatch.setattr(main, "_pipeline", main.SafirPipeline(cfg))

    store = AnalysisStore(tmp_path / "analyses.db")
    monkeypatch.setattr(main, "_analysis_store", store)

    llm = MockLLMClient()
    ask = AskService(analysis_store=store, rag_service=_FakeAskRAG(), llm_client=llm, rag_top_k=5)
    monkeypatch.setattr(main, "_ask_service", ask)
    return {"store": store, "ask": ask, "llm": llm}


@pytest.fixture
def video_in_data():
    Path("data").mkdir(exist_ok=True)
    path = Path("data") / "_ask_test.mp4"
    frames = [np.full((120, 160, 3), 30, np.uint8) for _ in range(60)]
    for i in range(20, 40):
        cv2.rectangle(frames[i], (20, 20), (140, 100), (210, 210, 210), -1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
    for f in frames:
        writer.write(f)
    writer.release()
    yield str(path)
    path.unlink(missing_ok=True)


def _run_job(client: TestClient, video: str) -> str:
    job_id = client.post("/analyze/jobs", json={"video_source": video, "user_prompt": "risk?"}).json()["job_id"]
    deadline = time.time() + 30
    while time.time() < deadline:
        if client.get(f"/analyze/jobs/{job_id}").json()["status"] in ("done", "error"):
            return job_id
        time.sleep(0.1)
    raise AssertionError("Is tamamlanmadi.")


# --------------------------- general ask ---------------------------


def test_general_ask_uses_rag_sources(ask_env):
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "Yuksekte calismada nelere dikkat edilmeli?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] and isinstance(body["answer"], str)
    assert body["job_id"] is None
    # sources GERCEK retrieval sonucundan (fake RAG'in text+score'u)
    regs = [s for s in body["sources"] if s["type"] == "regulation"]
    assert len(regs) == 2
    assert regs[0]["text"].startswith("ISG Yonetmeligi Madde 12") and regs[0]["score"] == 0.82
    assert "RAG mevzuat (2)" in body["context_used"]


# --------------------------- job-scoped ask ---------------------------


def test_job_scoped_ask_uses_analysis_context(ask_env, video_in_data):
    client = TestClient(app)
    job_id = _run_job(client, video_in_data)

    resp = client.post("/ask", json={"question": "Bu analiz neden bu risk seviyesinde?", "job_id": job_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert any("analiz raporu" in c for c in body["context_used"])
    # analiz kaynagi + mevzuat kaynaklari birlikte
    types = {s["type"] for s in body["sources"]}
    assert "analysis" in types and "regulation" in types
    analysis_src = next(s for s in body["sources"] if s["type"] == "analysis")
    assert analysis_src["label"] == "SAFIR analiz raporu"


def test_unknown_job_id_404(ask_env):
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "neden?", "job_id": "yok-boyle-job"})
    assert resp.status_code == 404


def test_empty_question_422(ask_env):
    client = TestClient(app)
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422


# --------------------------- safety / leakage ---------------------------


def test_no_internal_or_secret_leakage(ask_env, video_in_data):
    client = TestClient(app)
    job_id = _run_job(client, video_in_data)
    resp = client.post("/ask", json={"question": "Bu kararin dayanagi ne?", "job_id": job_id})
    blob = json.dumps(resp.json(), ensure_ascii=False)
    for forbidden in ("raw_response", "api_key", "API_KEY", "base64", "image_bytes", "system_prompt", "Traceback"):
        assert forbidden not in blob, f"sizinti: {forbidden}"


def test_llm_unavailable_returns_safe_503(ask_env, monkeypatch):
    client = TestClient(app)

    class _BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("gizli stack detay: SECRET")

    monkeypatch.setattr(main._ask_service, "_llm", _BoomLLM())
    resp = client.post("/ask", json={"question": "test"})
    assert resp.status_code == 503
    assert "SECRET" not in resp.text and "Traceback" not in resp.text
    assert resp.json()["detail"] == "SAFIR su anda cevap olusturamadi."


def test_ask_system_prompt_enforces_grounding():
    from src.assistant.ask_service import ASK_SYSTEM_PROMPT

    low = ASK_SYSTEM_PROMPT.lower()
    assert "uydurma" in low and "baglam" in low  # sadece baglamdan, uydurma yok
