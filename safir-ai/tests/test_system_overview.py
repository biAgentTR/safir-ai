"""Adim 2 - Data Center ("Sistem Verileri"): GET /system/overview + zenginlestirilmis
GET /history / GET /conversations alanlari icin testler.

Tamamen read-only bir ozet uc noktasi oldugu icin buradaki testler GERCEK
AnalysisStore/ConversationStore uzerinden GERCEK sayimlarin dogru geldigini
dogrular (mock veri degil) - mock olan yalnizca VLM/LLM (mevcut test
altyapisiyla ayni).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.main as main
from src.main import app
from src.memory.analysis_store import AnalysisStore
from src.memory.conversation_store import ConversationStore


class _FakeDoc:
    def __init__(self, text: str) -> None:
        self.text = text
        self.score = 1.0


class _FakeRAG:
    def seed_default_regulations(self) -> None:
        return None

    def query(self, question: str, top_k=None, keywords=None):
        return [_FakeDoc(f"[FAKE] {question}")]


@pytest.fixture
def data_center_env(monkeypatch, tmp_path):
    """Mock VLM/LLM + izole (tmp) AnalysisStore + izole (tmp) ConversationStore."""
    monkeypatch.setattr(main, "EmbeddingRAGService", lambda *a, **k: _FakeRAG())
    base = main.load_config()
    cfg = base.model_copy(update={"app": base.app.model_copy(update={"use_mock_vlm": True, "use_mock_llm": True})})
    monkeypatch.setattr(main, "_pipeline", main.SafirPipeline(cfg))

    analysis_store = AnalysisStore(tmp_path / "analyses.db")
    monkeypatch.setattr(main, "_analysis_store", analysis_store)
    conversation_store = ConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr(main, "_conversation_store", conversation_store)
    return analysis_store, conversation_store


@pytest.fixture
def video_in_data():
    Path("data").mkdir(exist_ok=True)
    path = Path("data") / "_system_overview_test.mp4"
    frames = [np.full((120, 160, 3), 30, np.uint8) for _ in range(60)]
    for i in range(20, 40):
        cv2.rectangle(frames[i], (20, 20), (140, 100), (210, 210, 210), -1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
    for f in frames:
        writer.write(f)
    writer.release()
    yield str(path)
    path.unlink(missing_ok=True)


def _run_to_completion(client: TestClient, video: str, timeout: float = 30.0) -> str:
    resp = client.post("/analyze/jobs", json={"video_source": video, "user_prompt": "risk?"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/analyze/jobs/{job_id}").json()
        if st["status"] in ("done", "error"):
            return job_id
        time.sleep(0.1)
    raise AssertionError("Is zamaninda tamamlanmadi.")


def test_system_overview_counts_real_analyses_and_conversations(data_center_env, video_in_data):
    _, conversation_store = data_center_env
    client = TestClient(app)

    job_id = _run_to_completion(client, video_in_data)

    conv = conversation_store.create(title="Test sohbet")
    conversation_store.add_message(conv.conversation_id, "user", "soru")
    conversation_store.add_message(conv.conversation_id, "assistant", "cevap")

    resp = client.get("/system/overview")
    assert resp.status_code == 200
    body = resp.json()
    totals = body["totals"]

    assert totals["total_analyses"] == 1
    assert totals["completed_analyses"] == 1
    assert totals["failed_analyses"] == 0
    assert totals["running_or_queued_analyses"] == 0
    assert totals["total_conversations"] == 1
    assert totals["total_messages"] == 2
    assert totals["analyses_with_trace"] == 1
    assert body["scan_limit"] == main._SYSTEM_OVERVIEW_SCAN_LIMIT
    assert "generated_at" in body

    # bu analiz gercekten VOD (canli kaynak degil) oldugu icin representative
    # frame cikarimi calisir; en az bir kare diske gercekten yazilmis olmali.
    assert totals["stored_representative_frame_count"] > 0

    # job_id kaydin gercekten var oldugunu (mock veri degil) capraz dogrular
    history = client.get("/history").json()
    assert any(r["job_id"] == job_id for r in history)


def test_system_overview_is_read_only_get_only(data_center_env):
    client = TestClient(app)
    # POST/PUT/DELETE tanimli degil -> FastAPI 405 doner (yazma ucu YOK)
    assert client.post("/system/overview").status_code == 405
    assert client.delete("/system/overview").status_code == 405


def test_system_overview_empty_state_is_all_zero(data_center_env):
    client = TestClient(app)
    body = client.get("/system/overview").json()["totals"]
    assert body == {
        "total_analyses": 0,
        "completed_analyses": 0,
        "failed_analyses": 0,
        "running_or_queued_analyses": 0,
        "total_conversations": 0,
        "total_messages": 0,
        "analyses_with_trace": 0,
        "stored_representative_frame_count": 0,
        # RAG + Prompt Injection Guard telemetrisi (bkz. _RagSecurityAggregate):
        # veri yoksa sayimlar 0, ortalamalar "N/A" anlaminda None kalir.
        "total_events_detected": 0,
        "critical_risk_analyses": 0,
        "rag_query_count": 0,
        "rag_zero_result_count": 0,
        "avg_embedding_latency_ms": None,
        "avg_rerank_latency_ms": None,
        "avg_total_rag_latency_ms": None,
        "guard_checks": 0,
        "guard_allowed": 0,
        "guard_quarantined": 0,
        "guard_failures": 0,
        "guard_fail_closed_blocks": 0,
        "avg_guard_latency_ms": None,
        "avg_guard_confidence": None,
    }


def test_history_list_includes_updated_at(data_center_env, video_in_data):
    client = TestClient(app)
    _run_to_completion(client, video_in_data)
    row = client.get("/history").json()[0]
    assert "updated_at" in row and row["updated_at"]


def test_conversations_list_includes_message_count(data_center_env):
    _, conversation_store = data_center_env
    client = TestClient(app)
    conv = conversation_store.create(title="t")
    conversation_store.add_message(conv.conversation_id, "user", "a")
    conversation_store.add_message(conv.conversation_id, "assistant", "b")
    conversation_store.add_message(conv.conversation_id, "user", "c")

    row = client.get("/conversations").json()[0]
    assert row["conversation_id"] == conv.conversation_id
    assert row["message_count"] == 3


def test_conversations_list_message_count_zero_for_empty_conversation(data_center_env):
    _, conversation_store = data_center_env
    client = TestClient(app)
    conversation_store.create(title="bos sohbet")
    row = client.get("/conversations").json()[0]
    assert row["message_count"] == 0
