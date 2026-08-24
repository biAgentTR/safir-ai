"""ADIM 2: trace serialization + JobState + frame endpoint + SSE testleri.

Pipeline mantigi DEGISMEDEN, gercek pipeline'in ara asama ciktilarinin
serilestirilip (base64'suz) SSE ile disari verilmesini dogrular. Gercek bir
sentetik video ile tam akis (sampler -> VLM -> events -> context -> decision ->
escalation -> report) kosulur; VLM/LLM mock'tur (harici Gemini anahtari gerektirmez).
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
from src.main import JobState, app
from src.event_analysis.risk_resolver import resolve_deterministic_risk_with_provenance
from src.event_analysis.schemas import RuleMatch
from src.observability.trace_serializer import (
    STAGE_ORDER,
    serialize_decision,
    serialize_decision_final,
    serialize_vlm,
)


# --------------------------- ortak yardimcilar ---------------------------


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
def mock_pipeline(monkeypatch):
    """Global pipeline'i mock VLM/LLM + sahte RAG ile kurar (Gemini/GPU/ag gerekmez)."""
    monkeypatch.setattr(main, "EmbeddingRAGService", lambda *a, **k: _FakeRAG())
    base = main.load_config()
    cfg = base.model_copy(update={"app": base.app.model_copy(update={"use_mock_vlm": True, "use_mock_llm": True})})
    monkeypatch.setattr(main, "_pipeline", main.SafirPipeline(cfg))
    return cfg


@pytest.fixture
def video_in_data(tmp_path):
    """20-38. kareler arasi hareket iceren sentetik videoyu data/ altina yazar (temizler)."""
    Path("data").mkdir(exist_ok=True)
    path = Path("data") / "_trace_sse_test.mp4"
    frames = [np.full((120, 160, 3), 30, np.uint8) for _ in range(60)]
    for i in range(20, 38):
        cv2.rectangle(frames[i], (20, 20), (140, 100), (210, 210, 210), -1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
    for f in frames:
        writer.write(f)
    writer.release()
    yield str(path)
    path.unlink(missing_ok=True)


def _run_job_to_completion(client: TestClient, video: str, timeout: float = 30.0) -> str:
    resp = client.post("/analyze/jobs", json={"video_source": video, "user_prompt": "risk var mi?"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/analyze/jobs/{job_id}").json()
        if st["status"] in ("done", "error"):
            return job_id
        time.sleep(0.1)
    raise AssertionError("Is zamaninda tamamlanmadi.")


# --------------------------- serializer unit testleri ---------------------------


def test_serialize_decision_excludes_raw_response_and_internal_reasoning():
    class _D:
        risk_score = 80
        risk_level = "yuksek"
        summary = "ozet"
        recommended_action = "aksiyon"
        actions = ["a", "b"]
        events = [{"time": "00:10", "event": "x"}]
        raw_response = "GIZLI internal reasoning + tool zinciri"

    summary, data, frames, status, error = serialize_decision({"decision": _D()}, "job1")
    assert "raw_response" not in data
    assert "GIZLI" not in json.dumps(data)
    assert data["risk_score"] == 80 and data["actions"] == ["a", "b"]
    assert status == "completed"


def test_serialize_decision_final_explains_rule_engine_override_of_agent_draft():
    """Kok neden testi (90 vs 88): 'decision_final' artik trace'e ULASIYOR ve
    RuleEngine'in NEDEN/HANGI kuralla Agent'in taslak skorunu ezdigini gosteriyor."""

    class _Decision:
        risk_score = 88
        risk_level = "kritik"
        risk_status = "assessed"

    rule_matches = [
        RuleMatch(
            rule_id="YG-03",
            rule_description="Yangin Guvenligi Talimati YG-03",
            event_type="yangin_duman",
            severity="kritik",
            source_event_id="evt_0",
        )
    ]
    provenance = resolve_deterministic_risk_with_provenance(rule_matches)

    summary, data, frames, status, error = serialize_decision_final(
        {"decision": _Decision(), "risk_provenance": provenance}, "job1"
    )

    assert status == "completed"
    assert data["risk_score"] == 88
    assert data["risk_provenance"]["risk_source"] == "rule_engine"
    assert data["risk_provenance"]["rule_ids"] == ["YG-03"]
    assert "YG-03" in data["risk_provenance"]["explanation"]
    assert "NIHAI" in summary and "YG-03" in summary


def test_serialize_decision_final_marks_agent_only_when_no_rule_matched():
    class _Decision:
        risk_score = 30
        risk_level = "dusuk"
        risk_status = "assessed"

    provenance = resolve_deterministic_risk_with_provenance([])

    summary, data, frames, status, error = serialize_decision_final(
        {"decision": _Decision(), "risk_provenance": provenance}, "job1"
    )

    assert data["risk_provenance"]["risk_source"] is None
    assert "Agent" in summary


def test_serialize_vlm_marks_degraded_as_failed():
    class _VR:
        description = "[HATA] VLM analizi yapilamadi (x). Manuel inceleme gerekli."
        model_name = "gemini-x"
        frame_count = 3
        latency_ms = 0.0
        structured_events = []

    summary, data, frames, status, error = serialize_vlm({"vlm_response": _VR(), "clusters": [], "user_prompt": "p"}, "j")
    assert status == "failed" and error
    assert "base64" not in json.dumps(data)


# --------------------------- tam akis: trace + no-base64 ---------------------------


def test_full_job_produces_all_stage_events_without_base64(mock_pipeline, video_in_data):
    client = TestClient(app)
    job_id = _run_job_to_completion(client, video_in_data)

    job = main._jobs[job_id]
    assert job.status == "done"
    stages = [e["stage"] for e in job.trace_events]
    # Tum gercek stage'ler sirayla uretildi
    assert stages == STAGE_ORDER

    blob = json.dumps(job.trace_events, ensure_ascii=False)
    for forbidden in ("base64", "image_bytes", "raw_response", "data:image"):
        assert forbidden not in blob, f"{forbidden} SSE trace payload'una sizmis!"

    # her olayda zorunlu alanlar
    for ev in job.trace_events:
        assert set(ev) >= {"stage", "status", "timestamp", "duration_ms", "summary", "data", "metadata", "error"}
        assert ev["status"] in ("queued", "running", "completed", "failed")
        assert ev["metadata"]["label"]  # presentation label

    # kareler ayri tutuluyor + referanslaniyor
    assert len(job.frames) > 0
    sampler_ev = next(e for e in job.trace_events if e["stage"] == "sampler")
    ref = sampler_ev["data"]["evidence_frames"][0]
    assert ref["thumbnail_url"].startswith(f"/analyze/jobs/{job_id}/frames/")


# --------------------------- frame endpoint ---------------------------


def test_frame_endpoint_serves_bytes_validates_and_404(mock_pipeline, video_in_data):
    client = TestClient(app)
    job_id = _run_job_to_completion(client, video_in_data)
    frame_id = next(iter(main._jobs[job_id].frames))

    ok = client.get(f"/analyze/jobs/{job_id}/frames/{frame_id}")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/jpeg" and ok.content

    assert client.get(f"/analyze/jobs/{job_id}/frames/yok_boyle_kare").status_code == 404
    assert client.get(f"/analyze/jobs/{job_id}/frames/..%2Fetc").status_code in (400, 404)
    assert client.get("/analyze/jobs/bilinmeyen/frames/ev0").status_code == 404


# --------------------------- SSE endpoint ---------------------------


def test_sse_replays_events_and_ends(mock_pipeline, video_in_data):
    client = TestClient(app)
    job_id = _run_job_to_completion(client, video_in_data)

    body = client.get(f"/analyze/jobs/{job_id}/stream").text
    # Tamamlanmis is: tum olaylar replay + 'end'
    for stage in STAGE_ORDER:
        assert f'"stage": "{stage}"' in body
    assert "event: end" in body
    assert '"status": "done"' in body
    assert "base64" not in body and "image_bytes" not in body


def test_sse_unknown_job_yields_error_event():
    client = TestClient(app)
    body = client.get("/analyze/jobs/bilinmeyen-id/stream").text
    assert "event: error" in body and "bulunamadi" in body.lower()


def test_existing_polling_endpoint_unchanged(mock_pipeline, video_in_data):
    """Mevcut GET /analyze/jobs/{id} sozlesmesi degismedi (SSE additive)."""
    client = TestClient(app)
    job_id = _run_job_to_completion(client, video_in_data)
    st = client.get(f"/analyze/jobs/{job_id}").json()
    assert set(st) == {"status", "stage_name", "step", "total_steps", "result", "error"}
    assert st["status"] == "done" and st["result"] is not None
    assert client.get("/analyze/jobs/bilinmeyen").status_code == 404
