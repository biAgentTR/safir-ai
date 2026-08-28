"""ADIM 5A (P0): 'risk=0' ile 'analiz basarisiz' (risk_status='unknown') ayrimi.

Onceki davranista `_degraded_decision()`/`_coerce_risk_score()` basarisizlikta
`risk_score=0` uretiyordu; bu, gercek dusuk riskle ayirt edilemiyordu. Bu
dosya, yeni `risk_status: "assessed" | "unknown"` alaninin `AgentDecision` ->
`EscalationPolicy` -> `SafirReport`/`AnalysisStore` -> `/history` API ->
export (JSON/HTML/PDF) zincirinde dogru ve kayipsiz tasindigini dogrular.
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
from src.decision.escalation import EscalationPolicy, EscalationTier
from src.main import app
from src.memory.analysis_store import AnalysisStore
from src.schemas.report import SafirReport
from src.ui.report_export import ReportExporter
from src.ui.theme import UNKNOWN_BADGE, resolve_risk_badge
from src.utils.config_loader import EscalationConfig


# ------------------------------------------------------------------
# 1) Gercek dusuk risk (risk_score=0) -> risk_status="assessed"
# ------------------------------------------------------------------


def test_genuine_zero_score_stays_assessed(safir_config) -> None:
    """Model gercekten 0 dondururse, bu 'assessed' kalir — 'unknown'a CEVRILMEZ."""
    from src.agent.langgraph_agent import SafirAgent

    agent = SafirAgent(llm_config=safir_config.llm, agent_config=safir_config.agent, use_mock_llm=True)
    decision = agent._parse_decision(
        json.dumps({"summary": "Risk yok.", "risk_score": 0, "actions": ["izlemeye devam"], "events": []})
    )

    assert decision.risk_score == 0
    assert decision.risk_status == "assessed"
    assert decision.risk_level == "dusuk"


# ------------------------------------------------------------------
# 2) LLM/ajan zinciri patlar -> risk_score=None, risk_status="unknown"
# ------------------------------------------------------------------


def test_agent_chain_failure_never_becomes_low_risk(safir_config) -> None:
    from src.agent.langgraph_agent import SafirAgent

    agent = SafirAgent(llm_config=safir_config.llm, agent_config=safir_config.agent, use_mock_llm=True)
    agent._llm.invoke = lambda _messages: (_ for _ in ()).throw(RuntimeError("LLM coktu"))

    decision = agent.run("## Gozlem\ntest")

    assert decision.risk_score is None
    assert decision.risk_status == "unknown"
    assert decision.risk_level == "unknown"


@pytest.mark.parametrize(
    "raw_text",
    [
        "bu metin ne JSON ne de eski RISK_SKORU formatinda",
        json.dumps({"summary": "eksik alan", "actions": ["x"]}),  # risk_score alani yok
        json.dumps({"summary": "gecersiz", "risk_score": "cok-riskli", "actions": []}),  # sayisal degil
    ],
)
def test_undecodable_output_becomes_unknown_not_zero(safir_config, raw_text: str) -> None:
    """JSON/regex hicbir gecerli risk_score cikaramazsa None doner, 0 DEGIL."""
    from src.agent.langgraph_agent import SafirAgent

    agent = SafirAgent(llm_config=safir_config.llm, agent_config=safir_config.agent, use_mock_llm=True)
    decision = agent._parse_decision(raw_text)

    assert decision.risk_score is None
    assert decision.risk_status == "unknown"
    assert decision.risk_level == "unknown"


# ------------------------------------------------------------------
# 3) VLM basarisizligi -> degraded path -> unknown (ayni _degraded_decision yolu)
# ------------------------------------------------------------------


def test_degraded_decision_used_when_reasoning_raises(safir_config) -> None:
    """VLM/LLM zincirindeki genel hata durumunda kullanilan `_degraded_decision`
    dogrudan 'unknown' uretir (bkz. test_resilience.py'deki uctan uca esdegeri)."""
    from src.agent.langgraph_agent import SafirAgent

    agent = SafirAgent(llm_config=safir_config.llm, agent_config=safir_config.agent, use_mock_llm=True)
    decision = agent._degraded_decision("simulated hata")

    assert decision.risk_score is None
    assert decision.risk_status == "unknown"
    assert "[HATA]" in decision.raw_response


# ------------------------------------------------------------------
# 5) Escalation: unknown ASLA low-risk/otomatik-alarm olarak islenmez
# ------------------------------------------------------------------


class _RecordingSink:
    def __init__(self) -> None:
        self.dispatched = []

    def dispatch(self, *, risk_score, risk_level, recommended_action, summary, auto) -> str:
        self.dispatched.append({"risk_score": risk_score, "auto": auto})
        return f"alert-{len(self.dispatched)}"

    def acknowledge(self, alert_id: str, operator_note: str = ""):  # pragma: no cover
        raise NotImplementedError


def test_escalation_unknown_status_forces_pending_review_never_alarm_or_silent_monitor() -> None:
    """2026-08-25 (Human-on-the-Loop siklastirma): belirsiz durum artik sessizce NOTIFY'a
    DEGIL, operatorun ACIK karari beklenen PENDING_REVIEW'e duser (bkz. escalation.py)."""
    config = EscalationConfig(notify_score=26, auto_alarm_score=51)
    sink = _RecordingSink()
    policy = EscalationPolicy(config, sink=sink)

    decision = policy.evaluate(
        risk_score=None,
        risk_level="unknown",
        recommended_action="Sahayi manuel inceleyin",
        summary="Risk belirsiz",
        risk_status="unknown",
    )

    assert decision.tier is EscalationTier.PENDING_REVIEW
    assert decision.auto_dispatched is False
    assert decision.alert_id is None
    assert sink.dispatched == []  # ASLA otomatik alarm tetiklenmez


def test_escalation_defaults_to_assessed_for_backward_compat() -> None:
    """`risk_status` verilmezse varsayilan 'assessed' — mevcut cagiranlar bozulmaz."""
    config = EscalationConfig(notify_score=26, auto_alarm_score=51)
    policy = EscalationPolicy(config, sink=_RecordingSink())

    decision = policy.evaluate(risk_score=10, risk_level="dusuk", recommended_action="izle", summary="s")
    assert decision.tier is EscalationTier.MONITOR


# ------------------------------------------------------------------
# 4) Persistence: unknown durumu DB <-> API arasinda kaybolmaz
# ------------------------------------------------------------------


def test_analysis_store_persists_and_reloads_unknown_risk_status(tmp_path) -> None:
    db = tmp_path / "a.db"
    s1 = AnalysisStore(db)
    s1.create("job-unknown", "data/v.mp4")
    s1.mark_completed(
        "job-unknown",
        report_json=json.dumps({"risk_score": None, "risk_status": "unknown"}),
        risk_level="unknown",
        risk_score=None,
        summary="Risk belirsiz",
        risk_status="unknown",
    )
    s1.close()

    s2 = AnalysisStore(db)  # restart
    rec = s2.get("job-unknown")
    assert rec is not None
    assert rec.risk_score is None
    assert rec.risk_status == "unknown"
    s2.close()


def test_analysis_store_old_rows_default_to_assessed(tmp_path) -> None:
    """Migrasyon oncesi (risk_status kolonu olmadan) yazilmis satirlar 'assessed' varsayilanini alir."""
    import sqlite3

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE analyses (
            job_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            status TEXT NOT NULL, video_source TEXT, risk_level TEXT, risk_score INTEGER,
            summary TEXT, report_json TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO analyses (job_id, created_at, updated_at, status, video_source, risk_level, risk_score, summary, report_json) "
        "VALUES ('legacy-1', 't', 't', 'completed', 'v.mp4', 'dusuk', 5, 's', '{}')"
    )
    conn.commit()
    conn.close()

    store = AnalysisStore(db)  # migration calisir
    rec = store.get("legacy-1")
    assert rec is not None
    assert rec.risk_status == "assessed"
    store.close()


# ------------------------------------------------------------------
# 4b) End-to-end API: agent basarisiz -> /history + /analyze/jobs unknown'i korur
# ------------------------------------------------------------------


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
def mock_pipeline_agent_fails(monkeypatch, tmp_path):
    """mock_pipeline (test_history.py ile ayni desen) + ajan LLM'i her zaman patlar."""
    monkeypatch.setattr(main, "EmbeddingRAGService", lambda *a, **k: _FakeRAG())
    # cfg sets both mock flags True -> SafirPipeline takes the
    # MockEmbeddingRAGService branch (see main.py), not EmbeddingRAGService.
    monkeypatch.setattr(main, "MockEmbeddingRAGService", lambda *a, **k: _FakeRAG())
    base = main.load_config()
    cfg = base.model_copy(update={"app": base.app.model_copy(update={"use_mock_vlm": True, "use_mock_llm": True})})
    pipeline = main.SafirPipeline(cfg)
    pipeline._agent._llm.invoke = lambda _messages: (_ for _ in ()).throw(RuntimeError("LLM coktu (test)"))
    monkeypatch.setattr(main, "_pipeline", pipeline)
    store = AnalysisStore(tmp_path / "analyses.db")
    monkeypatch.setattr(main, "_analysis_store", store)
    return store


@pytest.fixture
def video_in_data():
    Path("data").mkdir(exist_ok=True)
    path = Path("data") / "_risk_status_test.mp4"
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


def test_pipeline_agent_failure_yields_unknown_report_end_to_end(mock_pipeline_agent_fails, video_in_data) -> None:
    """Ajan basarisiz olsa bile pipeline COKMEZ; nihai rapor 0/dusuk risk DEGIL, unknown'dur."""
    client = TestClient(app)
    job_id = _run_to_completion(client, video_in_data)

    polled = client.get(f"/analyze/jobs/{job_id}").json()
    assert polled["status"] == "done"
    result = polled["result"]
    assert result["risk_score"] is None
    assert result["risk_status"] == "unknown"
    assert result["escalation_tier"] == "pending_review"  # ASLA otomatik alarm/monitor degil
    assert result["auto_dispatched"] is False


def test_history_preserves_unknown_risk_status_via_api(mock_pipeline_agent_fails, video_in_data) -> None:
    client = TestClient(app)
    job_id = _run_to_completion(client, video_in_data)

    lst = client.get("/history").json()
    row = next((x for x in lst if x["job_id"] == job_id), None)
    assert row is not None
    assert row["risk_score"] is None
    assert row["risk_status"] == "unknown"

    detail = client.get(f"/history/{job_id}").json()
    assert detail["report"]["risk_score"] is None
    assert detail["report"]["risk_status"] == "unknown"


def test_report_pdf_for_unknown_risk_job_still_generates_valid_pdf(mock_pipeline_agent_fails, video_in_data) -> None:
    """PDF export, risk_score=None oldugunda TypeError/500 firlatmaz; gecerli PDF uretir."""
    client = TestClient(app)
    job_id = _run_to_completion(client, video_in_data)

    resp = client.get(f"/history/{job_id}/report.pdf")
    assert resp.status_code == 200
    assert resp.content.startswith(b"%PDF-")
    assert resp.content.rstrip().endswith(b"%%EOF")


# ------------------------------------------------------------------
# 6) Export: JSON/HTML/PDF (theme + report_export helpers) unknown bilgisini korur
# ------------------------------------------------------------------


def test_safir_report_schema_allows_null_score_and_defaults_to_assessed() -> None:
    report_assessed = SafirReport(
        video_source="v.mp4",
        generated_at="2026-01-01T00:00:00Z",
        natural_language_summary="obs",
        risk_score=0,
        risk_level="dusuk",
        recommended_action="izle",
    )
    assert report_assessed.risk_status == "assessed"  # varsayilan, geriye-uyum
    assert report_assessed.to_sartname_json()["risk_status"] == "assessed"

    report_unknown = SafirReport(
        video_source="v.mp4",
        generated_at="2026-01-01T00:00:00Z",
        natural_language_summary="obs",
        risk_score=None,
        risk_level="unknown",
        risk_status="unknown",
        recommended_action="manuel inceleyin",
    )
    assert report_unknown.risk_score is None
    assert report_unknown.to_sartname_json()["risk_score"] is None
    assert report_unknown.to_sartname_json()["risk_status"] == "unknown"


def test_resolve_risk_badge_none_score_is_unknown_not_normal() -> None:
    """`risk_score=None` -> UNKNOWN_BADGE. ASLA NORMAL/dusuk rozetine dusmez (TypeError da ATMAZ)."""
    assert resolve_risk_badge("unknown", None) == UNKNOWN_BADGE


def _unknown_report_dict() -> dict:
    return {
        "video_source": "v.mp4",
        "generated_at": "2026-01-01T00:00:00Z",
        "natural_language_summary": "obs",
        "summary": "Risk belirsiz",
        "risk_score": None,
        "risk_level": "unknown",
        "risk_status": "unknown",
        "recommended_action": "Sahayi manuel inceleyin",
        "actions": ["Sahayi manuel inceleyin"],
        "detected_event_types": [],
        "timeline": [],
        "evidence_frames": [],
        "relevant_regulations": [],
        "escalation_tier": "notify",
        "auto_dispatched": False,
        "alert_id": None,
        "vlm_model": "mock",
        "llm_model": "mock",
    }


def test_html_export_never_shows_none_slash_100_for_unknown_risk() -> None:
    html = ReportExporter(_unknown_report_dict()).to_html()
    assert "None/100" not in html
    assert "Belirsiz" in html


def test_pdf_export_does_not_crash_on_none_risk_score() -> None:
    """Eski kod `risk_score <= 5` karsilastirmasinda `None` ile TypeError firlatirdi."""
    pdf_bytes = ReportExporter(_unknown_report_dict()).to_pdf()
    assert pdf_bytes.startswith(b"%PDF-")
