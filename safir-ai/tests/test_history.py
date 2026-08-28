"""ADIM 5A: analiz gecmisi (History) kaliciligi + /history API testleri.

Pipeline/sampler/VLM/karar mantigi DEGISMEDEN, her isin nihai `SafirReport`'unun
ayri bir SQLite deposuna (AnalysisStore) kalici yazildigini ve /history uc
noktalarindan (newest-first, limit, 404, restart-sonrasi) dogru sunuldugunu
dogrular. VLM/LLM/RAG mock'tur (mevcut test altyapisiyla ayni).
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
from src.memory.analysis_store import AnalysisStore


# --------------------------- fixtures ---------------------------


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
def mock_pipeline(monkeypatch, tmp_path):
    """Mock VLM/LLM + sahte RAG pipeline + izole (tmp) AnalysisStore."""
    monkeypatch.setattr(main, "EmbeddingRAGService", lambda *a, **k: _FakeRAG())
    # cfg sets both mock flags True -> SafirPipeline takes the
    # MockEmbeddingRAGService branch (see main.py), not EmbeddingRAGService.
    monkeypatch.setattr(main, "MockEmbeddingRAGService", lambda *a, **k: _FakeRAG())
    base = main.load_config()
    cfg = base.model_copy(update={"app": base.app.model_copy(update={"use_mock_vlm": True, "use_mock_llm": True})})
    monkeypatch.setattr(main, "_pipeline", main.SafirPipeline(cfg))
    # History deposunu tmp'e yonlendir (data/ kirletme, testler izole)
    store = AnalysisStore(tmp_path / "analyses.db")
    monkeypatch.setattr(main, "_analysis_store", store)
    return store


@pytest.fixture
def video_in_data():
    Path("data").mkdir(exist_ok=True)
    path = Path("data") / "_history_test.mp4"
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


# --------------------------- store unit (restart) ---------------------------


def test_store_persists_across_reopen(tmp_path):
    """Ayni dosyaya yeni baglanti (=restart) tamamlanmis kaydi hala gorur."""
    db = tmp_path / "a.db"
    s1 = AnalysisStore(db)
    s1.create("job-x", "data/v.mp4")
    s1.mark_completed("job-x", report_json=json.dumps({"risk_score": 70}), risk_level="yuksek", risk_score=70, summary="ozet")
    s1.close()

    s2 = AnalysisStore(db)  # yeniden acilis
    rec = s2.get("job-x")
    assert rec is not None and rec.status == "completed" and rec.risk_score == 70
    assert json.loads(rec.report_json)["risk_score"] == 70
    s2.close()


def test_store_newest_first_and_limit(tmp_path):
    s = AnalysisStore(tmp_path / "a.db")
    for i in range(5):
        s.create(f"job-{i}", f"data/{i}.mp4")
        time.sleep(0.005)  # created_at siralamasi ayirt edilebilsin
    ids = [r.job_id for r in s.list(limit=3)]
    assert ids == ["job-4", "job-3", "job-2"]  # newest-first + limit
    assert len(s.list(limit=2, offset=2)) == 2
    s.close()


# --------------------------- API: completed ---------------------------


def test_completed_job_appears_in_history_with_exact_report(mock_pipeline, video_in_data):
    client = TestClient(app)
    job_id = _run_to_completion(client, video_in_data)

    # canonical sonuc (mevcut polling endpoint) ile History detay birebir esit olmali
    polled = client.get(f"/analyze/jobs/{job_id}").json()
    assert polled["status"] == "done"

    lst = client.get("/history").json()
    row = next((x for x in lst if x["job_id"] == job_id), None)
    assert row is not None and row["status"] == "completed"
    assert row["video_source"] == video_in_data
    assert row["risk_score"] == polled["result"]["risk_score"]
    assert row["risk_level"] == polled["result"]["risk_level"]

    detail = client.get(f"/history/{job_id}").json()
    assert detail["status"] == "completed"
    # kalici SafirReport, canonical rapor ile ONEMLI alanlarda birebir esit
    for key in ("risk_score", "risk_level", "video_source", "actions", "timeline", "escalation_tier", "vlm_model"):
        assert detail["report"][key] == polled["result"][key], f"alan uyusmuyor: {key}"


def test_history_unknown_job_404(mock_pipeline):
    client = TestClient(app)
    assert client.get("/history/bilinmeyen-id").status_code == 404


def test_history_restart_persistence(mock_pipeline, video_in_data, tmp_path):
    """Tamamlanan analiz, deponun yeniden acilmasindan (=restart) sonra da History'de."""
    client = TestClient(app)
    job_id = _run_to_completion(client, video_in_data)
    db_path = tmp_path / "analyses.db"

    # "restart": ayni dosyaya yeni bir depo baglayip main'e tak
    reopened = AnalysisStore(db_path)
    main._analysis_store = reopened
    detail = client.get(f"/history/{job_id}").json()
    assert detail["status"] == "completed" and detail["report"]["risk_score"] is not None


# --------------------------- API: failed ---------------------------


def test_failed_job_recorded_as_failed(mock_pipeline):
    """Var olmayan video kaynagi -> pipeline hata -> History'de status=failed, report=None."""
    client = TestClient(app)
    resp = client.post("/analyze/jobs", json={"video_source": "yok_boyle_bir_video.mp4", "user_prompt": "x"})
    job_id = resp.json()["job_id"]
    deadline = time.time() + 20
    while time.time() < deadline:
        if client.get(f"/analyze/jobs/{job_id}").json()["status"] in ("done", "error"):
            break
        time.sleep(0.1)

    detail = client.get(f"/history/{job_id}").json()
    assert detail["status"] == "failed"
    assert detail["report"] is None


def test_history_ordering_and_limit_via_api(mock_pipeline, video_in_data):
    client = TestClient(app)
    j1 = _run_to_completion(client, video_in_data)
    j2 = _run_to_completion(client, video_in_data)
    lst = client.get("/history?limit=1").json()
    assert len(lst) == 1 and lst[0]["job_id"] == j2  # en yeni once
    two = client.get("/history?limit=2").json()
    assert [x["job_id"] for x in two] == [j2, j1]


# --------------------------- API: PDF export ---------------------------


def test_report_pdf_for_completed_job_is_a_real_pdf(mock_pipeline, video_in_data):
    """`ReportExporter.to_pdf()` gercekten cagriliyor mu — gercek PDF magic-byte + boyut kontrolu."""
    client = TestClient(app)
    job_id = _run_to_completion(client, video_in_data)

    resp = client.get(f"/history/{job_id}/report.pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.content.startswith(b"%PDF-")  # gercek PDF dosya imzasi
    assert resp.content.rstrip().endswith(b"%%EOF")  # gecerli bir PDF govdesi tamamlanmis
    assert len(resp.content) > 500  # bos/bozuk dosya degil


def test_report_pdf_with_turkish_characters_in_video_name(mock_pipeline):
    """Video adinda Turkce karakter (ör. 'ı') varsa PDF indirme 500'e DUSMEMELI.

    Kok neden: Content-Disposition baslik degeri Latin-1 ile sinirlidir
    (Starlette/ASGI dogrudan encode eder); Turkce 'ı'/'ş'/'ğ' Latin-1'e
    SIGMAZ ve `UnicodeEncodeError` firlatirdi (indirme TAMAMEN basarisiz
    olurdu) - bkz. get_history_report_pdf'teki RFC 6266 filename*/filename
    duzeltmesi.
    """
    Path("data").mkdir(exist_ok=True)
    path = Path("data") / "kamera_şantiye_ışıklı.mp4"
    frames = [np.full((120, 160, 3), 30, np.uint8) for _ in range(40)]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
    for f in frames:
        writer.write(f)
    writer.release()
    try:
        client = TestClient(app)
        job_id = _run_to_completion(client, str(path))

        resp = client.get(f"/history/{job_id}/report.pdf")
        assert resp.status_code == 200
        assert resp.content.startswith(b"%PDF-")
        disposition = resp.headers["content-disposition"]
        assert "attachment" in disposition
        assert "filename*=UTF-8''" in disposition
    finally:
        path.unlink(missing_ok=True)


def test_report_pdf_falls_back_to_persisted_history_when_job_leaves_memory(mock_pipeline, video_in_data):
    """Is `_jobs` bellek-ici sozlugunden CIKARILSA bile (ör. sunucu yeniden baslasa),
    kalici History kaydindan AYNI gecerli PDF uretilebilmeli."""
    client = TestClient(app)
    job_id = _run_to_completion(client, video_in_data)

    with main._jobs_lock:
        del main._jobs[job_id]

    resp = client.get(f"/history/{job_id}/report.pdf")
    assert resp.status_code == 200
    assert resp.content.startswith(b"%PDF-")


def test_report_pdf_unknown_job_404(mock_pipeline):
    client = TestClient(app)
    resp = client.get("/history/bilinmeyen-id/report.pdf")
    assert resp.status_code == 404


def test_report_pdf_invalid_job_id_400(mock_pipeline):
    client = TestClient(app)
    resp = client.get("/history/../../etc/report.pdf")
    assert resp.status_code in (400, 404)  # path-traversal denemesi hicbir sekilde 200 olmamali
