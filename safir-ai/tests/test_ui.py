"""Operator Paneli (OOP refactor) icin testler.

Saf mantik modulleri (theme, api_client, report_export) streamlit gerektirmez;
tam panel render'i (`AppTest`) yalnizca streamlit kuruluysa calisir
(`importorskip`). Backend cagrilari `httpx` monkeypatch'i ile taklit edilir.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.ui.api_client import SafirApiClient
from src.ui.report_export import ReportExporter
from src.ui.theme import resolve_risk_badge


def _sample_report() -> Dict[str, Any]:
    return {
        "video_source": "data/test.mp4",
        "generated_at": "2026-01-01T00:00:00Z",
        "natural_language_summary": "VLM ham gozlem metni.",
        "summary": "Operatore yonelik kisa ozet.",
        "risk_score": 90,
        "risk_level": "kritik",
        "recommended_action": "Saglik ekibini cagir",
        "actions": ["Saglik ekibini cagir", "Alani guvenlik altina al"],
        "escalation_tier": "alarm",
        "auto_dispatched": True,
        "alert_id": "alert-xyz",
        "timeline": [{"timestamp": 15.0, "description": "Forklift devrildi"}],
        "evidence_frames": [],
        "relevant_regulations": ["ISG Yonetmeligi Madde 12"],
        "sampler_stats": None,
        "vlm_model": "gemini-2.5-flash",
        "llm_model": "gemini-2.5-flash",
        "event_id": 1,
    }


def test_resolve_risk_badge_low_score_is_normal() -> None:
    assert resolve_risk_badge("kritik", 3)[0] == "NORMAL"
    assert resolve_risk_badge("kritik", 90)[0] == "CRITICAL"
    assert resolve_risk_badge("orta", 40)[0] == "MEDIUM"


def test_report_exporter_html_includes_summary_actions_escalation() -> None:
    html = ReportExporter(_sample_report()).to_html()
    assert "Operatore yonelik kisa ozet." in html
    assert "Saglik ekibini cagir" in html
    assert "otomatik alarm tetiklendi" in html


def test_report_exporter_file_stub_is_path_safe() -> None:
    stub = ReportExporter(_sample_report()).file_stub
    assert "/" not in stub and "\\" not in stub


def test_api_client_create_job_returns_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {"job_id": "job-123"}

    captured: Dict[str, Any] = {}

    def _fake_post(url: str, json: Dict[str, Any], timeout: float) -> _Resp:  # noqa: A002
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr("src.ui.api_client.httpx.post", _fake_post)
    client = SafirApiClient(base_url="http://backend:8000")
    job_id = client.create_analyze_job("data/x.mp4", "istem", 5, 0.011)

    assert job_id == "job-123"
    assert captured["url"] == "http://backend:8000/analyze/jobs"
    assert captured["json"]["video_source"] == "data/x.mp4"


def test_api_client_health_false_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def _fake_get(url: str, timeout: float):
        raise httpx.ConnectError("down")

    monkeypatch.setattr("src.ui.api_client.httpx.get", _fake_get)
    assert SafirApiClient().check_health() is False


def test_dashboard_apptest_renders_report_without_exception() -> None:
    """Tam panel, bir raporu (otomatik eskalasyon dahil) hatasiz render etmeli."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/ui/dashboard.py", default_timeout=30)
    at.run()
    assert not at.exception

    at.session_state["last_report"] = _sample_report()
    at.run()
    assert not at.exception
