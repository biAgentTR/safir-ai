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


# ---------------------------------------------------------------------------
# 2026-08-24 (SON PRODUCTION RUNTIME AUDIT): Agent onerisi (llm_proposed_score)
# ile resmi risk_score UI'da ASLA KARISTIRILMAMALI/BIRLESTIRILMEMELI - bu
# testler GERCEK render edilmis widget agacini (kaynak string taramasi
# DEGIL) inceler.
# ---------------------------------------------------------------------------


def _sample_report_with_diverging_agent_score() -> Dict[str, Any]:
    """Agent'in taslak skoru (85) ile resmi deterministik risk_score'un (58) BILEREK FARKLI oldugu bir rapor."""
    report = _sample_report()
    report.update(
        {
            "risk_score": 58,
            "risk_level": "yuksek",
            "llm_proposed_score": 85,
            "summary": "Ajan degerlendirmesi: sahada aktif bir tehlike gozlemlendi.",
            "semantic_rag_sources": [
                {
                    "rule_title": "BYKHY",
                    "content": "madde metni",
                    "score": 0.869,
                    "embedding_score": 0.869,
                    "chunk_id": "bykhy__madde_87",
                    "document_id": "bykhy",
                    "article_number": "87",
                    "source_url": "https://example.gov.tr/bykhy",
                    "relevance_score": 0.42,
                    "relevance_status": "accepted",
                    "cross_encoder_score": 0.91,
                    "final_rank": 1,
                    "source_verified": True,
                },
            ],
        }
    )
    return report


def _collect_all_rendered_text(at) -> str:
    """`AppTest` agacindaki tum yaygin metin-tasiyan widget turlerinin (metric/info/markdown/caption/subheader/header) icerigini TEK bir metinde birlestirir."""
    parts: list = []
    for collection_name in ("metric", "info", "warning", "error", "success", "markdown", "caption", "subheader", "header", "text"):
        collection = getattr(at, collection_name, None)
        if not collection:
            continue
        for element in collection:
            label = getattr(element, "label", None)
            if label:
                parts.append(str(label))
            value = getattr(element, "value", None)
            if value is not None:
                parts.append(str(value))
    return "\n".join(parts)


def test_ui_official_risk_metric_uses_canonical_risk_score_not_agent_proposal() -> None:
    """Ust bilgi panelindeki 'Risk Skoru' metric'i, `llm_proposed_score` (85) DEGIL, `report.risk_score` (58) degerini gostermeli."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/ui/dashboard.py", default_timeout=30)
    at.run()
    at.session_state["last_report"] = _sample_report_with_diverging_agent_score()
    at.run()
    assert not at.exception

    risk_metrics = [m for m in at.metric if m.label == "Risk Skoru"]
    assert len(risk_metrics) == 1
    assert risk_metrics[0].value == "58/100"
    assert "85" not in risk_metrics[0].value


def test_ui_agent_proposal_is_labeled_separately_and_never_as_official_risk() -> None:
    """Agent'in 85 taslak skoru, GORULUYORSA acikca 'Model Önerisi'/'llm_proposed_score' olarak etiketlenmeli - `report.risk_score` alaniyla AYNI widget'ta GORUNMEMELI."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/ui/dashboard.py", default_timeout=30)
    at.run()
    at.session_state["last_report"] = _sample_report_with_diverging_agent_score()
    at.run()
    assert not at.exception

    all_text = _collect_all_rendered_text(at)
    assert "llm_proposed_score" in all_text or "Model Önerisi" in all_text
    # "85" GORULEBILIR (Agent onerisi olarak) ama HICBIR "Risk Skoru" metric'i BUNU tasimamali.
    risk_metrics = [m for m in at.metric if m.label == "Risk Skoru"]
    assert all("85" not in m.value for m in risk_metrics)


def test_ui_never_renders_old_rerank_score_terminology() -> None:
    """Render edilmis panelde 'Rerank Skoru' / 'LLM Rerank' terminolojisi HICBIR YERDE gecmemeli - canonical alanlar (Embedding Skoru/Deterministic Relevance/Cross-Encoder Relevance/Seçildi) kullanilmali."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/ui/dashboard.py", default_timeout=30)
    at.run()
    at.session_state["last_report"] = _sample_report_with_diverging_agent_score()
    at.run()
    assert not at.exception

    all_text = _collect_all_rendered_text(at)
    assert "Rerank Skoru" not in all_text
    assert "LLM Rerank" not in all_text
    assert "Cross-Encoder Relevance" in all_text
    assert "Embedding Skoru" in all_text
    assert "Deterministic Relevance" in all_text
