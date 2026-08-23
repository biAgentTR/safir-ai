"""RAG + Prompt Injection Guard dashboard telemetrisi icin birim testleri.

Kapsam:
  - `src/observability/trace_serializer.py::serialize_rag_security` - RAG/guard
    telemetrisinin GUVENLI (ham metin icermeyen), yapilandirilmis JSON'a
    serilestirilmesi.
  - `src/main.py::_accumulate_rag_security_from_trace` - persisted trace_events
    uzerinden `/system/overview` KPI toplamlarina dogru sekilde katkida bulunmasi.
  - `EmbeddingRAGService`in gercek query telemetrisi (candidate/final count,
    latency, zero-result, avg score) uretmesi.

Hicbir yerde UYDURULMUS/rastgele bir deger dogrulanmaz - yalnizca gercekten
hesaplanan alanlar test edilir; opsiyonel telemetri eksikken de serializer'in
crash OLMADIGI ayrica dogrulanir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.main import _RagSecurityAggregate, _accumulate_rag_security_from_trace
from src.rag.embedding_rag_service import RagQueryTelemetry, RagResultTelemetry
from src.observability.trace_serializer import STAGE_ORDER, FORBIDDEN_KEYS, serialize_rag_security
from src.security.prompt_injection_guard import GuardResult


def _sample_rag_telemetry(zero_result: bool = False) -> RagQueryTelemetry:
    result = RagResultTelemetry(
        document_id="6331_isg_kanunu",
        document_title="6331 Sayılı İSG Kanunu",
        article_number="12",
        source_url="https://example.gov.tr/6331",
        embedding_score=0.81,
        rerank_score=0.72,
        selected=not zero_result,
    )
    return RagQueryTelemetry(
        query="fire_detected yangın tespit edildi",
        candidate_count=20,
        final_count=0 if zero_result else 1,
        zero_result=zero_result,
        retrieval_status="reranked",
        threshold=0.10,
        embedding_latency_ms=45.2,
        rerank_latency_ms=310.7,
        total_latency_ms=356.0,
        avg_embedding_score=0.75,
        avg_rerank_score=None if zero_result else 0.72,
        results=[] if zero_result else [result],
    )


def _sample_guard_result(action: str = "allow", guard_failed: bool = False, source: str = "vlm_description") -> GuardResult:
    return GuardResult(
        is_injection=(action == "quarantine"),
        confidence=0.95 if action == "quarantine" else 0.02,
        reason="Instruction override attempt." if action == "quarantine" else None,
        action=action,
        source=source,
        guard_failed=guard_failed,
        latency_ms=210.5,
    )


# --- serialize_rag_security: RAG metrics serialization ---
def test_serialize_rag_security_with_full_telemetry() -> None:
    payload = {"rag_telemetry": _sample_rag_telemetry(), "guard_results": [_sample_guard_result()]}
    summary, data, frames, status, error = serialize_rag_security(payload, "job1")

    assert status == "completed"
    assert error is None
    assert frames == {}
    assert data["rag"]["candidate_count"] == 20
    assert data["rag"]["final_count"] == 1
    assert data["rag"]["zero_result"] is False
    assert data["rag"]["embedding_latency_ms"] == 45.2
    assert data["rag"]["rerank_latency_ms"] == 310.7
    assert data["rag"]["results"][0]["document_id"] == "6331_isg_kanunu"
    assert data["rag"]["results"][0]["rerank_score"] == 0.72
    assert data["security"][0]["source"] == "vlm_description"
    assert data["security"][0]["action"] == "allow"
    assert "RAG" in summary


# --- zero-result retrieval ---
def test_serialize_rag_security_zero_result_is_reported_not_fabricated() -> None:
    payload = {"rag_telemetry": _sample_rag_telemetry(zero_result=True), "guard_results": []}
    _summary, data, _frames, _status, _error = serialize_rag_security(payload, "job1")

    assert data["rag"]["zero_result"] is True
    assert data["rag"]["final_count"] == 0
    assert data["rag"]["results"] == []
    assert data["rag"]["avg_rerank_score"] is None


def test_serialize_rag_security_flags_fallback_placeholder_corpus_loudly() -> None:
    """RAG P0 kok neden testi: gercek mevzuat indeksi yoksa (fallback), bu SESSIZCE gecilmemeli - trace'te acikca UYARI olarak gorunmeli."""
    telemetry = _sample_rag_telemetry()
    telemetry.corpus_source = "fallback_placeholder"
    payload = {"rag_telemetry": telemetry, "guard_results": []}

    summary, data, _frames, _status, _error = serialize_rag_security(payload, "job1")

    assert data["rag"]["corpus_source"] == "fallback_placeholder"
    assert "UYARI" in summary
    assert "PLACEHOLDER" in summary


def test_serialize_rag_security_persisted_index_corpus_is_not_flagged() -> None:
    telemetry = _sample_rag_telemetry()
    telemetry.corpus_source = "persisted_index"
    payload = {"rag_telemetry": telemetry, "guard_results": []}

    summary, data, _frames, _status, _error = serialize_rag_security(payload, "job1")

    assert data["rag"]["corpus_source"] == "persisted_index"
    assert "UYARI" not in summary


# --- guard detection serialization ---
def test_serialize_rag_security_guard_quarantine_is_reported() -> None:
    payload = {"rag_telemetry": None, "guard_results": [_sample_guard_result(action="quarantine")]}
    _summary, data, _frames, _status, _error = serialize_rag_security(payload, "job1")

    assert data["rag"] is None  # RAG hic calismadi - UYDURULMUS bir "rag" nesnesi YOK
    assert data["security"][0]["action"] == "quarantine"
    assert data["security"][0]["is_injection"] is True
    assert data["security"][0]["confidence"] == 0.95


# --- guard API failure serialization ---
def test_serialize_rag_security_guard_failure_is_flagged() -> None:
    payload = {
        "rag_telemetry": None,
        "guard_results": [_sample_guard_result(action="quarantine", guard_failed=True)],
    }
    _summary, data, _frames, _status, _error = serialize_rag_security(payload, "job1")

    assert data["security"][0]["guard_failed"] is True


# --- missing optional telemetry: hicbir sey crash olmamali ---
def test_serialize_rag_security_with_no_telemetry_at_all_does_not_crash() -> None:
    _summary, data, _frames, status, error = serialize_rag_security({"rag_telemetry": None, "guard_results": []}, "job1")
    assert data == {"rag": None, "security": []}
    assert status == "completed"
    assert error is None


def test_serialize_rag_security_with_missing_payload_keys_does_not_crash() -> None:
    _summary, data, _frames, status, _error = serialize_rag_security({}, "job1")
    assert data == {"rag": None, "security": []}
    assert status == "completed"


def test_rag_security_stage_never_leaks_forbidden_keys() -> None:
    payload = {"rag_telemetry": _sample_rag_telemetry(), "guard_results": [_sample_guard_result()]}
    _summary, data, _frames, _status, _error = serialize_rag_security(payload, "job1")
    serialized = json.dumps(data)
    for forbidden in FORBIDDEN_KEYS:
        assert forbidden not in serialized


def test_rag_security_is_registered_in_stage_order() -> None:
    assert "rag_security" in STAGE_ORDER


# --- reason truncation: guard'in gerekcesi kararin kaynagi olarak KULLANILMAZ, sadece bilgi amacli gosterilir ---
def test_serialize_rag_security_truncates_long_reason() -> None:
    long_reason = "x" * 5000
    guard_result = GuardResult(
        is_injection=True, confidence=0.9, reason=long_reason, action="quarantine", source="user_prompt"
    )
    _summary, data, _frames, _status, _error = serialize_rag_security(
        {"rag_telemetry": None, "guard_results": [guard_result]}, "job1"
    )
    assert len(data["security"][0]["reason"]) <= 200


# --- _accumulate_rag_security_from_trace: /system/overview aggregation ---
def _trace_json(rag_data, security_data, risk_level=None, detected_event_count=0) -> str:
    events = [
        {
            "stage": "events",
            "data": {"detected_events": [{"event_name": "e"}] * detected_event_count},
        },
        {"stage": "rag_security", "data": {"rag": rag_data, "security": security_data}},
    ]
    if risk_level is not None:
        events.append({"stage": "report", "data": {"risk_level": risk_level}})
    return json.dumps(events)


def test_accumulate_rag_security_counts_query_and_scores() -> None:
    agg = _RagSecurityAggregate()
    trace = _trace_json(
        rag_data={
            "zero_result": False,
            "embedding_latency_ms": 10.0,
            "rerank_latency_ms": 200.0,
            "total_latency_ms": 210.0,
        },
        security_data=[{"action": "allow", "guard_failed": False, "latency_ms": 100.0, "confidence": 0.1}],
        risk_level="kritik",
        detected_event_count=3,
    )
    _accumulate_rag_security_from_trace(trace, agg)

    assert agg.total_events_detected == 3
    assert agg.critical_risk_analyses == 1
    assert agg.rag_query_count == 1
    assert agg.rag_zero_result_count == 0
    assert agg.embedding_latencies == [10.0]
    assert agg.guard_checks == 1
    assert agg.guard_allowed == 1
    assert agg.guard_quarantined == 0


def test_accumulate_rag_security_counts_fail_closed_block() -> None:
    agg = _RagSecurityAggregate()
    trace = _trace_json(
        rag_data=None,
        security_data=[{"action": "quarantine", "guard_failed": True, "latency_ms": 50.0, "confidence": 1.0}],
    )
    _accumulate_rag_security_from_trace(trace, agg)

    assert agg.rag_query_count == 0  # RAG hic calismadi
    assert agg.guard_quarantined == 1
    assert agg.guard_failures == 1
    assert agg.guard_fail_closed_blocks == 1


def test_accumulate_rag_security_ignores_malformed_trace_json() -> None:
    agg = _RagSecurityAggregate()
    _accumulate_rag_security_from_trace("not valid json", agg)
    assert agg.rag_query_count == 0
    assert agg.guard_checks == 0


def test_accumulate_rag_security_ignores_missing_trace_json() -> None:
    agg = _RagSecurityAggregate()
    _accumulate_rag_security_from_trace(None, agg)
    assert agg.guard_checks == 0


def test_accumulate_rag_security_handles_legacy_trace_without_rag_security_stage() -> None:
    """Eski (bu ozellikten ONCEKI) trace kayitlarinda 'rag_security' stage'i YOKTUR - crash olmamali."""
    agg = _RagSecurityAggregate()
    legacy_trace = json.dumps([{"stage": "events", "data": {"detected_events": [{"event_name": "e"}]}}])
    _accumulate_rag_security_from_trace(legacy_trace, agg)
    assert agg.total_events_detected == 1
    assert agg.rag_query_count == 0
    assert agg.guard_checks == 0
