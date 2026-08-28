import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock
from src.vlm.schemas import (
    VLMSceneObservation,
    VLMObservationReport,
    ChunkAnalysisResult,
    VLMAnalysisStatus,
    TaxonomyStatus
)
from src.vlm.parser import parse_vlm_response
from src.main import SafirPipeline
from src.vlm.base_vlm import VLMResponse

def test_typed_model_validations():
    with pytest.raises(ValidationError):
        VLMSceneObservation(observed_label="Olay", relative_start_sec=10.0, relative_end_sec=5.0)
    with pytest.raises(ValidationError):
        VLMSceneObservation(observed_label="")
    with pytest.raises(ValidationError):
        VLMSceneObservation(observed_label="Olay", confidence=1.5)
    obs = VLMSceneObservation(
        observed_label="Valid", relative_start_sec=1.0, relative_end_sec=5.0, confidence=0.9
    )
    assert obs.taxonomy_status == TaxonomyStatus.UNCERTAIN

def test_model_cannot_spoof_app_fields():
    fake_json = '''EVENTS_JSON: [{"event_name": "Test", "analysis_status": "success", "repair_used": true}]'''
    desc, result = parse_vlm_response(fake_json)
    assert result.analysis_status == VLMAnalysisStatus.SUCCESS
    assert result.repair_used is False
    assert result.report.observations[0].observed_label == "Test"

def test_parser_success():
    """Legacy adapter gecerli EVENTS_JSON"""
    content = '''Harika bir video. EVENTS_JSON: [{"event_name": "Kask", "start_time": 1, "end_time": 2}]'''
    desc, result = parse_vlm_response(content)
    assert result.analysis_status == VLMAnalysisStatus.SUCCESS
    assert desc == "Harika bir video."
    assert len(result.report.observations) == 1

def test_parser_success_empty():
    content = '''Olay yok. EVENTS_JSON: []'''
    desc, result = parse_vlm_response(content)
    assert result.analysis_status == VLMAnalysisStatus.SUCCESS_EMPTY
    assert len(result.report.observations) == 0

def test_parser_invalid_json():
    """Legacy adapter bozuk EVENTS_JSON"""
    content = '''Bozuk json EVENTS_JSON: [{"event_name": "Kask" ]'''
    desc, result = parse_vlm_response(content)
    assert result.analysis_status == VLMAnalysisStatus.PARSE_FAILED
    assert result.repair_attempted is True
    assert result.repair_succeeded is False
    
def test_parser_partial_observation():
    """Bazi gecerli/bazi gecersiz observation -> partial"""
    content = '''EVENTS_JSON: [{"event_name": "A", "start_time": 1, "end_time": 2}, {"event_name": "B", "start_time": 10, "end_time": 5}]'''
    desc, result = parse_vlm_response(content)
    assert result.analysis_status == VLMAnalysisStatus.PARTIAL
    assert len(result.report.observations) == 1
    assert result.report.observations[0].observed_label == "A"

def test_parser_repair_success():
    content = '''EVENTS_JSON: [{'event_name': 'Kask'}]'''
    desc, result = parse_vlm_response(content)
    assert result.analysis_status == VLMAnalysisStatus.SUCCESS
    assert result.repair_used is True
    assert result.repair_succeeded is True
    
def test_parser_repair_failure():
    """Repair basarisizligi"""
    content = '''EVENTS_JSON: [{'event_name': 'Kask', "broken": True,,}]'''
    desc, result = parse_vlm_response(content)
    assert result.analysis_status == VLMAnalysisStatus.PARSE_FAILED
    assert result.repair_attempted is True
    assert result.repair_succeeded is False
    assert result.repair_failure_reason is not None

def test_parser_empty_content():
    """Bos model content'i"""
    desc, result = parse_vlm_response("")
    assert result.analysis_status == VLMAnalysisStatus.MODEL_FAILED
    assert result.parse_status == "empty_content"

def test_totally_invalid_observation_list():
    """Tamamen gecersiz observation listesi"""
    content = '''EVENTS_JSON: [{"event_name": "B", "start_time": 10, "end_time": 5}, {"event_name": "C", "start_time": 15, "end_time": 10}]'''
    desc, result = parse_vlm_response(content)
    # Both are invalid due to start > end
    assert result.analysis_status == VLMAnalysisStatus.PARSE_FAILED
    assert len(result.report.observations) == 0

def test_regex_fallback_not_found():
    """Regex fallback sonuc bulamazsa parse_failed"""
    content = '''Olay var ama json formatinda degil.'''
    desc, result = parse_vlm_response(content)
    assert result.analysis_status == VLMAnalysisStatus.PARSE_FAILED
    assert result.parse_status == "unrecognized_format"

def test_mixed_analysis_provenance_status(caplog):
    """Mixed analysis provenance status"""
    pipeline = object.__new__(SafirPipeline)
    pipeline._event_builder = MagicMock()
    pipeline._event_history = MagicMock()
    pipeline._last_stage_rag_telemetry = None
    pipeline._event_store = MagicMock()
    pipeline._agent = MagicMock()
    
    from src.event_analysis.schemas import DetectedEvent
    detected_events = [
        DetectedEvent(event_name="A", description="A", timestamp=5.0, confidence=0.9, source_analysis_id="ID1"),
        DetectedEvent(event_name="B", description="B", timestamp=5.0, confidence=0.9, source_analysis_id="ID2"),
    ]
    
    # We mock vlm_response with chunk_analysis_result
    vlm_response = VLMResponse(
        description="",
        model_name="test",
        frame_count=0,
        latency_ms=0.0,
        structured_events=[],
        chunk_analysis_result=ChunkAnalysisResult(
            analysis_status=VLMAnalysisStatus.SUCCESS,
            parse_status="success_typed",
            analysis_id="ID1"
        )
    )
    
    try:
        pipeline.build_report(
            video_source="test.mp4",
            sampler=MagicMock(),
            evidence_frames=[],
            vlm_response=vlm_response,
            context=MagicMock(),
            decision=MagicMock(summary="", actions=[], risk_score=0.0, risk_status="assessed"),
            escalation=MagicMock(),
            temporal_events=[],
            rule_matches=[],
            latest_timestamp=10.0,
            detected_events=detected_events,
            analysis_mode="vlm_direct"
        )
    except Exception:
        pass # We only care about the mutation of vlm_response
    
    assert vlm_response.chunk_analysis_result.analysis_status == VLMAnalysisStatus.PARTIAL
    assert vlm_response.chunk_analysis_result.parse_status == "provenance_integrity_failed"

def test_model_failure_active_pipeline():
    """HTTP/model cagrisi hatasi -> model_failed aktif cagri zincirinde uretilmeli"""
    pipeline = object.__new__(SafirPipeline)
    vlm_mock = MagicMock()
    vlm_mock.analyze_video.side_effect = Exception("HTTP Timeout")
    context_mock = MagicMock()
    context_mock.analysis_id = "test_analysis"
    
    response = pipeline._stage_vlm_video(
        vlm=vlm_mock,
        video_source="test.mp4",
        evidence_frames=[],
        user_prompt="test",
        context=context_mock
    )
    assert response.status == "failed"
    assert response.chunk_analysis_result.analysis_status == VLMAnalysisStatus.MODEL_FAILED
    assert response.chunk_analysis_result.parse_status == "pipeline_exception"
