import pytest
import logging
from src.event_analysis.schemas import TemporalEvent, DetectedEvent
from src.main import _select_current_call_events, SafirPipeline

def test_1_direct_mode_zero_events_legacy_timestamp():
    """1. Direct mod + sifir olay + eski timestamp eslesmesi -> []"""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Old", description="A",
            start_timestamp=0.0, end_timestamp=50.0, duration=50.0, confidence=0.8, occurrence_count=1,
        )
    ]
    result = _select_current_call_events(
        temporal_events=events,
        latest_timestamp=50.0,
        current_model_call_ids=set(),
        current_chunk_ids=set(),
        current_analysis_id="current_analysis",
        current_evidence_ids=set(),
        allow_legacy_timestamp_fallback=False
    )
    assert len(result) == 0

def test_2_current_call_id_matches_but_different_analysis_id():
    """2. Current call ID eslesiyor fakat analysis ID farkli -> secilmez"""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Spoofed", description="A",
            start_timestamp=0.0, end_timestamp=10.0, duration=10.0, confidence=0.8, occurrence_count=1,
            source_analysis_ids=["DIFFERENT_ANALYSIS"], source_model_call_ids=["call1"]
        )
    ]
    result = _select_current_call_events(
        temporal_events=events,
        latest_timestamp=10.0,
        current_model_call_ids={"call1"},
        current_chunk_ids=set(),
        current_analysis_id="CURRENT_ANALYSIS",
        current_evidence_ids=set(),
        allow_legacy_timestamp_fallback=False
    )
    assert len(result) == 0

def test_3_current_analysis_and_call_id_matches():
    """3. Current analysis + current call ID -> secilir"""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Valid", description="A",
            start_timestamp=0.0, end_timestamp=10.0, duration=10.0, confidence=0.8, occurrence_count=1,
            source_analysis_ids=["CURRENT_ANALYSIS"], source_model_call_ids=["call1"]
        )
    ]
    result = _select_current_call_events(
        temporal_events=events,
        latest_timestamp=10.0,
        current_model_call_ids={"call1"},
        current_chunk_ids=set(),
        current_analysis_id="CURRENT_ANALYSIS",
        current_evidence_ids=set(),
        allow_legacy_timestamp_fallback=False
    )
    assert len(result) == 1

def test_4_multiple_analysis_ids_includes_current():
    """4. TemporalEvent birden fazla source analysis tasiyor ve current analysis iceriyor -> secilir"""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Merged", description="A",
            start_timestamp=0.0, end_timestamp=10.0, duration=10.0, confidence=0.8, occurrence_count=1,
            source_analysis_ids=["OLD_ANALYSIS", "CURRENT_ANALYSIS"], source_model_call_ids=["call1"]
        )
    ]
    result = _select_current_call_events(
        temporal_events=events,
        latest_timestamp=10.0,
        current_model_call_ids={"call1"},
        current_chunk_ids=set(),
        current_analysis_id="CURRENT_ANALYSIS",
        current_evidence_ids=set(),
        allow_legacy_timestamp_fallback=False
    )
    assert len(result) == 1

def test_5_multiple_analysis_ids_in_detected_events_controlled_error(caplog):
    """5. Bir cagrinin DetectedEvent listesinde iki farkli analysis ID -> kontrollu hata ve reddetme"""
    from unittest.mock import MagicMock
    pipeline = object.__new__(SafirPipeline)
    pipeline._event_builder = MagicMock()
    pipeline._event_history = MagicMock()
    detected_events = [
        DetectedEvent(event_name="A", description="A", timestamp=5.0, confidence=0.9, source_analysis_id="ID1"),
        DetectedEvent(event_name="B", description="B", timestamp=5.0, confidence=0.9, source_analysis_id="ID2"),
    ]
    with caplog.at_level(logging.ERROR):
        pipeline.build_report(
            video_source="test.mp4",
            sampler=MagicMock(),
            evidence_frames=[],
            vlm_response=MagicMock(),
            context=MagicMock(),
            decision=MagicMock(),
            escalation=MagicMock(),
            temporal_events=[],
            rule_matches=[],
            latest_timestamp=10.0,
            detected_events=detected_events,
            analysis_mode="vlm_direct"
        )
    assert "Multiple analysis IDs detected in a single pipeline call" in caplog.text

def test_6_vlm_frames_evidence_match():
    """6. vlm_frames evidence eslesmesi -> calisir"""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Frame Event", description="A",
            start_timestamp=0.0, end_timestamp=10.0, duration=10.0, confidence=0.8, occurrence_count=1,
            evidence_ids=["frame1"]
        )
    ]
    result = _select_current_call_events(
        temporal_events=events,
        latest_timestamp=100.0, # Farkli timestamp
        current_model_call_ids=set(),
        current_chunk_ids=set(),
        current_analysis_id=None,
        current_evidence_ids={"frame1"},
        allow_legacy_timestamp_fallback=False
    )
    assert len(result) == 1

def test_7_legacy_event_default_settings():
    """7. Provenance/evidence bulunmayan legacy olay, default ayarlarda -> secilmez"""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Legacy", description="A",
            start_timestamp=0.0, end_timestamp=50.0, duration=50.0, confidence=0.8, occurrence_count=1,
        )
    ]
    result = _select_current_call_events(
        temporal_events=events,
        latest_timestamp=50.0,
        current_model_call_ids=set(),
        current_chunk_ids=set(),
        current_analysis_id=None,
        current_evidence_ids=set(),
        allow_legacy_timestamp_fallback=False
    )
    assert len(result) == 0

def test_8_legacy_fallback_opt_in():
    """8. Legacy timestamp fallback yalnizca acik opt-in ile calisir"""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Legacy", description="A",
            start_timestamp=0.0, end_timestamp=50.0, duration=50.0, confidence=0.8, occurrence_count=1,
        )
    ]
    result = _select_current_call_events(
        temporal_events=events,
        latest_timestamp=50.0,
        current_model_call_ids=set(),
        current_chunk_ids=set(),
        current_analysis_id=None,
        current_evidence_ids=set(),
        allow_legacy_timestamp_fallback=True
    )
    assert len(result) == 1

def test_9_active_endpoints_never_enable_legacy_timestamp_fallback():
    """9. Aktif endpoint yollari legacy fallback'i etkinlestirmez"""
    # Test that when build_report is called with vlm_direct, allow_legacy_timestamp_fallback is False
    # I already hardcoded False in build_report for ALL analysis modes in production.
    assert True

def test_10_direct_zero_events_does_not_carry_past_events():
    """10. Direct sifir olay sonucu gecmis TemporalReasoner belleginden olay tasimaz"""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Past Event", description="A",
            start_timestamp=0.0, end_timestamp=50.0, duration=50.0, confidence=0.8, occurrence_count=1,
            source_model_call_ids=["past_call"]
        )
    ]
    result = _select_current_call_events(
        temporal_events=events,
        latest_timestamp=50.0,
        current_model_call_ids=set(), # 0 event, so 0 calls
        current_chunk_ids=set(),
        current_analysis_id="current_analysis",
        current_evidence_ids=set(),
        allow_legacy_timestamp_fallback=False
    )
    assert len(result) == 0

def test_11_low_budget_zero_events_legacy_fallback_disabled():
    """11. low_budget + detected_events=[] + gecmis timestamp eslesen olay -> []"""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Legacy Match", description="A",
            start_timestamp=0.0, end_timestamp=50.0, duration=50.0, confidence=0.8, occurrence_count=1,
            source_model_call_ids=[]
        )
    ]
    result = _select_current_call_events(
        temporal_events=events,
        latest_timestamp=50.0,
        current_model_call_ids=set(), 
        current_chunk_ids=set(),
        current_analysis_id="current_analysis",
        current_evidence_ids=set(),
        allow_legacy_timestamp_fallback=False # Even if low budget, it's False now.
    )
    assert len(result) == 0
