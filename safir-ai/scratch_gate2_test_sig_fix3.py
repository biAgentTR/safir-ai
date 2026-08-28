import re

with open("tests/test_pipeline_integration.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix regression test explicitly
old_regression = """    result = _select_current_call_events(
        events,
        
        latest_timestamp=60.0, 
        current_model_call_ids={"call1"}, 
        current_chunk_ids={"chunk1"}, 
        current_analysis_id="analysis1", 
        current_evidence_ids=set(),
        is_vlm_direct=True
    ,
        current_model_call_ids=set(),
        current_chunk_ids=set(),
        current_analysis_id=None,
        current_evidence_ids={eid for d in (
        current_model_call_ids={"call1"} or []) for eid in d.evidence_ids} if "
        current_model_call_ids={"call1"}" != "[]" else set(),
        is_vlm_direct=False
    )"""

# Instead of relying on regex mess, let's just rewrite the test completely.

test_regression_code = """
def test_select_current_call_events_vlm_direct_multiple_events_loss_regression():
    \"\"\"Faz 1A (T021): vlm_direct modunda donen ve FARKLI end_time zamanlarina sahip coklu olaylarin '1e-6' latest_timestamp karsilastirmasi nedeniyle SESSIZCE KAYBOLMASINI test eder.\"\"\"
    events = [
        TemporalEvent(
            event_id="e1", event_name="Event A", description="Desc A",
            start_timestamp=5.0, end_timestamp=15.0, duration=10.0,
            confidence=0.9, occurrence_count=1, source_model_call_ids=["call1"]
        ),
        TemporalEvent(
            event_id="e2", event_name="Event B", description="Desc B",
            start_timestamp=20.0, end_timestamp=30.0, duration=10.0,
            confidence=0.8, occurrence_count=1, source_model_call_ids=["call1"]
        ),
        TemporalEvent(
            event_id="e3", event_name="Event C", description="Desc C",
            start_timestamp=40.0, end_timestamp=50.0, duration=10.0,
            confidence=0.7, occurrence_count=1, source_model_call_ids=["call1"]
        ),
    ]
    detected = []
    latest_timestamp = 50.0

    result = _select_current_call_events(
        events,
        latest_timestamp,
        current_model_call_ids={"call1"},
        current_chunk_ids=set(),
        current_analysis_id=None,
        current_evidence_ids=set(),
        is_vlm_direct=True
    )

    # Artik kaybolmamali
    assert len(result) == 3, "vlm_direct modunda erken biten olaylar sessizce kayboluyor!"

def test_select_current_call_events_preserves_input_order():
    \"\"\"Secilen olaylarin siralari gizlice degistirilmemeli, kronolojik (girdi) sirasi korunmalidir.\"\"\"
    events = [
        TemporalEvent(
            event_id="e1", event_name="Event A", description="Desc A",
            start_timestamp=5.0, end_timestamp=10.0, duration=5.0,
            confidence=0.7, occurrence_count=1, evidence_ids=["id1"]
        ),
        TemporalEvent(
            event_id="e2", event_name="Event B", description="Desc B",
            start_timestamp=20.0, end_timestamp=25.0, duration=5.0,
            confidence=0.9, occurrence_count=1, evidence_ids=["id2"]
        ),
    ]
    detected = [
        DetectedEvent(event_name="Event A", description="Desc", timestamp=5.0, confidence=0.7, evidence_ids=["id1"]),
        DetectedEvent(event_name="Event B", description="Desc", timestamp=20.0, confidence=0.9, evidence_ids=["id2"]),
    ]

    result = _select_current_call_events(
        events,
        25.0,
        current_model_call_ids=set(),
        current_chunk_ids=set(),
        current_analysis_id=None,
        current_evidence_ids={"id1", "id2"},
        is_vlm_direct=False
    )

    assert len(result) == 2
    assert result[0].event_name == "Event A", "Girdi sirasi (kronolojik) korunmuyor!"
"""

# We can replace the two tests.
content = re.sub(
    r'def test_select_current_call_events_vlm_direct_multiple_events_loss_regression\(\):.*?def test_api_context_generation\(\):',
    test_regression_code + "\n\ndef test_api_context_generation():",
    content,
    flags=re.DOTALL
)

with open("tests/test_pipeline_integration.py", "w", encoding="utf-8") as f:
    f.write(content)
