import re

with open("tests/test_pipeline_integration.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix test_multiple_real_vlm_events_with_distinct_start_end_times_all_survive_to_report
old_event = """                    {
                        "event_id": "e2",
                        "event_name": "kovada_alev_baslangici",
                        "canonical_event_type": "yangin_duman",
                        "start_time": 38.0,
                        "end_time": 75.0,
                        "evidence_ids": [ef.evidence_id for ef in evidence_frames[3:]] or [],
                        "description": "Kovada kucuk bir alev basladi.",
                        "keywords": ["kovada alev", "kucuk yangin baslangici", "duman izi"],
                        "risk_score": 65,
                        "confidence": 0.75,
                    },"""

new_event = """                    {
                        "event_id": "e2",
                        "event_name": "kovada_alev_baslangici",
                        "canonical_event_type": "yangin_duman",
                        "start_time": 38.0,
                        "end_time": 75.0,
                        "evidence_ids": [evidence_frames[0].evidence_id] if evidence_frames else [],
                        "description": "Kovada kucuk bir alev basladi.",
                        "keywords": ["kovada alev", "kucuk yangin baslangici", "duman izi"],
                        "risk_score": 65,
                        "confidence": 0.75,
                    },"""

content = content.replace(old_event, new_event)

with open("tests/test_pipeline_integration.py", "w", encoding="utf-8") as f:
    f.write(content)


# Check test_selector_isolation.py
with open("tests/test_selector_isolation.py", "r", encoding="utf-8") as f:
    content2 = f.read()

if "pipeline._last_stage_rag_telemetry = None" not in content2:
    print("WARNING: pipeline._last_stage_rag_telemetry = None NOT FOUND in test_selector_isolation.py")
    
    # Try another replace
    old_test_5 = """        from unittest.mock import MagicMock
        pipeline = object.__new__(SafirPipeline)
        pipeline._event_builder = MagicMock()
        pipeline._event_history = MagicMock()
        detected_events = ["""

    new_test_5 = """        from unittest.mock import MagicMock
        pipeline = object.__new__(SafirPipeline)
        pipeline._event_builder = MagicMock()
        pipeline._event_history = MagicMock()
        pipeline._last_stage_rag_telemetry = None
        pipeline._event_store = MagicMock()
        pipeline._agent = MagicMock()
        detected_events = ["""

    content2 = content2.replace(old_test_5, new_test_5)

    with open("tests/test_selector_isolation.py", "w", encoding="utf-8") as f:
        f.write(content2)

