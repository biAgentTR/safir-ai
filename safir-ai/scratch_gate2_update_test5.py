import re

with open("tests/test_selector_isolation.py", "r", encoding="utf-8") as f:
    content = f.read()

new_test_5 = """    def test_5_multiple_analysis_ids_in_detected_events_raises_error():
        \"\"\"5. Bir cagrinin DetectedEvent listesinde iki farkli analysis ID -> acik hata\"\"\"
        from unittest.mock import MagicMock
        pipeline = object.__new__(SafirPipeline)
        pipeline._rag_service = MagicMock()
        pipeline._context_builder = MagicMock()
        pipeline._event_engine = MagicMock()
        detected_events = [
            DetectedEvent(event_name="A", description="A", timestamp=5.0, confidence=0.9, source_analysis_id="ID1"),
            DetectedEvent(event_name="B", description="B", timestamp=5.0, confidence=0.9, source_analysis_id="ID2"),
        ]
        pipeline._event_engine.detect.return_value = detected_events
        
        with pytest.raises(ValueError, match="iki farkli analysis ID bulundu"):
            pipeline.stage_context(
                vlm_response=MagicMock(),
                user_prompt="",
                latest_timestamp=10.0,
                rule_matches=[],
                temporal_events=[],
                analysis_mode="vlm_direct",
                context=None
            )"""

content = re.sub(
    r'    def test_5_multiple_analysis_ids_in_detected_events_raises_error\(\):.*?context=None\n        \)',
    new_test_5,
    content,
    flags=re.DOTALL
)

with open("tests/test_selector_isolation.py", "w", encoding="utf-8") as f:
    f.write(content)
