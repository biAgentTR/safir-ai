import re

with open("tests/test_chunk_provenance.py", "r", encoding="utf-8") as f:
    content = f.read()

# Let's fix the test to actually invoke EvrenVLM parsing
replacement = """
def test_model_provenance_spoofing_protection():
    \"\"\"Model ciktisindaki sahte _provenance objesinin tamamen ezilmesini test eder (Kapi 1).\"\"\"
    from src.vlm.evren_vlm import EvrenVLM
    from src.vlm.video_chunker import VideoChunk, AnalysisContext
    from unittest.mock import patch, MagicMock

    # EvrenVLM'in mock VLM cevabi (Sahte _provenance iceriyor)
    fake_json = '''{
        "description": "test",
        "structured_events": [
            {
                "event_name": "test_event",
                "start_time": 5.0,
                "end_time": 10.0,
                "chunk_id": "SAHTE_CHUNK",
                "_provenance": {
                    "source_analysis_id": "SAHTE_ANALIZ",
                    "source_chunk_id": "SAHTE_CHUNK"
                }
            }
        ]
    }'''

    vlm = EvrenVLM(model_name="test-model")
    
    # Mock _send_single_video to return a VLMResponse that has the parsed fake_json
    # Wait, _parse_structured_events is what creates the dict
    from src.vlm.vlm_client import parse_structured_events
    desc, structured_events = parse_structured_events(fake_json)
    
    mock_response = MagicMock()
    mock_response.latency_ms = 100
    mock_response.description = "test"
    mock_response.structured_events = structured_events
    
    chunk = VideoChunk(index=0, start_offset_sec=0.0, end_offset_sec=60.0, path="dummy.mp4", duration_sec=60.0, encoder="test")
    chunk.context = AnalysisContext(analysis_id="GERCEK_ANALIZ", video_id="GERCEK_VIDEO")
    chunk.chunk_id = "GERCEK_CHUNK"
    
    with patch.object(vlm, '_send_single_video', return_value=mock_response):
        response = vlm._analyze_video_chunks([chunk], "test prompt")
        
    event = response.structured_events[0]
    
    # 1. EvrenVLM sahte alanlari ezdi mi?
    assert event["_provenance"]["source_analysis_id"] == "GERCEK_ANALIZ"
    assert event["_provenance"]["source_chunk_id"] == "GERCEK_CHUNK"
    
    # 2. EventEngine sahte alanlari yutar mi?
    engine_input = EventEngineInput(
        vlm_description="Test",
        source_model="test",
        timestamp=0.0,
        evidence_timestamps={},
        structured_events=response.structured_events
    )
    engine = EventEngine()
    detected = engine._detect_from_structured(engine_input)
    
    assert detected[0].source_analysis_id == "GERCEK_ANALIZ"
    assert detected[0].source_chunk_id == "GERCEK_CHUNK"
    assert detected[0].source_observation_id.startswith("GERCEK_CHUNK:observation:")
"""

# Replace the previously added test
content = re.sub(r"def test_model_provenance_spoofing_protection\(\):.*", replacement, content, flags=re.DOTALL)

with open("tests/test_chunk_provenance.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated Gate 1 test")
