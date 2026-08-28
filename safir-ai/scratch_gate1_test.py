import re

with open("tests/test_chunk_provenance.py", "r", encoding="utf-8") as f:
    content = f.read()

new_test = """
def test_model_provenance_spoofing_protection():
    \"\"\"Model ciktisindaki sahte _provenance objesinin tamamen ezilmesini test eder (Kapi 1).\"\"\"
    # Modelden gelen ham karsilik, modeli "akilli" davranip bizi hacklemeye calisirsa:
    fake_model_event = {
        "event_name": "test_event",
        "start_time": 5.0,
        "end_time": 10.0,
        "_provenance": {
            "source_analysis_id": "SAHTE_ANALIZ",
            "source_chunk_id": "SAHTE_CHUNK",
            "source_model_call_id": "SAHTE_CALL",
            "source_observation_id": "SAHTE_OBSERVATION"
        }
    }
    
    # EvrenVLM dongusundeki islem (shifted["_provenance"] = {...}) sahteyi EZMELIDIR.
    # setdefault veya update KULLANILMAMALIDIR.
    shifted = dict(fake_model_event)
    shifted["_provenance"] = {
        "source_analysis_id": "GERCEK_ANALIZ",
        "source_chunk_id": "GERCEK_CHUNK",
        "source_model_call_id": "GERCEK_CALL",
        "source_observation_id": "GERCEK_OBSERVATION"
    }
    
    assert shifted["_provenance"]["source_analysis_id"] == "GERCEK_ANALIZ"
    assert shifted["_provenance"]["source_chunk_id"] == "GERCEK_CHUNK"
    
    # EventEngine bunu dogru ayiklar mi?
    engine_input = EventEngineInput(
        vlm_description="Test",
        source_model="test",
        timestamp=0.0,
        evidence_timestamps={},
        structured_events=[shifted]
    )
    engine = EventEngine()
    detected = engine._detect_from_structured(engine_input)
    
    assert detected[0].source_analysis_id == "GERCEK_ANALIZ"
    assert detected[0].source_chunk_id == "GERCEK_CHUNK"
"""

content = content + new_test

with open("tests/test_chunk_provenance.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Added Gate 1 spoof test")
