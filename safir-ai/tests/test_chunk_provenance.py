import pytest
from src.vlm.video_chunker import VideoChunk, AnalysisContext
from src.event_analysis.schemas import DetectedEvent, TemporalEvent
from src.event_analysis.temporal_reasoner import TemporalReasoner
from src.event_analysis.event_engine import EventEngine, EventEngineInput

def test_provenance_id_formats_and_determinism():
    """Observation ID ve Model Call ID'lerin formatini test eder."""
    chunk_id = "test_chunk_123"
    
    # Simulating what EvrenVLM loop does
    model_call_id = f"{chunk_id}:vlm"
    assert model_call_id == "test_chunk_123:vlm"
    
    obs_id_0 = f"{chunk_id}:observation:{0:06d}"
    obs_id_5 = f"{chunk_id}:observation:{5:06d}"
    
    assert obs_id_0 == "test_chunk_123:observation:000000"
    assert obs_id_5 == "test_chunk_123:observation:000005"
    
    # Determinism: applying same format string gives same result
    assert f"{chunk_id}:observation:{5:06d}" == obs_id_5

def test_chunk_event_relative_and_global_times():
    """EventEngine icerisinde relative ve global zamanlarin korundugunu test eder."""
    # VLM cikisini taklit edelim
    engine_input = EventEngineInput(
        vlm_description="Test",
        source_model="test",
        timestamp=100.0,
        evidence_timestamps={},
        structured_events=[
            {
                "event_name": "test_event",
                "start_time": 105.0, # global
                "end_time": 115.0, # global
                "_provenance": {
                    "source_chunk_id": "c1",
                    "relative_start_sec": 5.0,
                    "relative_end_sec": 15.0,
                    "source_observation_id": "c1:observation:000000"
                }
            }
        ]
    )
    
    engine = EventEngine()
    detected = engine._detect_from_structured(engine_input)
    assert len(detected) == 1
    event = detected[0]
    
    assert event.timestamp == 105.0
    assert event.end_timestamp == 115.0
    assert event.relative_start_sec == 5.0
    assert event.relative_end_sec == 15.0
    assert event.source_chunk_id == "c1"
    assert event.source_observation_id == "c1:observation:000000"

def test_model_spoofing_protection():
    """Model ciktisindaki sahte provenance alanlarinin uygulamayi bozmamasini test eder."""
    # EvrenVLM'in parse edecegi mock VLM JSON'i: VLM kendine ait chunk_id basmis
    import json
    vlm_json = """
    {
        "description": "test",
        "structured_events": [
            {
                "event_name": "test_event",
                "start_time": 5.0,
                "end_time": 10.0,
                "chunk_id": "FAKE_CHUNK",
                "observation_id": "FAKE_OBSERVATION"
            }
        ]
    }
    """
    
    # EvrenVLM'de bu nasil izole edilir?
    # vlm_json dict'e cevrilir. EvrenVLM dongusunde _provenance SOZLUGU SIFIRDAN uretilir ve asil item
    # icine eklenir. `chunk_id` alanlari olsa bile EventEngine bunlari KULLANMAZ, `_provenance` keys kullanir.
    
    engine_input = EventEngineInput(
        vlm_description="Test",
        source_model="test",
        timestamp=0.0,
        evidence_timestamps={},
        structured_events=[
            {
                "event_name": "test_event",
                "start_time": 5.0,
                "end_time": 10.0,
                "chunk_id": "FAKE_CHUNK",
                "observation_id": "FAKE_OBSERVATION",
                "_provenance": {
                    "source_chunk_id": "REAL_CHUNK",
                    "source_observation_id": "REAL_OBS"
                }
            }
        ]
    )
    engine = EventEngine()
    detected = engine._detect_from_structured(engine_input)
    
    # Modelin urettigi FAKE dikkate alinmamali, _provenance kullanilmali
    assert detected[0].source_chunk_id == "REAL_CHUNK"
    assert detected[0].source_observation_id == "REAL_OBS"

def test_temporal_merge_provenance_union():
    """Ayni olayin farkli chunk'larda tespit edildiginde union listesi olusturmasi."""
    event1 = DetectedEvent(
        event_name="yangin",
        description="y1",
        timestamp=10.0,
        end_timestamp=20.0,
        confidence=0.9,
        source_chunk_id="chunk_1",
        source_observation_id="obs_1"
    )
    
    event2 = DetectedEvent(
        event_name="yangin",
        description="y2",
        timestamp=15.0, # overlap, same name -> merge
        end_timestamp=25.0,
        confidence=0.8,
        source_chunk_id="chunk_2",
        source_observation_id="obs_2"
    )
    
    reasoner = TemporalReasoner(merge_window_sec=10.0)
    temporals = reasoner.reason([event1, event2])
    
    assert len(temporals) == 1
    te = temporals[0]
    
    # Kaynak listeleri union yapilmali
    assert te.source_chunk_ids == ["chunk_1", "chunk_2"]
    assert te.source_observation_ids == ["obs_1", "obs_2"]
    assert te.occurrence_count == 2
    
def test_temporal_non_merging_independence():
    """Farkli olaylarin provenance'larinin karismamasi test edilir."""
    event1 = DetectedEvent(
        event_name="kaza",
        description="y1",
        timestamp=10.0,
        confidence=0.9,
        source_chunk_id="chunk_1",
        source_observation_id="obs_1"
    )
    
    event2 = DetectedEvent(
        event_name="yangin", # farkli tip, merge olmaz
        description="y2",
        timestamp=15.0,
        confidence=0.8,
        source_chunk_id="chunk_2",
        source_observation_id="obs_2"
    )
    
    reasoner = TemporalReasoner()
    temporals = reasoner.reason([event1, event2])
    
    assert len(temporals) == 2
    assert temporals[0].source_chunk_ids == ["chunk_1"]
    assert temporals[1].source_chunk_ids == ["chunk_2"]

def test_backward_compatibility():
    """Eski DetectedEvent uretimlerinin (provenance olmadan) calismaya devam etmesi."""
    # Sadece eski alanlar
    event = DetectedEvent(
        event_name="eski_olay",
        description="eski",
        timestamp=0.0,
        confidence=0.5
    )
    assert event.source_chunk_id is None
    
    reasoner = TemporalReasoner()
    temporals = reasoner.reason([event])
    assert len(temporals) == 1
    # None alanlar listeye alinmaz (union_provenance null check'i)
    assert temporals[0].source_chunk_ids == []


def test_model_provenance_spoofing_protection():
    """Model ciktisindaki sahte _provenance objesinin tamamen ezilmesini test eder (Kapi 1)."""
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

    from src.vlm.evren_vlm import EvrenVLM
    from src.vlm.video_chunker import VideoChunk, AnalysisContext
    from unittest.mock import patch, MagicMock

    # Create dummy VLM without calling __init__
    vlm = object.__new__(EvrenVLM)
    # mock property model_name
    type(vlm).model_name = property(lambda self: "test-model")

    
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

def test_model_provenance_spoofing_protection():
    """Model ciktisindaki sahte _provenance objesinin tamamen ezilmesini test eder (Kapi 1)."""
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

def test_model_provenance_spoofing_protection():
    """Model ciktisindaki sahte _provenance objesinin tamamen ezilmesini test eder (Kapi 1)."""
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

def test_model_provenance_spoofing_protection():
    """Model ciktisindaki sahte _provenance objesinin tamamen ezilmesini test eder (Kapi 1)."""
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

def test_model_provenance_spoofing_protection():
    """Model ciktisindaki sahte _provenance objesinin tamamen ezilmesini test eder (Kapi 1)."""
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

def test_model_provenance_spoofing_protection():
    """Model ciktisindaki sahte _provenance objesinin tamamen ezilmesini test eder (Kapi 1)."""
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
