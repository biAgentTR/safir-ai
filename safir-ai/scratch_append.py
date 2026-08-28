with open('tests/test_pipeline_integration.py', 'a', encoding='utf-8') as f:
    f.write('''

# ------------------------------------------------------------------
# FAZ 1A: _select_current_call_events Veri Kaybi Regresyon Testleri
# ------------------------------------------------------------------

from src.main import _select_current_call_events
from src.event_analysis.schemas import TemporalEvent, DetectedEvent

def test_select_current_call_events_vlm_direct_multiple_events_loss_regression():
    """T021/Faz1A: vlm_direct modunda, ayni cagridan uretilen ancak FARKLI bitis 
    zamanlarina sahip coklu olaylarin '1e-6' latest_timestamp karsilastirmasi
    nedeniyle SESSIZCE KAYBOLMASINI (yalnizca en son biten olayin secilmesini) uretir."""
    
    events = [
        TemporalEvent(
            event_id="e1", event_name="Event A", description="Desc A",
            start_timestamp=5.0, end_timestamp=15.0, duration=10.0,
            confidence=0.9, occurrence_count=1
        ),
        TemporalEvent(
            event_id="e2", event_name="Event B", description="Desc B",
            start_timestamp=20.0, end_timestamp=30.0, duration=10.0,
            confidence=0.8, occurrence_count=1
        ),
        TemporalEvent(
            event_id="e3", event_name="Event C", description="Desc C",
            start_timestamp=40.0, end_timestamp=50.0, duration=10.0,
            confidence=0.7, occurrence_count=1
        ),
    ]
    # vlm_direct always sets empty evidence frames for selection phase
    detected = [] 
    latest_timestamp = 50.0
    
    result = _select_current_call_events(events, latest_timestamp, detected)
    
    # BEKLENTI: Butun olaylar cagriya aittir, 3'u de donmelidir.
    # MEVCUT DURUM (FAIL EDECEK): Yalnizca 'Event C' (bitis: 50.0) doner.
    assert len(result) == 3, "vlm_direct modunda erken biten olaylar sessizce kayboluyor!"
    names = [e.event_name for e in result]
    assert "Event A" in names
    assert "Event B" in names
    assert "Event C" in names


def test_select_current_call_events_single_event():
    """Tek bir olay geldiginde (zaman eslesmesi saglanacagindan) dogru secilmelidir."""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Event A", description="Desc A",
            start_timestamp=5.0, end_timestamp=15.0, duration=10.0,
            confidence=0.9, occurrence_count=1
        )
    ]
    result = _select_current_call_events(events, 15.0, [])
    assert len(result) == 1
    assert result[0].event_name == "Event A"


def test_select_current_call_events_empty_input():
    """Girdi bos ise cikti da bos olmalidir."""
    result = _select_current_call_events([], 10.0, [])
    assert result == []


def test_select_current_call_events_same_end_time():
    """Bitis zamanlari ayni olan farkli olaylarin hepsi secilmelidir."""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Event A", description="Desc A",
            start_timestamp=5.0, end_timestamp=50.0, duration=45.0,
            confidence=0.9, occurrence_count=1
        ),
        TemporalEvent(
            event_id="e2", event_name="Event B", description="Desc B",
            start_timestamp=20.0, end_timestamp=50.0, duration=30.0,
            confidence=0.8, occurrence_count=1
        )
    ]
    result = _select_current_call_events(events, 50.0, [])
    assert len(result) == 2


def test_select_current_call_events_vlm_frames_uses_evidence_ids():
    """vlm_frames (low_budget) modunda, olaylarin gercek kanit kimlikleri (evidence_ids) 
    eslesiyorsa, latest_timestamp uymasa bile SECTIGINI dogrular."""
    events = [
        TemporalEvent(
            event_id="e1", event_name="Event Old", description="Desc",
            start_timestamp=0.0, end_timestamp=10.0, duration=10.0,
            confidence=0.9, occurrence_count=1, evidence_ids=["old_id"]
        ),
        TemporalEvent(
            event_id="e2", event_name="Event New", description="Desc",
            start_timestamp=15.0, end_timestamp=25.0, duration=10.0,
            confidence=0.9, occurrence_count=1, evidence_ids=["new_id_1"]
        ),
    ]
    detected = [
        DetectedEvent(
            event_name="Detected New", timestamp=15.0, 
            confidence=0.9, evidence_ids=["new_id_1"]
        )
    ]
    
    # latest_timestamp tamamen farkli bir deger olsa bile
    result = _select_current_call_events(events, 100.0, detected)
    
    assert len(result) == 1
    assert result[0].event_name == "Event New"


def test_select_current_call_events_preserves_input_order():
    """Secilen olaylarin siralari gizlice degistirilmemeli, kronolojik (girdi) 
    sirasi korunmalidir."""
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
        DetectedEvent(event_name="Event A", timestamp=5.0, confidence=0.7, evidence_ids=["id1"]),
        DetectedEvent(event_name="Event B", timestamp=20.0, confidence=0.9, evidence_ids=["id2"]),
    ]
    
    result = _select_current_call_events(events, 25.0, detected)
    
    assert len(result) == 2
    # Mevcut kod confidence=0.9 olan Event B'yi basa atarak SIRAYI BOZAR.
    assert result[0].event_name == "Event A", "Girdi sirasi (kronolojik) korunmuyor!"
    assert result[1].event_name == "Event B"

''')
