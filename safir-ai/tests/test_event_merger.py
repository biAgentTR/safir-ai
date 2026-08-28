import pytest
import time
from typing import List

from src.utils.config_loader import EventMergerConfig
from src.event_analysis.schemas import TemporalEvent
from src.event_analysis.event_merger import EventMerger

def create_event(
    id: str,
    start: float,
    end: float,
    name: str,
    analysis_id: str,
    video_id: str,
    chunk_id: str,
    obs_id: str,
    confidence: float = 0.8,
    occurrence_count: int = 1,
    event_type: str = "test",
    desc: str = "desc"
) -> TemporalEvent:
    return TemporalEvent(
        event_id=id,
        event_name=name,
        event_type=event_type,
        description=desc,
        start_timestamp=start,
        end_timestamp=end,
        duration=end-start,
        confidence=confidence,
        occurrence_count=occurrence_count,
        matched_keywords=[],
        source_model="test_model",
        related_events=[],
        evidence_ids=[f"ev_{id}"],
        source_analysis_ids=[analysis_id],
        source_video_ids=[video_id],
        source_chunk_ids=[chunk_id],
        source_model_call_ids=[f"mc_{chunk_id}"],
        source_observation_ids=[obs_id],
        risk_hint=50
    )

def test_merger_same_event_two_chunks():
    e1 = create_event("1", 55, 60, "Ates", "a1", "v1", "c1", "o1")
    e2 = create_event("2", 55, 65, "Ates", "a1", "v1", "c2", "o2")
    merger = EventMerger(EventMergerConfig())
    res = merger.merge([e1, e2])
    assert len(res) == 1
    assert res[0].start_timestamp == 55
    assert res[0].end_timestamp == 65
    assert set(res[0].source_chunk_ids) == {"c1", "c2"}
    assert set(res[0].source_observation_ids) == {"o1", "o2"}
    assert set(res[0].evidence_ids) == {"ev_1", "ev_2"}
    assert res[0].occurrence_count == 1  # Occurrence doesn't sum
    assert res[0].confidence == 0.8

def test_merger_different_analysis():
    e1 = create_event("1", 55, 60, "Ates", "a1", "v1", "c1", "o1")
    e2 = create_event("2", 55, 65, "Ates", "a2", "v1", "c2", "o2")
    merger = EventMerger(EventMergerConfig())
    res = merger.merge([e1, e2])
    assert len(res) == 2

def test_merger_different_video():
    e1 = create_event("1", 55, 60, "Ates", "a1", "v1", "c1", "o1")
    e2 = create_event("2", 55, 65, "Ates", "a1", "v2", "c2", "o2")
    merger = EventMerger(EventMergerConfig())
    res = merger.merge([e1, e2])
    assert len(res) == 2

def test_merger_same_chunk():
    e1 = create_event("1", 55, 60, "Ates", "a1", "v1", "c1", "o1")
    e2 = create_event("2", 55, 65, "Ates", "a1", "v1", "c1", "o2")
    merger = EventMerger(EventMergerConfig())
    res = merger.merge([e1, e2])
    assert len(res) == 2

def test_merger_legacy_event():
    e1 = create_event("1", 55, 60, "Ates", "", "", "", "")
    e2 = create_event("2", 55, 65, "Ates", "", "", "", "")
    merger = EventMerger(EventMergerConfig())
    res = merger.merge([e1, e2])
    assert len(res) == 2

def test_merger_three_chunks_transitive():
    e1 = create_event("1", 55, 60, "Ates", "a1", "v1", "c1", "o1")
    e2 = create_event("2", 55, 65, "Ates", "a1", "v1", "c2", "o2")
    e3 = create_event("3", 60, 70, "Ates", "a1", "v1", "c3", "o3")
    merger = EventMerger(EventMergerConfig())
    res = merger.merge([e1, e2, e3])
    assert len(res) == 1
    assert set(res[0].source_chunk_ids) == {"c1", "c2", "c3"}

def test_merger_deterministic_id():
    e1 = create_event("1", 55, 60, "Ates", "a1", "v1", "c1", "o1")
    e2 = create_event("2", 55, 65, "Ates", "a1", "v1", "c2", "o2")
    merger = EventMerger(EventMergerConfig())
    res1 = merger.merge([e1, e2])[0].event_id
    res2 = merger.merge([e2, e1])[0].event_id
    assert res1 == res2

def test_merger_chronological_output():
    e1 = create_event("1", 55, 60, "Ates", "a1", "v1", "c1", "o1")
    e2 = create_event("2", 55, 65, "Ates", "a1", "v1", "c2", "o2")
    e3 = create_event("3", 10, 20, "Dusme", "a1", "v1", "c3", "o3")
    merger = EventMerger(EventMergerConfig())
    res = merger.merge([e1, e2, e3])
    assert len(res) == 2
    assert res[0].start_timestamp == 10
    assert res[1].start_timestamp == 55

def test_merger_performance():
    events = []
    for i in range(500):
        events.append(create_event(str(i), i, i+5, "Ates", "a1", "v1", f"c{i}", f"o{i}"))
    
    merger = EventMerger(EventMergerConfig(max_boundary_gap_sec=2.0))
    t0 = time.time()
    res = merger.merge(events)
    t1 = time.time()
    assert (t1 - t0) < 1.0
    # Expected ~ 1 or multiple depending on transitive boundary
    assert len(res) > 0

def test_merger_disabled():
    e1 = create_event("1", 55, 60, "Ates", "a1", "v1", "c1", "o1")
    e2 = create_event("2", 55, 65, "Ates", "a1", "v1", "c2", "o2")
    merger = EventMerger(EventMergerConfig(enabled=False))
    res = merger.merge([e1, e2])
    assert len(res) == 2

def test_merger_contradictions():
    e1 = create_event("1", 55, 60, "Ates", "a1", "v1", "c1", "o1", desc="duman yok")
    e2 = create_event("2", 55, 65, "Ates", "a1", "v1", "c2", "o2", desc="duman var")
    merger = EventMerger(EventMergerConfig())
    res = merger.merge([e1, e2])
    # uncertainties feature check: wait, we haven't implemented it in merger code! Let's update merger code to populate uncertainties with different descriptions.

    assert len(res) == 1
    assert len(res[0].uncertainties) > 0
    assert 'Contradicting' in res[0].uncertainties[0]

