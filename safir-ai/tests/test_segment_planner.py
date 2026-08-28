import math
import pytest
import uuid
from typing import Optional
from src.vlm.video_chunker import (
    AnalysisContext, 
    SegmentPlan, 
    plan_segments, 
    generate_chunk_id,
    VideoChunk
)

def test_analysis_context_and_ids():
    analysis_id = str(uuid.uuid4())
    video_id = str(uuid.uuid4())
    ctx = AnalysisContext(analysis_id=analysis_id, video_id=video_id)
    
    assert ctx.analysis_id == analysis_id
    assert ctx.video_id == video_id
    assert analysis_id != video_id
    
    chunk_id_1 = generate_chunk_id(analysis_id, 0)
    chunk_id_2 = generate_chunk_id(analysis_id, 1)
    
    assert chunk_id_1 != chunk_id_2
    assert chunk_id_1 == f"{analysis_id}:chunk:000000"
    
    other_analysis_id = str(uuid.uuid4())
    other_chunk_id = generate_chunk_id(other_analysis_id, 0)
    assert chunk_id_1 != other_chunk_id

def test_plan_segments_exact_and_small_boundaries():
    # 0.00001
    plans = plan_segments(0.00001, 60.0, 0.0)
    assert len(plans) == 1
    assert plans[0].start_sec == 0.0
    assert plans[0].end_sec == 0.00001
    
    # 0.1
    plans = plan_segments(0.1, 60.0, 0.0)
    assert len(plans) == 1
    assert plans[0].end_sec == 0.1
    
    # 30
    plans = plan_segments(30.0, 60.0, 0.0)
    assert len(plans) == 1
    assert plans[0].end_sec == 30.0
    
    # 60
    plans = plan_segments(60.0, 60.0, 0.0)
    assert len(plans) == 1
    assert plans[0].end_sec == 60.0
    
    # 60.00001
    plans = plan_segments(60.00001, 60.0, 0.0)
    assert len(plans) == 2
    assert plans[0].start_sec == 0.0
    assert plans[0].end_sec == 60.0
    assert plans[1].start_sec == 60.0
    assert plans[1].end_sec == 60.00001
    
    # 61
    plans = plan_segments(61.0, 60.0, 0.0)
    assert len(plans) == 2
    assert plans[1].end_sec == 61.0
    
    # 120
    plans = plan_segments(120.0, 60.0, 0.0)
    assert len(plans) == 2
    assert plans[1].end_sec == 120.0
    
    # 120.00001
    plans = plan_segments(120.00001, 60.0, 0.0)
    assert len(plans) == 3
    assert plans[2].start_sec == 120.0
    assert plans[2].end_sec == 120.00001
    
    # 125
    plans = plan_segments(125.0, 60.0, 0.0)
    assert len(plans) == 3
    assert plans[2].end_sec == 125.0
    
    # Verify strict constraints
    for p in plans:
        assert p.start_sec >= 0
        assert p.end_sec > p.start_sec
        assert p.end_sec <= 125.0
        assert p.duration_sec > 0

def test_plan_segments_float_precision():
    # 0.3, 0.1, 0
    plans = plan_segments(0.3, 0.1, 0.0)
    assert len(plans) == 3
    assert math.isclose(plans[-1].end_sec, 0.3)
    
    # 0.30000000000000004, 0.1, 0
    plans = plan_segments(0.30000000000000004, 0.1, 0.0)
    assert len(plans) == 3
    assert math.isclose(plans[-1].end_sec, 0.30000000000000004)
    
    # 1.0, 0.1, 0.03
    plans = plan_segments(1.0, 0.1, 0.03)
    # Using formula step = 0.07. 1.0 / 0.07 is roughly 14.something -> 15 chunks
    assert plans[0].start_sec == 0.0
    assert plans[-1].end_sec == 1.0
    
    # Check gapless coverage (with overlap)
    for i in range(1, len(plans)):
        assert plans[i].start_sec < plans[i-1].end_sec

def test_plan_segments_overlap_metadata():
    # duration=61, window=60, overlap=5
    plans = plan_segments(61.0, 60.0, 5.0)
    assert len(plans) == 2
    # chunk 0: 0-60
    assert plans[0].overlap_left_sec == 0.0
    assert plans[0].overlap_right_sec == 5.0
    # chunk 1: 55-61
    assert plans[1].overlap_left_sec == 5.0
    assert plans[1].overlap_right_sec == 0.0
    
    # duration=115, window=60, overlap=5
    plans = plan_segments(115.0, 60.0, 5.0)
    assert len(plans) == 2
    # chunk 1: 55-115 -> length 60, overlap right is 0
    assert plans[1].overlap_right_sec == 0.0
    
    # duration=120, window=60, overlap=5
    plans = plan_segments(120.0, 60.0, 5.0)
    assert len(plans) == 3
    assert plans[1].overlap_right_sec == 5.0
    assert plans[2].overlap_left_sec == 5.0
    
    # duration=0.03, window=0.02, overlap=0.015
    plans = plan_segments(0.03, 0.02, 0.015)
    assert len(plans) == 3
    # 0 -> 0 to 0.02
    # 1 -> 0.005 to 0.025
    # 2 -> 0.010 to 0.03
    assert math.isclose(plans[0].overlap_right_sec, 0.015)
    assert math.isclose(plans[1].overlap_left_sec, 0.015)
    assert math.isclose(plans[1].overlap_right_sec, 0.015)
    assert math.isclose(plans[2].overlap_left_sec, 0.015)

def test_plan_segments_edge_cases():
    with pytest.raises(ValueError):
        plan_segments(0.0, 60.0)
    with pytest.raises(ValueError):
        plan_segments(-1.0, 60.0)
    with pytest.raises(ValueError):
        plan_segments(float('nan'), 60.0)
    with pytest.raises(ValueError):
        plan_segments(float('inf'), 60.0)
    with pytest.raises(ValueError):
        plan_segments(float('-inf'), 60.0)
        
    with pytest.raises(ValueError):
        plan_segments(60.0, 0.0)
    with pytest.raises(ValueError):
        plan_segments(60.0, float('nan'))
    with pytest.raises(ValueError):
        plan_segments(60.0, float('inf'))
        
    with pytest.raises(ValueError):
        plan_segments(60.0, 60.0, -1.0)
    with pytest.raises(ValueError):
        plan_segments(60.0, 60.0, 60.0)
    with pytest.raises(ValueError):
        plan_segments(60.0, 60.0, 61.0)
    with pytest.raises(ValueError):
        plan_segments(60.0, 60.0, float('nan'))
    with pytest.raises(ValueError):
        plan_segments(60.0, 60.0, float('inf'))

def test_video_chunk_model_backward_compatibility():
    chunk = VideoChunk(
        path="dummy.mp4",
        start_offset_sec=0.0,
        end_offset_sec=60.0,
        index=0,
        is_original=False
    )
    assert chunk.analysis_id is None
    assert chunk.video_id is None
    assert chunk.chunk_id is None
    assert chunk.plan is None

def _verify_plan_invariants(plans, duration_sec, window_sec, overlap_sec):
    assert len(plans) > 0, "Plan listesi bos olmamali"
    assert plans[0].start_sec == 0.0, "Ilk segment 0'dan baslamali"
    assert math.isclose(plans[-1].end_sec, duration_sec) or plans[-1].end_sec == duration_sec, "Son segment duration_sec'te bitmeli"
    
    for i, p in enumerate(plans):
        assert p.index == i, "Index sirali olmali"
        assert p.end_sec > p.start_sec, "Segment suresi pozitif olmali (end <= start olamaz)"
        assert p.end_sec <= duration_sec, "Segment duration_sec'i asmamali"
        
        if overlap_sec == 0.0 and i > 0:
            assert math.isclose(p.start_sec, plans[i-1].end_sec), "overlap=0 iken bosluk olmamali"
            
        if i > 0:
            assert p.start_sec <= plans[i-1].end_sec or math.isclose(p.start_sec, plans[i-1].end_sec), "Overlap veya bitisiklik saglanmali (bosluk olmamali)"

def test_plan_segments_invariants_and_extreme_small():
    # Cok kucuk degerler
    for small in [1e-12, 1e-9, 1e-6, 1e-5]:
        plans = plan_segments(small, 60.0, 0.0)
        _verify_plan_invariants(plans, small, 60.0, 0.0)
        assert len(plans) == 1
        assert plans[0].end_sec == small

def test_plan_segments_float_quirks():
    configs = [
        (0.3, 0.1, 0.0),
        (0.30000000000000004, 0.1, 0.0),
        (1.0, 0.1, 0.03),
        (60.0000000001, 60.0, 0.0)
    ]
    for d, w, o in configs:
        plans = plan_segments(d, w, o)
        _verify_plan_invariants(plans, d, w, o)

def test_plan_segments_determinism():
    plans1 = plan_segments(123.456, 10.0, 2.5)
    plans2 = plan_segments(123.456, 10.0, 2.5)
    
    assert len(plans1) == len(plans2)
    for p1, p2 in zip(plans1, plans2):
        assert p1.index == p2.index
        assert p1.start_sec == p2.start_sec
        assert p1.end_sec == p2.end_sec
        assert p1.overlap_left_sec == p2.overlap_left_sec
        assert p1.overlap_right_sec == p2.overlap_right_sec

def test_plan_segments_safety_limit():
    from pytest import raises
    with raises(ValueError, match="Too many segments estimated"):
        # 0.001 window with no overlap over 1000 seconds = 1,000,000 chunks
        plan_segments(1000.0, 0.001, 0.0)
