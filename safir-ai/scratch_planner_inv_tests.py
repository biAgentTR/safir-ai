import os
import math

with open("tests/test_segment_planner.py", "r", encoding="utf-8") as f:
    content = f.read()

# I will append new tests. 
new_tests = """
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
            assert p.start_sec < plans[i-1].end_sec, "Overlap veya bitisiklik saglanmali (bosluk olmamali)"

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
"""

with open("tests/test_segment_planner.py", "a", encoding="utf-8") as f:
    f.write(new_tests)

print("Appended invariants to test_segment_planner.py")
