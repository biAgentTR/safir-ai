from src.vlm.video_chunker import plan_segments

tests_overlap_0 = [0, -1, 0.00001, 30, 60, 60.00001, 61, 120, 125]
print("--- Overlap 0 ---")
for d in tests_overlap_0:
    try:
        res = plan_segments(d, 60, 0)
        print(f"duration={d} -> {[ (p.start_sec, p.end_sec) for p in res ]}")
    except Exception as e:
        print(f"duration={d} -> {type(e).__name__}: {e}")

tests_overlap_5 = [30, 60, 61, 115, 120, 125]
print("\n--- Overlap 5 ---")
for d in tests_overlap_5:
    try:
        res = plan_segments(d, 60, 5)
        print(f"duration={d} -> {[ (p.start_sec, p.end_sec, p.overlap_left_sec, p.overlap_right_sec) for p in res ]}")
    except Exception as e:
        print(f"duration={d} -> {type(e).__name__}: {e}")

tests_decimal = [
    (1.0, 0.1, 0),
    (1.0, 0.1, 0.03),
    (0.3, 0.1, 0),
    (0.30000000000000004, 0.1, 0)
]
print("\n--- Ondalikli ---")
for d, w, o in tests_decimal:
    try:
        res = plan_segments(d, w, o)
        print(f"duration={d}, window={w}, overlap={o} -> {[ (p.start_sec, p.end_sec) for p in res ]}")
    except Exception as e:
        print(f"duration={d}, window={w}, overlap={o} -> {type(e).__name__}: {e}")

import math
print("\n--- Equivalency (math.ceil vs plan_segments) ---")
for d in [0.1, 1, 30, 59.999, 60, 60.001, 61, 119.999, 120, 120.001, 125, 180]:
    num_chunks = math.ceil(d / 60.0)
    old_starts = [i * 60.0 for i in range(num_chunks)]
    old_ends = [min(60.0, d - s) + s for s in old_starts]
    old_tuples = list(zip(old_starts, old_ends))
    
    new_plans = plan_segments(d, 60, 0)
    new_tuples = [(p.start_sec, p.end_sec) for p in new_plans]
    print(f"d={d:7.3f} | Old: {len(old_tuples)} chunks | New: {len(new_tuples)} chunks | EQUAL: {old_tuples == new_tuples}")
    if old_tuples != new_tuples:
        print(f"  OLD: {old_tuples}")
        print(f"  NEW: {new_tuples}")

