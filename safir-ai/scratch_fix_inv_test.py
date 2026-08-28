import re

with open("tests/test_segment_planner.py", "r", encoding="utf-8") as f:
    content = f.read()

old_assert = '''        if i > 0:
            assert p.start_sec < plans[i-1].end_sec, "Overlap veya bitisiklik saglanmali (bosluk olmamali)"'''

new_assert = '''        if i > 0:
            assert p.start_sec <= plans[i-1].end_sec or math.isclose(p.start_sec, plans[i-1].end_sec), "Overlap veya bitisiklik saglanmali (bosluk olmamali)"'''

content = content.replace(old_assert, new_assert)
with open("tests/test_segment_planner.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated test_segment_planner.py assertion.")
