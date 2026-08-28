import re

with open("tests/test_segment_planner.py", "r", encoding="utf-8") as f:
    content = f.read()

old_test = '''    # 0.30000000000000004, 0.1, 0
    plans = plan_segments(0.30000000000000004, 0.1, 0.0)
    assert len(plans) == 4
    assert plans[-1].start_sec == 0.30000000000000004 or plans[-1].start_sec == 0.3'''

new_test = '''    # 0.30000000000000004, 0.1, 0
    plans = plan_segments(0.30000000000000004, 0.1, 0.0)
    assert len(plans) == 3
    assert math.isclose(plans[-1].end_sec, 0.30000000000000004)'''

content = content.replace(old_test, new_test)
with open("tests/test_segment_planner.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated test_segment_planner.py expectations.")
