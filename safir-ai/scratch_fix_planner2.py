import re

with open("src/vlm/video_chunker.py", "r", encoding="utf-8") as f:
    content = f.read()

old_loop_end = '''        plans.append(SegmentPlan(
            index=index,
            start_sec=start,
            end_sec=end,
            overlap_left_sec=overlap_left_sec,
            overlap_right_sec=overlap_right_sec
        ))
        
        index += 1'''

new_loop_end = '''        plans.append(SegmentPlan(
            index=index,
            start_sec=start,
            end_sec=end,
            overlap_left_sec=overlap_left_sec,
            overlap_right_sec=overlap_right_sec
        ))
        
        if end >= duration_sec:
            break
            
        index += 1'''

content = content.replace(old_loop_end, new_loop_end)
with open("src/vlm/video_chunker.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Added end >= duration_sec break.")
