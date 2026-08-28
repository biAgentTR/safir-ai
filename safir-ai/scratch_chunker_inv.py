import re

with open('src/vlm/video_chunker.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''        # Matematiksel olarak hic ilerleme yoksa sonsuz donguyu kir
        if end <= start:
            break'''

new_logic = '''        # Matematiksel olarak hic ilerleme yoksa sonsuz donguyu kir
        if end <= start:
            raise RuntimeError(
                f"Segment planner failed to make progress: index={index} start={start} end={end} "
                f"duration_sec={duration_sec} window_sec={window_sec} overlap_sec={overlap_sec} step_sec={step_sec}"
            )'''

old_start = '''    step_sec = window_sec - overlap_sec
    plans = []'''

new_start = '''    step_sec = window_sec - overlap_sec
    
    # Güvenlik üst sınırı: Çok küçük step ile devasa segment oluşumunu engelle
    # 60 saniyelik standart operasyonda 24 saatlik video bile 1440 segment üretir.
    # 100,000 güvenli ve pratik bir donanımsal üst limittir.
    estimated_segments = (duration_sec / step_sec) + 1
    if estimated_segments > 100000:
        raise ValueError(f"Too many segments estimated ({estimated_segments}). duration={duration_sec}, step={step_sec}")

    plans = []'''

content = content.replace(old_logic, new_logic)
content = content.replace(old_start, new_start)

with open('src/vlm/video_chunker.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated video_chunker.py with invariants and safety limits.")
