import re

with open("src/vlm/base_vlm.py", "r", encoding="utf-8") as f:
    content = f.read()

# I don't know if analyze_video is in BaseVLM. Let's check where it's defined in base_vlm.py.
