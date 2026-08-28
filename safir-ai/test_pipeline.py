import sys
import os

from src.main import get_pipeline, normalize_video_source

try:
    pipeline = get_pipeline()
    normalized = normalize_video_source("data/test.mp4")
    report = pipeline.run(normalized, "risk analizi yap")
    print(report.json())
except Exception as e:
    import traceback
    traceback.print_exc()
