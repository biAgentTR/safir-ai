import re

# 1. Fix plan_segments overlap logic
with open("src/vlm/video_chunker.py", "r", encoding="utf-8") as f:
    content = f.read()

old_overlap = '''        next_start = (index + 1) * step_sec
        if next_start < duration_sec:
            next_end = min(next_start + window_sec, duration_sec)
            overlap_right_sec = max(0.0, min(end, next_end) - max(start, next_start))
        else:
            overlap_right_sec = 0.0'''
            
new_overlap = '''        if end >= duration_sec:
            overlap_right_sec = 0.0
        else:
            next_start = (index + 1) * step_sec
            if next_start < duration_sec:
                next_end = min(next_start + window_sec, duration_sec)
                overlap_right_sec = max(0.0, min(end, next_end) - max(start, next_start))
            else:
                overlap_right_sec = 0.0'''

content = content.replace(old_overlap, new_overlap)
with open("src/vlm/video_chunker.py", "w", encoding="utf-8") as f:
    f.write(content)


# 2. Fix import in test_pipeline_integration.py
with open("tests/test_pipeline_integration.py", "r", encoding="utf-8") as f:
    t_content = f.read()

t_content = t_content.replace(
    "from src.main import create_analyze_job, analyze_video, AnalyzeRequest, app",
    "from src.main import create_analyze_job, analyze, AnalyzeRequest, app"
)
t_content = t_content.replace(
    "def analyze_video(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> Union[SafirReport, JobStatusResponse]:",
    "def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> Union[SafirReport, JobStatusResponse]:"
)

with open("tests/test_pipeline_integration.py", "w", encoding="utf-8") as f:
    f.write(t_content)

print("Fixed planner overlap logic and test imports.")
