import re

with open("src/vlm/video_chunker.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update split_video_into_chunks signature and body
content = content.replace(
    "video_path: str, chunk_duration_sec: float, out_dir: Optional[str] = None, context: Optional[AnalysisContext] = None",
    "video_path: str, chunk_duration_sec: float, chunk_overlap_sec: float = 0.0, out_dir: Optional[str] = None, context: Optional[AnalysisContext] = None"
)

content = content.replace(
    "chunks = _split_with_ffmpeg(video_path, chunk_duration_sec, total_duration, out_dir, context)",
    "chunks = _split_with_ffmpeg(video_path, chunk_duration_sec, total_duration, chunk_overlap_sec, out_dir, context)"
)

content = content.replace(
    "return _split_with_opencv(video_path, chunk_duration_sec, out_dir, context)",
    "return _split_with_opencv(video_path, chunk_duration_sec, chunk_overlap_sec, out_dir, context)"
)

# 2. Update _split_with_ffmpeg
old_ff_sig = """def _split_with_ffmpeg(
    video_path: str, chunk_duration_sec: float, total_duration_sec: float, out_dir: str,
    context: Optional[AnalysisContext] = None
) -> Optional[List[VideoChunk]]:"""

new_ff_sig = """def _split_with_ffmpeg(
    video_path: str, chunk_duration_sec: float, total_duration_sec: float, chunk_overlap_sec: float, out_dir: str,
    context: Optional[AnalysisContext] = None
) -> Optional[List[VideoChunk]]:"""
content = content.replace(old_ff_sig, new_ff_sig)

content = content.replace(
    "plans = plan_segments(total_duration_sec, chunk_duration_sec, 0.0)",
    "plans = plan_segments(total_duration_sec, chunk_duration_sec, chunk_overlap_sec)"
)

content = content.replace(
    "num_chunks = max(1, int(math.ceil(total_duration_sec / chunk_duration_sec)))",
    "num_chunks = len(plans)"
)

# 3. Rewrite _split_with_opencv
old_cv_sig = """def _split_with_opencv(video_path: str, chunk_duration_sec: float, out_dir: Optional[str] = None, context: Optional[AnalysisContext] = None) -> List[VideoChunk]:"""
new_cv_sig = """def _split_with_opencv(video_path: str, chunk_duration_sec: float, chunk_overlap_sec: float = 0.0, out_dir: Optional[str] = None, context: Optional[AnalysisContext] = None) -> List[VideoChunk]:"""
content = content.replace(old_cv_sig, new_cv_sig)

with open("src/vlm/video_chunker.py", "w", encoding="utf-8") as f:
    f.write(content)
