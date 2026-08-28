import re

with open("src/vlm/video_chunker.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update _split_with_ffmpeg
old_ffmpeg = '''def _split_with_ffmpeg(
    video_path: str, chunk_duration_sec: float, total_duration_sec: float, out_dir: str
) -> Optional[List[VideoChunk]]:'''
new_ffmpeg = '''def _split_with_ffmpeg(
    video_path: str, chunk_duration_sec: float, total_duration_sec: float, out_dir: str,
    context: Optional[AnalysisContext] = None
) -> Optional[List[VideoChunk]]:'''
content = content.replace(old_ffmpeg, new_ffmpeg)

old_ffmpeg_body = '''    num_chunks = math.ceil(total_duration_sec / chunk_duration_sec)
    chunks: List[VideoChunk] = []

    for i in range(num_chunks):
        start = i * chunk_duration_sec
        duration = min(chunk_duration_sec, total_duration_sec - start)
        chunk_path = os.path.join(out_dir, f"chunk_{i:03d}.mp4")'''
new_ffmpeg_body = '''    plans = plan_segments(total_duration_sec, chunk_duration_sec, 0.0)
    chunks: List[VideoChunk] = []

    for plan in plans:
        start = plan.start_sec
        duration = plan.duration_sec
        i = plan.index
        chunk_path = os.path.join(out_dir, f"chunk_{i:03d}.mp4")'''
content = content.replace(old_ffmpeg_body, new_ffmpeg_body)

old_ffmpeg_append = '''        chunks.append(
            VideoChunk(
                path=chunk_path,
                start_offset_sec=start,
                end_offset_sec=start + duration,
                index=i,
                encoder="cuda" if use_cuda else "cpu",
            )
        )'''
new_ffmpeg_append = '''        chunks.append(
            VideoChunk(
                path=chunk_path,
                start_offset_sec=start,
                end_offset_sec=start + duration,
                index=i,
                encoder="cuda" if use_cuda else "cpu",
                analysis_id=context.analysis_id if context else None,
                video_id=context.video_id if context else None,
                chunk_id=generate_chunk_id(context.analysis_id, i) if context else None,
                plan=plan,
                planned_start_sec=plan.start_sec,
                planned_end_sec=plan.end_sec,
                overlap_left_sec=plan.overlap_left_sec,
                overlap_right_sec=plan.overlap_right_sec
            )
        )'''
content = content.replace(old_ffmpeg_append, new_ffmpeg_append)

# 2. Update _split_with_opencv
old_opencv = '''def _split_with_opencv(video_path: str, chunk_duration_sec: float, out_dir: Optional[str] = None) -> List[VideoChunk]:'''
new_opencv = '''def _split_with_opencv(video_path: str, chunk_duration_sec: float, out_dir: Optional[str] = None, context: Optional[AnalysisContext] = None) -> List[VideoChunk]:'''
content = content.replace(old_opencv, new_opencv)

old_opencv_body = '''    total_duration = frame_count / fps
    num_chunks = math.ceil(total_duration / chunk_duration_sec)
    frames_per_chunk = int(fps * chunk_duration_sec)'''
new_opencv_body = '''    total_duration = frame_count / fps
    plans = plan_segments(total_duration, chunk_duration_sec, 0.0)
    frames_per_chunk = int(fps * chunk_duration_sec)'''
content = content.replace(old_opencv_body, new_opencv_body)

old_opencv_loop = '''    for i in range(num_chunks):
        chunk_path = os.path.join(out_dir, f"chunk_{i:03d}.mp4")
        writer = cv2.VideoWriter(chunk_path, _FOURCC, fps, (width, height))
        
        frames_written = 0
        while frames_written < frames_per_chunk:'''
new_opencv_loop = '''    for plan in plans:
        i = plan.index
        chunk_path = os.path.join(out_dir, f"chunk_{i:03d}.mp4")
        writer = cv2.VideoWriter(chunk_path, _FOURCC, fps, (width, height))
        
        # OpenCv mantigi plan uzerinden (overlap desteksiz simdilik)
        frames_to_write = int((plan.end_sec - plan.start_sec) * fps)
        
        frames_written = 0
        while frames_written < frames_to_write:'''
content = content.replace(old_opencv_loop, new_opencv_loop)

old_opencv_append = '''        chunks.append(
            VideoChunk(
                path=chunk_path,
                start_offset_sec=i * chunk_duration_sec,
                end_offset_sec=min((i + 1) * chunk_duration_sec, total_duration),
                index=i,
                encoder="opencv",
            )
        )'''
new_opencv_append = '''        chunks.append(
            VideoChunk(
                path=chunk_path,
                start_offset_sec=plan.start_sec,
                end_offset_sec=plan.end_sec,
                index=i,
                encoder="opencv",
                analysis_id=context.analysis_id if context else None,
                video_id=context.video_id if context else None,
                chunk_id=generate_chunk_id(context.analysis_id, i) if context else None,
                plan=plan,
                planned_start_sec=plan.start_sec,
                planned_end_sec=plan.end_sec,
                overlap_left_sec=plan.overlap_left_sec,
                overlap_right_sec=plan.overlap_right_sec
            )
        )'''
content = content.replace(old_opencv_append, new_opencv_append)

# 3. Update split_video_into_chunks
old_split = '''def split_video_into_chunks(
    video_path: str, chunk_duration_sec: float, out_dir: Optional[str] = None
) -> List[VideoChunk]:'''
new_split = '''def split_video_into_chunks(
    video_path: str, chunk_duration_sec: float, out_dir: Optional[str] = None, context: Optional[AnalysisContext] = None
) -> List[VideoChunk]:'''
content = content.replace(old_split, new_split)

old_split_short = '''    if chunk_duration_sec <= 0:
        return [VideoChunk(path=video_path, start_offset_sec=0.0, end_offset_sec=0.0, index=0, is_original=True)]'''
new_split_short = '''    if chunk_duration_sec <= 0:
        chunk_id = generate_chunk_id(context.analysis_id, 0) if context else None
        return [VideoChunk(
            path=video_path, start_offset_sec=0.0, end_offset_sec=0.0, index=0, is_original=True,
            analysis_id=context.analysis_id if context else None,
            video_id=context.video_id if context else None,
            chunk_id=chunk_id
        )]'''
content = content.replace(old_split_short, new_split_short)

old_split_short2 = '''    if total_duration <= chunk_duration_sec:
        return [
            VideoChunk(
                path=video_path, start_offset_sec=0.0, end_offset_sec=total_duration, index=0, is_original=True
            )
        ]'''
new_split_short2 = '''    if total_duration <= chunk_duration_sec:
        chunk_id = generate_chunk_id(context.analysis_id, 0) if context else None
        return [
            VideoChunk(
                path=video_path, start_offset_sec=0.0, end_offset_sec=total_duration, index=0, is_original=True,
                analysis_id=context.analysis_id if context else None,
                video_id=context.video_id if context else None,
                chunk_id=chunk_id,
                plan=SegmentPlan(0, 0.0, total_duration, 0.0, 0.0),
                planned_start_sec=0.0,
                planned_end_sec=total_duration
            )
        ]'''
content = content.replace(old_split_short2, new_split_short2)

old_split_calls = '''    chunks = _split_with_ffmpeg(video_path, chunk_duration_sec, total_duration, out_dir)
    if chunks is None:
        return _split_with_opencv(video_path, chunk_duration_sec, out_dir)'''
new_split_calls = '''    chunks = _split_with_ffmpeg(video_path, chunk_duration_sec, total_duration, out_dir, context)
    if chunks is None:
        return _split_with_opencv(video_path, chunk_duration_sec, out_dir, context)'''
content = content.replace(old_split_calls, new_split_calls)

old_split_opencv_early = '''        return _split_with_opencv(video_path, chunk_duration_sec, out_dir)'''
new_split_opencv_early = '''        return _split_with_opencv(video_path, chunk_duration_sec, out_dir, context)'''
content = content.replace(old_split_opencv_early, new_split_opencv_early)


with open("src/vlm/video_chunker.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Refactoring complete.")
