import re

with open("src/vlm/video_chunker.py", "r", encoding="utf-8") as f:
    content = f.read()

# Pattern to replace
pattern = r"def _split_with_opencv\(.*?\).*?finally:\n        cap\.release\(\)"

new_opencv = """def _split_with_opencv(video_path: str, chunk_duration_sec: float, chunk_overlap_sec: float = 0.0, out_dir: Optional[str] = None, context: Optional[AnalysisContext] = None) -> List[VideoChunk]:
    \"\"\"Eski, kare-kare OpenCV implementasyonu - YALNIZCA `ffmpeg` kullanilamadiginda/basarisiz oldugunda geri-dusme (fallback) olarak kullanilir.\"\"\"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video dosyasi acilamadi: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0.0

        if total_frames <= 0 or total_duration <= chunk_duration_sec:
            return [
                VideoChunk(
                    path=video_path, start_offset_sec=0.0, end_offset_sec=total_duration, index=0, is_original=True
                )
            ]

        plans = plan_segments(total_duration, chunk_duration_sec, chunk_overlap_sec)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_dir = out_dir or tempfile.mkdtemp(prefix="safir_vlm_chunks_")
        os.makedirs(out_dir, exist_ok=True)

        chunks: List[VideoChunk] = []
        
        for plan in plans:
            chunk_path = os.path.join(out_dir, f"chunk_{plan.index:03d}.mp4")
            writer = cv2.VideoWriter(chunk_path, _FOURCC, fps, (width, height))
            
            start_frame = int(round(plan.start_sec * fps))
            end_frame = int(round(plan.end_sec * fps))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames_written = 0
            
            for _ in range(end_frame - start_frame):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                frames_written += 1
                
            writer.release()
            
            chunks.append(
                VideoChunk(
                    path=chunk_path,
                    start_offset_sec=plan.start_sec,
                    end_offset_sec=plan.start_sec + (frames_written / fps if fps > 0 else 0.0),
                    index=plan.index,
                    encoder="opencv",
                    analysis_id=context.analysis_id if context else None,
                    video_id=context.video_id if context else None,
                    chunk_id=generate_chunk_id(context.video_id, plan.index) if context else None,
                    plan=plan,
                    planned_start_sec=plan.start_sec,
                    planned_end_sec=plan.end_sec,
                    overlap_left_sec=plan.overlap_left_sec,
                    overlap_right_sec=plan.overlap_right_sec
                )
            )

        return chunks

    finally:
        cap.release()"""

content = re.sub(pattern, new_opencv, content, flags=re.DOTALL)

with open("src/vlm/video_chunker.py", "w", encoding="utf-8") as f:
    f.write(content)
