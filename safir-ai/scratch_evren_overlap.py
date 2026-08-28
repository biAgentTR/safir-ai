with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    evren = f.read()

evren = evren.replace(
    "chunks = split_video_into_chunks(video_source, chunk_duration_sec, context=context)",
    "chunk_overlap_sec = getattr(self._endpoint, 'chunk_overlap_sec', 0.0)\n        chunks = split_video_into_chunks(video_source, chunk_duration_sec, chunk_overlap_sec=chunk_overlap_sec, context=context)"
)

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.write(evren)
