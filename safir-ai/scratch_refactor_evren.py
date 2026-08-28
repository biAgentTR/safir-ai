import re

with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    evren_content = f.read()

# Add AnalysisContext import
if "AnalysisContext" not in evren_content:
    evren_content = evren_content.replace(
        "from src.vlm.video_chunker import VideoChunk, cleanup_chunks, split_video_into_chunks",
        "from src.vlm.video_chunker import VideoChunk, cleanup_chunks, split_video_into_chunks, AnalysisContext"
    )

old_analyze_video = '''    def analyze_video(
        self,
        video_source: str,
        evidence_frames: List[EvidenceFrame],
        prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
    ) -> VLMResponse:'''
new_analyze_video = '''    def analyze_video(
        self,
        video_source: str,
        evidence_frames: List[EvidenceFrame],
        prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
        context: Optional[AnalysisContext] = None,
    ) -> VLMResponse:'''
evren_content = evren_content.replace(old_analyze_video, new_analyze_video)

old_split = '''        chunks = split_video_into_chunks(video_source, chunk_duration_sec)'''
new_split = '''        chunks = split_video_into_chunks(video_source, chunk_duration_sec, context=context)'''
evren_content = evren_content.replace(old_split, new_split)

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.write(evren_content)

print("evren_vlm.py refactored.")
