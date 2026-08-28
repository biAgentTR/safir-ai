import re

with open("src/vlm/schemas.py", "r", encoding="utf-8") as f:
    content = f.read()

# Update ChunkAnalysisResult
old_chunk = """class ChunkAnalysisResult(BaseModel):
    analysis_status: VLMAnalysisStatus
    parse_status: str
    repair_used: bool = False
    fallback_used: bool = False
    report: Optional[VLMObservationReport] = None
    analysis_id: Optional[str] = None
    video_id: Optional[str] = None
    chunk_id: Optional[str] = None
    model_call_id: Optional[str] = None
    attempt: int = 1"""

new_chunk = """class ChunkAnalysisResult(BaseModel):
    analysis_status: VLMAnalysisStatus
    parse_status: str
    legacy_adapter_used: bool = False
    repair_attempted: bool = False
    repair_used: bool = False
    repair_succeeded: bool = False
    repair_failure_reason: Optional[str] = None
    fallback_used: bool = False
    report: Optional[VLMObservationReport] = None
    analysis_id: Optional[str] = None
    video_id: Optional[str] = None
    chunk_id: Optional[str] = None
    model_call_id: Optional[str] = None
    attempt: int = 1"""

content = content.replace(old_chunk, new_chunk)

with open("src/vlm/schemas.py", "w", encoding="utf-8") as f:
    f.write(content)
