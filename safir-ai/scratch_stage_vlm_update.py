import re

with open("src/main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Make sure VLMAnalysisStatus and ChunkAnalysisResult are available in main.py
if "ChunkAnalysisResult" not in content:
    content = content.replace("from typing import Any", "from src.vlm.schemas import ChunkAnalysisResult, VLMAnalysisStatus\nfrom typing import Any")

old_stage_vlm = """            return VLMResponse(
                description=f"[HATA] VLM analizi yapilamadi ({exc}). Manuel inceleme gerekli.",
                model_name=getattr(vlm, "model_name", "unknown"),
                frame_count=len(evidence_frames),
                latency_ms=0.0,
                structured_events=[],
                status="failed",
            )"""

new_stage_vlm = """            return VLMResponse(
                description=f"[HATA] VLM analizi yapilamadi ({exc}). Manuel inceleme gerekli.",
                model_name=getattr(vlm, "model_name", "unknown"),
                frame_count=len(evidence_frames),
                latency_ms=0.0,
                structured_events=[],
                status="failed",
                chunk_analysis_result=ChunkAnalysisResult(
                    analysis_status=VLMAnalysisStatus.MODEL_FAILED,
                    parse_status="pipeline_exception",
                    analysis_id=context.analysis_id if context else None
                )
            )"""

content = content.replace(old_stage_vlm, new_stage_vlm)

old_stage_vlm_frames = """            return VLMResponse(
                description=f"[HATA] VLM batch analizi basarisiz ({exc}). Manuel inceleme gerekli.",
                model_name=getattr(vlm, "model_name", "unknown"),
                frame_count=len(evidence_frames),
                latency_ms=0.0,
                structured_events=[],
                status="failed",
            )"""

new_stage_vlm_frames = """            return VLMResponse(
                description=f"[HATA] VLM batch analizi basarisiz ({exc}). Manuel inceleme gerekli.",
                model_name=getattr(vlm, "model_name", "unknown"),
                frame_count=len(evidence_frames),
                latency_ms=0.0,
                structured_events=[],
                status="failed",
                chunk_analysis_result=ChunkAnalysisResult(
                    analysis_status=VLMAnalysisStatus.MODEL_FAILED,
                    parse_status="pipeline_exception",
                    analysis_id=context.analysis_id if context else None
                )
            )"""

content = content.replace(old_stage_vlm_frames, new_stage_vlm_frames)

with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(content)
