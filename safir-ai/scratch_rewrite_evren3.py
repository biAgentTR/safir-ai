import re

with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Imports
import_stmt = """from src.vlm.analysis_aggregator import AnalysisAggregator
from src.vlm.schemas import ChunkAnalysisResult, VLMAnalysisStatus
from src.utils.config_loader import get_config
"""
content = content.replace("from typing import Any, Callable, Dict, List, Optional, Tuple", "from typing import Any, Callable, Dict, List, Optional, Tuple\n" + import_stmt)

# 2. analyze_video overlap_sec fix
analyze_vid_old = """            return self._analyze_video_chunks(chunks, prompt, on_progress)
        else:
            return self._send_single_video(
                video_source, prompt, on_progress=on_progress, range_label="[Tam Video]"
            )"""
            
analyze_vid_new = """            return self._analyze_video_chunks(chunks, prompt, on_progress)
        else:
            return self._send_single_video(
                video_source, prompt, on_progress=on_progress, range_label="[Tam Video]"
            )"""
            
# Wait, let's fix the overlap_sec first in analyze_video
old_chunker = """        chunker = VideoChunker(max_duration_sec=self.chunk_duration_sec)"""
new_chunker = """        cfg = get_config()
        chunker = VideoChunker(max_duration_sec=self.chunk_duration_sec, overlap_sec=cfg.system.vlm.chunking.overlap_sec)"""
content = content.replace(old_chunker, new_chunker)

# 3. rewrite _analyze_video_chunks
# We will just replace the entire method
new_analyze_chunks = """    def _analyze_video_chunks(
        self, chunks: List[VideoChunk], prompt: str, on_progress: Optional[VlmProgressCallback] = None
    ) -> VLMResponse:
        total_latency_ms = 0.0
        succeeded_count = 0
        per_chunk_elapsed_sec: List[Optional[float]] = []
        
        chunk_results: List[ChunkAnalysisResult] = []
        chunk_events_map: Dict[str, List[dict]] = {}
        chunk_summaries_map: Dict[str, str] = {}
        model_name = "evren-vlm-unknown"

        for chunk in chunks:
            label = f"[{_format_mmss(chunk.start_offset_sec)}-{_format_mmss(chunk.end_offset_sec)}]"
            try:
                response = self._send_single_video(
                    chunk.path,
                    prompt,
                    on_progress=on_progress,
                    chunk_index=chunk.index + 1,
                    total_chunks=len(chunks),
                    range_label=label,
                    chunk_start_offset_sec=chunk.start_offset_sec,
                    chunk_duration_sec=chunk.end_offset_sec - chunk.start_offset_sec,
                )
                model_name = response.model_name
                
                # Fetch result
                res = getattr(response, "chunk_analysis_result", None)
                if not res:
                    # Fallback for mocked cases
                    res = ChunkAnalysisResult(
                        analysis_status=VLMAnalysisStatus.SUCCESS,
                        parse_status="success",
                        chunk_id=chunk.chunk_id,
                        analysis_id=getattr(chunk, 'context', None).analysis_id if getattr(chunk, 'context', None) else None,
                        video_id=getattr(chunk, 'context', None).video_id if getattr(chunk, 'context', None) else None
                    )
                else:
                    if not res.chunk_id: res.chunk_id = chunk.chunk_id
                    if not res.analysis_id and getattr(chunk, 'context', None): res.analysis_id = getattr(chunk, 'context', None).analysis_id
                    if not res.video_id and getattr(chunk, 'context', None): res.video_id = getattr(chunk, 'context', None).video_id
                
                chunk_results.append(res)
                
                # Extract and shift events
                shifted_events = []
                for event_index, event in enumerate(response.structured_events):
                    shifted = dict(event)
                    
                    if getattr(chunk, 'context', None) is not None:
                        trusted_provenance = {
                            "source_analysis_id": chunk.context.analysis_id,
                            "source_video_id": chunk.context.video_id,
                            "source_chunk_id": chunk.chunk_id,
                            "source_model_call_id": res.model_call_id or f"mc_{chunk.chunk_id}",
                            "source_observation_id": f"obs_{chunk.chunk_id}_{event_index}",
                            "relative_start_sec": shifted.get("normalized_relative_start_sec"),
                            "relative_end_sec": shifted.get("normalized_relative_end_sec"),
                        }
                        shifted.pop("_provenance", None)
                        shifted["_provenance"] = trusted_provenance
                    
                    if "_provenance" not in shifted:
                        shifted["_provenance"] = {}
                    shifted["_provenance"]["normalized_relative_start_sec"] = shifted.get("normalized_relative_start_sec")
                    shifted["_provenance"]["normalized_relative_end_sec"] = shifted.get("normalized_relative_end_sec")
                    shifted["_provenance"]["was_adjusted"] = shifted.get("was_adjusted")
                    shifted["_provenance"]["adjustment_reasons"] = shifted.get("adjustment_reasons")
                    shifted["_provenance"]["time_status"] = shifted.get("time_status")
                    shifted["_provenance"]["time_base"] = shifted.get("time_base")
                    
                    shifted_events.append(shifted)
                
                chunk_events_map[chunk.chunk_id] = shifted_events
                chunk_summaries_map[chunk.chunk_id] = response.description
                
                succeeded_count += 1
                total_latency_ms += response.latency_ms
                per_chunk_elapsed_sec.append(round(response.latency_ms / 1000, 1))

            except Exception as exc:  # noqa: BLE001
                import logging
                logger = logging.getLogger(__name__)
                logger.exception("EVREN VLM: video parcasi basarisiz (index=%d, %s)", chunk.index, label)
                res = ChunkAnalysisResult(
                    analysis_status=VLMAnalysisStatus.MODEL_FAILED,
                    parse_status=f"pipeline_exception: {exc}",
                    chunk_id=chunk.chunk_id,
                    analysis_id=getattr(chunk, 'context', None).analysis_id if getattr(chunk, 'context', None) else None,
                    video_id=getattr(chunk, 'context', None).video_id if getattr(chunk, 'context', None) else None
                )
                chunk_results.append(res)
                per_chunk_elapsed_sec.append(None)
                continue

        aggregator = AnalysisAggregator()
        context = chunks[0].context if chunks and hasattr(chunks[0], 'context') else None
        
        agg_result = aggregator.aggregate(
            context=context,
            chunks=chunks,
            chunk_results=chunk_results,
            chunk_events_map=chunk_events_map,
            chunk_summaries_map=chunk_summaries_map,
            model_name=model_name,
            total_latency_ms=total_latency_ms
        )
        
        # Format description for legacy backward compatibility
        description_parts = []
        for summary in agg_result.scene_summaries:
            # find chunk label
            chunk_obj = next((c for c in chunks if c.chunk_id == summary.chunk_id), None)
            if chunk_obj:
                label = f"[{_format_mmss(chunk_obj.start_offset_sec)}-{_format_mmss(chunk_obj.end_offset_sec)}]"
                description_parts.append(f"{label} {summary.summary_text}".strip())
        
        for failed_id in agg_result.failed_chunk_ids:
            chunk_obj = next((c for c in chunks if c.chunk_id == failed_id), None)
            if chunk_obj:
                label = f"[{_format_mmss(chunk_obj.start_offset_sec)}-{_format_mmss(chunk_obj.end_offset_sec)}]"
                description_parts.append(f"{label} [ANALYSIS_FAILED] Bu parca icin VLM analizi basarisiz")

        # Telemetry
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            "AnalysisAggregator produced status: %s for %d chunks (events: %d).",
            agg_result.analysis_status,
            agg_result.total_chunk_count,
            agg_result.event_count_after_merge
        )

        final_description = "\\n".join(description_parts) if description_parts else "Analiz sonucsuz."
        
        return VLMResponse(
            description=final_description,
            model_name=model_name,
            frame_count=0, # doesn't matter for video
            latency_ms=total_latency_ms,
            structured_events=agg_result.merged_events,
            status="completed" if agg_result.analysis_status in (VLMAnalysisStatus.SUCCESS, VLMAnalysisStatus.SUCCESS_EMPTY) else "partial_failure" if agg_result.analysis_status == VLMAnalysisStatus.PARTIAL else "failed",
            chunking_summary={"chunks": len(chunks), "elapsed_sec": per_chunk_elapsed_sec},
            aggregate_result=agg_result
        )"""

pattern = r"    def _analyze_video_chunks\((.*?)\) -> VLMResponse:(.*?)        return VLMResponse\((.*?)\)"
content = re.sub(pattern, new_analyze_chunks, content, flags=re.DOTALL)

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.write(content)
