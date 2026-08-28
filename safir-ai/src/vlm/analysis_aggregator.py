from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
import logging
from src.vlm.schemas import ChunkAnalysisResult, VLMAnalysisStatus, VLMQualitySummary
from src.vlm.video_chunker import VideoChunk
# AnalysisContext, video_chunker ile birlikte tanimlidir (event_analysis.schemas DEGIL).
from src.vlm.video_chunker import AnalysisContext

logger = logging.getLogger(__name__)

class SceneSummary(BaseModel):
    chunk_id: str
    summary_text: str

class AnalysisAggregateResult(BaseModel):
    analysis_id: str
    video_id: str
    analysis_status: VLMAnalysisStatus
    chunk_results: List[ChunkAnalysisResult] = Field(default_factory=list)
    merged_events: List[dict] = Field(default_factory=list)
    successful_chunk_ids: List[str] = Field(default_factory=list)
    empty_chunk_ids: List[str] = Field(default_factory=list)
    partial_chunk_ids: List[str] = Field(default_factory=list)
    failed_chunk_ids: List[str] = Field(default_factory=list)
    quality_insufficient_chunk_ids: List[str] = Field(default_factory=list)
    scene_summaries: List[SceneSummary] = Field(default_factory=list)
    quality_summary: Optional[VLMQualitySummary] = None
    model_metadata: Dict[str, Any] = Field(default_factory=dict)
    total_chunk_count: int = 0
    processed_chunk_count: int = 0
    event_count_before_merge: int = 0
    event_count_after_merge: int = 0

class AnalysisAggregator:
    def __init__(self):
        pass
        
    def aggregate(
        self,
        context: Optional[AnalysisContext],
        chunks: List[VideoChunk],
        chunk_results: List[ChunkAnalysisResult],
        chunk_events_map: Dict[str, List[dict]],
        chunk_summaries_map: Dict[str, str],
        model_name: str,
        total_latency_ms: float
    ) -> AnalysisAggregateResult:
        if not chunks:
            # Kontrollu internal failure
            return AnalysisAggregateResult(
                analysis_id=context.analysis_id if context else "unknown",
                video_id=context.video_id if context else "unknown",
                analysis_status=VLMAnalysisStatus.MODEL_FAILED,
                model_metadata={"error": "empty_chunk_list"}
            )
            
        a_id = context.analysis_id if context else "unknown"
        v_id = context.video_id if context else "unknown"
        
        agg = AnalysisAggregateResult(
            analysis_id=a_id,
            video_id=v_id,
            analysis_status=VLMAnalysisStatus.SUCCESS, # placeholder
            total_chunk_count=len(chunks),
            processed_chunk_count=len(chunk_results),
            model_metadata={"model_name": model_name, "total_latency_ms": total_latency_ms}
        )
        
        # Validation & Grouping
        valid_events = []
        qualities = []
        
        # Sort chunks by plan index deterministically
        chunk_results.sort(key=lambda x: getattr(next((c for c in chunks if c.chunk_id == x.chunk_id), None), "index", 0) if x.chunk_id else 0)
        
        analysis_ids = set()
        
        for res in chunk_results:
            if not res.chunk_id:
                agg.failed_chunk_ids.append("unknown")
                continue
                
            if res.analysis_id:
                analysis_ids.add(res.analysis_id)
                if res.analysis_id != a_id and a_id != "unknown":
                    # Reject chunks from other analysis
                    continue
                    
            agg.chunk_results.append(res)
            
            if res.analysis_status == VLMAnalysisStatus.SUCCESS:
                agg.successful_chunk_ids.append(res.chunk_id)
                events = chunk_events_map.get(res.chunk_id, [])
                valid_events.extend(events)
            elif res.analysis_status == VLMAnalysisStatus.SUCCESS_EMPTY:
                agg.empty_chunk_ids.append(res.chunk_id)
            elif res.analysis_status == VLMAnalysisStatus.PARTIAL:
                agg.partial_chunk_ids.append(res.chunk_id)
                events = chunk_events_map.get(res.chunk_id, [])
                valid_events.extend(events)
            elif res.analysis_status in (VLMAnalysisStatus.MODEL_FAILED, VLMAnalysisStatus.PARSE_FAILED):
                agg.failed_chunk_ids.append(res.chunk_id)
            elif res.analysis_status == VLMAnalysisStatus.QUALITY_INSUFFICIENT:
                agg.quality_insufficient_chunk_ids.append(res.chunk_id)
                
            if res.report and res.report.quality:
                qualities.append(res.report.quality)
                
            summary = chunk_summaries_map.get(res.chunk_id)
            if summary:
                agg.scene_summaries.append(SceneSummary(chunk_id=res.chunk_id, summary_text=summary))
                
        # Status decision matrix
        if len(analysis_ids) > 1:
            agg.analysis_status = VLMAnalysisStatus.PARTIAL
        elif not agg.chunk_results:
            agg.analysis_status = VLMAnalysisStatus.MODEL_FAILED
        elif len(agg.successful_chunk_ids) == len(agg.chunk_results):
            agg.analysis_status = VLMAnalysisStatus.SUCCESS
        elif len(agg.empty_chunk_ids) == len(agg.chunk_results):
            agg.analysis_status = VLMAnalysisStatus.SUCCESS_EMPTY
        elif len(agg.empty_chunk_ids) + len(agg.successful_chunk_ids) == len(agg.chunk_results):
            agg.analysis_status = VLMAnalysisStatus.SUCCESS if agg.successful_chunk_ids else VLMAnalysisStatus.SUCCESS_EMPTY
        elif len(agg.quality_insufficient_chunk_ids) == len(agg.chunk_results):
            agg.analysis_status = VLMAnalysisStatus.QUALITY_INSUFFICIENT
        elif len(agg.failed_chunk_ids) == len(agg.chunk_results):
            # Parse vs Model failed
            all_parse = all(r.analysis_status == VLMAnalysisStatus.PARSE_FAILED for r in agg.chunk_results)
            if all_parse:
                agg.analysis_status = VLMAnalysisStatus.PARSE_FAILED
            else:
                agg.analysis_status = VLMAnalysisStatus.MODEL_FAILED
        elif agg.successful_chunk_ids or agg.partial_chunk_ids or agg.empty_chunk_ids:
            agg.analysis_status = VLMAnalysisStatus.PARTIAL
        else:
            agg.analysis_status = VLMAnalysisStatus.MODEL_FAILED

        # Quality aggregation
        #
        # DIKKAT: `VLMObservationQuality.visibility` modelin serbest metni de
        # olabilir ("clear", "partial", ...) ve `coverage_confidence` hic
        # gelmeyebilir (None). Bu yuzden sadece SAYISAL degerler toplulastirilir;
        # sayisal olmayan/eksik degerler sessizce atlanir (eskiden `str < float`
        # karsilastirmasi ve `None` toplama TypeError firlatiyordu).
        if qualities:
            limitations = set()
            visibilities: List[float] = []
            coverages: List[float] = []
            for q in qualities:
                if q.limitations:
                    limitations.update(q.limitations)
                if isinstance(q.visibility, (int, float)) and not isinstance(q.visibility, bool):
                    visibilities.append(float(q.visibility))
                if isinstance(q.coverage_confidence, (int, float)) and not isinstance(q.coverage_confidence, bool):
                    coverages.append(float(q.coverage_confidence))

            agg.quality_summary = VLMQualitySummary(
                # En kotu (minimum) gorunurluk = muhafazakar ozet.
                visibility=min(visibilities) if visibilities else None,
                limitations=sorted(limitations),
                coverage_confidence=(sum(coverages) / len(coverages)) if coverages else None,
            )
            
        # Chronological sort of valid events
        valid_events.sort(key=lambda x: (x.get("_provenance", {}).get("normalized_relative_start_sec", 0.0), x.get("event_name", "")))
        
        agg.merged_events = valid_events
        agg.event_count_before_merge = len(valid_events)
        agg.event_count_after_merge = len(valid_events) # Will be updated if merger runs here. Since merger runs in main.py, this will be updated later.
        
        return agg
