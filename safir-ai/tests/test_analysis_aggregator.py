import pytest
from src.vlm.analysis_aggregator import AnalysisAggregator, AnalysisAggregateResult, SceneSummary
from src.vlm.schemas import ChunkAnalysisResult, VLMAnalysisStatus, VLMQualitySummary
from src.vlm.video_chunker import VideoChunk
from src.event_analysis.schemas import AnalysisContext

def create_chunk(id, idx):
    return VideoChunk(
        chunk_id=id,
        index=idx,
        path=f"path_{id}",
        start_offset_sec=0,
        end_offset_sec=10,
        context=AnalysisContext(analysis_id="a1", video_id="v1")
    )

def test_aggregator_all_success_with_events():
    chunks = [create_chunk("c1", 0), create_chunk("c2", 1)]
    chunk_results = [
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.SUCCESS, parse_status="ok", chunk_id="c1", analysis_id="a1", video_id="v1"),
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.SUCCESS, parse_status="ok", chunk_id="c2", analysis_id="a1", video_id="v1")
    ]
    chunk_events_map = {"c1": [{"event_name": "E1"}], "c2": [{"event_name": "E2"}]}
    
    agg = AnalysisAggregator().aggregate(chunks[0].context, chunks, chunk_results, chunk_events_map, {}, "test-model", 100.0)
    assert agg.analysis_status == VLMAnalysisStatus.SUCCESS
    assert len(agg.merged_events) == 2

def test_aggregator_all_success_empty():
    chunks = [create_chunk("c1", 0), create_chunk("c2", 1)]
    chunk_results = [
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.SUCCESS_EMPTY, parse_status="ok", chunk_id="c1", analysis_id="a1", video_id="v1"),
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.SUCCESS_EMPTY, parse_status="ok", chunk_id="c2", analysis_id="a1", video_id="v1")
    ]
    
    agg = AnalysisAggregator().aggregate(chunks[0].context, chunks, chunk_results, {}, {}, "test-model", 100.0)
    assert agg.analysis_status == VLMAnalysisStatus.SUCCESS_EMPTY
    assert len(agg.merged_events) == 0

def test_aggregator_success_and_model_failed():
    chunks = [create_chunk("c1", 0), create_chunk("c2", 1)]
    chunk_results = [
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.SUCCESS, parse_status="ok", chunk_id="c1", analysis_id="a1", video_id="v1"),
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.MODEL_FAILED, parse_status="err", chunk_id="c2", analysis_id="a1", video_id="v1")
    ]
    chunk_events_map = {"c1": [{"event_name": "E1"}]}
    
    agg = AnalysisAggregator().aggregate(chunks[0].context, chunks, chunk_results, chunk_events_map, {}, "test-model", 100.0)
    assert agg.analysis_status == VLMAnalysisStatus.PARTIAL
    assert len(agg.merged_events) == 1
    assert agg.failed_chunk_ids == ["c2"]

def test_aggregator_all_model_failed():
    chunks = [create_chunk("c1", 0), create_chunk("c2", 1)]
    chunk_results = [
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.MODEL_FAILED, parse_status="err", chunk_id="c1", analysis_id="a1", video_id="v1"),
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.MODEL_FAILED, parse_status="err", chunk_id="c2", analysis_id="a1", video_id="v1")
    ]
    
    agg = AnalysisAggregator().aggregate(chunks[0].context, chunks, chunk_results, {}, {}, "test-model", 100.0)
    assert agg.analysis_status == VLMAnalysisStatus.MODEL_FAILED

def test_aggregator_all_parse_failed():
    chunks = [create_chunk("c1", 0), create_chunk("c2", 1)]
    chunk_results = [
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.PARSE_FAILED, parse_status="err", chunk_id="c1", analysis_id="a1", video_id="v1"),
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.PARSE_FAILED, parse_status="err", chunk_id="c2", analysis_id="a1", video_id="v1")
    ]
    
    agg = AnalysisAggregator().aggregate(chunks[0].context, chunks, chunk_results, {}, {}, "test-model", 100.0)
    assert agg.analysis_status == VLMAnalysisStatus.PARSE_FAILED

def test_aggregator_mixed_model_and_parse_failed():
    chunks = [create_chunk("c1", 0), create_chunk("c2", 1)]
    chunk_results = [
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.MODEL_FAILED, parse_status="err", chunk_id="c1", analysis_id="a1", video_id="v1"),
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.PARSE_FAILED, parse_status="err", chunk_id="c2", analysis_id="a1", video_id="v1")
    ]
    
    agg = AnalysisAggregator().aggregate(chunks[0].context, chunks, chunk_results, {}, {}, "test-model", 100.0)
    # Kontrollu failure
    assert agg.analysis_status == VLMAnalysisStatus.MODEL_FAILED

def test_aggregator_quality_insufficient():
    chunks = [create_chunk("c1", 0)]
    chunk_results = [
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.QUALITY_INSUFFICIENT, parse_status="ok", chunk_id="c1", analysis_id="a1", video_id="v1")
    ]
    
    agg = AnalysisAggregator().aggregate(chunks[0].context, chunks, chunk_results, {}, {}, "test-model", 100.0)
    assert agg.analysis_status == VLMAnalysisStatus.QUALITY_INSUFFICIENT

def test_aggregator_empty_chunks_list():
    agg = AnalysisAggregator().aggregate(AnalysisContext(analysis_id="a", video_id="v"), [], [], {}, {}, "test-model", 100.0)
    assert agg.analysis_status == VLMAnalysisStatus.MODEL_FAILED
    assert agg.model_metadata["error"] == "empty_chunk_list"

def test_aggregator_farkli_analysis_id():
    chunks = [create_chunk("c1", 0), create_chunk("c2", 1)]
    chunk_results = [
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.SUCCESS, parse_status="ok", chunk_id="c1", analysis_id="a1", video_id="v1"),
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.SUCCESS, parse_status="ok", chunk_id="c2", analysis_id="A-FARKLI", video_id="v1")
    ]
    
    agg = AnalysisAggregator().aggregate(chunks[0].context, chunks, chunk_results, {}, {}, "test-model", 100.0)
    # 2. chunk atlanmali
    assert len(agg.chunk_results) == 1
    assert agg.chunk_results[0].chunk_id == "c1"

def test_aggregator_quality_union():
    chunks = [create_chunk("c1", 0), create_chunk("c2", 1)]
    
    from src.vlm.schemas import VLMObservationReport
    r1 = VLMObservationReport(quality=VLMQualitySummary(visibility=0.8, limitations=["L1"], coverage_confidence=0.9), events=[])
    r2 = VLMObservationReport(quality=VLMQualitySummary(visibility=0.6, limitations=["L1", "L2"], coverage_confidence=0.5), events=[])
    
    chunk_results = [
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.SUCCESS, parse_status="ok", chunk_id="c1", analysis_id="a1", video_id="v1", report=r1),
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.SUCCESS, parse_status="ok", chunk_id="c2", analysis_id="a1", video_id="v1", report=r2)
    ]
    
    agg = AnalysisAggregator().aggregate(chunks[0].context, chunks, chunk_results, {}, {}, "test-model", 100.0)
    assert agg.quality_summary.visibility == 0.6 # min
    assert set(agg.quality_summary.limitations) == {"L1", "L2"} # union
    assert agg.quality_summary.coverage_confidence == 0.7 # average (0.9+0.5)/2

def test_aggregator_scene_summaries():
    chunks = [create_chunk("c1", 0), create_chunk("c2", 1)]
    chunk_results = [
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.SUCCESS, parse_status="ok", chunk_id="c1", analysis_id="a1", video_id="v1"),
        ChunkAnalysisResult(analysis_status=VLMAnalysisStatus.SUCCESS, parse_status="ok", chunk_id="c2", analysis_id="a1", video_id="v1")
    ]
    
    summaries = {"c1": "Scene 1", "c2": "Scene 2"}
    agg = AnalysisAggregator().aggregate(chunks[0].context, chunks, chunk_results, {}, summaries, "test-model", 100.0)
    
    assert len(agg.scene_summaries) == 2
    assert agg.scene_summaries[0].chunk_id == "c1"
    assert agg.scene_summaries[0].summary_text == "Scene 1"
    assert agg.scene_summaries[1].chunk_id == "c2"
    assert agg.scene_summaries[1].summary_text == "Scene 2"
