import os

new_tests = """

# ------------------------------------------------------------------
# FAZ C1A.1: API, Context ve Pipeline Entegrasyon Smoke Testleri
# ------------------------------------------------------------------
from src.main import create_analyze_job, analyze_video, AnalyzeRequest, app
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid

client = TestClient(app)

def test_api_context_generation():
    \"\"\"API endpointlerinin AnalysisContext'i dogru sekilde urettigini test eder.\"\"\"
    req = AnalyzeRequest(video_source="test.mp4", prompt="Test")
    
    # 1. Job endpoint
    with patch('src.main.SafirPipeline.run') as mock_run:
        response = client.post("/analyze/jobs", json={"video_source": "test.mp4", "prompt": "Test"})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        assert mock_run.called
        kwargs = mock_run.call_args.kwargs
        context = kwargs.get("context")
        assert context is not None
        assert context.analysis_id == job_id
        assert context.video_id != job_id
        assert len(context.video_id) == 36

    # 2. Senkron endpoint
    with patch('src.main.SafirPipeline.run') as mock_run_sync:
        response = client.post("/analyze", json={"video_source": "test.mp4", "prompt": "Test"})
        
        assert mock_run_sync.called
        kwargs = mock_run_sync.call_args.kwargs
        context = kwargs.get("context")
        assert context is not None
        assert context.analysis_id != job_id
        assert context.video_id != context.analysis_id
        assert len(context.analysis_id) == 36
        assert len(context.video_id) == 36

def test_chunker_and_vlm_context_propagation():
    \"\"\"VideoChunker'in urettigi chunklarin ayni ID'leri tasidigini ve context'in EvrenVLM ile
    iletildigini test eder.\"\"\"
    from src.vlm.video_chunker import split_video_into_chunks, AnalysisContext
    
    analysis_id = str(uuid.uuid4())
    video_id = str(uuid.uuid4())
    context = AnalysisContext(analysis_id=analysis_id, video_id=video_id)
    
    with patch('src.vlm.video_chunker._probe_duration_sec', return_value=120.0), \\
         patch('src.vlm.video_chunker._ffmpeg_extract_chunk', return_value=True):
        
        chunks = split_video_into_chunks("dummy.mp4", 60.0, out_dir=".", context=context)
        assert len(chunks) == 2
        
        for i, chunk in enumerate(chunks):
            assert chunk.analysis_id == analysis_id
            assert chunk.video_id == video_id
            assert chunk.chunk_id == f"{analysis_id}:chunk:{i:06d}"
            assert chunk.index == i

    from src.vlm.evren_vlm import EvrenVLM
    vlm = EvrenVLM("vlm")
    with patch.object(vlm, '_send_single_video') as mock_send, \\
         patch('src.vlm.video_chunker._probe_duration_sec', return_value=30.0):
        
        # Orijinal video kisaysa tek parca doner. Mock yapip test edelim
        vlm.analyze_video("dummy.mp4", [], "Test prompt", context=context)
        args, kwargs = mock_send.call_args
        assert args[0] == "dummy.mp4"
        assert "context" not in kwargs
"""

with open("tests/test_pipeline_integration.py", "a", encoding="utf-8") as f:
    f.write(new_tests)

print("Appended tests to test_pipeline_integration.py")
