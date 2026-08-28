import pytest
from src.utils.config_loader import ChunkingConfig
from src.vlm.video_chunker import plan_segments

def test_chunking_config_defaults():
    cfg = ChunkingConfig()
    assert cfg.window_sec == 60.0
    assert cfg.overlap_sec == 5.0

def test_chunking_config_invalid():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ChunkingConfig(window_sec=-1.0)
    with pytest.raises(ValidationError):
        ChunkingConfig(overlap_sec=-1.0)
    with pytest.raises(ValidationError):
        ChunkingConfig(window_sec=60.0, overlap_sec=60.0)
    with pytest.raises(ValidationError):
        ChunkingConfig(window_sec=60.0, overlap_sec=65.0)

def test_plan_segments_overlap_5_120s():
    plans = plan_segments(120.0, 60.0, 5.0)
    assert len(plans) == 3
    assert plans[0].start_sec == 0.0
    assert plans[0].end_sec == 60.0
    assert plans[1].start_sec == 55.0
    assert plans[1].end_sec == 115.0
    assert plans[2].start_sec == 110.0
    assert plans[2].end_sec == 120.0

def test_plan_segments_overlap_5_125s():
    plans = plan_segments(125.0, 60.0, 5.0)
    assert len(plans) == 3
    assert plans[2].start_sec == 110.0
    assert plans[2].end_sec == 125.0

def test_plan_segments_overlap_5_180s():
    plans = plan_segments(180.0, 60.0, 5.0)
    assert len(plans) == 4
    assert plans[3].start_sec == 165.0
    assert plans[3].end_sec == 180.0

def test_plan_segments_overlap_0_125s():
    plans = plan_segments(125.0, 60.0, 0.0)
    assert len(plans) == 3
    assert plans[0].start_sec == 0.0
    assert plans[1].start_sec == 60.0
    assert plans[2].start_sec == 120.0
    assert plans[2].end_sec == 125.0
