import re

with open("tests/test_chunk_provenance.py", "r", encoding="utf-8") as f:
    content = f.read()

replacement = """    from src.vlm.evren_vlm import EvrenVLM
    from src.vlm.video_chunker import VideoChunk, AnalysisContext
    from unittest.mock import patch, MagicMock

    # Create dummy VLM without calling __init__
    vlm = object.__new__(EvrenVLM)
    vlm.model_name = "test-model"
"""

content = re.sub(
    r'    from src\.vlm\.evren_vlm import EvrenVLM\n    from src\.vlm\.vlm_client import EndpointConfig\n    vlm = EvrenVLM\(endpoint_config=EndpointConfig\(name="test", model_name="test-model"\)\)',
    replacement,
    content
)

with open("tests/test_chunk_provenance.py", "w", encoding="utf-8") as f:
    f.write(content)
