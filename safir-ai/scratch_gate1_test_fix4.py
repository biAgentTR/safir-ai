import re

with open("tests/test_chunk_provenance.py", "r", encoding="utf-8") as f:
    content = f.read()

replacement = """    from src.vlm.evren_vlm import EvrenVLM
    from src.vlm.video_chunker import VideoChunk, AnalysisContext
    from unittest.mock import patch, MagicMock

    # Create dummy VLM without calling __init__
    vlm = object.__new__(EvrenVLM)
    # mock property model_name
    type(vlm).model_name = property(lambda self: "test-model")
"""

content = re.sub(
    r'    from src\.vlm\.evren_vlm import EvrenVLM\n    from src\.vlm\.video_chunker import VideoChunk, AnalysisContext\n    from unittest\.mock import patch, MagicMock\n\n    # Create dummy VLM without calling __init__\n    vlm = object\.__new__\(EvrenVLM\)\n    vlm\.model_name = "test-model"\n',
    replacement,
    content,
    flags=re.DOTALL
)

with open("tests/test_chunk_provenance.py", "w", encoding="utf-8") as f:
    f.write(content)
