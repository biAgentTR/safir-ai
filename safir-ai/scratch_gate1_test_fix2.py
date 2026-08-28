import re

with open("tests/test_chunk_provenance.py", "r", encoding="utf-8") as f:
    content = f.read()

replacement = """    from src.vlm.evren_vlm import EvrenVLM
    from src.vlm.vlm_client import EndpointConfig
    vlm = EvrenVLM(endpoint_config=EndpointConfig(name="test", model_name="test-model"))"""

content = content.replace('    vlm = EvrenVLM(model_name="test-model")', replacement)

with open("tests/test_chunk_provenance.py", "w", encoding="utf-8") as f:
    f.write(content)
