import re

with open("tests/test_chunk_provenance.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("evidence_timestamps=[],", "evidence_timestamps={},")

with open("tests/test_chunk_provenance.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed test_chunk_provenance.py")
