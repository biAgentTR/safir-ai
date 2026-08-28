import re

with open("tests/test_vlm_contracts.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix mock
content = content.replace(
    "pipeline._event_history = MagicMock()\n    pipeline._last_stage_rag_telemetry = None",
    "pipeline._event_history = MagicMock()\n    pipeline._last_stage_rag_telemetry = None\n    pipeline._event_store = MagicMock()"
)

with open("tests/test_vlm_contracts.py", "w", encoding="utf-8") as f:
    f.write(content)
