import re

with open("tests/test_vlm_contracts.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    "pipeline._event_store = MagicMock()",
    "pipeline._event_store = MagicMock()\n    pipeline._agent = MagicMock()"
)

with open("tests/test_vlm_contracts.py", "w", encoding="utf-8") as f:
    f.write(content)
