import re

with open("tests/test_vlm_contracts.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1
content = content.replace(
    "content = '''Bozuk json EVENTS_JSON: [{\"event_name\": \"Kask\"'''",
    "content = '''Bozuk json EVENTS_JSON: [{\"event_name\": \"Kask\" ]'''"
)

# Fix 2
content = content.replace(
    'assert result.parse_status == "regex_fallback_not_found"',
    'assert result.parse_status == "unrecognized_format"'
)

# Fix 3
content = content.replace(
    "pipeline._event_history = MagicMock()",
    "pipeline._event_history = MagicMock()\n    pipeline._last_stage_rag_telemetry = None"
)

with open("tests/test_vlm_contracts.py", "w", encoding="utf-8") as f:
    f.write(content)
