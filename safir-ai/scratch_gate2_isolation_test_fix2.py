import re

with open("tests/test_selector_isolation.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add occurrence_count=1 to TemporalEvent instantiations
content = re.sub(
    r'(confidence=[0-9.]+),',
    r'\1, occurrence_count=1,',
    content
)

with open("tests/test_selector_isolation.py", "w", encoding="utf-8") as f:
    f.write(content)
