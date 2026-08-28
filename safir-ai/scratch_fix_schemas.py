import re

with open("src/event_analysis/schemas.py", "r", encoding="utf-8") as f:
    content = f.read()

pattern = r"uncertainties: List\[str\].*?matched_keywords: List\[str\] = Field\("

new_content = """uncertainties: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    attributes: List[str] = Field(default_factory=list)
    matched_keywords: List[str] = Field("""

content = re.sub(pattern, new_content, content, flags=re.DOTALL)

with open("src/event_analysis/schemas.py", "w", encoding="utf-8") as f:
    f.write(content)
