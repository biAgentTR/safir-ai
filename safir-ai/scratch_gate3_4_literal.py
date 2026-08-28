import re

with open("src/main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add Literal import and update signature for run and build_report
if "from typing import Literal" not in content:
    content = content.replace("from typing import Any", "from typing import Any, Literal")

content = content.replace('analysis_mode: str = "vlm_direct",', 'analysis_mode: Literal["vlm_direct", "low_budget"] = "vlm_direct",')

with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(content)
