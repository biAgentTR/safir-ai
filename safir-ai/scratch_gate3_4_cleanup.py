import re

with open("src/event_analysis/schemas.py", "r", encoding="utf-8") as f:
    content = f.read()

# Remove the VLM stuff I added
content = re.sub(r'from enum import Enum\nfrom pydantic import Field\n\nclass TaxonomyStatus.*?report: Optional\[VLMObservationReport\] = None\n', '', content, flags=re.DOTALL)

with open("src/event_analysis/schemas.py", "w", encoding="utf-8") as f:
    f.write(content)
