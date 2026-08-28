import re

with open("src/vlm/schemas.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add root_validator or model_validator
validator = """    @pydantic.model_validator(mode='after')
    def check_start_end(self):
        if self.relative_start_sec is not None and self.relative_end_sec is not None:
            if self.relative_start_sec > self.relative_end_sec:
                raise ValueError("start_sec cannot be greater than end_sec")
        return self
"""

content = content.replace("    uncertainties: List[str] = Field(default_factory=list)\n", f"    uncertainties: List[str] = Field(default_factory=list)\n\n{validator}\n")
content = "import pydantic\n" + content

with open("src/vlm/schemas.py", "w", encoding="utf-8") as f:
    f.write(content)
