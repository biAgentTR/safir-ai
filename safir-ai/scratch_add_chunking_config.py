import re

with open("src/utils/config_loader.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add ChunkingConfig
chunking_config = """class ChunkingConfig(BaseModel):
    window_sec: float = 60.0
    overlap_sec: float = 5.0
    
    @model_validator(mode="after")
    def validate_overlap(self):
        import math
        if not math.isfinite(self.window_sec) or self.window_sec <= 0:
            raise ValueError("window_sec finite ve > 0 olmali")
        if not math.isfinite(self.overlap_sec) or self.overlap_sec < 0:
            raise ValueError("overlap_sec finite ve >= 0 olmali")
        if self.overlap_sec >= self.window_sec:
            raise ValueError("overlap_sec < window_sec olmali")
        return self

class VLMConfig(BaseModel):"""

content = content.replace("class VLMConfig(BaseModel):", chunking_config)

# Add chunking field to VLMConfig
vlm_config_fields = """class VLMConfig(BaseModel):
    \"\"\"Aktif VLM secimini ve tum VLM tanimlarini tutar (Factory Pattern icin).\"\"\"

    active_model: str
    chunking: Optional[ChunkingConfig] = None"""

content = content.replace("""class VLMConfig(BaseModel):
    \"\"\"Aktif VLM secimini ve tum VLM tanimlarini tutar (Factory Pattern icin).\"\"\"

    active_model: str""", vlm_config_fields)

with open("src/utils/config_loader.py", "w", encoding="utf-8") as f:
    f.write(content)
