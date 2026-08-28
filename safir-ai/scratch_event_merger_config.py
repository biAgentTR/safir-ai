import re

with open("src/utils/config_loader.py", "r", encoding="utf-8") as f:
    content = f.read()

cfg = """class EventMergerConfig(BaseModel):
    enabled: bool = True
    min_temporal_iou: float = 0.0
    max_boundary_gap_sec: float = 5.0
    min_label_similarity: float = 0.5
    require_type_compatibility: bool = True

    @model_validator(mode="after")
    def validate_merger(self):
        if self.min_temporal_iou < 0.0 or self.min_temporal_iou > 1.0:
            raise ValueError("min_temporal_iou 0.0 ile 1.0 arasinda olmali")
        if self.max_boundary_gap_sec < 0.0:
            raise ValueError("max_boundary_gap_sec negatif olamaz")
        if self.min_label_similarity < 0.0 or self.min_label_similarity > 1.0:
            raise ValueError("min_label_similarity 0.0 ile 1.0 arasinda olmali")
        return self

class SystemConfig(BaseModel):
    merger: EventMergerConfig = Field(default_factory=EventMergerConfig)"""

content = re.sub(r"class SystemConfig\(BaseModel\):", cfg, content)

with open("src/utils/config_loader.py", "w", encoding="utf-8") as f:
    f.write(content)
