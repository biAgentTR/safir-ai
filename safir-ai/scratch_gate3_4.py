import re

with open("src/event_analysis/schemas.py", "r", encoding="utf-8") as f:
    content = f.read()

schemas_addition = """
from enum import Enum
from pydantic import Field

class TaxonomyStatus(str, Enum):
    MATCHED = "matched"
    NOVEL = "novel"
    UNCERTAIN = "uncertain"

class VLMObservationQuality(BaseModel):
    limitations: List[str] = Field(default_factory=list)

class VLMSceneObservation(BaseModel):
    observed_label: str
    canonical_type: Optional[str] = None
    taxonomy_status: TaxonomyStatus = TaxonomyStatus.UNCERTAIN
    relative_start_sec: Optional[float] = None
    relative_end_sec: Optional[float] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    visibility: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    attributes: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)

class VLMObservationReport(BaseModel):
    schema_version: str = "1.0"
    scene_summary: str = ""
    observations: List[VLMSceneObservation] = Field(default_factory=list)
    quality: VLMObservationQuality = Field(default_factory=VLMObservationQuality)

class VLMAnalysisStatus(str, Enum):
    SUCCESS = "success"
    SUCCESS_EMPTY = "success_empty"
    PARTIAL = "partial"
    PARSE_FAILED = "parse_failed"
    MODEL_FAILED = "model_failed"
    QUALITY_INSUFFICIENT = "quality_insufficient"

class ChunkAnalysisResult(BaseModel):
    analysis_status: VLMAnalysisStatus
    parse_status: str
    repair_used: bool = False
    fallback_used: bool = False
    report: Optional[VLMObservationReport] = None
"""

if "VLMObservationReport" not in content:
    content += schemas_addition

with open("src/event_analysis/schemas.py", "w", encoding="utf-8") as f:
    f.write(content)
