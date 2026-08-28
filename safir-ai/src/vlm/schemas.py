import pydantic
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class TaxonomyStatus(str, Enum):
    MATCHED = "matched"
    NOVEL = "novel"
    UNCERTAIN = "uncertain"

class VLMObservationQuality(BaseModel):
    limitations: List[str] = Field(default_factory=list)
    visibility: Optional[str] = None
    coverage_confidence: Optional[float] = None

class VLMQualitySummary(VLMObservationQuality):
    """Cok parcali (chunk) bir analizde parca kalitelerinin BIRLESTIRILMIS ozeti.

    `VLMObservationQuality` tek bir chunk'in ham model ciktisidir ve `visibility`
    orada modelin yazdigi serbest metindir ("clear", "partial", ...). Toplulastirma
    ise sayisal bir esik uretir; bu yuzden burada `visibility` SAYISAL olarak
    yeniden tanimlanir (bkz. `analysis_aggregator.AnalysisAggregator.aggregate`).
    """

    visibility: Optional[float] = None


class VLMSceneObservation(BaseModel):
    observed_label: str = Field(min_length=1)
    canonical_type: Optional[str] = None
    taxonomy_status: TaxonomyStatus = TaxonomyStatus.UNCERTAIN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    relative_start_sec: Optional[float] = None
    relative_end_sec: Optional[float] = None
    entities: List[str] = Field(default_factory=list)
    attributes: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)

    @pydantic.model_validator(mode='after')
    def check_start_end(self):
        if self.relative_start_sec is not None and self.relative_end_sec is not None:
            if self.relative_start_sec > self.relative_end_sec:
                raise ValueError("start_sec cannot be greater than end_sec")
        return self


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
    legacy_adapter_used: bool = False
    repair_attempted: bool = False
    repair_used: bool = False
    repair_succeeded: bool = False
    repair_failure_reason: Optional[str] = None
    fallback_used: bool = False
    report: Optional[VLMObservationReport] = None
    analysis_id: Optional[str] = None
    video_id: Optional[str] = None
    chunk_id: Optional[str] = None
    model_call_id: Optional[str] = None
    attempt: int = 1
