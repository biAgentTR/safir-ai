import logging
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from src.vlm.schemas import VLMSceneObservation


logger = logging.getLogger(__name__)

class NormalizedObservationTime(BaseModel):
    original_relative_start_sec: Optional[float] = None
    original_relative_end_sec: Optional[float] = None
    normalized_relative_start_sec: Optional[float] = None
    normalized_relative_end_sec: Optional[float] = None
    global_start_sec: Optional[float] = None
    global_end_sec: Optional[float] = None
    was_adjusted: bool = False
    adjustment_reasons: List[str] = Field(default_factory=list)
    time_status: str = "valid"  # valid, missing, invalid, partial
    time_base: str = "planned_chunk_offset"

# Config constants for tolerance
TIME_TOLERANCE_NEGATIVE_SEC = 2.0
TIME_TOLERANCE_EXCEED_CHUNK_SEC = 2.0

from typing import Optional
def normalize_observation_time(
    observation: VLMSceneObservation,
    chunk_start_offset_sec: float,
    chunk_duration_sec: Optional[float]
) -> NormalizedObservationTime:
    orig_start = observation.relative_start_sec
    orig_end = observation.relative_end_sec
    
    result = NormalizedObservationTime(
        original_relative_start_sec=orig_start,
        original_relative_end_sec=orig_end,
    )
    
    if orig_start is None and orig_end is None:
        result.time_status = "missing"
        return result
        
    if orig_start is None and orig_end is not None:
        result.time_status = "invalid"
        result.adjustment_reasons.append("Missing start time with present end time")
        return result

    # Point event
    start = orig_start
    end = orig_end
    if orig_end is None and start is not None:
        end = start
        result.was_adjusted = True
        result.adjustment_reasons.append("End time missing, set equal to start time (point event)")
        
    import math
    if math.isnan(start) or math.isnan(end) or math.isinf(start) or math.isinf(end):
        result.time_status = "invalid"
        result.adjustment_reasons.append("NaN or Infinity encountered")
        return result
        
    if start > end:
        result.time_status = "invalid"
        result.adjustment_reasons.append(f"start ({start}) > end ({end})")
        return result
        
    # Check bounds
    if start < 0:
        if start >= -TIME_TOLERANCE_NEGATIVE_SEC:
            start = 0.0
            result.was_adjusted = True
            result.adjustment_reasons.append("Negative start time within tolerance clamped to 0")
        else:
            result.time_status = "invalid"
            result.adjustment_reasons.append("Negative start time beyond tolerance")
            return result
            
    if chunk_duration_sec is not None and end > chunk_duration_sec:
        if end <= chunk_duration_sec + TIME_TOLERANCE_EXCEED_CHUNK_SEC:
            end = chunk_duration_sec
            result.was_adjusted = True
            result.adjustment_reasons.append("End time exceeding chunk duration within tolerance clamped")
        else:
            result.time_status = "invalid"
            result.adjustment_reasons.append("End time exceeding chunk duration beyond tolerance")
            return result
            
    result.normalized_relative_start_sec = start
    result.normalized_relative_end_sec = end
    result.global_start_sec = chunk_start_offset_sec + start
    result.global_end_sec = chunk_start_offset_sec + end
    
    return result

