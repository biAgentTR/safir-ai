import re
import json
import logging
import math
from typing import Tuple, List, Dict, Any, Optional

from src.vlm.schemas import (
    VLMObservationReport,
    VLMSceneObservation,
    VLMObservationQuality,
    TaxonomyStatus,
    VLMAnalysisStatus,
    ChunkAnalysisResult
)

logger = logging.getLogger(__name__)

_EVENTS_JSON_PATTERN = re.compile(r"EVENTS_JSON:\s*(\[.*\])", re.DOTALL | re.IGNORECASE)

class LegacyAdapter:
    """Eski EVENTS_JSON formatini yeni VLMObservationReport'a donusturur."""

    @staticmethod
    def parse(raw_content: str, model_call_id: Optional[str] = None, chunk_id: Optional[str] = None, analysis_id: Optional[str] = None) -> Tuple[str, ChunkAnalysisResult]:
        match = _EVENTS_JSON_PATTERN.search(raw_content)
        if not match:
            # Parse failed
            return raw_content.strip(), ChunkAnalysisResult(
                analysis_status=VLMAnalysisStatus.PARSE_FAILED,
                parse_status="regex_fallback_not_found",
                fallback_used=True,
                legacy_adapter_used=True,
                analysis_id=analysis_id,
                chunk_id=chunk_id,
                model_call_id=model_call_id
            )

        json_str = match.group(1)
        clean_description = raw_content[: match.start()].strip()
        
        repair_attempted = False
        repair_succeeded = False
        repair_failure_reason = None
        
        try:
            events_raw = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Ilkel onarim denemesi (Lokal JSON temizleme: tirnaklari duzeltme)
            repair_attempted = True
            try:
                fixed_str = json_str.replace("'", '"')
                events_raw = json.loads(fixed_str)
                repair_succeeded = True
            except Exception as ex:
                repair_succeeded = False
                repair_failure_reason = str(ex)
                return clean_description, ChunkAnalysisResult(
                    analysis_status=VLMAnalysisStatus.PARSE_FAILED,
                    parse_status="json_decode_failed",
                    fallback_used=True,
                    legacy_adapter_used=True,
                    repair_attempted=True,
                    repair_used=True,
                    repair_succeeded=False,
                    repair_failure_reason=repair_failure_reason,
                    analysis_id=analysis_id,
                    chunk_id=chunk_id,
                    model_call_id=model_call_id
                )

        if not isinstance(events_raw, list):
            return clean_description, ChunkAnalysisResult(
                analysis_status=VLMAnalysisStatus.PARSE_FAILED,
                parse_status="expected_list",
                fallback_used=True,
                legacy_adapter_used=True,
                repair_attempted=repair_attempted,
                repair_used=repair_attempted,
                repair_succeeded=repair_succeeded,
                analysis_id=analysis_id,
                chunk_id=chunk_id,
                model_call_id=model_call_id
            )

        if not events_raw:
            return clean_description, ChunkAnalysisResult(
                analysis_status=VLMAnalysisStatus.SUCCESS_EMPTY,
                parse_status="success_empty",
                legacy_adapter_used=True,
                repair_attempted=repair_attempted,
                repair_used=repair_attempted,
                repair_succeeded=repair_succeeded,
                analysis_id=analysis_id,
                chunk_id=chunk_id,
                model_call_id=model_call_id,
                report=VLMObservationReport(
                    scene_summary=clean_description,
                    observations=[]
                )
            )

        observations = []
        has_invalid = False
        for raw in events_raw:
            try:
                # Validasyon kurallari: start_time, end_time finite
                start = float(raw.get("start_time", 0.0))
                end = float(raw.get("end_time", 0.0))
                if not (math.isfinite(start) and math.isfinite(end)):
                    has_invalid = True
                    continue
                # start > end TimeNormalizer veya model validatorunda acik hatadir (pydantic yakalar)
                
                obs = VLMSceneObservation(
                    observed_label=str(raw.get("event_name") or raw.get("type", "unknown")),
                    confidence=float(raw.get("confidence", 0.8)),
                    relative_start_sec=start,
                    relative_end_sec=end,
                    evidence=raw.get("evidence_ids", []) or []
                )
                observations.append(obs)
            except Exception:
                has_invalid = True
        
        status = VLMAnalysisStatus.SUCCESS
        if has_invalid:
            status = VLMAnalysisStatus.PARTIAL
        if not observations and events_raw:
            status = VLMAnalysisStatus.PARSE_FAILED

        report = VLMObservationReport(
            scene_summary=clean_description,
            observations=observations
        )

        return clean_description, ChunkAnalysisResult(
            analysis_status=status,
            parse_status="parsed_with_legacy_adapter" if not has_invalid else "partial_valid",
            legacy_adapter_used=True,
            repair_attempted=repair_attempted,
            repair_used=repair_attempted,
            repair_succeeded=repair_succeeded,
            report=report,
            analysis_id=analysis_id,
            chunk_id=chunk_id,
            model_call_id=model_call_id
        )

def parse_vlm_response(raw_content: str, model_call_id: Optional[str] = None, chunk_id: Optional[str] = None, analysis_id: Optional[str] = None) -> Tuple[str, ChunkAnalysisResult]:
    if not raw_content or not raw_content.strip():
        return "", ChunkAnalysisResult(
            analysis_status=VLMAnalysisStatus.MODEL_FAILED,
            parse_status="empty_content",
            legacy_adapter_used=False,
            model_call_id=model_call_id,
            chunk_id=chunk_id,
            analysis_id=analysis_id
        )
        
    # Yeni format henuz prompt'larda aktif olmadigi icin simdilik dogrudan legacy_adapter_used=True 
    # mantigina dusulmeden once bir pydantic VLMSceneObservation denemesi yapilabilir 
    # ama prompt'lar "EVENTS_JSON" uretiyor.
    if "EVENTS_JSON:" in raw_content:
        return LegacyAdapter.parse(raw_content, model_call_id=model_call_id, chunk_id=chunk_id, analysis_id=analysis_id)
        
    # Yeni format (saf JSON fallback veya gelecekteki format)
    try:
        data = json.loads(raw_content)
        # Eger basariliysa ve dict ise, VLMObservationReport dogrudan parse edilebilir.
        report = VLMObservationReport(**data)
        return report.scene_summary, ChunkAnalysisResult(
            analysis_status=VLMAnalysisStatus.SUCCESS if report.observations else VLMAnalysisStatus.SUCCESS_EMPTY,
            parse_status="success_typed",
            legacy_adapter_used=False,
            report=report,
            model_call_id=model_call_id,
            chunk_id=chunk_id,
            analysis_id=analysis_id
        )
    except Exception:
        # Ne yeni format, ne de EVENTS_JSON
        return raw_content.strip(), ChunkAnalysisResult(
            analysis_status=VLMAnalysisStatus.PARSE_FAILED,
            parse_status="unrecognized_format",
            legacy_adapter_used=False,
            model_call_id=model_call_id,
            chunk_id=chunk_id,
            analysis_id=analysis_id
        )
