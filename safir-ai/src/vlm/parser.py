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

# Modeller, "markdown kullanma" talimatina ragmen JSON'u cogu zaman bir kod
# citi (```json ... ```) icine sarar. Bu citi temizlemeden `json.loads`
# cagirmak, GECERLI bir cevabi "unrecognized_format" sayip olaylari SESSIZCE
# dusurur (ve sistem sabit anahtar-kelime fallback'ine iner) - bkz.
# `_extract_json_object`.
_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(raw_content: str) -> Optional[str]:
    """Model ciktisindan typed JSON nesnesini (varsa) cikarir.

    Sirayla dener: (1) metnin tamami zaten bir JSON nesnesi mi, (2) bir kod
    citi (```json ... ```) icinde mi, (3) metnin icinde ilk `{` ile son `}`
    arasinda kalan blok. Hicbiri tutmazsa `None` doner ve cagiran taraf
    `unrecognized_format` yoluna gider.

    Args:
        raw_content: Modelin ham metin ciktisi.

    Returns:
        Ayristirilmaya aday JSON metni veya `None`.
    """
    stripped = raw_content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced = _CODE_FENCE_PATTERN.search(raw_content)
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate.startswith("{"):
            return candidate

    start = raw_content.find("{")
    end = raw_content.rfind("}")
    if start != -1 and end > start:
        return raw_content[start : end + 1]
    return None


def observations_to_structured_events(
    chunk_res: ChunkAnalysisResult,
    chunk_offset_sec: float = 0.0,
    evidence_timestamps: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Typed `VLMSceneObservation` listesini `EventEngine`in bekledigi dict listesine cevirir.

    NEDEN TEK BIR YERDE (regresyon notu): bu esleme daha once
    `base_vlm._post_chat_completion` icinde SATIR ICI yapiliyordu ve modelin
    urettigi `attributes`/`entities`/`canonical_type` alanlarini DISARIDA
    BIRAKIYORDU. Sonuc: `EventEngine`e giden her olayin `keywords`u BOS
    geliyor, bu yuzden `DetectedEvent.matched_keywords` de bos kaliyor ve
    sistem sabit `_KEYWORD_RULES` taksonomisine (10 kategori) geri dusuyordu -
    yani VLM'in KENDI urettigi serbest anahtar kelimeler hicbir zaman
    kullanilmiyordu. Ayrica `matched_keywords` bos oldugunda semantik RAG
    sorgusu HIC atilmaz (bkz. `SafirPipeline`), dolayisiyla mevzuat
    getirme de sessizce devre disi kaliyordu.

    Bu fonksiyon, esleme mantigini TEK bir yerde toplar ve VLM'in kendi
    kelimelerini KORUR; hicbir sabit liste UYGULANMAZ, hicbir terim
    UYDURULMAZ (model bir sey uretmediyse liste bos kalir).

    Args:
        chunk_res: `parse_vlm_response` ciktisi.
        chunk_offset_sec: Bu parcanin orijinal videodaki baslangic saniyesi
            (bolunmemis video icin 0.0).
        evidence_timestamps: Kare-tabanli yolda `evidence_id -> saniye`
            eslemesi; zaman normalizasyonuna aktarilir (yoksa `None`).

    Returns:
        `EventEngine._parse_structured_events`in tukettigi dict listesi.
    """
    from src.vlm.time_normalizer import normalize_observation_time

    if not chunk_res.report or not chunk_res.report.observations:
        return []

    structured_events: List[Dict[str, Any]] = []
    all_invalid = True
    for obs in chunk_res.report.observations:
        norm = normalize_observation_time(obs, chunk_offset_sec, evidence_timestamps)
        if norm.time_status == "invalid":
            chunk_res.analysis_status = VLMAnalysisStatus.PARTIAL
            continue
        all_invalid = False

        # VLM'in KENDI urettigi serbest terimler: `attributes` (ör. "baretsiz",
        # "hizli hareket") once, `entities` (ör. "isci", "forklift") sonra;
        # sira korunur, tekrarlar elenir. Sabit bir sozluge BAKILMAZ.
        keywords: List[str] = []
        for term in list(obs.attributes or []) + list(obs.entities or []):
            cleaned = str(term).strip()
            if cleaned and cleaned not in keywords:
                keywords.append(cleaned)

        structured_events.append(
            {
                "event_name": obs.observed_label,
                # `canonical_event_type`: modelin bilinen taksonomiye
                # eslestirdigi tip (eslestirmediyse None - ZORLANMAZ).
                "canonical_event_type": obs.canonical_type,
                "taxonomy_status": getattr(obs.taxonomy_status, "value", obs.taxonomy_status),
                "keywords": keywords,
                "description": obs.observed_label,
                "confidence": obs.confidence,
                "start_time": norm.global_start_sec,
                "end_time": norm.global_end_sec,
                "evidence_ids": obs.evidence,
                "uncertainties": list(obs.uncertainties or []),
                "normalized_relative_start_sec": norm.normalized_relative_start_sec,
                "normalized_relative_end_sec": norm.normalized_relative_end_sec,
                "was_adjusted": norm.was_adjusted,
                "adjustment_reasons": norm.adjustment_reasons,
                "time_status": norm.time_status,
                "time_base": norm.time_base,
            }
        )

    if all_invalid and chunk_res.report.observations:
        chunk_res.analysis_status = VLMAnalysisStatus.PARTIAL
        chunk_res.parse_status = "all_times_invalid"

    return structured_events

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
                
                # VLM'in EVENTS_JSON blogunda urettigi SERBEST anahtar
                # kelimeler `attributes`e tasinir (onceden tamamen
                # DUSURULUYORDU - bkz. `observations_to_structured_events`
                # regresyon notu). `canonical_event_type` de korunur; model
                # bilinen taksonomiye eslestirmediyse None kalir, hicbir tip
                # ZORLANMAZ.
                raw_keywords = raw.get("keywords") or []
                attributes = [
                    str(kw).strip() for kw in raw_keywords if isinstance(kw, (str, int)) and str(kw).strip()
                ]
                canonical = raw.get("canonical_event_type") or raw.get("type")
                obs = VLMSceneObservation(
                    observed_label=str(raw.get("event_name") or raw.get("type", "unknown")),
                    canonical_type=str(canonical).strip().lower() if canonical else None,
                    confidence=float(raw.get("confidence", 0.8)),
                    relative_start_sec=start,
                    relative_end_sec=end,
                    attributes=attributes,
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
        
    # Yeni (typed) format. Modeller "markdown kullanma" talimatina ragmen
    # JSON'u siklikla ```json citi icine sarar veya oncesine/sonrasina kisa
    # bir cumle ekler; ham metni DOGRUDAN `json.loads`a vermek bu gecerli
    # cevaplari "unrecognized_format" sayip TUM olaylari sessizce dusururdu
    # (ve sistem sabit anahtar-kelime taksonomisine geri duserdi).
    candidate = _extract_json_object(raw_content)
    if candidate is None:
        return raw_content.strip(), ChunkAnalysisResult(
            analysis_status=VLMAnalysisStatus.PARSE_FAILED,
            parse_status="unrecognized_format",
            legacy_adapter_used=False,
            model_call_id=model_call_id,
            chunk_id=chunk_id,
            analysis_id=analysis_id
        )
    try:
        data = json.loads(candidate)
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
