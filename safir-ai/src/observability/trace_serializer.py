"""Pipeline trace olaylarini frontend-uyumlu, guvenli JSON'a ceviren serializer katmani.

Pipeline is mantigini (SafirPipeline stage'leri, VLM/agent/RAG kararlari) DEGISTIRMEZ.
Yalnizca `SafirPipeline.run(trace=...)` gozlem kancasindan gelen GERCEK objeleri
JSON-uyumlu bir `TraceEvent` sozlugune cevirir. Kurallar:

- Base64/`image_bytes` gibi buyuk goruntu verisi trace olayina KONULMAZ; bunun
  yerine kucuk bir referans (`frame_id` + `thumbnail_url`) uretilir ve gercek
  bayt, cagiran tarafindan (JobState) ayri tutulup frame endpoint'inden sunulur.
- Ajanin `raw_response`'u ve gizli/internal reasoning'i frontend'e VERILMEZ.
- Gemini'nin gercek HTTP govdesi / tam sistem promptu / ham orijinal yaniti
  mevcut trace'te olmadigi icin UYDURULMAZ.
"""

from __future__ import annotations

import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# Gercek pipeline stage sirasi (SafirPipeline.run icindeki emit sirasi ile ayni).
STAGE_ORDER: List[str] = ["sampler", "vlm", "events", "agent_context", "decision", "escalation", "report"]

# Yalnizca sunum (presentation) metadata'si — pipeline isimleri/isleyisi degismez.
STAGE_LABELS: Dict[str, str] = {
    "sampler": "Frame Sampling",
    "vlm": "Multimodal Analysis",
    "events": "Event Analysis",
    "agent_context": "Context & RAG",
    "decision": "Agent Decision",
    "escalation": "Risk Escalation",
    "report": "Final Report",
}

# Bir is icin bellek-ici tutulacak maksimum frame sayisi (base64 sisirmesini onler).
MAX_FRAMES_PER_JOB = 240

# Frontend'e serilestirilen bir asama olayinda ASLA bulunmamasi gereken alanlar
# (test bu anahtarlarin sizmadigini dogrular).
FORBIDDEN_KEYS = ("base64_image", "image_bytes", "raw_response")


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _thumb_url(job_id: str, frame_id: str) -> str:
    return f"/analyze/jobs/{job_id}/frames/{frame_id}"


def make_event(
    stage: str,
    status: str,
    summary: str,
    data: Dict[str, Any],
    *,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Transport/observation modeli: pipeline business logic'inden bagimsiz TraceEvent sozlugu.

    Args:
        stage: Ic stage anahtari (bkz. `STAGE_ORDER`).
        status: `queued` | `running` | `completed` | `failed`.
        summary: Kisa, kullaniciya gosterilebilir ozet.
        data: Stage'e ozgu, JSON-uyumlu, guvenli veri.
        duration_ms: Bu stage icin gecen sure (ms), varsa.
        error: `failed` durumunda kisa, guvenli hata mesaji (stack trace/secret YOK).
        metadata: Ek presentation metadata'si (label vb.).

    Returns:
        JSON-serilestirilebilir TraceEvent sozlugu.
    """
    meta = {"label": STAGE_LABELS.get(stage, stage), "order": STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1}
    if metadata:
        meta.update(metadata)
    return {
        "stage": stage,
        "status": status,
        "timestamp": _now_iso(),
        "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
        "summary": summary,
        "data": data,
        "metadata": meta,
        "error": error,
    }


# --- Stage serializer'lari: (payload) -> (summary, data, frames, status, error) ---------


def _bbox_to_list(bbox) -> Optional[List[int]]:
    return [int(x) for x in bbox] if bbox else None


def serialize_sampler(
    sampler_payload: Dict[str, Any], job_id: str
) -> Tuple[str, Dict[str, Any], Dict[str, bytes], str, Optional[str]]:
    """`sampler` emit'ini 'Frame Sampling' verisine cevirir (base64 haric).

    ONEMLI (mimari): Sampler artik hicbir olay kumelemesi YAPMAZ; bu yuzden
    burada "event_groups"/"representative_frames" YOKTUR - esik-gecmis TUM
    evidence kareleri (kumeleme VLM katmaninda yapilir) tek bir kronolojik
    listede sunulur.
    """
    evidence_frames = sampler_payload.get("evidence_frames", []) or []
    stats = sampler_payload.get("stats")

    frames: Dict[str, bytes] = {}

    def _add_frame(fid: str, raw: bytes) -> Optional[str]:
        if not raw or len(frames) >= MAX_FRAMES_PER_JOB:
            return None
        frames[fid] = raw
        return _thumb_url(job_id, fid)

    evidence_refs = []
    for ef in evidence_frames:
        fid = f"ev{ef.frame_id}"
        evidence_refs.append(
            {
                "evidence_id": ef.evidence_id,
                "frame_id": fid,
                "timestamp_sec": round(ef.timestamp_sec, 2),
                "timestamp_str": ef.timestamp_str,
                "change_score": round(ef.change_score, 4),
                "is_fallback": ef.is_fallback,
                "selection_reason": ef.selection_reason,
                "motion_bbox": _bbox_to_list(ef.motion_bbox),
                "thumbnail_url": _add_frame(fid, ef.image_bytes),
            }
        )

    stats_dict = (
        {
            "total_frames_scanned": stats.total_frames_scanned,
            "sampled_frames_evaluated": stats.sampled_frames_evaluated,
            "evidence_frame_count": stats.evidence_frame_count,
            "eliminated_frame_count": stats.eliminated_frame_count,
            "gpu_savings_ratio_pct": stats.eliminated_ratio_pct,
            "elapsed_sec": stats.elapsed_sec,
        }
        if stats
        else {}
    )
    summary = (
        f"{stats_dict.get('total_frames_scanned', 0)} kare -> "
        f"{len(evidence_frames)} kanit karesi (kumeleme VLM katmaninda yapilacak) "
        f"(%{stats_dict.get('gpu_savings_ratio_pct', 0)} GPU tasarrufu)"
    )
    data = {"stats": stats_dict, "evidence_frames": evidence_refs}
    return summary, data, frames, "completed", None


def serialize_vlm(
    payload: Dict[str, Any], job_id: str
) -> Tuple[str, Dict[str, Any], Dict[str, bytes], str, Optional[str]]:
    """`vlm` emit'ini guvenli alanlarla serialize eder (raw HTTP/tam sistem promptu UYDURULMAZ)."""
    vr = payload["vlm_response"]
    evidence_frames = payload.get("evidence_frames", []) or []
    # `status` (bkz. `VLMResponse`/`stage_vlm`), metin-tabanli `[HATA]` on-eki
    # kontrolunden DAHA GUVENILIRDIR (batch dagitiminda tum batch'ler
    # basarisiz olmasa bile aciklama metni degisebilir); yine de eski
    # cagiranlarla (status alani olmayan sahte nesneler) geriye-donuk uyumluluk
    # icin metin on-eki de yedek olarak kontrol edilir.
    vlm_status = getattr(vr, "status", None)
    degraded = vlm_status == "failed" or (vlm_status is None and vr.description.startswith("[HATA]"))
    partial_failure = vlm_status == "partial_failure"
    data = {
        "model_name": vr.model_name,
        "frame_count": vr.frame_count,
        "latency_ms": round(vr.latency_ms, 1),
        "user_prompt": payload.get("user_prompt", ""),
        "frames_sent": vr.frame_count or len(evidence_frames),
        "description": vr.description,           # temizlenmis insan-okur gozlem (mevcut, guvenli)
        "structured_events": vr.structured_events,  # parse edilmis EVENTS_JSON (mevcut, guvenli; evidence_ids dahil)
        "vlm_status": vlm_status or ("failed" if degraded else "completed"),
    }
    if degraded:
        summary = f"{vr.model_name}: VLM basarisiz (degraded)"
        return summary, data, {}, "failed", "VLM analizi basarisiz — pipeline degraded modda devam etti."
    if partial_failure:
        summary = f"{vr.model_name}: bazi olaylar analiz edilemedi (partial_failure)"
        return summary, data, {}, "completed", None
    summary = f"{vr.model_name}: {len(vr.structured_events)} olay, {vr.latency_ms:.0f} ms"
    return summary, data, {}, "completed", None


def serialize_events(
    payload: Dict[str, Any], job_id: str
) -> Tuple[str, Dict[str, Any], Dict[str, bytes], str, Optional[str]]:
    """`events` emit'ini serialize eder (olay listesi + zaman + confidence + kurallar)."""
    detected = payload.get("detected_events", []) or []
    temporal = payload.get("temporal_events", []) or []
    rules = payload.get("rule_matches", []) or []
    data = {
        "detected_events": [
            {
                "event_type": d.event_type,
                "timestamp": round(d.timestamp, 2),
                "confidence": round(d.confidence, 2),
                "matched_keywords": d.matched_keywords,
            }
            for d in detected
        ],
        "temporal_events": [
            {
                "event_type": t.event_type,
                "occurrence_count": t.occurrence_count,
                "duration": round(t.duration, 2),
                "start_timestamp": round(t.start_timestamp, 2),
                "end_timestamp": round(t.end_timestamp, 2),
                "confidence": round(t.confidence, 2),
            }
            for t in temporal
        ],
        "rule_matches": [
            {"rule_id": r.rule_id, "rule_description": r.rule_description, "severity": r.severity, "event_type": r.event_type}
            for r in rules
        ],
    }
    summary = f"{len(detected)} olay tespit edildi, {len(rules)} ISG kurali tetiklendi"
    return summary, data, {}, "completed", None


def serialize_agent_context(
    payload: Dict[str, Any], job_id: str
) -> Tuple[str, Dict[str, Any], Dict[str, bytes], str, Optional[str]]:
    """`agent_context` emit'ini serialize eder (yalnizca mevcut `prompt_block` metni)."""
    prompt_block = payload.get("prompt_block", "") or ""
    data = {"prompt_block": prompt_block, "length": len(prompt_block)}
    summary = f"Ajan baglami hazirlandi ({len(prompt_block)} karakter)"
    return summary, data, {}, "completed", None


def serialize_decision(
    payload: Dict[str, Any], job_id: str
) -> Tuple[str, Dict[str, Any], Dict[str, bytes], str, Optional[str]]:
    """`decision` emit'ini GUVENLI alanlarla serialize eder (`raw_response` DAHIL EDILMEZ)."""
    d = payload["decision"]
    risk_status = getattr(d, "risk_status", "assessed")
    data = {
        "risk_score": d.risk_score,
        "risk_level": d.risk_level,
        "risk_status": risk_status,
        "summary": d.summary,
        "recommended_action": d.recommended_action,
        "actions": d.actions,
        "events": d.events,
    }
    if risk_status == "unknown" or d.risk_score is None:
        summary = "Risk BELIRSIZ (analiz guvenilir sekilde tamamlanamadi)"
    else:
        summary = f"Risk {d.risk_level.upper()} ({d.risk_score}/100)"
    return summary, data, {}, "completed", None


def serialize_escalation(
    payload: Dict[str, Any], job_id: str
) -> Tuple[str, Dict[str, Any], Dict[str, bytes], str, Optional[str]]:
    """`escalation` emit'ini serialize eder (kademe + otomatik tetik + alert_id + gerekce)."""
    e = payload["escalation"]
    data = {
        "tier": e.tier.value,
        "auto_dispatched": e.auto_dispatched,
        "alert_id": e.alert_id,
        "reason": e.reason,
    }
    auto = " (alarm OTOMATIK tetiklendi)" if e.auto_dispatched else ""
    summary = f"Eskalasyon: {e.tier.value}{auto}"
    return summary, data, {}, "completed", None


def serialize_report(
    payload: Dict[str, Any], job_id: str
) -> Tuple[str, Dict[str, Any], Dict[str, bytes], str, Optional[str]]:
    """`report` emit'ini base64 icermeyen kompakt bir ozete cevirir (tam rapor polling endpoint'inde)."""
    r = payload["report"]
    risk_status = getattr(r, "risk_status", "assessed")
    data = {
        "event_id": r.event_id,
        "risk_score": r.risk_score,
        "risk_level": r.risk_level,
        "risk_status": risk_status,
        "escalation_tier": r.escalation_tier,
        "auto_dispatched": r.auto_dispatched,
        "alert_id": r.alert_id,
        "detected_event_types": r.detected_event_types,
        "vlm_model": r.vlm_model,
        "llm_model": r.llm_model,
        "timeline": [{"timestamp": round(e.timestamp, 2), "description": e.description} for e in r.timeline],
        "sartname_json": r.to_sartname_json(),
    }
    if risk_status == "unknown" or r.risk_score is None:
        summary = "Final rapor uretildi — risk BELIRSIZ (manuel inceleme gerekli)"
    else:
        summary = f"Final rapor uretildi — risk {r.risk_level.upper()} ({r.risk_score}/100)"
    return summary, data, {}, "completed", None


class PipelineTraceCollector:
    """`SafirPipeline.run(trace=...)` icin cagrilabilir toplayici.

    Her stage emit'ini serialize eder, sure hesaplar, goruntu baytlarini
    `on_frames` ile ayri verir, olayi `on_event` ile disari akitir. Pipeline
    mantigina dokunmaz.
    """

    _SERIALIZERS: Dict[str, Callable] = {
        "sampler": serialize_sampler,
        "vlm": serialize_vlm,
        "events": serialize_events,
        "agent_context": serialize_agent_context,
        "decision": serialize_decision,
        "escalation": serialize_escalation,
        "report": serialize_report,
    }

    def __init__(
        self,
        job_id: str,
        on_event: Callable[[Dict[str, Any]], None],
        on_frames: Callable[[Dict[str, bytes]], None],
        clock: Callable[[], float] = None,
    ) -> None:
        import time

        self._job_id = job_id
        self._on_event = on_event
        self._on_frames = on_frames
        self._clock = clock or time.perf_counter
        self._last = self._clock()

    def __call__(self, stage: str, payload: Dict[str, Any]) -> None:
        serializer = self._SERIALIZERS.get(stage)
        if serializer is None:
            return
        summary, data, frames, status, error = serializer(payload, self._job_id)
        self._flush(stage, summary, data, frames, status, error)

    def _flush(self, stage, summary, data, frames, status, error) -> None:
        now = self._clock()
        duration_ms = (now - self._last) * 1000.0
        self._last = now
        if frames:
            self._on_frames(frames)
        self._on_event(make_event(stage, status, summary, data, duration_ms=duration_ms, error=error))
