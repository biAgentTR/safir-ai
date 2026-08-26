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
#
# `decision_final` KOK NEDEN DUZELTMESI (risk explainability P0): `main.py::run()`
# `_emit("decision", ...)` (Agent'in KENDI taslak risk_score'u, orn. 90) VE
# ayrica `_emit("decision_final", ...)` (RuleEngine'in EZDIGI, nihai/authoritative
# risk_score, orn. 88) uretiyordu - ama bu STAGE_ORDER/`_SERIALIZERS`de
# "decision_final" hic TANIMLI DEGILDI; `PipelineTraceCollector.__call__`
# taninmayan bir stage icin SESSIZCE `return` ediyordu (asagida). Sonuc: 90
# gorunuyordu, 88'e NEDEN/NASIL donustugunu acikca soyleyen olay trace'e HIC
# ULASMIYORDU - operator icin "90 mi 88 mi authoritative?" sorusu koddan
# CEVAPLANAMAZ hale geliyordu. Bu artik "decision" ile "escalation" arasina
# eklenerek (gercek emit sirasiyla BIREBIR ayni) duzeltildi.
STAGE_ORDER: List[str] = [
    "sampler",
    "vlm",
    "events",
    "rag_security",
    "decision",
    "decision_final",
    "escalation",
    "report",
]
"""2026-08-25: "agent_context" (eski "Baglam ve RAG") stage'i KALDIRILDI - bu asama
sadece `prompt_block` metnini VE RuleEngine'in deterministik `relevant_regulations`
eslesmesini (kisa/dahili referans etiketleri, GERCEK skorlu semantik RAG telemetrisi
DEGIL) goruyordu; ayni ekranda hemen ardindan gelen "rag_security" ("RAG ve Guvenlik
Telemetrisi") zaten GERCEK, skorlu/reranked semantik RAG sonucunu gosteriyor - iki
ayri panel operatoru "hangisi gercek RAG?" sorusuyla kafasi karisik birakiyordu.
`stage_context()`in kendisi (prompt_block/context uretimi, RuleEngine eslesmesi,
gercek semantik RAG sorgusu) pipeline icinde HALA calisir - yalnizca AYRI bir trace
paneli olarak GORUNMESI kaldirildi (bkz. `src/main.py::run()`)."""

# Yalnizca sunum (presentation) metadata'si — pipeline isimleri/isleyisi degismez.
STAGE_LABELS: Dict[str, str] = {
    "sampler": "Frame Sampling",
    "vlm": "Multimodal Analysis",
    "events": "Event Analysis",
    "rag_security": "RAG & Security Telemetry",
    "decision": "Agent Decision (draft)",
    "decision_final": "Final Risk (RuleEngine-authoritative)",
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


_VLM_PROGRESS_SUMMARY: Dict[str, str] = {
    "chunking": "Video {total_chunks} parçaya bölünüyor",
    "chunk_start": "Parça {chunk_index}/{total_chunks} gönderiliyor {range_label}",
    "chunk_done": "Parça {chunk_index}/{total_chunks} tamamlandı {range_label} ({elapsed_sec:.0f}s)",
    "chunk_failed": "Parça {chunk_index}/{total_chunks} başarısız {range_label} ({elapsed_sec:.0f}s)",
}


def _summarize_vlm_progress(progress: Dict[str, Any]) -> str:
    """Adim-adim VLM ilerleme olayi icin kisa, kullaniciya-gosterilebilir Turkce ozet uretir.

    `stage_vlm`in `on_progress` kancasindan gelen ham `dict`i (bkz.
    `src/vlm/evren_vlm.py::VlmProgressCallback`) bicimlendirir; bilinmeyen bir
    `phase` icin (gelecekte eklenebilecek yeni bir faz) HAM veriyi UYDURMADAN
    genel bir "işleniyor" metnine duser - operatore YANLIS/eksik bir sayi
    gostermek yerine.
    """
    phase = progress.get("phase", "")
    template = _VLM_PROGRESS_SUMMARY.get(phase)
    if template is None:
        return "VLM analizi işleniyor"
    fields = {
        "total_chunks": progress.get("total_chunks", "?"),
        "chunk_index": progress.get("chunk_index", "?"),
        "range_label": progress.get("range_label") or "",
        "elapsed_sec": progress.get("elapsed_sec") or 0.0,
    }
    try:
        return template.format(**fields).strip()
    except (KeyError, ValueError):
        return "VLM analizi işleniyor"


def serialize_vlm(
    payload: Dict[str, Any], job_id: str
) -> Tuple[str, Dict[str, Any], Dict[str, bytes], str, Optional[str]]:
    """`vlm` emit'ini guvenli alanlarla serialize eder (raw HTTP/tam sistem promptu UYDURULMAZ).

    Ara "adim-adim ilerleme" emit'leri (bkz. `src/main.py::run`in `_on_vlm_progress`
    kancasi) `payload["progress"]` tasir - nihai `vlm_response` HENUZ YOKTUR,
    bu yuzden asagidaki normal (tamamlanmis) yol calismadan ONCE ayriliyor.
    Bu olaylar HER ZAMAN `status="running"` doner - operator video parcalanip
    gonderilirken canli ilerleme gorur, ama bu heniz stage'in NIHAI sonucu
    degildir (asil "completed"/"failed" olay, `vlm_response` geldiginde AYRICA
    ve HER ZAMAN yayinlanir).
    """
    if "progress" in payload:
        progress = payload["progress"] or {}
        return _summarize_vlm_progress(progress), {"progress": progress}, {}, "running", None

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
                "event_name": d.event_name,
                "event_type": d.event_type,
                "timestamp": round(d.timestamp, 2),
                "confidence": round(d.confidence, 2),
                "matched_keywords": d.matched_keywords,
            }
            for d in detected
        ],
        "temporal_events": [
            {
                "event_name": t.event_name,
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


_MAX_REASON_CHARS = 200
"""Guard'in kisa gerekce metni icin azami karakter sayisi - ham injection payload'u
DEGIL, Gemini'nin KISA siniflandirma gerekcesi (bkz. GuardResult.reason); yine
de operatorun goreceginden emin olmak icin defansif olarak kirpilir."""


def serialize_rag_security(
    payload: Dict[str, Any], job_id: str
) -> Tuple[str, Dict[str, Any], Dict[str, bytes], str, Optional[str]]:
    """`rag_security` emit'ini serialize eder: RAG retrieval + Prompt Injection Guard telemetrisi.

    GUVENLIK: API anahtari/secret ASLA bu stage'e KONULMAZ. `query`/chunk
    `text` alanlari ONCEDEN kasitli olarak disaridaydi (yalnizca metadata/skor);
    2026-08-24 RAG PIPELINE RECONSTRUCTION'da (gorev tanimi 13. bolum) operator'un
    "bu madde neden secildi?" sorusunu trace'ten cevaplayabilmesi icin BILEREK
    eklendi - `query`, VLM'in ZATEN `events` stage'inde gorunen matched_keywords/
    aciklamasindan turetilir (yeni bir bilgi sizdirmiyor); chunk `text`i, ZATEN
    ACIK/resmi mevzuat metnidir (gizli/kisisel veri degil). `rag_telemetry`/
    `guard_results` yoksa (RAG/Guard bu cagrida hic calismadiysa) ilgili alan
    `None`/bos liste olarak birakilir - UYDURULMUS bir deger KONULMAZ.
    """
    rag_telemetry = payload.get("rag_telemetry")
    guard_results = payload.get("guard_results") or []
    relevance_weights = payload.get("relevance_weights")

    rag_data: Optional[Dict[str, Any]] = None
    if rag_telemetry is not None:
        rag_data = {
            "query": rag_telemetry.query,
            "query_length": len(rag_telemetry.query),
            "candidate_count": rag_telemetry.candidate_count,
            "final_count": rag_telemetry.final_count,
            "zero_result": rag_telemetry.zero_result,
            "retrieval_status": rag_telemetry.retrieval_status,
            "corpus_source": getattr(rag_telemetry, "corpus_source", "unseeded"),
            "threshold": rag_telemetry.threshold,
            "embedding_latency_ms": rag_telemetry.embedding_latency_ms,
            "rerank_latency_ms": rag_telemetry.rerank_latency_ms,
            "total_latency_ms": rag_telemetry.total_latency_ms,
            "avg_embedding_score": rag_telemetry.avg_embedding_score,
            "avg_relevance_score": rag_telemetry.avg_relevance_score,
            # RAG RERANKER DETERMINIZATION: bu iki alan SABITTIR - relevance
            # skorlamasinin ARTIK bir LLM'e SORULMADIGINI operator/UI icin
            # ACIKCA belirtir (bkz. `deterministic_reranker.py`).
            "reranker": "deterministic",
            "relevance_method": "weighted_hybrid",
            # 2026-08-24 (explainability - desktop Vue UI'nin GERCEKTEN okudugu
            # canli trace payload'u): agirliklar + Cross-Encoder durumu bu
            # sorgunun GENELINE ait (tek deger, satir basina DEGIL) - koddan
            # (config'ten okunmus `RelevanceWeights`) tasinir, UYDURULMAZ.
            "relevance_weights": (
                {
                    "semantic": relevance_weights.semantic,
                    "lexical": relevance_weights.lexical,
                    "keyword": relevance_weights.keyword,
                    "metadata": relevance_weights.metadata,
                    "phrase": relevance_weights.phrase,
                }
                if relevance_weights is not None
                else None
            ),
            "cross_encoder_status": getattr(rag_telemetry, "cross_encoder_status", "disabled"),
            "results": [
                {
                    "rank": getattr(r, "rank", None),
                    "final_rank": getattr(r, "final_rank", None),
                    "chunk_id": getattr(r, "chunk_id", None),
                    "document_id": r.document_id,
                    "document_title": r.document_title,
                    "article_number": r.article_number,
                    "source_url": r.source_url,
                    "embedding_score": r.embedding_score,
                    "relevance_score": r.relevance_score,
                    "semantic_score": getattr(r, "semantic_score", None),
                    "lexical_score": getattr(r, "lexical_score", None),
                    "keyword_score": getattr(r, "keyword_score", None),
                    "metadata_score": getattr(r, "metadata_score", None),
                    "phrase_score": getattr(r, "phrase_score", None),
                    "cross_encoder_score": getattr(r, "cross_encoder_score", None),
                    "relevance_status": getattr(r, "relevance_status", None),
                    "relevance_reason": getattr(r, "relevance_reason", None),
                    "selected": r.selected,
                    "text": getattr(r, "text", ""),
                }
                for r in rag_telemetry.results
            ],
        }

    security_data = [
        {
            "source": g.source,
            "is_injection": g.is_injection,
            "confidence": g.confidence,
            "action": g.action,
            "reason": (g.reason[:_MAX_REASON_CHARS] if g.reason else None),
            "guard_failed": g.guard_failed,
            "latency_ms": g.latency_ms,
        }
        for g in guard_results
    ]

    data = {"rag": rag_data, "security": security_data}

    if rag_telemetry is None:
        rag_summary = "RAG sorgusu yapilmadi (keyword yok)"
    elif rag_telemetry.retrieval_status == "reranker_unavailable":
        # GERIYE-UYUMLULUK: bu deger artik `EmbeddingRAGService.query()`
        # tarafindan URETILMEZ (relevance skorlama ARTIK bir LLM/API'ye
        # bagli DEGIL - bkz. `deterministic_reranker.py`, gorev tanimi 8.
        # bolum) - ama GECMIS (bu degisiklikten ONCEKI) persisted trace
        # kayitlarinda hala GORULEBILIR; bu dal SADECE o eski kayitlarin
        # anlamli gosterilmesi icin KORUNUR.
        rag_summary = (
            f"RAG retrieval yapildi ancak reranker kullanilamadi "
            f"({rag_telemetry.candidate_count} aday bulundu, reranker basarisiz oldugu icin "
            "hicbiri dogrulanmadi — GUVENLIK GEREGI 0 sonuc donduruldu, embedding siralamasi "
            "SESSIZCE final sonuc olarak sunulmadi) [GECMIS KAYIT - artik uretilmiyor]"
        )
    elif rag_telemetry.retrieval_status == "insufficient_evidence":
        # Deterministik relevance skorlama GERCEKTEN calisti (LLM/API
        # basarisizligi YOK) ama HICBIR aday `score_threshold`u gecemedi -
        # bu "retrieval calismadi" ile KARISTIRILMAMALI: adaylar bulundu ve
        # skorlandi, sadece hicbiri yeterince alakali degildi.
        rag_summary = (
            f"RAG: yetersiz kanit ({rag_telemetry.candidate_count} aday skorlandi, "
            f"hicbiri threshold'u gecemedi) - relevance skorlama basariyla calisti, LLM/API hatasi YOK"
        )
    elif getattr(rag_telemetry, "corpus_source", None) == "fallback_placeholder":
        rag_summary = (
            f"RAG: {rag_telemetry.final_count}/{rag_telemetry.candidate_count} sonuc "
            f"({rag_telemetry.retrieval_status}) — UYARI: GERCEK mevzuat corpus'u YOK, "
            "8 maddelik PLACEHOLDER'dan geliyor"
        )
    else:
        rag_summary = (
            f"RAG: {rag_telemetry.final_count}/{rag_telemetry.candidate_count} sonuc "
            f"({rag_telemetry.retrieval_status})"
        )
    quarantined = sum(1 for g in guard_results if g.action == "quarantine")
    guard_summary = f"{len(guard_results)} guvenlik kontrolu ({quarantined} quarantine)" if guard_results else "Guard devre disi"
    summary = f"{rag_summary}; {guard_summary}"
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


def serialize_decision_final(
    payload: Dict[str, Any], job_id: str
) -> Tuple[str, Dict[str, Any], Dict[str, bytes], str, Optional[str]]:
    """`decision_final` emit'ini serialize eder: RuleEngine'in EZDIGI (veya degistirmeden biraktiği) nihai risk.

    Bkz. `STAGE_ORDER` yorumu: bu stage'in trace'e HIC ULASMAMASI, "90 vs 88"
    aciklanabilirlik P0'inin kok nedeniydi. `data["risk_provenance"]`,
    `risk_resolver.RiskProvenance`den (LLM'e SORULMADAN, deterministik)
    turer - hangi kural(lar)in bu karari urettigini acikca gosterir.
    """
    d = payload["decision"]
    provenance = payload.get("risk_provenance")
    risk_status = getattr(d, "risk_status", "assessed")

    provenance_data = None
    if provenance is not None:
        provenance_data = {
            "risk_source": "rule_engine" if provenance.rule_ids else None,
            "rule_ids": provenance.rule_ids,
            "rule_severities": provenance.rule_severities,
            "contributing_event_ids": provenance.contributing_event_ids,
            "explanation": provenance.explanation(),
            # RISK ENGINE V2 (2026-08-24): eski sabit-bucket skorlamanin yerini alan
            # matematiksel modelin TAM izlenebilirligi - "88 nereden geldi?" sorusu
            # artik yalnizca HANGI kural degil, HANGI feature'lardan/agirliklardan
            # geldigini de gosterir (bkz. `risk_model.py`).
            "scoring_method": provenance.scoring_method,
            "final_score": provenance.final_score,
            "feature_values": provenance.features,
            "feature_contributions": provenance.feature_contributions,
            "llm_proposed_score": provenance.llm_proposed_score,
            "regulatory_evidence_ids": provenance.regulatory_evidence_ids,
        }

    data = {
        "risk_score": d.risk_score,
        "risk_level": d.risk_level,
        "risk_status": risk_status,
        "risk_provenance": provenance_data,
    }

    if risk_status == "unknown" or d.risk_score is None:
        summary = "Nihai risk BELIRSIZ (analiz guvenilir sekilde tamamlanamadi)"
    elif provenance_data and provenance_data["risk_source"] == "rule_engine":
        llm_bit = (
            f", Agent'in taslak tahmini={provenance_data['llm_proposed_score']} (DIKKATE ALINMADI)"
            if provenance_data.get("llm_proposed_score") is not None
            else ""
        )
        summary = (
            f"NIHAI (authoritative) risk: {d.risk_level.upper()} ({d.risk_score}/100, "
            f"scoring_method={provenance_data.get('scoring_method')}) "
            f"- kaynak: RuleEngine [{', '.join(provenance_data['rule_ids'])}]{llm_bit}"
        )
    else:
        summary = (
            f"NIHAI (authoritative) risk: {d.risk_level.upper()} ({d.risk_score}/100) "
            "- hicbir kural eslesmedi, Agent'in kendi tahmini KORUNDU (dogrulanmamis)"
        )
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
        "detected_event_names": r.detected_event_names,
        "detected_event_types": r.detected_event_types,
        "events": [
            {
                "event_name": ev.event_name,
                "event_type": ev.event_type,
                "keywords": ev.keywords,
                "risk_level": ev.risk_level,
                "risk_score": ev.risk_score,
            }
            for ev in r.events
        ],
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
        "rag_security": serialize_rag_security,
        "decision": serialize_decision,
        "decision_final": serialize_decision_final,
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
