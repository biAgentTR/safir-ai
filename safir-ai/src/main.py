"""SAFIR uctan uca pipeline giris noktasi (FastAPI servisi).

Akis: Video Input -> AdaptiveFrameSampler (CPU) -> VLM (vLLM) -> ContextBuilder
(SQLite + FAISS RAG) -> SafirAgent (LangGraph, vLLM) -> SafirReport (JSON).

Operator paneli (`src/ui/dashboard.py`) icin canli ilerleme takibi
`/analyze/jobs` uzerinden asenkron (arka plan thread'i + durum sorgulama)
olarak saglanir; `/analyze` ise tek seferlik senkron kullanim icin korunur.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
import logging
import os
from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import re
import threading
import time
import urllib.parse
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from src.vlm.schemas import ChunkAnalysisResult, VLMAnalysisStatus
from typing import Any, Literal, Callable, Deque, Dict, List, Literal, Optional, Tuple

import cv2

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.observability.trace_serializer import MAX_FRAMES_PER_JOB, PipelineTraceCollector

from src.agent.langgraph_agent import SafirAgent
from src.agent.tools import RetrieverTool
from src.assistant.ask_service import AskService, JobNotFound
from src.assistant.suggestion_engine import build_dynamic_suggestions
from src.decision.escalation import EscalationPolicy, FieldAlarmDispatcher
from src.event_analysis.event_builder import EventBuilder
from src.event_analysis.event_engine import EventEngine
from src.event_analysis.event_history import EventHistory
from src.event_analysis.risk_resolver import RiskProvenance, resolve_deterministic_risk_with_provenance
from src.event_analysis.rule_engine import RuleEngine
from src.event_analysis.schemas import DetectedEvent, EventEngineInput, RuleMatch, TemporalEvent
from src.event_analysis.temporal_reasoner import DEFAULT_RELATION_WINDOW_SEC, TemporalReasoner
from src.event_analysis.event_merger import EventMerger
from src.memory.analysis_store import AnalysisStore
from src.memory.context_builder import ContextBuilder
from src.memory.conversation_store import ConversationStore
from src.memory import document_extraction
from src.rag.embedding_rag_service import EmbeddingRAGService, MockEmbeddingRAGService
from src.memory.event_store import EventStore
from src.sampler.adaptive_sampler import EvidenceFrame, sampler_from_config
from src.security.prompt_injection_guard import build_prompt_injection_guard

from src.schemas.report import EventSummary, EvidenceFrameOut, RagContext, SafirReport, SamplerStats, TimelineEntry
from src.utils.config_loader import SafirConfig, load_config
from src.vlm.base_vlm import BaseVLM, VLMResponse
from src.vlm.factory import get_llm_client, get_vlm_client
from src.vlm.sudden_event_analyzer import analyze_sudden_events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAFIR", description="Saha Analiz ve Farkindalik Karar Sistemi API'si")

_DATA_DIR = "data"

# Operator panelinde gosterilecek 3 asamali ilerleme adlari (sira onemli).
STAGE_SAMPLER = ("VLM Oncesi Katman (Adaptive Frame Sampler)", 1, 3)
STAGE_VLM = ("Gorsel Dil Modeli (vLLM Gorsel Anlama)", 2, 3)
STAGE_AGENT = ("LangGraph Ajan (RAG & Karar)", 3, 3)

OnStageCallback = Callable[[str, int, int], None]

# Her ana asamanin GERCEK ara ciktisini disari veren gozlem (trace) geri cagirmasi.
# `(asama_adi, payload_sozlugu)` alir; yalnizca gozlem/gorunurluk icindir
# (orn. Jupyter demo), pipeline davranisini DEGISTIRMEZ. Varsayilan None -> sifir maliyet.
TraceCallback = Callable[[str, Dict[str, object]], None]

# `stage_vlm` icin ADIM-ADIM (video parcalanirken/gonderilirken) ilerleme
# kancasi - bkz. `src/vlm/evren_vlm.py::VlmProgressCallback`. `run()` bunu
# `TraceCallback`e (stage="vlm", status="running") baglayarak operatore
# canli gorunurluk saglar; varsayilan None -> sifir maliyet.
VlmProgressCallback = Callable[[Dict[str, object]], None]

# `TemporalEvent.end_timestamp`i bu pipeline cagrisinin `latest_timestamp`ina
# esitlerken kullanilan tolerans (kayan nokta karsilastirmasi icin).
_CURRENT_CALL_TIMESTAMP_TOLERANCE = 1e-6


def _prune_stale_events(
    buffer: "Deque[DetectedEvent]", latest_timestamp: float, retention_sec: float
) -> None:
    """Buffer'daki, en yeni zaman damgasindan `retention_sec`den daha eski olaylari yerinde (in-place) atar.

    `SafirPipeline._event_history_buffer`, pipeline cagrilari arasinda
    SIFIRLANMAZ (Temporal Reasoning'in gecmis baglama ihtiyaci vardir); bu
    fonksiyon, buffer'in saatlerce calisan bir sistemde sinirsiz buyumesini
    zaman bazli bir budama ile engeller. Sayi bazli bir `maxlen` yerine zaman
    bazli budama secilmistir; boylece VLM cagri sikligindan bagimsiz olarak
    `TemporalReasoner`in ihtiyac duyabilecegi tum gecmis (varsayilan
    `relation_window_sec=30s`in birkaç kati) her zaman buffer'da kalir.

    Args:
        buffer: `EventEngine.detect(...)` ciktilarinin biriktigi, zaman
            sirali (soldan sagizli - en eski solda) `deque`.
        latest_timestamp: Bu pipeline cagrisinin gozlem zaman damgasi.
        retention_sec: Bu zaman damgasindan ne kadar eski olaylarin
            tutulacagi (saniye); daha eskiler atilir.
    """
    cutoff = latest_timestamp - retention_sec
    while buffer and buffer[0].timestamp < cutoff:
        buffer.popleft()


def _select_current_call_events(
    temporal_events: List[TemporalEvent],
    latest_timestamp: float,
    current_model_call_ids: set[str],
    current_chunk_ids: set[str],
    current_analysis_id: Optional[str],
    current_evidence_ids: set[str],
    allow_legacy_timestamp_fallback: bool = False,
) -> List[TemporalEvent]:
    """Bu pipeline cagrisinda uretilen/guncellenen TUM `TemporalEvent`leri secer.

    BUG FIX (T021): eskiden BUTUN `TemporalEvent`lerin bu cagriya ait
    olup olmadigi TEK BIR ORTAK `end_timestamp == latest_timestamp` esitligiyle
    belirleniyordu - bu, VLM'in TEK bir cagride, FARKLI bitis zamanlarina
    sahip BIRDEN FAZLA olay urettigi (gercek Gemini ciktisinda son derece
    yaygin - orn. "00:06-00:22" ve "00:38-01:15") durumlarda YANLIS calisiyordu:
    yalnizca EN GEC biten olay `latest_timestamp`a esit oluyor, digerleri
    SESSIZCE ELENIYORDU (bkz. `SafirReport.events`in bos donmesi hatasi).

    Duzeltme: bir `TemporalEvent`, ONCELIKLE bu cagrinin `detected_events`
    (`stage_events` ciktisi) ile PAYLASTIGI en az bir `evidence_id` varsa
    "bu cagriya ait" sayilir (kesin, evidence-tabanli baglanti - zaman
    damgasi esitligine bagli DEGILDIR). `evidence_ids` olmayan
    (structured VLM ciktisi degil, eski anahtar-kelime fallback yolundan
    gelen) olaylar icin ESKI zaman damgasi-tolerans karsilastirmasina
    GERI DUSULUR (bu yolun evidence_id'si zaten hep bostur).

    Args:
        temporal_events: `TemporalReasoner.reason(...)` ciktisi (tum buffer
            uzerinden hesaplanmis, bu cagriya ozel olmayan tam liste).
        latest_timestamp: Bu pipeline cagrisinin gozlem zaman damgasi.
        detected_events: Bu cagrinin `EventEngine.detect(...)` ciktisi
            (`stage_events`); evidence-tabanli eslesme icin kullanilir.
            `None`/bos ise dogrudan zaman damgasi-tolerans karsilastirmasina
            duser (geriye uyumluluk).

    Returns:
        Bu cagriya ait `TemporalEvent`lerin, guvene gore azalan sirali
        listesi. Bos olabilir (teorik olarak; `EventEngine` her zaman en az
        bir `DetectedEvent` urettigi icin pratikte bos donmez).
    """
    def _belongs_to_current_call(te: TemporalEvent) -> bool:
        # Guvenlik Kontrolu: Eger olay baska bir analysis ID'ye aitse KESINLIKLE reddet!
        if te.source_analysis_ids:
            if not current_analysis_id:
                return False
            if current_analysis_id not in te.source_analysis_ids:
                return False

        # 1. Model Call ID kesisimi (Kesin)
        if current_model_call_ids and te.source_model_call_ids:
            if set(te.source_model_call_ids) & current_model_call_ids:
                return True
                
        # 2. Chunk ID kesisimi
        if current_chunk_ids and te.source_chunk_ids:
            if set(te.source_chunk_ids) & current_chunk_ids:
                return True
                
        # 3. Evidence ID kesisimi (Eski kare modu veya fallback)
        if current_evidence_ids and te.evidence_ids:
            if set(te.evidence_ids) & current_evidence_ids:
                return True
                
        # 4. Geriye uyumluluk (Yalnizca opt-in)
        if allow_legacy_timestamp_fallback and not te.source_model_call_ids and not te.source_chunk_ids and not te.evidence_ids:
            import logging
            logging.getLogger(__name__).warning("Legacy timestamp fallback is being used!")
            return abs(te.end_timestamp - latest_timestamp) <= _CURRENT_CALL_TIMESTAMP_TOLERANCE
            
        return False

    current_call_events = [te for te in temporal_events if _belongs_to_current_call(te)]
    
    # Siralama sorumlulugu: Selector yalnizca uyelik secer, gizli siralamayi iptal ediyoruz
    # Siralama veya kronolojik order'i baska yerde yapacagiz veya burada saf kronolojik dondurecegiz.
    # Prompt: "Timeline kronolojik siralanmali. Selector icinde gizli confidence sort bulunmamali."
    current_call_events.sort(key=lambda te: te.start_timestamp)
    return current_call_events


def _summarize_rule_matches(rule_matches: List[RuleMatch]) -> str:
    """Tetiklenen `RuleMatch`leri, ajan istemine eklenebilecek kisa bir madde listesine cevirir.

    Args:
        rule_matches: `RuleEngine.evaluate(...)` ciktisi (bu cagriya kadar
            biriken tum buffer uzerinden hesaplanmis).

    Returns:
        Her satiri bir kural olan madde listesi; `rule_matches` bossa bos string.
    """
    if not rule_matches:
        return ""
    return "\n".join(f"- [{match.rule_id}] ({match.severity}) {match.rule_description}" for match in rule_matches)


# VLM'in urettigi SERBEST BICIMLI risk keyword'lerini (orn. "fire_detected"),
# semantik RAG sorgusunu IYILESTIRMEK icin (yalnizca bu amacla - event
# siniflandirmasini veya risk skorunu ETKILEMEZ, bkz. RAG gorev tanimi 8.
# bolum) Turkce ifadelerle genisleten KUCUK, elle-derlenmis harita. Bilinen
# bir keyword YOKSA, alt cizgiyi bosluga cevirerek generic bir genisletme
# yapilir (uydurma anlam eklenmez).
_KEYWORD_QUERY_EXPANSIONS: Dict[str, str] = {
    "fire_detected": "yangın tespit edildi, yangın riski, açık alev",
    "smoke_detected": "duman tespit edildi, duman ve yangın riski",
    "unattended_fire": "gözetimsiz yangın, kontrolsüz yangın",
    "unattended_fire_hazard": "gözetimsiz yangın tehlikesi",
    "uncontrolled_open_flame": "kontrolsüz açık alev",
    "indoor_fire_risk": "kapalı alanda yangın riski",
    "person_without_helmet": "baret kullanılmaması, baş koruyucu KKD ihlali",
    "person_without_safety_vest": "yansıtıcı yelek kullanılmaması, KKD ihlali",
    "ppe_violation": "kişisel koruyucu donanım kullanılmaması",
    "forklift_person_collision_risk": "forklift ve yaya çarpışma riski, araç-yaya yakınlığı",
    "person_in_forklift_path": "yayanın forklift güzergahında bulunması",
    "fallen_person": "yerde hareketsiz kişi, düşme sonrası kaza",
    "unauthorized_presence_in_critical_area": "yetkisiz kişinin kritik alanda bulunması",
}


def _expand_keyword_for_query(keyword: str) -> str:
    """Bir VLM risk keyword'unu (bilinen bir esleme varsa) Turkce ifadeye genisletir; yoksa alt cizgiyi bosluga cevirir."""
    return _KEYWORD_QUERY_EXPANSIONS.get(keyword, keyword.replace("_", " "))


def _aggregate_matched_keywords(temporal_events: List) -> List[str]:
    """Bu cagriya ait `TemporalEvent`lerin `matched_keywords`lerini, ilk-gorulme sirali ve tekrarsiz olarak birlestirir."""
    seen: List[str] = []
    for event in temporal_events:
        for keyword in getattr(event, "matched_keywords", None) or []:
            if keyword not in seen:
                seen.append(keyword)
    return seen


def _build_semantic_query(matched_keywords: List[str], vlm_description: str, max_description_chars: int = 200) -> Optional[str]:
    """VLM'in dinamik risk keyword'lerinden (+ kisa VLM aciklamasindan) semantik RAG sorgusu kurar.

    Args:
        matched_keywords: `_aggregate_matched_keywords(...)` ciktisi.
        vlm_description: VLM'in serbest metin gozlemi (baglam icin kisaltilarak eklenir).
        max_description_chars: `vlm_description`den sorguya eklenecek azami karakter sayisi.

    Returns:
        Bos degilse sorgu metni; `matched_keywords` bossa `None` (bu durumda
        semantik RAG cagrisi HIC YAPILMAZ - keyword yoksa sorgulanacak bir
        risk sinyali de yoktur, bos/rastgele bir sorgu UYDURULMAZ).
    """
    if not matched_keywords:
        return None
    expanded = [_expand_keyword_for_query(k) for k in matched_keywords]
    short_description = vlm_description.strip()[:max_description_chars]
    parts = expanded + ([short_description] if short_description else [])
    return ". ".join(parts)


_CITATION_PATTERNS = [
    # "<Isim> Yonetmeligi/Kanunu/Tuzugu Madde <no>" (orn. "Is Ekipmanlari Yonetmeligi Madde 12").
    re.compile(
        r"([A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü\s]{2,60}?(?:Yönetmeliği|Kanunu|Tüzüğü))\s+Madde\s+([IVXLCM\d]+(?:\.\d+)*)"
    ),
    # "<Isim> Talimati <KOD>" (orn. "Yangin Guvenligi Talimati YG-03").
    re.compile(r"([A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü\s]{2,50}?Talimat[ıi])\s+([A-ZÇĞİÖŞÜ]{1,6}-\d+)"),
]
"""Ajanin SERBEST METIN ciktisinda (summary/actions) gecebilecek, mevzuat-atfi
GIBI GORUNEN kaliplari yakalayan, KASITLI DAR regex'ler (bkz.
`_extract_potential_citations`/`_unverified_citations`, gorev tanimi 10. bolum).
Bu, TAM bir NLP atif-cikarma sistemi DEGILDIR - yalnizca gorev tanimindaki somut
ornekle (orn. 'Yangin Guvenligi Talimati YG-03') AYNI kaliptaki, kolayca
UYDURULABILECEK referanslari YAKALAMAK icin yeterince spesifiktir; eslesmeyen
serbest metin SESSIZCE gecilir (yanlis-pozitif UYDURULMAZ)."""


def _extract_potential_citations(text: str) -> List[Tuple[str, str, str]]:
    """Serbest metinden mevzuat-atfi GIBI GORUNEN kaliplari cikarir.

    Returns:
        `(tam_eslesme_metni, dokuman_adi, madde_veya_kod)` uclulerinin listesi.
    """
    matches: List[Tuple[str, str, str]] = []
    for pattern in _CITATION_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m.group(0), m.group(1).strip(), m.group(2).strip()))
    return matches


def _unverified_citations(text: str, semantic_rag_sources: List) -> List[str]:
    """Ajanin serbest metninde gecen mevzuat-benzeri atiflardan, GERCEKTEN RETRIEVED EVIDENCE

    ile eslesmeyenleri dondurur (gorev tanimi 10. bolum: "corpus lookup -> bulunamadi ->
    source_verified=false"). Eslesme, dokuman adinin (kucuk harfe cevrilmis, bosluklar
    sadelestirilmis) evidence'in `rule_title`iyle KISMEN ortusmesi + (varsa) madde/kod
    numarasinin BIREBIR ayni olmasi uzerinden yapilir - deterministik, LLM'e SORULMAZ.

    Args:
        text: Ajanin serbest metni (orn. `decision.summary` + tum `decision.actions`).
        semantic_rag_sources: Bu cagrida GERCEKTEN retrieved olan `RagContext` listesi
            (bkz. `build_report`) - yalnizca BUNLARA gore dogrulama yapilir (corpus'un
            TAMAMINA degil, bu sorgunun GERCEKTEN getirdigi kanitlara gore).

    Returns:
        Dogrulanamamis (evidence'ta karsiligi bulunamayan) atiflarin TAM metni
        (orijinal, kirpilmamis) - tekrarsiz, ilk-gorulme sirali.
    """
    if not text:
        return []

    def _normalize(s: str) -> str:
        return " ".join(s.lower().split())

    evidence_titles = [_normalize(getattr(s, "rule_title", "") or "") for s in semantic_rag_sources]
    evidence_articles = {
        (_normalize(getattr(s, "rule_title", "") or ""), _normalize(getattr(s, "article_number", "") or ""))
        for s in semantic_rag_sources
    }

    unverified: List[str] = []
    seen: set = set()
    for full_match, doc_name, article_or_code in _extract_potential_citations(text):
        if full_match in seen:
            continue
        seen.add(full_match)
        norm_doc = _normalize(doc_name)
        norm_article = _normalize(article_or_code)
        if not any(norm_doc in t or t in norm_doc for t in evidence_titles):
            unverified.append(full_match)
            continue
        # Dokuman adi (kismen) eslesti - madde/kod numarasi da BIREBIR eslesiyor mu?
        article_matches = any((norm_doc in t or t in norm_doc) and a == norm_article for (t, a) in evidence_articles)
        if not article_matches:
            unverified.append(full_match)

    return unverified


_WINDOWS_ABS_PATH_RE = re.compile(r"^[a-zA-Z]:[\\/]")


def _is_absolute_path(path: str) -> bool:
    """POSIX (`/...`), Windows suruculu (`C:\\...`/`C:/...`) veya UNC (`\\\\sunucu\\pay`) mutlak yolu tanir.

    `os.path.isabs` host isletim sistemine gore davranir (ornegin Linux
    backend'de `C:\\...` mutlak SAYILMAZ); bu fonksiyon host'tan BAGIMSIZ
    olarak her iki bicimi de taniyip mutlak kabul eder.
    """
    if os.path.isabs(path):
        return True
    if _WINDOWS_ABS_PATH_RE.match(path):
        return True
    if path.startswith("\\\\"):
        return True
    return False


def normalize_video_source(video_source: str) -> str:
    """`video_source` degerini, canli yayin URI'lerini ve MUTLAK yollari koruyarak normalize eder.

    Operator panelinden veya harici istemcilerden gelen deger su sekilde ele alinir:
      - RTSP/HTTP(S) canli yayin adresleri: OLDUGU GIBI birakilir.
      - MUTLAK yol (POSIX `/...`, Windows `C:\\...`/`C:/...`, UNC `\\\\sunucu\\pay`):
        OLDUGU GIBI KORUNUR — kullanicinin isaret ettigi GERCEK dosya okunur.
        (Onceden bu durumda da dosya adi ayiklanip `data/<ad>` altina
        yonlendiriliyordu; bu, ayni dosya adina sahip FARKLI videolarin
        sunucuda CAKISIP yanlislikla ayni/eski dosyanin okunmasina yol
        aciyordu — bkz. kok neden analizi, Hata #1.)
      - BAGIL yol veya yalnizca dosya adi: MEVCUT (degismemis) davranis
        korunur — dosya adi ayiklanip `data/<dosya_adi>` seklinde yeniden
        yazilir; boylece konteyner icindeki `./data` bind mount'una gore
        calisma DEVAM eder.

    Args:
        video_source: `/analyze` istegindeki ham `video_source` degeri.

    Returns:
        Canli yayin URI'si veya mutlak yol ise degistirilmeden; bagil/dosya-adi
        girdiyse `data/<dosya_adi>` seklinde normalize edilmis yol.
    """
    lowered = video_source.strip().lower()
    if lowered.startswith(("rtsp://", "http://", "https://")):
        return video_source

    stripped = video_source.strip()
    if _is_absolute_path(stripped):
        return stripped

    normalized_slashes = stripped.replace("\\", "/")
    filename = os.path.basename(normalized_slashes)
    return os.path.join(_DATA_DIR, filename)


class AnalyzeRequest(BaseModel):
    """`/analyze` ve `/analyze/jobs` uc noktalari icin ortak istek govdesi."""

    video_source: str
    user_prompt: str = "Sahnede riskli bir durum var mi degerlendir."
    analysis_mode: Literal["vlm_direct", "low_budget"] = "vlm_direct"
    """Operator panelindeki iki BAGIMSIZ analiz sistemi (bkz. `useAnalysisMode.ts`):
    "vlm_direct" video kaynagini DOGRUDAN, TEK istekte VLM'e gonderir (`vlm.
    active_model`, ör. EVREN "vlm"); "low_budget" Adaptive Frame Sampler'i
    calistirip urettigi kareleri `vlm.frames_model`e (ör. EVREN "llm-large",
    <=2 goruntu/istek) gonderir (bkz. `SafirPipeline.run`/`stage_vlm`).
    Varsayilan "vlm_direct" - eski istemciler (bu alani hic gondermeyen)
    davranisi TAMAMEN AYNI kalir."""
    sample_fps: Optional[int] = Field(
        default=None, ge=1, le=10, description="Operator panelinden gelen ornekleme FPS override'i (1-10)."
    )
    min_change_threshold: Optional[float] = Field(
        default=None,
        ge=0.001,
        le=0.050,
        description="Operator panelinden gelen hassasiyet esigi override'i (0.001-0.050).",
    )


class AlertTriggerRequest(BaseModel):
    """`/alerts/trigger` icin istek govdesi (operatorun manuel/override alarm tetiklemesi).

    NOT: Yuksek/kritik risk artik pipeline tarafindan OTOMATIK tetiklenir; bu uc
    nokta yalnizca operatorun otomatik akis disinda manuel alarm baslatmasi
    (override) icindir.
    """

    risk_score: int
    risk_level: str
    recommended_action: str
    operator_note: str = ""


class AlertTriggerResponse(BaseModel):
    """`/alerts/trigger` yaniti."""

    acknowledged: bool
    alert_id: str
    message: str


class AlertAcknowledgeRequest(BaseModel):
    """`/alerts/{alert_id}/acknowledge` icin istek govdesi (Human-on-the-Loop denetimi)."""

    operator_note: str = ""


class AlertAcknowledgeResponse(BaseModel):
    """`/alerts/{alert_id}/acknowledge` yaniti."""

    alert_id: str
    acknowledged: bool
    message: str


class FeedbackRequest(BaseModel):
    """`/events/{event_id}/feedback` icin istek govdesi (Human-in-the-Loop dogrulamasi)."""

    feedback: str = Field(description="'true_positive' veya 'false_positive'.")


class FeedbackResponse(BaseModel):
    """`/events/{event_id}/feedback` yaniti."""

    event_id: int
    feedback: str
    message: str


@dataclass
class JobState:
    """Bir `/analyze/jobs` isinin canli durumu (asama, sonuc, hata, trace)."""

    status: str = "queued"                    # queued | running | done | error
    stage_name: str = ""
    step: int = 0
    total_steps: int = 3
    result: Optional[SafirReport] = None
    error: Optional[str] = None
    # 08 - Gozlemlenebilirlik (SSE): pipeline'in GERCEK ara asama ciktilarinin
    # frontend-uyumlu, base64 icermeyen serilestirilmis olaylari. `frames`,
    # trace olaylarinin referans verdigi kucuk JPEG baytlarini bellek-ici tutar
    # (frame endpoint'i buradan sunar; base64 SSE payload'una girmez).
    trace_events: List[dict] = field(default_factory=list)
    frames: Dict[str, bytes] = field(default_factory=dict)
    current_stage: str = ""


class JobStatusResponse(BaseModel):
    """`/analyze/jobs/{job_id}` yaniti."""

    status: str
    stage_name: str
    step: int
    total_steps: int
    result: Optional[SafirReport] = None
    error: Optional[str] = None


class AnalyzeJobResponse(BaseModel):
    """`/analyze/jobs` (POST) yaniti."""

    job_id: str


class HistoryListItem(BaseModel):
    """`GET /history` liste ogesi (History listesi icin gereken ozet alanlar)."""

    job_id: str
    created_at: str
    updated_at: str
    video_source: Optional[str] = None
    status: str
    risk_level: Optional[str] = None
    risk_score: Optional[int] = None
    risk_status: str = "assessed"
    summary: Optional[str] = None


class HistoryDetail(BaseModel):
    """`GET /history/{job_id}` yaniti: metadata + kalici `SafirReport` (varsa)."""

    job_id: str
    created_at: str
    updated_at: str
    status: str
    video_source: Optional[str] = None
    report: Optional[SafirReport] = None
    trace_events: List[Dict[str, Any]] = Field(default_factory=list)
    """Kalici trace olay listesi (varsa); `PipelineTraceCollector` tarafindan
    canli analiz sirasinda uretilenle AYNI `TraceEvent` sozlugu semasini tasir
    (bkz. `src/observability/trace_serializer.py::make_event`). Bu analiz icin
    hic kaydedilmemisse (eski kayit veya persistence basarisiz olduysa) bos liste."""


class AskRequest(BaseModel):
    """`POST /ask` istek govdesi."""

    question: str = Field(min_length=1, description="Kullanicinin dogal dil sorusu.")
    job_id: Optional[str] = Field(default=None, description="Opsiyonel: baglam alinacak analiz kimligi.")
    use_video: bool = Field(
        default=False,
        description=(
            "True ve job_id verilmisse, soru ONCE dogrudan videoya sorulur (EVREN prefix-cache "
            "avantaji); basarisiz olursa sessizce metin-tabanli akisa doner (bkz. AskService)."
        ),
    )

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Soru bos olamaz.")
        return v


class AskSource(BaseModel):
    """`POST /ask` yanitindaki tek bir kaynak (yalnizca backend'in verdigi alanlar)."""

    type: str  # "analysis" | "regulation"
    text: Optional[str] = None
    score: Optional[float] = None
    label: Optional[str] = None


class AskResponse(BaseModel):
    """`POST /ask` yaniti: dayanakli cevap + kaynaklar + kullanilan baglam."""

    answer: str
    sources: List[AskSource] = Field(default_factory=list)
    job_id: Optional[str] = None
    context_used: List[str] = Field(default_factory=list)


class AskSuggestionsResponse(BaseModel):
    """`GET /ask/suggestions` yaniti: bu rapora OZGU takip sorusu onerileri (bkz. suggestion_engine.py)."""

    suggestions: List[str] = Field(default_factory=list)


# --- SAFIR Asistan: sohbet gecmisi (Conversation) - yalnizca UI/gecmis icindir,
# LLM'e GERI BESLENMEZ (bkz. src/assistant/ask_service.py modul dokustringi). ---


class ConversationCreateRequest(BaseModel):
    """`POST /conversations` istek govdesi."""

    title: Optional[str] = Field(default=None, description="Kisa baslik (genelde ilk sorudan kirpilir).")
    job_id: Optional[str] = Field(default=None, description="Bu sohbetin bagli oldugu analiz kimligi (opsiyonel).")


class ConversationSummary(BaseModel):
    """Sohbet listesi/ozet gorunumu (mesajlar HARIC)."""

    conversation_id: str
    created_at: str
    updated_at: str
    title: Optional[str] = None
    job_id: Optional[str] = None
    message_count: int = 0


class ConversationMessageOut(BaseModel):
    """Tek bir sohbet mesaji (`GET`/`POST /conversations/.../messages` yanitlarinda)."""

    id: int
    role: str  # "user" | "assistant"
    content: str
    created_at: str


_MAX_CONTEXT_CONTENT_LEN = 4000
"""Kullanici tarafindan eklenen tek bir baglam (not) icin azami karakter
sayisi. Sinirsiz metin -> sinirsiz prompt buyumesi/token maliyeti demek;
operator notu icin bu sinir fazlasiyla yeterlidir (bkz. Adim 3 istegi:
'uzun metinler icin makul bir limit')."""
_MAX_CONTEXT_LABEL_LEN = 80


class ConversationContextOut(BaseModel):
    """Bir sohbete eklenmis tek bir ek baglam kaydi (`POST`/`GET .../context`)."""

    id: int
    kind: str
    label: Optional[str] = None
    content: str
    created_at: str


class ConversationDocumentOut(BaseModel):
    """Bir sohbete yuklenmis tek bir belgenin metadata + durumu (Adim 4). Ham dosya icerigi ASLA doner."""

    document_id: str
    filename: str
    file_ext: str
    file_size_bytes: int
    page_count: Optional[int] = None
    status: str  # "processing" | "ready" | "error"
    error_message: Optional[str] = None
    created_at: str


class ConversationDetailResponse(ConversationSummary):
    """`GET /conversations/{id}` yaniti: ozet + tam mesaj gecmisi + ek baglamlar + belgeler."""

    messages: List[ConversationMessageOut] = Field(default_factory=list)
    context: List[ConversationContextOut] = Field(default_factory=list)
    documents: List[ConversationDocumentOut] = Field(default_factory=list)


class ConversationMessageCreateRequest(BaseModel):
    """`POST /conversations/{id}/messages` istek govdesi."""

    role: str = Field(description="'user' veya 'assistant'.")
    content: str = Field(min_length=1)


class ConversationContextCreateRequest(BaseModel):
    """`POST /conversations/{id}/context` istek govdesi (Adim 3: yalnizca metin/not)."""

    content: str = Field(min_length=1, max_length=_MAX_CONTEXT_CONTENT_LEN)
    label: Optional[str] = Field(default=None, max_length=_MAX_CONTEXT_LABEL_LEN)


def _naive_concat_batches(batch_responses: List[VLMResponse]) -> VLMResponse:
    """Reconciliation VLM cagrisi BASARISIZ olursa kullanilan, kayipsiz-ama-DEGRADE fallback.

    Batch-yerel olaylari, `event_id` cakismasini onlemek icin `b{batch_index}_`
    on-ekiyle birlestirir (GERCEK global birlestirme yapmaz - ayni gercek olay
    iki batch'te ayri kayit olarak kalabilir); yine de HICBIR evidence_id
    kaybolmaz.

    Args:
        batch_responses: `BaseVLM.analyze_evidence_batched` ciktisi (en az bir
            basarili batch icermelidir).

    Returns:
        On-ekli `event_id`lerle birlestirilmis, tek bir `VLMResponse`.
    """
    succeeded = [r for r in batch_responses if r.status != "failed"]
    structured_events: List[Dict[str, Any]] = []
    description_parts: List[str] = []
    for i, r in enumerate(succeeded):
        for ev in r.structured_events:
            stamped = dict(ev)
            stamped["event_id"] = f"b{i}_{stamped.get('event_id', 'e')}"
            structured_events.append(stamped)
        description_parts.append(r.description)
    return VLMResponse(
        description="\n\n".join(description_parts),
        model_name=succeeded[0].model_name if succeeded else batch_responses[0].model_name,
        frame_count=sum(r.frame_count for r in batch_responses),
        latency_ms=sum(r.latency_ms for r in batch_responses),
        structured_events=structured_events,
    )


def _reconcile_unassigned_evidence(
    structured_events: List[Dict[str, Any]],
    evidence_frames: List[EvidenceFrame],
    batch_responses: List[VLMResponse],
) -> List[Dict[str, Any]]:
    """VLM'in evidence_ids atamalarini DOGRULAR ve atanamayan evidence'i acikca korur.

    Iki kurali uygular (bkz. gorev tanimi):
    1. "Uydurma veya kayip evidence ID kabul etme": her olayin `evidence_ids`si,
       gercekten gonderilen `EvidenceFrame.evidence_id` kumesiyle KESISTIRILIR;
       gecersiz/uydurma kimlikler sessizce (yalnizca log ile) ATILIR. Bir
       evidence_id BIRDEN FAZLA olaya atanmissa yalnizca ILK atama korunur
       (cakisma onlenir).
    2. "Atanamayan evidence'lari unassigned/uncertain olarak koru": hicbir
       olaya atanmamis (VLM tarafindan atlanmis VEYA batch'i tamamen basarisiz
       olmus) evidence, mevcut bir `unassigned`/`siniflandirilamadi` kaydiyla
       BIRLESTIRILIR (yoksa yeni bir tane olusturulur). Evidence hicbir zaman
       SESSIZCE kaybolmaz.

    Args:
        structured_events: VLM'in (reconciliation sonrasi veya tek-batch)
            urettigi ham olay listesi.
        evidence_frames: Bu calisma icin gonderilen TUM evidence kareleri
            (gecerli evidence_id kumesinin ve zaman damgalarinin kaynagi).
        batch_responses: `analyze_evidence_batched` ciktisi (hangi
            evidence_id'lerin hangi batch'e ait oldugunu ve basarisiz
            batch'leri belirlemek icin).

    Returns:
        `evidence_ids` alanlari dogrulanmis/temizlenmis, gerekirse bir
        `unassigned` kaydi eklenmis/genisletilmis olay listesi.
    """
    valid_ids_in_order = [ef.evidence_id for ef in evidence_frames]
    valid_ids = set(valid_ids_in_order)
    by_id = {ef.evidence_id: ef for ef in evidence_frames}

    assigned: set = set()
    cleaned_events: List[Dict[str, Any]] = []
    for ev in structured_events:
        cleaned = dict(ev)
        kept_ids: List[str] = []
        for raw_id in cleaned.get("evidence_ids") or []:
            eid = str(raw_id)
            if eid not in valid_ids:
                logger.warning("VLM gecersiz/uydurma evidence_id uretti, atlaniyor: %r", eid)
                continue
            if eid in assigned:
                logger.warning("VLM evidence_id'yi birden fazla olaya atamis, ilk atama korunuyor: %r", eid)
                continue
            kept_ids.append(eid)
            assigned.add(eid)
        cleaned["evidence_ids"] = kept_ids
        cleaned_events.append(cleaned)

    leftover_ids = [eid for eid in valid_ids_in_order if eid not in assigned]
    if leftover_ids:
        leftover_frames = [by_id[eid] for eid in leftover_ids]
        existing_unassigned = next(
            (
                e
                for e in cleaned_events
                if e.get("event_id") == "unassigned" or e.get("canonical_event_type") == "siniflandirilamadi"
            ),
            None,
        )
        if existing_unassigned is not None:
            existing_unassigned["evidence_ids"] = sorted(set(existing_unassigned.get("evidence_ids", [])) | set(leftover_ids))
        else:
            cleaned_events.append(
                {
                    "event_id": "unassigned",
                    # `event_name`/`canonical_event_type` - GERCEK VLM olaylariyla
                    # AYNI sema (bkz. prompts/vlm_prompts.py EVENTS_JSON semasi);
                    # eski `"type"` alani hicbir zaman var olmayan bir anahtardi
                    # (frontend'in VlmEventList.vue "Tur" kolonu hep bunu okuyup
                    # hep bos donuyordu - bkz. desktop/app/types/api.ts VlmEvent).
                    "event_name": "siniflandirilamadi",
                    "canonical_event_type": "siniflandirilamadi",
                    "start_time": min(f.timestamp_sec for f in leftover_frames),
                    "end_time": max(f.timestamp_sec for f in leftover_frames),
                    "evidence_ids": leftover_ids,
                    "description": "Bu evidence kareleri herhangi bir olaya atanamadi (belirsiz veya analiz edilemedi).",
                    "risk_score": None,
                    "confidence": 0.0,
                }
            )
        logger.info(
            "Atanamayan %d evidence karesi 'unassigned/siniflandirilamadi' olarak korundu: %s",
            len(leftover_ids),
            leftover_ids,
        )

    return cleaned_events


def _select_report_evidence_frames(
    vlm_response: VLMResponse, evidence_frames: List[EvidenceFrame]
) -> List[EvidenceFrameOut]:
    """Rapor/UI icin, VLM'in kumeledigi HER olay icin bir temsili evidence karesi secer.

    Her olay icin, o olaya atanmis `evidence_ids` arasindan en yuksek
    `change_score`e sahip kare secilir (video YENIDEN ACILMAZ, kare yeniden
    KODLANMAZ - zaten bellekteki `EvidenceFrame.base64_image` kullanilir).
    `evidence_ids` bos/gecersiz olan (ör. hicbir evidence karesi eslesmeyen)
    olaylar atlanir.

    Args:
        vlm_response: `stage_vlm` ciktisi (`structured_events` dogrulanmis/
            temizlenmis olmalidir - bkz. `_reconcile_unassigned_evidence`).
        evidence_frames: Bu calisma icin gonderilen TUM evidence kareleri.

    Returns:
        Her VLM olayi icin bir `EvidenceFrameOut` (olay sirasinda).
    """
    evidence_by_id = {ef.evidence_id: ef for ef in evidence_frames}
    report_frames: List[EvidenceFrameOut] = []
    for ev in vlm_response.structured_events:
        candidates = [evidence_by_id[eid] for eid in (ev.get("evidence_ids") or []) if eid in evidence_by_id]
        if not candidates:
            continue
        best = max(candidates, key=lambda f: f.change_score)
        report_frames.append(
            EvidenceFrameOut(
                event_id=str(ev.get("event_id", "")),
                timestamp_sec=best.timestamp_sec,
                timestamp_str=best.timestamp_str,
                change_score=best.change_score,
                base64_image=best.base64_image,
                saved_path=best.saved_path,
                is_fallback=best.is_fallback,
            )
        )
    return report_frames


def _extract_vlm_direct_evidence_frames(
    video_source: str, structured_events: List[Dict[str, object]], jpeg_quality: int = 85
) -> List[EvidenceFrameOut]:
    """VLM Direct icin kanit karelerini, Adaptive Frame Sampler'DAN BAGIMSIZ,
    dogrudan kaynak videodan uretir.

    VLM Direct modunda (`analysis_mode="vlm_direct"`) sampler hic calismaz
    (bkz. `SafirPipeline.run` yorumu) - dolayisiyla `_select_report_evidence_
    frames` her zaman BOS doner (eslesecek `EvidenceFrame` yok). EVREN'in
    donusu (`structured_events`) da hicbir kare goruntusu icermez, yalnizca
    zaman damgasi/aciklama tasir. Bu yuzden HER VLM olayi icin, `cv2.
    VideoCapture` ile videoda dogrudan o saniyeye ATLANARAK tek bir kare
    gercekten yakalanir (ffmpeg KULLANILMAZ - kare cikarimi tamamen OpenCV
    ile, ek bir surec/CLI cagrisi olmadan yapilir) ve JPEG/base64'e kodlanir
    - boylece rapor/PDF/HTML/JSON kanit karesi olarak GERCEK, o olaya ait bir
    goruntu icerir (onceki, dusuk-butceli sisteme ait sampler kareleri
    DEGIL).

    Hangi saniye? Olayin ORTALAMASI/orta noktasi DEGIL - `start_time`in
    KENDISI: VLM prompt'u (`src/prompts/vlm_prompts.py`, "Olay zaman
    damgasini HER ZAMAN olayin ILK BASLADIGI saniyeye (start_time) gore ver")
    modele HER olay icin zaman damgasini olayin GERCEKTEN basladigi ani
    bildirmesini talimatlandirir; bu yuzden kanit karesi de modelin
    raporladigi O TAM saniyeden alinir - rapordaki metnin ("Başlangıç: MM:SS")
    anlattigi anla, gosterilen goruntu BIREBIR eslesir.

    Canli yayin (RTSP) kaynaklarinda veya kare yakalanamayan durumlarda o
    olay sessizce atlanir (rapor cokusu yerine eksik kanit karesi tercih
    edilir).
    """
    if not structured_events:
        return []
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.warning("VLM Direct kanit karesi uretilemedi: video acilamadi (%s)", video_source)
        return []
    try:
        report_frames: List[EvidenceFrameOut] = []
        for ev in structured_events:
            timestamp_sec = float(ev.get("start_time") or 0.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                continue
            base64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
            minutes, seconds = divmod(int(timestamp_sec), 60)
            report_frames.append(
                EvidenceFrameOut(
                    event_id=str(ev.get("event_id", "")),
                    timestamp_sec=round(timestamp_sec, 2),
                    timestamp_str=f"{minutes:02d}:{seconds:02d}",
                    change_score=0.0,
                    base64_image=f"data:image/jpeg;base64,{base64_str}",
                    saved_path=None,
                    is_fallback=False,
                )
            )
        return report_frames
    finally:
        cap.release()


class SafirPipeline:
    """Tum SAFIR katmanlarini tek bir uctan uca akista birlestiren orkestrator."""

    def __init__(self, config: SafirConfig) -> None:
        """Pipeline'i konfigurasyondan tum alt sistemleri kurarak baslatir.

        Args:
            config: `load_config()` ile uretilmis dogrulanmis `SafirConfig`.
        """
        self._config = config
        self._default_sample_fps = config.sampler.sample_fps
        # 2026-08-26 (IKI BAGIMSIZ ANALIZ MODU): operator panelindeki "VLM
        # Direct"/"Dusuk Butceli" secimi ONCEDEN sadece GORSELDI - her iki UI
        # modu da AYNI (tek) `self._vlm`i kullaniyordu (`config.vlm.
        # active_model`). Artik `AnalyzeRequest.analysis_mode` ile GERCEKTEN
        # secilen iki AYRI, BAGIMSIZ `BaseVLM` orneği kuruluyor - `run()`/
        # `stage_vlm` bu ikisinden HANGISININ kullanilacagina `analysis_mode`a
        # gore karar verir (bkz. asagidaki metotlar).
        #
        # Mock modda (`use_mock_vlm=true`) BILEREK TEK bir `MockVLMClient`
        # orneği hem `_vlm_video` hem `_vlm_frames` icin PAYLASILIR (iki AYRI
        # `MockVLMClient()` cagrisi YAPILMAZ): mock istemcinin `analyze_video`/
        # `analyze_evidence`/`requires_frame_sampling`i test tarafindan
        # monkeypatch'lenebilir kalmasi icin - `self._vlm` (asagida) geriye-
        # donuk uyumluluk icin bu paylasilan orneğe (gercek moddaysa
        # `_vlm_video`ya) isaret eder (bkz. `tests/test_pipeline_integration.py`
        # gibi `pipeline._vlm`i DOGRUDAN patch'leyen testler).
        if config.app.use_mock_vlm:
            shared_mock_vlm = get_vlm_client(config.vlm, use_mock=True)
            self._vlm_video = shared_mock_vlm
            self._vlm_frames = shared_mock_vlm
        else:
            self._vlm_video = get_vlm_client(config.vlm, use_mock=False)
            self._vlm_frames = get_vlm_client(
                config.vlm.model_copy(update={"active_model": config.vlm.frames_model}),
                use_mock=False,
            )
        self._vlm = self._vlm_video
        self._event_store = EventStore(config.memory.sqlite)
        # 2026-08-26 (RERANK KALDIRILDI): EVREN dokumantasyonu (SS 10), saf
        # yogun getirmenin (R@1=0.95) HER reranking varyantindan (hibrit,
        # dedike rerank ucu, LLM-as-judge) daha iyi performans gosterdigini,
        # yeniden siralamanin R@1'i 0.55'e kadar DUSURDUGUNU acikca
        # gostermektedir ve "yeniden siralama adiminin eklenmemesini"
        # ONERMEKTEDIR. Bu yuzden `EvrenReranker` (LLM-as-judge, bkz.
        # `src/rag/reranker.py`) ARTIK cagrilmiyor - `cross_encoder=None`
        # ile `EmbeddingRAGService`, gate'ten GECMIS adaylari SADECE
        # deterministic relevance skoruna (embedding + lexical/keyword/
        # metadata/phrase agirlikli toplam) gore sunar; ekstra bir LLM
        # cagrisi/ag bagimliligi/gecikme eklenmez.
        # Bug duzeltmesi (2026-08-27): `EmbeddingRAGService.__init__` her zaman
        # GERCEK bir Qdrant istemcisi kurar - `use_mock_vlm`/`use_mock_llm`den
        # BAGIMSIZDI, bu yuzden GELISTIRME/TEST yorumunun ("harici hicbir sey
        # gerekmez", bkz. configs/config.yaml ~satir 283) aksine, yalnizca
        # mock modda calistirmak bile `EVREN_QDRANT_KEY` olmadan pipeline
        # KURULUSUNDA `ConfigurationError` ile patliyordu. Her iki mock
        # bayragi da acikken (belgelenen TEK desteklenen "harici bagimlilik
        # yok" kombinasyonu) sahte, ag-bagimsiz `MockEmbeddingRAGService`
        # kullanilir (bkz. o sinifin dokustringi).
        if config.app.use_mock_vlm and config.app.use_mock_llm:
            self._rag_service = MockEmbeddingRAGService()
        else:
            self._rag_service = EmbeddingRAGService(
                config.memory.embedding,
                config.memory.qdrant,
                config.memory.reranker,
            )
        self._rag_service.seed_default_regulations()
        # Prompt Injection Guard (bkz. src/security/prompt_injection_guard.py):
        # `guard.enabled=false` ise `self._guard=None` -> ContextBuilder davranisi
        # TAMAMEN degismez. RuleEngine/EventEngine bu satirdan hicbir sekilde
        # etkilenmez (guard yalnizca ContextBuilder'a verilir).
        self._guard = build_prompt_injection_guard(config.guard) if config.guard.enabled else None
        self._context_builder = ContextBuilder(self._event_store, guard=self._guard)
        self._last_stage_rag_telemetry = None  # bkz. stage_context(): yalnizca GERCEK bir query() sonrasi dolar
        self._agent = SafirAgent(
            llm_config=config.llm,
            agent_config=config.agent,
            event_store=self._event_store,
            rag_service=self._rag_service,
            use_mock_llm=config.app.use_mock_llm,
        )

        # 07 - Olay Analizi Katmani (T008-T012): Context Builder ile LangGraph
        # Ajani arasindaki ara katman. `RetrieverTool`, `rag_service` None
        # olsa bile guvenlidir (mock veriye duser).
        # 2026-08-25 (kucuk-LLM fallback siniflandirici, bkz. event_engine.py
        # modul dokustringi): EventEngine'in `classifier`i, `05 LangGraph
        # Ajani`nin KENDI LLM istemcisinden AYRI, dedike bir `LLMClient`
        # (ayni "llm-fast" EVREN uc noktasi) - yalnizca structured+keyword
        # yolunun HER IKISI de basarisiz oldugu SON CARE durumda cagrilir.
        # Mock LLM modunda (`use_mock_llm=True`) HIC enjekte edilmez -
        # `MockLLMClient` siniflandirma JSON'u URETMEZ (RISK_SKORU formatinda
        # sabit bir karar dondurur), bu yuzden fallback devre disi birakilir
        # ve davranis eski (genel_gozlem'e duser) haliyle AYNI kalir.
        event_classifier_llm = (
            get_llm_client(config.llm, use_mock=False) if not config.app.use_mock_llm else None
        )
        self._event_engine = EventEngine(classifier=event_classifier_llm)
        self._temporal_reasoner = TemporalReasoner(relation_window_sec=DEFAULT_RELATION_WINDOW_SEC)
        self._event_merger = EventMerger(config.system.merger)
        # 2026-08-25: RuleEngine ARTIK hicbir RAG/AG cagrisi yapmaz (retriever
        # kaldirildi) - tamamen yerel, deterministik event_type -> kisa mevzuat
        # etiketi/siddet eslemesidir. Mevzuatin GERCEK metni SADECE gercek
        # semantik RAG sorgusundan (asagida `stage_context`) gelir (bkz.
        # `src/event_analysis/rule_engine.py` modul dokustringi).
        self._rule_engine = RuleEngine()
        self._event_builder = EventBuilder()
        self._event_history = EventHistory(self._event_store)

        # 06 - Otomatik Eskalasyon (Human-on-the-Loop): risk skoruna gore aksiyon
        # kademesini KENDISI belirler ve yuksek/kritik durumda saha alarmini
        # operator onayi beklemeden OTOMATIK tetikler. `FieldAlarmDispatcher`e
        # zaten mevcut olan `self._event_store` (SQLite) verilir - alarmlar
        # artik surec yeniden baslasa bile kaybolmaz (audit P0 duzeltmesi,
        # bkz. `FieldAlarmDispatcher`/`EventStore.record_alert` docstring'i).
        self._escalation = EscalationPolicy(
            config.escalation, sink=FieldAlarmDispatcher(store=self._event_store)
        )

        # Pipeline cagrilari arasinda SIFIRLANMAYAN, zaman bazli budanan
        # buffer (bkz. `_prune_stale_events`). `relation_window_sec`in 3
        # kati kadar (varsayilan 90s) gecmis her zaman korunur.
        self._event_buffer_retention_sec = DEFAULT_RELATION_WINDOW_SEC * 3
        self._event_history_buffer: Deque[DetectedEvent] = deque()
        # `run()` cagrilari arasinda video_source DEGISTIGINDE buffer sifirlanir
        # (bagimsiz video analizleri arasi contamination'i onler); ayni
        # video_source ile ARDISIK cagrilarda (surekli kamera/stream izleme)
        # buffer KORUNUR (bkz. `run()` basindaki kontrol).
        self._last_video_source: Optional[str] = None

    def run(
        self,
        video_source: str,
        user_prompt: str,
        on_stage: Optional[OnStageCallback] = None,
        sample_fps_override: Optional[int] = None,
        min_change_threshold_override: Optional[float] = None,
        trace: Optional[TraceCallback] = None,
        analysis_mode: Literal["vlm_direct", "low_budget"] = "vlm_direct",
        context: Optional[AnalysisContext] = None,
    ) -> SafirReport:
        """Video kaynagindan nihai `SafirReport`'a kadar tum pipeline'i calistirir.

        Her cagri icin `configs/config.yaml` (ve varsa override'lar) ile taze
        bir `AdaptiveFrameSampler` orneği kurulur; boylece ardisik analizler
        birbirinin `prev_gray`/gurultu gecmisi durumunu paylasmaz.

        Args:
            video_source: `.mp4` dosya yolu veya RTSP/HTTP URI'si.
            user_prompt: Ajanin odaklanmasi istenen kullanici istemi.
            on_stage: Her ana asamadan once cagrilan, operator paneli icin
                canli ilerleme bildiren istege bagli geri cagirma
                (`(asama_adi, adim, toplam_adim)`).
            sample_fps_override: Operator panelindeki slider'dan gelen
                ornekleme FPS degeri; verilmezse config degeri kullanilir.
            min_change_threshold_override: Operator panelindeki slider'dan
                gelen hassasiyet esigi; verilmezse config degeri kullanilir.
            analysis_mode: `"vlm_direct"` (varsayilan) veya `"low_budget"`
                (bkz. `AnalyzeRequest.analysis_mode` dokustringi) - hangi
                `BaseVLM` orneğinin (`self._vlm_video`/`self._vlm_frames`)
                kullanilacagini VE sampler'in calistirilip calistirilmayacagini
                belirler.

        Returns:
            Doga dil ozeti, risk skoru/seviyesi, kanit kareleri, ilgili
            mevzuat ve CPU suzgec istatistiklerini iceren rapor.

        Raises:
            RuntimeError: Video kaynagindan hic Evidence Frame/Olay Grubu uretilemezse.
        """
        pipeline_started_at = time.perf_counter()

        # Bagimsiz video analizleri arasi olay-bellegi (temporal reasoning
        # buffer) contamination'ini onler: video_source bir onceki cagridan
        # FARKLIYSA buffer temizlenir. Ayni video_source ile ARDISIK
        # cagrilarda (surekli kamera/stream izleme) buffer bilerek korunur.
        if video_source != self._last_video_source:
            self._event_history_buffer.clear()
        self._last_video_source = video_source

        def _emit(stage: str, payload: Dict[str, object]) -> None:
            """Gozlem kancasini (varsa) bir asamanin gercek ciktisiyla cagirir (yan etkisiz)."""
            if trace is not None:
                trace(stage, payload)

        if on_stage:
            on_stage(*STAGE_SAMPLER)

        # Her asama, GERCEK production metoduna delege edilir; boylece hem run()
        # tek cagriyla calisir, hem de Jupyter demo bu metotlari tek tek cagirip
        # her asamanin gercek ciktisini gosterebilir (implementasyon kopyalanmaz).
        #
        # Video-tabanli bir VLM (EVREN) aktifken Adaptive Frame Sampler
        # TAMAMEN ATLANIR: ciktisi (evidence_frames) bu saglayicilarda ZATEN
        # VLM girdisi olarak KULLANILMAZ (bkz. `BaseVLM.requires_frame_
        # sampling` dokustringi, `EvrenVLM.analyze_video`in `del evidence_
        # frames`i) - calistirmak, videonun HER native karesini CPU'da tarayan
        # (uzunlukla orantili, ölçülen: 75.6s'lik bir video icin 92.8s) saf bir
        # zaman kaybi olur, hicbir raporlanan degeri ETKILEMEZ. `analysis_mode`
        # burada da belirleyicidir: "vlm_direct" -> `self._vlm_video`
        # (`requires_frame_sampling=False`, sampler ATLANIR); "low_budget" ->
        # `self._vlm_frames` (`requires_frame_sampling=True`, sampler CALISIR).
        active_vlm = self._vlm_video if analysis_mode == "vlm_direct" else self._vlm_frames
        if getattr(active_vlm, "requires_frame_sampling", True):
            sampler, evidence_frames = self.stage_sample(
                video_source, sample_fps_override, min_change_threshold_override
            )
            _emit("sampler", {"evidence_frames": evidence_frames, "stats": sampler.last_run_stats})
        else:
            sampler = None
            evidence_frames = []
            _emit("sampler", {"evidence_frames": evidence_frames, "stats": None, "skipped": True})

        if on_stage:
            on_stage(*STAGE_VLM)

        def _on_vlm_progress(progress: Dict[str, object]) -> None:
            # Ayni "vlm" asama anahtarinda, status="running" ile BIRDEN FAZLA
            # trace olayi yayinlanir (nihai "completed" olayindan ONCE) -
            # `PipelineTraceCollector`/SSE/`JobState.trace_events` bunu zaten
            # DESTEKLER (bir asama icin tek olay ZORUNLULUGU yoktur); operator
            # video parcalara bolunup gonderilirken ADIM ADIM ilerlemeyi
            # (ör. "Parca 1/2 gonderiliyor") canli gorur.
            _emit("vlm", {"progress": progress})

        vlm_response = self.stage_vlm(
            video_source, evidence_frames, user_prompt, on_progress=_on_vlm_progress, analysis_mode=analysis_mode
        )
        _emit("vlm", {"vlm_response": vlm_response, "evidence_frames": evidence_frames, "user_prompt": user_prompt})

        # Ani Olay Tespiti (bkz. src/vlm/sudden_event_detector.py +
        # sudden_event_analyzer.py): VLM Direct'te ana VLM'e TUM video TEK
        # seferde gonderilir - modelin kendisi kare-kare fark HESAPLAMAZ, bu
        # yuzden cok kisa/ani gecen olaylar (bir kapinin aniden kapanmasi,
        # bir kisinin sikismasi vb.) uzun anlatinin icinde KAYBOLABILIR. Bu
        # katman TAMAMEN EK/BAGIMSIZDIR: OpenCV ile videoyu ONCEDEN tarayip
        # ani kare-farki "sicramalari" bulur, her adayi (ONCESI/SONRASI kare
        # cifti) `self._vlm_frames` (zaten var olan, dusuk-butceli mod icin
        # kurulmus goruntu-kabul eden istemci) ile DOGRULATIR ve yalnizca
        # GERCEKTEN onemli bulunanlari ana VLM'in `structured_events`
        # listesine EKLER - asagi akis (EventEngine/RuleEngine/rapor) HICBIR
        # DEGISIKLIGE UGRAMAZ, cunku eklenen sozlukler AYNI EVENTS_JSON
        # semasindadir. Canli yayin (rtsp/http) kaynaklari HIC TARANMAZ
        # (tam-video taramasi sonsuz surer); zaten VLM Direct'in aktif
        # saglayicisi (EvrenVLM) bu kaynaklari `vlm_response` asamasindan
        # ONCE reddeder, ama burada da AYRICA guvenlik icin kontrol edilir.
        if analysis_mode == "vlm_direct" and not video_source.strip().lower().startswith(
            ("rtsp://", "http://", "https://")
        ):
            try:
                confirmed_sudden_events = analyze_sudden_events(video_source, self._vlm_frames)
            except Exception:  # noqa: BLE001 - bu katman ASLA ana analizi COKERTMEMELI
                logger.exception("Ani olay tespit/dogrulama katmani basarisiz; ana VLM sonucu etkilenmedi.")
                confirmed_sudden_events = []
            if confirmed_sudden_events:
                vlm_response.structured_events = list(vlm_response.structured_events) + confirmed_sudden_events
                logger.info(
                    "Ani olay tespiti: %d dogrulanmis olay ana olay listesine eklendi.",
                    len(confirmed_sudden_events),
                )

        # VLM Direct'te sampler hic calismadigi (evidence_frames=[]) icin
        # kanit karesi burada, dogrudan kaynak videodan uretilir - bkz.
        # `_extract_vlm_direct_evidence_frames` dokustringi.
        vlm_direct_evidence_frames = (
            _extract_vlm_direct_evidence_frames(video_source, vlm_response.structured_events)
            # `evidence_frames` bossa sampler GERCEKTEN atlandi (bkz. yukarida) -
            # yalnizca o zaman devreye girer; sampler CALISTIYSA (ör. mock/test
            # ortaminda paylasilan istemcinin `requires_frame_sampling=True`
            # kalmasi durumunda) sampler'in ZATEN GERCEK olan kareleri korunur.
            if analysis_mode == "vlm_direct" and not evidence_frames
            else None
        )

        # 2026-08-26 ("Analiz Hattı" surekliligi): VLM Direct modunda ("sampler"
        # asamasi baslangicta "skipped" olarak isaretlenir, yukarida) video
        # parcalama BILGISI (kac parcaya bolundu, CUDA/CPU, sureler) YALNIZCA
        # `stage_vlm` TAMAMLANDIKTAN SONRA bilinir (bkz. `EvrenVLM.analyze_video`
        # -> `VLMResponse.chunking_summary`). Operator VLM Direct kullaniciysa
        # "Kare Örnekleme" karti tamamlanma SONRASI da (canli GORUNUMDE veya
        # Gecmis'ten tekrar acildiginda) BOS/jenerik bir cumlede KALMASIN diye
        # (bkz. mentor/operator eleştirisi: "analiz hattı frontend'de kopuk
        # oluyor"), "sampler" asamasi BURADA ikinci (GUNCELLENMIS, KALICI)
        # bir olayla YENIDEN yayinlanir - `PipelineTraceCollector` bunu normal
        # sekilde `trace_events`e ekler, History'ye AYNI mekanizmayla kaydedilir.
        if getattr(vlm_response, "chunking_summary", None):
            _emit(
                "sampler",
                {"evidence_frames": [], "stats": None, "skipped": True, "chunking": vlm_response.chunking_summary},
            )

        if on_stage:
            on_stage(*STAGE_AGENT)

        detected_events, temporal_events, rule_matches, latest_timestamp = self.stage_events(
            vlm_response, evidence_frames
        )
        _emit(
            "events",
            {
                "detected_events": detected_events,
                "temporal_events": temporal_events,
                "rule_matches": rule_matches,
            },
        )

        prompt_block, context = self.stage_context(
            vlm_response, user_prompt, latest_timestamp, rule_matches, temporal_events,
            analysis_mode=analysis_mode, context=context
        )
        # 2026-08-25: "agent_context" (eski "Baglam ve RAG") trace stage'i ARTIK
        # EMIT EDILMIYOR - `stage_context()`in KENDISI (prompt_block/context uretimi,
        # RuleEngine mevzuat eslesmesi, GERCEK semantik RAG sorgusu) HALA CALISIYOR
        # (bkz. asagidaki `rag_security` emit'inin okudugu `self._last_stage_rag_telemetry`,
        # yine bu cagridan gelir); yalnizca bu asamanin AYRI bir trace/UI paneli olarak
        # GORUNMESI kaldirildi - operatore artik TEK, gercek skorlu/telemetrili RAG
        # goruntusu ("rag_security"/"RAG ve Guvenlik Telemetrisi") gosteriliyor.
        # `report.relevant_regulations` (Rapor sekmesindeki "Ilgili ISG Mevzuati")
        # ETKILENMEDI - o alan hala RuleEngine'in deterministik eslesmesinden dolar.
        _emit(
            "rag_security",
            {
                "rag_telemetry": self._last_stage_rag_telemetry,
                "guard_results": context.guard_results,
                "relevance_weights": (
                    self._rag_service.relevance_weights
                    if self._last_stage_rag_telemetry is not None and hasattr(self._rag_service, "relevance_weights")
                    else None
                ),
            },
        )

        decision = self.stage_decide(prompt_block)
        _emit("decision", {"decision": decision})

        decision, risk_provenance = self.stage_finalize_risk(
            decision, rule_matches, temporal_events=temporal_events, semantic_rag_sources=context.semantically_related_chunks
        )
        _emit("decision_final", {"decision": decision, "risk_provenance": risk_provenance})

        escalation = self.stage_escalate(decision, vlm_response, risk_provenance)
        _emit("escalation", {"escalation": escalation})

        report = self.build_report(
            video_source=video_source,
            sampler=sampler,
            evidence_frames=evidence_frames,
            vlm_response=vlm_response,
            context=context,
            decision=decision,
            escalation=escalation,
            temporal_events=temporal_events,
            rule_matches=rule_matches,
            latest_timestamp=latest_timestamp,
            detected_events=detected_events,
            risk_provenance=risk_provenance,
            precomputed_evidence_frames=vlm_direct_evidence_frames,
        )
        logger.info(
            "SAFIR pipeline tamamlandi: video=%s risk=%s(%s) status=%s sure=%.3fs",
            video_source,
            decision.risk_score,
            decision.risk_level,
            decision.risk_status,
            time.perf_counter() - pipeline_started_at,
        )
        _emit("report", {"report": report})
        return report

    # -------------------------------------------------------------------------
    # Asama metotlari: `run()`in her adimini ayri, GERCEK production metodu
    # olarak sunar. Jupyter demo (notebooks/safir_end_to_end_demo.ipynb) bu
    # metotlari tek tek cagirip her asamanin gercek ciktisini gosterir;
    # implementasyon notebook'a KOPYALANMAZ, yalnizca bu metotlar cagrilir.
    # -------------------------------------------------------------------------

    def stage_sample(
        self,
        video_source: str,
        sample_fps_override: Optional[int] = None,
        min_change_threshold_override: Optional[float] = None,
    ):
        """01: Adaptive Frame Sampler (olay kumelemesi YAPMAZ - bkz. modul docstring'i).

        Returns:
            `(sampler, evidence_frames)` — `evidence_frames`, esik-gecmis TUM
            evidence karelerinin video geneli, kronolojik, kayipsiz listesidir.
        """
        sampler = sampler_from_config(self._config.sampler, min_change_threshold_override)
        sample_fps = sample_fps_override or self._default_sample_fps

        evidence_frames: List[EvidenceFrame] = sampler.process_video(video_source, sample_fps=sample_fps)
        if not evidence_frames:
            raise RuntimeError(f"Video kaynagindan kanit karesi uretilemedi: {video_source}")

        logger.info(
            "VLM oncesi katman ozeti: %d Kanit Karesi VLM'e gonderiliyor (kumeleme VLM katmaninda yapilacak)",
            len(evidence_frames),
        )
        return sampler, evidence_frames

    def stage_vlm(
        self,
        video_source: str,
        evidence_frames: List[EvidenceFrame],
        user_prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
        analysis_mode: Literal["vlm_direct", "low_budget"] = "vlm_direct",
        context: Optional[AnalysisContext] = None,
    ) -> VLMResponse:
        """03: VLM gorsel anlama + OLAY KUMELEME.

        `analysis_mode`e gore (bkz. `AnalyzeRequest.analysis_mode`
        dokustringi, `run()`daki AYNI secim) iki AYRI, birbirinden BAGIMSIZ
        `BaseVLM` orneği/yola dallanir - biri DEGISTIRILDIGINDE digeri
        ETKILENMEZ:

        - `"vlm_direct"` (varsayilan - `self._vlm_video`, EVREN icin
          `EvrenVLM`/model="vlm"): video DOGRUDAN, TEK istekte gonderilir
          (bkz. `_stage_vlm_video`). `evidence_frames` yalnizca
          `evidence_ids`/zaman damgasi baglami icin `stage_events`e tasinir -
          VLM girdisi olarak KULLANILMAZ.
        - `"low_budget"` (`self._vlm_frames`, EVREN icin `EvrenFramesVLM`/
          model="llm-large"): sampler'in urettigi evidence kareleri
          `vlm.batch_size` buyuklugunde batch'lere bolunup HER batch AYRI bir
          istekte gonderilir, birden fazla batch varsa TEK bir reconciliation
          cagrisiyla global olay listesine birlestirilir (bkz.
          `_stage_vlm_frames`).

        Her iki yolda da VLM cagrisi basarisiz olursa eski (geriye-donuk
        uyumlu) `[HATA]` degrade formatina dusulur; risk ASLA `0`/basarili
        gibi yorumlanmaz.

        Args:
            on_progress: Opsiyonel adim-adim ilerleme kancasi (`EvrenVLM`nin
                video-parcalama/gonderim ilerlemesini rapor eder - bkz.
                `src/vlm/evren_vlm.py::VlmProgressCallback`); yalnizca video
                yolunda (`_stage_vlm_video`) etkilidir.
            analysis_mode: `"vlm_direct"` (varsayilan) veya `"low_budget"`.
        """
        vlm = self._vlm_video if analysis_mode == "vlm_direct" else self._vlm_frames
        if getattr(vlm, "requires_frame_sampling", True):
            return self._stage_vlm_frames(vlm, evidence_frames, user_prompt, context)
        return self._stage_vlm_video(vlm, video_source, evidence_frames, user_prompt, on_progress, context)

    def _stage_vlm_video(
        self,
        vlm: BaseVLM,
        video_source: str,
        evidence_frames: List[EvidenceFrame],
        user_prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
        context: Optional[AnalysisContext] = None,
    ) -> VLMResponse:
        """VLM Direct yolu (DEGISMEDI): video kaynagi TEK bir istekte, dogrudan VLM'e gonderilir."""
        try:
            analyze = getattr(vlm, "analyze_video")
            try:
                response = analyze(video_source, evidence_frames, prompt=user_prompt, on_progress=on_progress, context=context)
            except TypeError:
                # Geriye-donuk uyumluluk: `on_progress` desteklemeyen bir
                # implementasyon (ör. testlerdeki eski sahte/mock VLM).
                response = analyze(video_source, evidence_frames, prompt=user_prompt)
        except Exception as exc:  # noqa: BLE001 - beklenmedik hata da degraded rapora tasinir
            logger.exception("VLM video analizi basarisiz; degraded raporla devam ediliyor.")
            return VLMResponse(
                description=f"[HATA] VLM analizi yapilamadi ({exc}). Manuel inceleme gerekli.",
                model_name=getattr(vlm, "model_name", "unknown"),
                frame_count=len(evidence_frames),
                latency_ms=0.0,
                structured_events=[],
                status="failed",
                chunk_analysis_result=ChunkAnalysisResult(
                    analysis_status=VLMAnalysisStatus.MODEL_FAILED,
                    parse_status="pipeline_exception",
                    analysis_id=context.analysis_id if context else None
                )
            )

        response.evidence_ids = [ef.evidence_id for ef in evidence_frames]
        return response

    def _stage_vlm_frames(
        self,
        vlm: BaseVLM,
        evidence_frames: List[EvidenceFrame],
        user_prompt: str,
        context: Optional[AnalysisContext] = None,
    ) -> VLMResponse:
        """Dusuk butceli yol: evidence kareleri `vlm.batch_size` buyuklugunde batch'lere bolunup analiz edilir.

        `BaseVLM.analyze_evidence_batched` ile kronolojik, kayipsiz batch'lere
        bolunur (bir batch'in basarisiz olmasi digerlerini ETKILEMEZ, hicbir
        evidence kaybolmaz); birden fazla batch uretildiyse
        `BaseVLM.reconcile_events` ile TEK bir global olay listesine
        birlestirilir (tek batch zaten globaldir, ekstra cagri YAPILMAZ).
        `EvrenFramesVLM` icin `vlm.batch_size` EVREN dokumantasyonunun
        (SS 7.5) istek basina azami 2 goruntu kuralina uymak zorundadir -
        bu, `configs/config.yaml` sorumlulugundadir (`EvrenFramesVLM.
        analyze_evidence` yanlis ayarlanirsa ACIKCA hata verir).
        """
        if not evidence_frames:
            return VLMResponse(
                description="[HATA] Kare-tabanli VLM analizi icin evidence karesi uretilemedi.",
                model_name=getattr(vlm, "model_name", "unknown"),
                frame_count=0,
                latency_ms=0.0,
                structured_events=[],
                status="failed",
            )
        try:
            batch_responses = vlm.analyze_evidence_batched(
                evidence_frames, user_prompt, batch_size=self._config.vlm.batch_size
            )
            return vlm.reconcile_events(batch_responses, user_prompt)
        except Exception as exc:  # noqa: BLE001 - beklenmedik hata da degraded rapora tasinir
            logger.exception("Kare-tabanli VLM analizi basarisiz; degraded raporla devam ediliyor.")
            return VLMResponse(
                description=f"[HATA] VLM analizi yapilamadi ({exc}). Manuel inceleme gerekli.",
                model_name=getattr(vlm, "model_name", "unknown"),
                frame_count=len(evidence_frames),
                latency_ms=0.0,
                structured_events=[],
                status="failed",
                chunk_analysis_result=ChunkAnalysisResult(
                    analysis_status=VLMAnalysisStatus.MODEL_FAILED,
                    parse_status="pipeline_exception",
                    analysis_id=context.analysis_id if context else None
                )
            )

    def stage_events(self, vlm_response: VLMResponse, evidence_frames: List[EvidenceFrame]):
        """07: EventEngine -> buffer/budama -> TemporalReasoner -> RuleEngine.

        Returns:
            `(detected_events, temporal_events, rule_matches, latest_timestamp)`.
            `temporal_events`/`rule_matches`, cagrilar arasi kalici buffer uzerinden hesaplanir.
        """
        onset_timestamp = evidence_frames[0].timestamp_sec if evidence_frames else 0.0
        evidence_timestamps = {ef.evidence_id: ef.timestamp_sec for ef in evidence_frames}
        engine_input = EventEngineInput.from_vlm_response(
            vlm_response, timestamp=onset_timestamp, evidence_timestamps=evidence_timestamps
        )
        detected_events = self._event_engine.detect(engine_input)
        self._event_history_buffer.extend(detected_events)

        # VLM'in kendi ata gidigi olay zaman damgalari (start_time/end_time)
        # artik video'nun SON evidence karesiyle AYNI olmak zorunda degildir
        # (bir olay video ortasinda bitebilir). "Bu cagrinin" en yeni zaman
        # damgasi, hem evidence karelerinin hem de tespit edilen olaylarin
        # gercek uctan (end_timestamp/timestamp) MAKSIMUMUdur; boylece
        # `_select_current_call_events`in tolerans karsilastirmasi dogru
        # calisir (bkz. asagida) ve budama penceresi hicbir olayi erken kesmez.
        latest_timestamp = evidence_frames[-1].timestamp_sec if evidence_frames else 0.0
        if detected_events:
            latest_timestamp = max(
                latest_timestamp,
                max(
                    (d.end_timestamp if d.end_timestamp is not None else d.timestamp)
                    for d in detected_events
                ),
            )
        _prune_stale_events(self._event_history_buffer, latest_timestamp, self._event_buffer_retention_sec)

        temporal_events = self._temporal_reasoner.reason(list(self._event_history_buffer))
        temporal_events = self._event_merger.merge(temporal_events)
        rule_matches = self._rule_engine.evaluate(temporal_events)
        return detected_events, temporal_events, rule_matches, latest_timestamp

    def stage_context(
        self,
        vlm_response: VLMResponse,
        user_prompt: str,
        latest_timestamp: float,
        rule_matches,
        temporal_events: Optional[List] = None,
        analysis_mode: Literal["vlm_direct", "low_budget"] = "vlm_direct",
        context: Optional[AnalysisContext] = None,
    ):
        """04: ContextBuilder (SQLite + GERCEK semantik RAG) -> ajan istem blogu.

        2026-08-25: mevzuat/kanit icerigi ARTIK TEK bir kaynaktan gelir -
        `temporal_events`in `matched_keywords`inden (VLM'in dinamik risk
        keyword'leri) kurulan GERCEK semantik RAG sorgusu (bkz.
        `_build_semantic_query`), sonucu `ContextBuilder`in
        `semantically_related_chunks` alanina gider. Onceden AYRICA
        `RuleEngine.evaluate(...)`nin (`rule_matches`) deterministik kisa
        etiketlerinden tureyen IKINCI, telemetrisiz bir "mevzuat" listesi de
        Context'e ekleniyordu - bu, operatore GORUNURDE iki ayri RAG kanali
        gibi gorunuyordu ve KALDIRILDI (bkz. `src/event_analysis/rule_engine.py`
        modul dokustringi). `rule_matches` HALA `_summarize_rule_matches` ile
        (asagida) ajana KISA, deterministik siniflandirma sinyali olarak
        gecirilir - bu bir "mevzuat metni" DEGILDIR, yalnizca "hangi ISG
        kategorisi/siddeti tetiklendi" bilgisidir.

        Semantik sorgu basarisiz olursa (API anahtari eksik, reranker
        "unavailable" vb.) HATA YUTULUR ve bos liste ile devam edilir - bu,
        agent akisini KESMEZ (bkz. `EmbeddingRAGService.query` docstring'i:
        bu zaten kendi icinde 'unavailable -> []' davranisina sahiptir;
        buradaki try/except yalnizca beklenmedik istisnalara karsi ek bir
        guvenlik agi).

        Returns:
            `(prompt_block, context)`.
        """
        matched_keywords = _aggregate_matched_keywords(temporal_events or [])
        semantic_query = _build_semantic_query(matched_keywords, vlm_response.description)
        semantically_related_chunks: List = []
        # Bu cagrida GERCEKTEN bir RAG sorgusu yapildiysa (ONCEKI cagridan kalan
        # `get_last_query_telemetry()` sonucunu "bu cagrinin telemetrisi" gibi
        # SUNMAMAK icin) yalnizca query() GERCEKTEN calistiysa dolar - bkz. `run()`.
        self._last_stage_rag_telemetry = None
        if semantic_query:
            try:
                semantically_related_chunks = self._rag_service.query(semantic_query, keywords=matched_keywords)
                # `getattr(..., None)` KASITLI: test/mock RAG servisleri (yalnizca
                # `.query()` tasiyan duck-typed sahteler) icin de GUVENLI CALISIR -
                # bkz. `context_builder.py::_format_semantic_chunks` ile ayni gerekce.
                get_telemetry = getattr(self._rag_service, "get_last_query_telemetry", None)
                self._last_stage_rag_telemetry = get_telemetry() if get_telemetry else None
            except Exception:  # noqa: BLE001 - semantik RAG best-effort'tur, agent akisini KESMEZ
                logger.exception("Semantik RAG sorgusu basarisiz; semantik kaynaklar bos birakiliyor.")
                semantically_related_chunks = []

        context = self._context_builder.build(
            vlm_description=vlm_response.description,
            user_prompt=user_prompt,
            timestamp=latest_timestamp,
            semantically_related_chunks=semantically_related_chunks,
        )
        prompt_block = context.to_prompt_block()
        event_analysis_summary = _summarize_rule_matches(rule_matches)
        if event_analysis_summary:
            prompt_block = (
                f"{prompt_block}\n\n## Olay Analizi Katmani Sinyalleri (T008-T012)\n{event_analysis_summary}"
            )
        return prompt_block, context

    def stage_decide(self, prompt_block: str):
        """05: LangGraph ajani muhakeme -> `AgentDecision` (risk skoru, ozet, aksiyonlar)."""
        return self._agent.run(prompt_block)

    def stage_finalize_risk(self, decision, rule_matches, temporal_events=None, semantic_rag_sources=None):
        """07c: RISK ENGINE V2.1 - deterministik (RuleEngine+`risk_model.py`) skor ile Agent'in taslak skorunun ORTALAMASINI nihai risk olarak uygular.

        Mimari karar (2026-08-27 guncellemesi - kullanici talebi): VLM ve
        `05 LangGraph Agent`taki LLM tek basina risk KARARI vermez, ama
        kendi taslak tahmini (`RiskProvenance.llm_proposed_score`) ARTIK
        nihai skora KATKIDA BULUNUR - `RuleEngine.evaluate(...)` ciktisindan
        (`rule_matches`) VE mevcut diger yapilandirilmis kanittan
        (`temporal_events`, `semantic_rag_sources`) turetilen deterministik
        skor ile ORTALAMASI alinir (bkz. `risk_resolver._pick_provenance`).
        Deterministik deger, blended degerden BAGIMSIZ olarak
        `RiskProvenance.deterministic_score`/`deterministic_level` alanlarinda
        AYRICA saklanir - iki deger operatore/rapora AYRI AYRI gosterilir.
        Kritik bir fiziksel tehlike icin guvenlik tabani (bkz.
        `risk_model._CRITICAL_HAZARD_FLOOR`) devredeyse, ortalama bu tabanin
        ALTINA DUSURULMEZ. Bu cagriya ait HICBIR kural eslesmediyse
        (`rule_matches` bos veya taninmayan siddette), LLM Agent'in karari
        (`decision.risk_score`/`risk_level`/`risk_status`) DEGISTIRILMEDEN
        korunur - boylece deterministik sinyal yokken sistem sessizce
        "dusuk risk" UYDURMAZ, LLM'in (varsa) degerlendirmesine ya da
        "unknown" durumuna duser.

        Onemli guvenlik istisnasi: `decision.risk_status == "unknown"` ise
        (05 LangGraph Agent muhakemesi TAMAMEN basarisiz oldu, bkz.
        `SafirAgent._degraded_decision`) override YAPILMAZ. Bu, mevcut
        "ajan cokerse hicbir zaman otomatik alarm tetiklenmez, operator
        manuel incelemeye yonlendirilir" guvenlik davranisini korur - ajanin
        kendisi calismiyorken (potansiyel sistem-genelinde bir sorunun
        isareti olabilir) rule engine sinyaline dayanarak fiziksel bir
        alarmi OTOMATIK tetiklemek istenmez; bu durumda operator "unknown"
        durumunu gorup MANUEL karar vermelidir.

        Args:
            decision: `stage_decide(...)` ciktisi (`AgentDecision`).
            rule_matches: `stage_events(...)` ciktisindan bu cagriya ait
                tum `RuleMatch` listesi (tekli + kombinasyon).
            temporal_events: (Opsiyonel) bu cagriya ait `TemporalEvent`ler -
                likelihood/duration/recurrence feature'larini besler.
            semantic_rag_sources: (Opsiyonel) bu cagrinin semantik RAG
                sonuclari - regulatory_support feature'ini besler.

        Returns:
            `(decision, risk_provenance)`: `risk_score`/`risk_level`/`risk_status`
            alanlari (varsa, ajan basarili muhakeme urettiyse) guncellenmis ayni
            `decision` nesnesi + hesaplamanin TAM izlenebilirligini tasiyan
            `RiskProvenance` (unknown durumunda `risk_level=None` provenance doner).
        """
        if decision.risk_status == "unknown":
            return decision, RiskProvenance(risk_level=None, risk_score=None)

        llm_proposed_score = decision.risk_score
        risk_provenance = resolve_deterministic_risk_with_provenance(
            rule_matches,
            temporal_events=temporal_events,
            semantic_rag_sources=semantic_rag_sources,
            llm_proposed_score=llm_proposed_score,
        )
        if risk_provenance.risk_level is not None:
            decision.risk_score = risk_provenance.risk_score
            decision.risk_level = risk_provenance.risk_level
            decision.risk_status = "assessed"
        return decision, risk_provenance

    def stage_escalate(self, decision, vlm_response: VLMResponse, risk_provenance: Optional[RiskProvenance] = None):
        """06: Otomatik eskalasyon -> `EscalationDecision`.

        GERCEK, RuleEngine-dogrulanmis (bkz. `risk_provenance.rule_ids`)
        yuksek/kritik risk otomatik olarak alarm tetikler. `risk_provenance`
        verilmezse (orn. eski cagiran kod) VEYA `rule_ids` bossa, deterministik
        kanit YOK sayilir - bu durumda (dusuk risk haricinde) otomatik islem/
        alarm YAPILMAZ, `PENDING_REVIEW`e dusulur (bkz. `EscalationPolicy.
        evaluate` docstring'i).

        RISK ENGINE V2.1 (2026-08-27) guvenlik notu: `report.risk_score`
        (operatore GOSTERILEN nihai deger) artik deterministik skorla Agent'in
        taslak skorunun ORTALAMASIDIR (bkz. `risk_resolver.py`) - ama OTOMATIK
        alarm tetikleme KARARI kasitli olarak BUNU KULLANMAZ: Agent'in dusuk
        (yanlis/naif) bir taslak tahmini, ZATEN RuleEngine tarafindan
        dogrulanmis kritik bir fiziksel tehlikenin otomatik alarmini
        ORTALAMA yoluyla SESSIZCE BASTIRAMAZ. Bu yuzden mumkunse
        `risk_provenance.deterministic_score/deterministic_level` (HAM,
        Agent'ten bagimsiz deger) kullanilir; yalnizca hicbir deterministik
        kural eslesmediyse (`deterministic_score is None`) Agent'in kendi
        (blended'a esit) degerine dusulur.
        """
        has_deterministic_backing = bool(risk_provenance and risk_provenance.rule_ids)
        escalation_score = decision.risk_score
        escalation_level = decision.risk_level
        if risk_provenance is not None and risk_provenance.deterministic_score is not None:
            escalation_score = risk_provenance.deterministic_score
            escalation_level = risk_provenance.deterministic_level
        escalation = self._escalation.evaluate(
            risk_score=escalation_score,
            risk_level=escalation_level,
            recommended_action=decision.recommended_action,
            summary=decision.summary or vlm_response.description,
            risk_status=decision.risk_status,
            has_deterministic_backing=has_deterministic_backing,
        )
        logger.info(
            "Otomatik eskalasyon: kademe=%s otomatik_tetik=%s alert_id=%s (%s)",
            escalation.tier.value,
            escalation.auto_dispatched,
            escalation.alert_id,
            escalation.reason,
        )
        return escalation

    def build_report(
        self,
        *,
        video_source: str,
        sampler,
        evidence_frames: List[EvidenceFrame],
        vlm_response: VLMResponse,
        context,
        decision,
        escalation,
        temporal_events,
        rule_matches,
        latest_timestamp: float,
        detected_events: Optional[List[DetectedEvent]] = None,
        risk_provenance: Optional[RiskProvenance] = None,
        precomputed_evidence_frames: Optional[List[EvidenceFrameOut]] = None,
        analysis_mode: Literal["vlm_direct", "low_budget"] = "vlm_direct",
    ) -> SafirReport:
        """06-07: Olay kaydi (EventBuilder/History/Store) + nihai `SafirReport` insasi.

        `precomputed_evidence_frames`: verilirse (VLM Direct - bkz. `run()`
        icindeki `_extract_vlm_direct_evidence_frames` cagrisi), rapordaki
        `evidence_frames` alani ICIN DOGRUDAN kullanilir; `_select_report_
        evidence_frames`in sampler-tabanli eslesmesi (dusuk-butceli sistem
        icin, `evidence_frames` sampler ciktisindan) ATLANIR - VLM Direct'te
        sampler hic calismadigindan bu eslesme ZATEN her zaman bos donerdi.

        `risk_provenance`: verilmezse (orn. Jupyter demo'da `build_report`
        dogrudan/tek basina cagrilirsa) `rule_matches`/`temporal_events`/
        `context.semantically_related_chunks`den GUVENLI sekilde yeniden
        hesaplanir - `run()`daki degerle BIREBIR AYNI sonucu verir (bkz.
        `risk_resolver._pick_provenance`, tek kaynak).
        """
        if risk_provenance is None:
            risk_provenance = resolve_deterministic_risk_with_provenance(
                rule_matches,
                temporal_events=temporal_events,
                semantic_rag_sources=getattr(context, "semantically_related_chunks", None),
                llm_proposed_score=getattr(decision, "risk_score", None),
            )
        rag_telemetry = self._last_stage_rag_telemetry
        cross_encoder_status = getattr(rag_telemetry, "cross_encoder_status", None) if rag_telemetry is not None else None
        relevance_threshold = getattr(rag_telemetry, "threshold", None) if rag_telemetry is not None else None
        # RAG explainability (gorev tanimi 8. bolum): agirliklar KODDAN okunur - bu cagrida
        # semantik RAG sorgusu GERCEKTEN yapildiysa (rag_telemetry mevcut), servisin KENDI
        # `RelevanceWeights`inden (config'ten okunmus, HARD-CODE DEGIL) tasinir.
        relevance_weights = (
            {
                "semantic": self._rag_service.relevance_weights.semantic,
                "lexical": self._rag_service.relevance_weights.lexical,
                "keyword": self._rag_service.relevance_weights.keyword,
                "metadata": self._rag_service.relevance_weights.metadata,
                "phrase": self._rag_service.relevance_weights.phrase,
            }
            if rag_telemetry is not None and hasattr(self._rag_service, "relevance_weights")
            else None
        )

        current_model_call_ids = set()
        current_chunk_ids = set()
        current_analysis_ids = set()
        current_evidence_ids = set()
        
        for d in (detected_events or []):
            if d.source_model_call_id:
                current_model_call_ids.add(d.source_model_call_id)
            if d.source_chunk_id:
                current_chunk_ids.add(d.source_chunk_id)
            if d.source_analysis_id:
                current_analysis_ids.add(d.source_analysis_id)
            if d.evidence_ids:
                current_evidence_ids.update(d.evidence_ids)
                
        if len(current_analysis_ids) > 1:
            import logging
            logging.getLogger(__name__).error(f"Security/Invariant violation: Multiple analysis IDs detected in a single pipeline call: {current_analysis_ids}")
            # Controlled failure: Drop mixed events to prevent contamination.
            detected_events = []
            current_analysis_ids.clear()
            if hasattr(vlm_response, "chunk_analysis_result") and vlm_response.chunk_analysis_result:
                from src.vlm.schemas import VLMAnalysisStatus
                vlm_response.chunk_analysis_result.analysis_status = VLMAnalysisStatus.PARTIAL
                vlm_response.chunk_analysis_result.parse_status = "provenance_integrity_failed"

            
        final_analysis_id = current_analysis_ids.pop() if current_analysis_ids else None
        
        current_call_events = _select_current_call_events(
            temporal_events=temporal_events,
            latest_timestamp=latest_timestamp,
            current_model_call_ids=current_model_call_ids,
            current_chunk_ids=current_chunk_ids,
            current_analysis_id=final_analysis_id,
            current_evidence_ids=current_evidence_ids,
            allow_legacy_timestamp_fallback=False
        )
        if getattr(vlm_response, 'aggregate_result', None):
            vlm_response.aggregate_result.event_count_after_merge = len(current_call_events)
        detected_event_names = sorted({te.event_name for te in current_call_events})
        detected_event_types = sorted({te.event_type for te in current_call_events if te.event_type})
        structured_events = self._event_builder.build_batch(current_call_events, rule_matches)
        # Her StructuredEvent, EventBuilder tarafindan KENDI rule_matches'inden
        # deterministik olarak risk tasir (bkz. risk_resolver.py); bir olay
        # icin hicbir kural eslesmediyse (risk_score/level None), cagri-geneli
        # (artik rule-engine-oncelikli, bkz. stage_finalize_risk) decision'a duser.
        recorded_event_ids = self._event_history.record_batch(
            structured_events,
            risk_scores=[e.risk_score if e.risk_score is not None else decision.risk_score for e in structured_events],
            risk_levels=[e.risk_level if e.risk_level is not None else decision.risk_level for e in structured_events],
            video_source=video_source,
        )
        logger.info(
            "Event Gecmisi: %d StructuredEvent kaydedildi (ids=%s)", len(recorded_event_ids), recorded_event_ids
        )
        event_id = recorded_event_ids[0] if recorded_event_ids else None
        onset_timestamp = evidence_frames[0].timestamp_sec if evidence_frames else 0.0
        timeline = self._event_store.get_timeline(
            start_ts=onset_timestamp, end_ts=latest_timestamp, video_source=video_source
        )

        # Risk explainability (P0 duzeltmesi): "risk_score NEREDEN geldi?"
        # sorusunu LLM'e SORMADAN, `risk_provenance`den (deterministik)
        # cevaplayan uc alan. `decision.risk_score` (nihai/authoritative
        # deger) ile HANGI kural(lar)in onu urettigini rapor duzeyinde de
        # gorunur kilar (bkz. `risk_resolver.RiskProvenance`).
        if decision.risk_status == "unknown":
            risk_source = "unknown"
            risk_explanation = "Analiz guvenilir sekilde tamamlanamadi; risk belirlenemedi."
        elif risk_provenance.rule_ids:
            risk_source = "rule_engine"
            risk_explanation = risk_provenance.explanation()
        else:
            risk_source = "agent"
            risk_explanation = (
                "Hicbir deterministik kural eslesmedi; risk seviyesi Agent'in kendi "
                "degerlendirmesinden alindi (RuleEngine tarafindan dogrulanmamis)."
            )

        # `getattr(..., None)` KASITLI: test/mock RAG servisleri (yalnizca `text`/`score`
        # tasiyan duck-typed sahteler) icin de GUVENLI CALISIR - bkz. ayni gerekce
        # `context_builder.py::_format_semantic_chunks` ve `tools.py::RetrieverTool.run`da.
        semantic_rag_sources = [
            RagContext(
                rule_title=(
                    getattr(chunk, "document_title", None)
                    or getattr(chunk, "document_id", None)
                    or "(bilinmeyen kaynak)"
                ),
                content=chunk.text,
                score=getattr(chunk, "embedding_score", None) or getattr(chunk, "score", 0.0),
                embedding_score=getattr(chunk, "embedding_score", None),
                chunk_id=getattr(chunk, "chunk_id", None),
                document_id=getattr(chunk, "document_id", None),
                article_number=getattr(chunk, "article_number", None),
                source_url=getattr(chunk, "source_url", None),
                relevance_score=getattr(chunk, "relevance_score", None),
                semantic_score=getattr(chunk, "semantic_score", None),
                lexical_score=getattr(chunk, "lexical_score", None),
                keyword_score=getattr(chunk, "keyword_score", None),
                metadata_score=getattr(chunk, "metadata_score", None),
                phrase_score=getattr(chunk, "phrase_score", None),
                relevance_status=getattr(chunk, "relevance_status", None),
                relevance_reason=getattr(chunk, "relevance_reason", None),
                cross_encoder_score=getattr(chunk, "cross_encoder_score", None),
                final_rank=getattr(chunk, "final_rank", None),
                source_verified=getattr(chunk, "source_verified", True),
            )
            for chunk in context.semantically_related_chunks
        ]
        # 2026-08-25: `relevant_regulations` (Rapor -> "Ilgili ISG Mevzuati") ARTIK
        # TEK kaynaktan turer - `semantic_rag_sources`in KENDISINDEN (yani
        # `context.semantically_related_chunks`den, GERCEK semantik RAG sorgusu +
        # deterministik relevance/guvenlik esiginden GECMIS "accepted" adaylar).
        # Onceden BURADA RuleEngine'in AYRI, deterministik kisa-etiket listesi
        # (`context.relevant_regulations`) kullaniliyordu - bu, gercek RAG
        # telemetrisine SAHIP DEGILDI ve operatore ikinci/sahte bir "RAG" kanali
        # gibi gorunuyordu (bkz. `src/event_analysis/rule_engine.py` modul
        # dokustringi). Esik-uzeri hicbir kanit yoksa liste BOS kalir - bir
        # mevzuat UYDURULMAZ.
        relevant_regulations = [
            f"{source.rule_title}" + (f" (Madde {source.article_number})" if source.article_number else "") + f": {source.content}"
            for source in semantic_rag_sources
        ]
        # Source validation (gorev tanimi 10. bolum): Ajanin serbest metninde (summary +
        # actions) gecen mevzuat-benzeri atiflardan, BU CAGRIDA GERCEKTEN retrieved olan
        # kanitlarla eslesmeyenleri isaretler - deterministik, regex-tabanli (LLM'e SORULMAZ).
        agent_free_text = " ".join([decision.summary or ""] + list(decision.actions or []))
        unverified_references = _unverified_citations(agent_free_text, semantic_rag_sources)

        return SafirReport(
            event_id=event_id,
            video_source=video_source,
            generated_at=datetime.datetime.utcnow().isoformat() + "Z",
            natural_language_summary=vlm_response.description,
            summary=decision.summary or vlm_response.description,
            risk_score=decision.risk_score,
            risk_level=decision.risk_level,
            risk_status=decision.risk_status,
            risk_source=risk_source,
            risk_explanation=risk_explanation,
            contributing_rule_ids=risk_provenance.rule_ids,
            scoring_method=risk_provenance.scoring_method if risk_provenance.rule_ids else None,
            risk_features=risk_provenance.features,
            risk_feature_contributions=risk_provenance.feature_contributions,
            llm_proposed_score=risk_provenance.llm_proposed_score,
            deterministic_score=risk_provenance.deterministic_score,
            deterministic_level=risk_provenance.deterministic_level,
            cross_encoder_status=cross_encoder_status,
            relevance_weights=relevance_weights,
            relevance_threshold=relevance_threshold,
            recommended_action=decision.recommended_action,
            actions=decision.actions,
            triggered_mock_actions=getattr(decision, "triggered_mock_actions", []),
            onset_timestamp_str=getattr(decision, "onset_timestamp", None)
            or (f"{int(onset_timestamp // 60):02d}:{int(onset_timestamp % 60):02d}" if evidence_frames else None),
            safe_timestamps=getattr(decision, "safe_timestamps", []),
            incident_timestamps=getattr(decision, "incident_timestamps", []),
            escalation_tier=escalation.tier.value,
            auto_dispatched=escalation.auto_dispatched,
            alert_id=escalation.alert_id,
            detected_event_names=detected_event_names,
            detected_event_types=detected_event_types,
            events=[
                EventSummary(
                    event_name=se.event_name,
                    event_type=se.event_type,
                    keywords=se.keywords,
                    risk_level=se.risk_level,
                    risk_score=se.risk_score,
                    evidence_ids=se.evidence_ids,
                    rule_ids=[m.rule_id for m in se.related_rule_matches],
                )
                for se in structured_events
            ],
            timeline=[
                TimelineEntry(timestamp=e["timestamp"], description=e["description"]) for e in timeline
            ],
            evidence_frames=(
                precomputed_evidence_frames
                if precomputed_evidence_frames is not None
                else _select_report_evidence_frames(vlm_response, evidence_frames)
            ),
            relevant_regulations=relevant_regulations,
            semantic_rag_sources=semantic_rag_sources,
            unverified_references=unverified_references,
            sampler_stats=(
                SamplerStats(
                    total_frames_scanned=sampler.last_run_stats.total_frames_scanned,
                    sampled_frames_evaluated=sampler.last_run_stats.sampled_frames_evaluated,
                    evidence_frame_count=sampler.last_run_stats.evidence_frame_count,
                    eliminated_frame_count=sampler.last_run_stats.eliminated_frame_count,
                    gpu_savings_ratio_pct=sampler.last_run_stats.eliminated_ratio_pct,
                    elapsed_sec=sampler.last_run_stats.elapsed_sec,
                )
                # `sampler` video-tabanli bir VLM (EVREN) aktifken `None`dir
                # (bkz. `run()` - bu asama TAMAMEN ATLANIR); "0" gibi
                # YANILTICI istatistik UYDURMAK yerine dogru sekilde `None`
                # ("N/A") kalir.
                if sampler is not None and sampler.last_run_stats
                else None
            ),
            vlm_model=vlm_response.model_name,
            llm_model=self._agent.model_name,
        )

    def acknowledge_alert(self, alert_id: str, operator_note: str = ""):
        """Operatorun otomatik tetiklenmis bir saha alarmini onaylamasini/geri almasini isler (Human-on-the-Loop).

        Args:
            alert_id: `SafirReport.alert_id` (otomatik tetiklenen alarmin kimligi).
            operator_note: Operatorun opsiyonel notu.

        Returns:
            Guncellenmis `AlertRecord`.

        Raises:
            KeyError: `alert_id` bilinmiyorsa.
        """
        return self._escalation.sink.acknowledge(alert_id, operator_note)

    def trigger_manual_alert(
        self, risk_score: int, risk_level: str, recommended_action: str, summary: str = ""
    ) -> str:
        """Operatorun manuel/zorunlu saha alarmi tetiklemesini isler (otomatik akisin disinda override).

        Args:
            risk_score: Operatorun bildirdigi risk skoru.
            risk_level: Risk seviyesi.
            recommended_action: Alarma iliştirilecek aksiyon.
            summary: Opsiyonel durum ozeti.

        Returns:
            Tetiklenen alarmin `alert_id`'si.
        """
        return self._escalation.sink.dispatch(
            risk_score=risk_score,
            risk_level=risk_level,
            recommended_action=recommended_action,
            summary=summary,
            auto=False,
        )

    def record_feedback(self, event_id: int, feedback: str) -> None:
        """Operatorun Human-in-the-Loop dogrulamasini SQLite'a isler.

        Args:
            event_id: `SafirReport.event_id` (bu analizin SQLite kaydi).
            feedback: `"true_positive"` veya `"false_positive"`.

        Raises:
            ValueError: `feedback` gecersiz bir deger olursa veya `event_id` bulunamazsa.
        """
        self._event_history.mark_feedback(event_id, feedback)


_pipeline: SafirPipeline | None = None
_jobs: Dict[str, JobState] = {}
_jobs_lock = threading.Lock()
_MAX_JOBS = 50  # Bellek sinir: bu sayidan fazla is tutulmaz (en eski budanir).


def get_pipeline() -> SafirPipeline:
    """Uygulama omru boyunca tek bir `SafirPipeline` orneği olusturur/dondurur (lazy singleton).

    Returns:
        Ilklendirilmis `SafirPipeline` orneği.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = SafirPipeline(load_config())
    return _pipeline


# 09 - Analiz Gecmisi (History): dosya-tabanli, KALICI depo. Mevcut `EventStore`'a
# ve pipeline mantigina DOKUNMAZ; yalnizca is yasam dongusu + nihai `SafirReport`
# kalici olarak saklanir. Yol, mevcut `memory.sqlite.db_path` ile ayni klasorde
# (`analyses.db`) tutulur (yeni config anahtari gerektirmez).
_analysis_store: Optional[AnalysisStore] = None


def get_analysis_store() -> AnalysisStore:
    """Uygulama omru boyunca tek bir `AnalysisStore` orneği olusturur/dondurur (lazy singleton)."""
    global _analysis_store
    if _analysis_store is None:
        db_path = Path(load_config().memory.sqlite.db_path).with_name("analyses.db")
        _analysis_store = AnalysisStore(db_path)
    return _analysis_store


# 09B - Ask SAFIR: mevcut RAG + LLM soyutlamalarini yeniden kullanan asistan
# servisi (yeni LLM/RAG sistemi KURMAZ). Pipeline'in seed'lenmis RAG'ini ve
# config'teki aktif LLM'i kullanir; testler/E2E `_ask_service`'i (tipki
# `_pipeline`/`_analysis_store` gibi) mock ile override edebilir.
_ask_service: Optional[AskService] = None


def get_ask_service() -> AskService:
    """Uygulama omru boyunca tek bir `AskService` orneği olusturur/dondurur (lazy singleton)."""
    global _ask_service
    if _ask_service is None:
        cfg = load_config()
        llm_client = get_llm_client(cfg.llm, use_mock=cfg.app.use_mock_llm)
        # `use_video` icin: ayni video-tabanli VLM istemcisi (pipeline'inkiyle AYNI config/mod) -
        # yalnizca EVREN gibi answer_video_question destekleyen bir saglayicida ise yarar; diger
        # saglayicilarda AskService bunu SESSIZCE metin-tabanli akisa geri dondurur.
        vlm_client = get_vlm_client(cfg.vlm, use_mock=cfg.app.use_mock_vlm)
        _ask_service = AskService(
            analysis_store=get_analysis_store(),
            rag_service=get_pipeline()._rag_service,  # pipeline'in seed'lenmis RAG'i (yeniden kullanim)
            llm_client=llm_client,
            rag_top_k=cfg.memory.qdrant.top_k,
            vlm_client=vlm_client,
        )
    return _ask_service


# 09C - SAFIR Asistan sohbet gecmisi: `AnalysisStore`/`EventStore` ile AYNI
# hafif desen (dosya-tabanli SQLite, lazy singleton) — yeni bir depolama
# soyutlamasi KURULMAZ, mevcut deseni tekrar eder.
_conversation_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Uygulama omru boyunca tek bir `ConversationStore` orneği olusturur/dondurur (lazy singleton).

    Veritabani dosyasi, `analyses.db` ile AYNI klasorde (`memory.sqlite.db_path`
    ile ayni dizin) ayri bir dosyadadir; mevcut `events`/`analyses` semalarina
    HIC dokunmaz.
    """
    global _conversation_store
    if _conversation_store is None:
        db_path = Path(load_config().memory.sqlite.db_path).with_name("conversations.db")
        _conversation_store = ConversationStore(db_path)
    return _conversation_store


# History icin kalici evidence-frame depolamasi: VLM'e fiilen gonderilen
# TUM evidence kareleri (artik "representative" alt kumesi YOK - kumeleme
# VLM katmaninda yapiliyor, sampler evidence esigini gecen HER kareyi
# gonderiyor). `data/` altinda, projenin mevcut kalici veri dizin desenine
# (analyses.db/events.db) uygun bir alt klasor.
_HISTORY_FRAMES_DIR = "history_frames"


def _history_frame_path(job_id: str, frame_id: str) -> Path:
    """Bir HISTORY evidence-frame'inin diskteki yolunu dondurur (dosya var olmayabilir)."""
    return Path(_DATA_DIR) / _HISTORY_FRAMES_DIR / job_id / f"{frame_id}.jpg"


# Adim 4: SAFIR Asistan'a yuklenen belgelerin ham baytlari (History
# representative-frame'lerle AYNI desen: `data/` altinda, kimlige gore
# izole alt klasorler). `document_id` sunucu tarafinda uuid4 olarak
# uretilir (bkz. ConversationStore.create_document) -> dosya YOLU HICBIR
# ZAMAN kullanicinin verdigi `filename`den turetilmez; path traversal
# riski yapisal olarak yoktur (yalnizca kimlik + sabit uzanti kullanilir).
_CONVERSATION_FILES_DIR = "conversation_files"


def _conversation_file_path(conversation_id: str, document_id: str, file_ext: str) -> Path:
    """Bir yuklenen belgenin diskteki yolunu dondurur (dosya var olmayabilir)."""
    return Path(_DATA_DIR) / _CONVERSATION_FILES_DIR / conversation_id / f"{document_id}.{file_ext}"


def _persist_representative_frames_best_effort(
    job_id: str, trace_events: List[Dict[str, Any]], frames: Dict[str, bytes]
) -> None:
    """VLM'e fiilen gonderilen evidence karelerinin TAMAMINI History icin diske yazar.

    `frames`, `_run_job` sirasinda `_on_frames` ile zaten doldurulmus, JPEG
    baytlarini tasiyan ayni sozluktur; burada yeniden kodlama/yeniden isleme
    YOKTUR — hangi anahtarlarin persist edilecegi, ayni calismada zaten
    uretilmis `trace_events`in `sampler` asamasindaki `evidence_frames[].
    frame_id` listesinden (bkz. `trace_serializer.serialize_sampler`) okunur.
    Kumeleme artik VLM katmaninda yapildigindan, "yalnizca temsili kareler"
    alt kumesi YOKTUR - esik-gecmis TUM evidence kareleri VLM'e gonderildigi
    icin hepsi burada da persist edilir.

    Best-effort: herhangi bir hata yalnizca loglanir; analiz sonucunu ASLA
    etkilemez (bu fonksiyon hicbir zaman disari exception firlatmaz).

    Args:
        job_id: Kalici klasor adi olarak kullanilacak is kimligi.
        trace_events: Bu calismanin tam trace olay listesi (`job.trace_events`in kopyasi).
        frames: Bu calismada biriken frame_id -> JPEG baytlari sozlugu (`job.frames`in kopyasi).
    """
    try:
        sampler_event = next((e for e in trace_events if e.get("stage") == "sampler"), None)
        if sampler_event is None:
            return
        rep_frame_ids = {
            ef["frame_id"] for ef in sampler_event.get("data", {}).get("evidence_frames", [])
        }
        if not rep_frame_ids:
            return
        job_dir = Path(_DATA_DIR) / _HISTORY_FRAMES_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in rep_frame_ids:
            raw = frames.get(frame_id)
            if raw is None:
                continue
            (job_dir / f"{frame_id}.jpg").write_bytes(raw)
    except Exception:  # noqa: BLE001 - best-effort; analiz sonucu ASLA etkilenmez
        logger.exception("History: representative frame'ler diske yazilamadi: job_id=%s", job_id)


def _run_job(
    job_id: str,
    video_source: str,
    user_prompt: str,
    sample_fps_override: Optional[int] = None,
    min_change_threshold_override: Optional[float] = None,
    analysis_mode: Literal["vlm_direct", "low_budget"] = "vlm_direct",
) -> None:
    """Arka plan thread'inde pipeline'i calistirip `JobState`'i gunceller.

    Args:
        job_id: `_jobs` sozlugundeki hedef isin kimligi.
        video_source: Ham (henuz normalize edilmemis) video kaynagi.
        user_prompt: Ajanin odaklanmasi istenen kullanici istemi.
        sample_fps_override: Operator panelinden gelen ornekleme FPS override'i.
        min_change_threshold_override: Operator panelinden gelen hassasiyet esigi override'i.
        analysis_mode: `AnalyzeRequest.analysis_mode` ("vlm_direct"/"low_budget").
    """

    def on_stage(stage_name: str, step: int, total: int) -> None:
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "running"
            job.stage_name = stage_name
            job.step = step
            job.total_steps = total

    # 08 - Gozlemlenebilirlik: her GERCEK asama ciktisini serilestirip JobState'e
    # yazan trace toplayici. Pipeline mantigi degismez; yalnizca `trace` kancasi
    # beslenir. `frames` (kucuk JPEG baytlar) trace olaylarindan AYRI tutulur.
    def _on_event(event: dict) -> None:
        with _jobs_lock:
            job = _jobs[job_id]
            job.trace_events.append(event)
            job.current_stage = event.get("stage", "")

    def _on_frames(frames: Dict[str, bytes]) -> None:
        with _jobs_lock:
            store = _jobs[job_id].frames
            for fid, raw in frames.items():
                if len(store) >= MAX_FRAMES_PER_JOB:
                    break
                store[fid] = raw

    trace_collector = PipelineTraceCollector(job_id, on_event=_on_event, on_frames=_on_frames)

    try:
        pipeline = get_pipeline()
        normalized_source = normalize_video_source(video_source)
        logger.info(
            "Analiz icin cozulen video yolu: girdi=%r -> cozulen=%r (job_id=%s)",
            video_source,
            normalized_source,
            job_id,
        )
        report = pipeline.run(
            normalized_source,
            user_prompt,
            on_stage=on_stage,
            sample_fps_override=sample_fps_override,
            min_change_threshold_override=min_change_threshold_override,
            trace=trace_collector,
            analysis_mode=analysis_mode,
        )
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "done"
            job.result = report
            job.step = job.total_steps
            trace_events_snapshot = list(job.trace_events)
            frames_snapshot = dict(job.frames)
        # History: VLM'e gonderilen representative frame'leri (yalnizca onlari)
        # diske yaz — kendi ici tamamen best-effort (hicbir exception disari
        # cikmaz), analiz sonucunu etkilemez.
        _persist_representative_frames_best_effort(job_id, trace_events_snapshot, frames_snapshot)
        # History: tamamlanmis analizi (gercek SafirReport + trace olay listesi ile)
        # kalici olarak sakla. Persistence hatasi is sonucunu ASLA etkilemez
        # (best-effort) — trace_json serilestirmesi de ayni try/except icinde,
        # bir sorun olursa yalnizca History'nin PipelineTimeline gorunumu eksik
        # kalir, analiz "basarisiz" sayilmaz.
        try:
            get_analysis_store().mark_completed(
                job_id,
                report_json=report.model_dump_json(),
                risk_level=report.risk_level,
                risk_score=report.risk_score,
                risk_status=report.risk_status,
                summary=report.summary or report.natural_language_summary,
                trace_json=json.dumps(trace_events_snapshot, ensure_ascii=False),
            )
        except Exception:  # noqa: BLE001
            logger.exception("History: tamamlanmis analiz kaydedilemedi: job_id=%s", job_id)
    except Exception as exc:  # noqa: BLE001 - hata is durumuna tasinip UI'a iletilir
        logger.exception("Analiz isi basarisiz oldu: job_id=%s", job_id)
        try:
            get_analysis_store().mark_failed(job_id)
        except Exception:  # noqa: BLE001
            logger.exception("History: basarisiz analiz durumu kaydedilemedi: job_id=%s", job_id)
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "error"
            # Guvenli hata mesaji (stack trace/secret UI'a donmez).
            job.error = "Analiz basarisiz oldu. Ayrintilar sunucu loglarinda."
            failed_stage = job.current_stage or "sampler"
            job.trace_events.append(
                {
                    "stage": failed_stage,
                    "status": "failed",
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "duration_ms": None,
                    "summary": "Asama basarisiz oldu.",
                    "data": {},
                    "metadata": {},
                    "error": job.error,
                }
            )


@app.on_event("startup")
def _warmup_evren_models_on_startup() -> None:
    """Her uygulama baslangicinda `scripts/model_warmup.py::warmup_all_models`i BIR KEZ calistirir.

    Amac, EVREN'in fiilen kullanilan dort model takma adinda (`vlm`,
    `llm-fast`, `llm-large`, `bge-m3-embed`) "soguk" olmasindan kaynaklanan
    ilk-istek gecikmesini, operatorun GERCEK ilk analizinden ONCE absorbe
    etmektir (bkz. `scripts/model_warmup.py` modul dokustringi). Isinma
    BASARISIZ olursa (ag hatasi, eksik `EVREN_API_KEY`, vb.) uygulama
    BASLAMAYA DEVAM EDER - bu yalnizca gozlemlenebilirlik/latency-onleme
    icindir, bir davranis KAPISI DEGILDIR (mock modda dahi calisir ama
    gercek EVREN cagrisi yapamayacagi icin ACIKCA basarisiz loglanir).
    """
    try:
        from scripts.model_warmup import warmup_all_models

        report = warmup_all_models(load_config())
        for r in report.results:
            level = logging.INFO if r.ok else logging.WARNING
            logger.log(level, "Model isinma: %s (%s) -> %s (%.0fms) %s", r.model_alias, r.role, "OK" if r.ok else "HATA", r.latency_ms, r.detail)
    except Exception:  # noqa: BLE001 - isinma ASLA uygulama baslangicini engellemez
        logger.exception("Model isinma basarisiz oldu; uygulama yine de baslatiliyor.")


@app.get("/health")
def health() -> dict:
    """Servisin ayakta oldugunu bildiren basit saglik kontrolu uc noktasi."""
    return {"status": "ok", "system": "SAFIR"}


@app.post("/analyze", response_model=SafirReport)
def analyze(request: AnalyzeRequest) -> SafirReport:
    """Verilen video kaynagini uctan uca (senkron) isleyip yapilandirilmis rapor uretir.

    Args:
        request: Video kaynagi ve kullanici istemini iceren istek govdesi.

    Returns:
        `SafirReport` JSON yaniti.

    Raises:
        HTTPException: Pipeline calistirilirken hata olusursa (400).
    """
    try:
        pipeline = get_pipeline()
        normalized_source = normalize_video_source(request.video_source)
        logger.info(
            "Analiz icin cozulen video yolu: girdi=%r -> cozulen=%r",
            request.video_source,
            normalized_source,
        )
        return pipeline.run(
            normalized_source,
            request.user_prompt,
            sample_fps_override=request.sample_fps,
            min_change_threshold_override=request.min_change_threshold,
            analysis_mode=request.analysis_mode,
        )
    except Exception as exc:  # noqa: BLE001 - API tuketicisine anlamli hata donmek icin
        logger.exception("Analiz pipeline hatasi")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analyze/jobs", response_model=AnalyzeJobResponse)
def create_analyze_job(request: AnalyzeRequest) -> AnalyzeJobResponse:
    """Analiz pipeline'ini arka planda baslatir ve takip icin bir is kimligi dondurur.

    Operator paneli, canli ilerleme cubugunu beslemek icin donen `job_id` ile
    `/analyze/jobs/{job_id}` uc noktasini duzenli araliklarla sorgular.

    Args:
        request: Video kaynagi ve kullanici istemini iceren istek govdesi.

    Returns:
        Yeni olusturulan isin kimligi.
    """
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        # Bellek sinir: en eski isleri budayarak `_jobs`'in (ve icindeki frame
        # baytlarinin) sinirsiz buyumesini engelle.
        while len(_jobs) >= _MAX_JOBS:
            _jobs.pop(next(iter(_jobs)))
        _jobs[job_id] = JobState()

    # History: kalici kaydi (created_at + video_source ile) olustur. Persistence
    # hatasi analiz akisini ASLA bozmamali (best-effort).
    try:
        get_analysis_store().create(job_id, request.video_source, status="queued")
    except Exception:  # noqa: BLE001
        logger.exception("History kaydi olusturulamadi (analiz yine de baslatiliyor): job_id=%s", job_id)

    thread = threading.Thread(
        target=_run_job,
        args=(
            job_id,
            request.video_source,
            request.user_prompt,
            request.sample_fps,
            request.min_change_threshold,
            request.analysis_mode,
        ),
        daemon=True,
    )
    thread.start()
    return AnalyzeJobResponse(job_id=job_id)


@app.get("/analyze/jobs/{job_id}", response_model=JobStatusResponse)
def get_analyze_job(job_id: str) -> JobStatusResponse:
    """Bir analiz isinin guncel asamasini ve (varsa) sonucunu dondurur.

    Args:
        job_id: `/analyze/jobs` tarafindan donen is kimligi.

    Returns:
        Isin guncel durumu (`queued`/`running`/`done`/`error`), asama bilgisi
        ve tamamlandiysa `SafirReport` sonucu.

    Raises:
        HTTPException: Verilen `job_id` bilinmiyorsa (404).
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Is bulunamadi: {job_id}")

    return JobStatusResponse(
        status=job.status,
        stage_name=job.stage_name,
        step=job.step,
        total_steps=job.total_steps,
        result=job.result,
        error=job.error,
    )


# Frame/is kimlikleri yalnizca in-memory sozluk anahtari veya (HISTORY icin)
# `data/history_frames/{job_id}/{frame_id}.jpg` dosya yolu bilesenidir; her iki
# kullanimda da path traversal'i onlemek icin bicim kisitlanir (slash/nokta yok).
_FRAME_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


@app.get("/analyze/jobs/{job_id}/frames/{frame_id}")
def get_job_frame(job_id: str, frame_id: str) -> Response:
    """Bir isin trace olaylarinda referans verilen kareyi (JPEG) sunar (base64 yerine).

    Iki kaynak sirayla denenir:
    1. CANLI: bellek-ici `JobState.frames` (is hala `_jobs` icindeyse) — mevcut
       davranis, DEGISMEDI.
    2. HISTORY: is bellekte yoksa (veya bu frame_id o iste yoksa), `_run_job`
       tarafindan best-effort yazilmis kalici representative-frame dosyasina
       (`data/history_frames/{job_id}/{frame_id}.jpg`) bakilir. Yalnizca VLM'e
       fiilen gonderilen representative frame'ler icin doludur; tum evidence
       frame'ler icin bu dosya yoktur (bkz. `_persist_representative_frames_best_effort`).

    Args:
        job_id: Analiz isinin kimligi.
        frame_id: Trace olayindaki `frame_id`/`thumbnail_url` referansi.

    Returns:
        `image/jpeg` yaniti.

    Raises:
        HTTPException: Is/kare bulunamazsa (404) veya kimlik bicimi gecersizse (400).
    """
    if not _FRAME_ID_RE.match(frame_id):
        raise HTTPException(status_code=400, detail="Gecersiz frame_id.")
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Gecersiz job_id.")

    with _jobs_lock:
        job = _jobs.get(job_id)
        raw = job.frames.get(frame_id) if job else None
    if raw is not None:
        return Response(content=raw, media_type="image/jpeg")

    disk_path = _history_frame_path(job_id, frame_id)
    try:
        if disk_path.is_file():
            return Response(content=disk_path.read_bytes(), media_type="image/jpeg")
    except OSError:
        logger.exception("History: kare diskten okunamadi: job_id=%s frame_id=%s", job_id, frame_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Is bulunamadi: {job_id}")
    raise HTTPException(status_code=404, detail=f"Kare bulunamadi: {frame_id}")


@app.get("/analyze/jobs/{job_id}/stream")
async def stream_analyze_job(job_id: str) -> StreamingResponse:
    """Bir isin GERCEK pipeline trace olaylarini canli (SSE) yayinlar.

    Additive uc nokta: mevcut polling (`GET /analyze/jobs/{job_id}`) YERINE GECMEZ,
    onu tamamlar. Baglanan istemci once o ana kadar birikmis tum trace olaylarini
    (replay) alir, sonra yenileri geldikce yayinlanir; is `done`/`error` olunca
    `event: end` ile stream duzgun sonlandirilir.

    SSE bicimi: her olay `data: <json>\\n\\n`; kapanis `event: end` + durum.
    Yalnizca serilestirilmis, base64/secret icermeyen TraceEvent'ler gonderilir.
    """

    async def event_generator():
        with _jobs_lock:
            exists = job_id in _jobs
        if not exists:
            yield f"event: error\ndata: {json.dumps({'detail': 'Is bulunamadi'})}\n\n"
            return

        sent = 0
        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
                events = list(job.trace_events) if job else []
                status = job.status if job else "error"

            for event in events[sent:]:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            sent = len(events)

            if status in ("done", "error"):
                yield f"event: end\ndata: {json.dumps({'status': status})}\n\n"
                return

            await asyncio.sleep(0.2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/alerts/trigger", response_model=AlertTriggerResponse)
def trigger_alert(request: AlertTriggerRequest) -> AlertTriggerResponse:
    """Operatorun manuel/override saha alarmi tetiklemesini isler (mock).

    Yuksek/kritik risk artik pipeline tarafindan OTOMATIK tetiklenir; bu uc
    nokta, operatorun otomatik akis disinda manuel alarm baslatmasi (override)
    icindir. Gercek bir saha entegrasyonunda alarm SMS/anons/SCADA'ya baglanir.

    Args:
        request: Manuel tetiklenen risk skoru/seviyesi ve operator notunu iceren istek.

    Returns:
        Alarmin kabul edildigini bildiren yanit (takip icin `alert_id`).
    """
    pipeline = get_pipeline()
    alert_id = pipeline.trigger_manual_alert(
        risk_score=request.risk_score,
        risk_level=request.risk_level,
        recommended_action=request.recommended_action,
        summary=request.operator_note,
    )
    return AlertTriggerResponse(
        acknowledged=True,
        alert_id=alert_id,
        message="Manuel saha alarmi tetiklendi ve kayit altina alindi.",
    )


@app.post("/alerts/{alert_id}/acknowledge", response_model=AlertAcknowledgeResponse)
def acknowledge_alert(alert_id: str, request: AlertAcknowledgeRequest) -> AlertAcknowledgeResponse:
    """Operatorun otomatik tetiklenmis bir saha alarmini denetlemesini/geri almasini isler.

    Bu, Human-on-the-Loop denetim noktasidir: alarm zaten OTOMATIK tetiklenmistir;
    operator burada onu yalnizca onaylar/geri alir, tetiklenmesini engellemez.

    Args:
        alert_id: `SafirReport.alert_id` (otomatik tetiklenen alarmin kimligi).
        request: Operator notunu iceren istek govdesi.

    Returns:
        Alarmin onaylandigini bildiren yanit.

    Raises:
        HTTPException: `alert_id` bilinmiyorsa (404).
    """
    try:
        pipeline = get_pipeline()
        pipeline.acknowledge_alert(alert_id, request.operator_note)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return AlertAcknowledgeResponse(
        alert_id=alert_id,
        acknowledged=True,
        message="Saha alarmi operator tarafindan onaylandi/denetlendi.",
    )


@app.post("/events/{event_id}/feedback", response_model=FeedbackResponse)
def submit_event_feedback(event_id: int, request: FeedbackRequest) -> FeedbackResponse:
    """Operatorun bir analiz sonucuna verdigi Human-in-the-Loop geri bildirimini kaydeder.

    Bu, otomatik bir RLHF/ince-ayar dongusunu tetiklemez; yalnizca
    `true_positive`/`false_positive` etiketini SQLite'a kalici olarak yazar.

    Args:
        event_id: `SafirReport.event_id` (analiz sonucundaki SQLite kaydi).
        request: `"true_positive"` veya `"false_positive"` iceren istek govdesi.

    Returns:
        Kaydin basariyla islendigini bildiren yanit.

    Raises:
        HTTPException: `feedback` gecersizse (422) veya `event_id` bulunamazsa (404).
    """
    try:
        pipeline = get_pipeline()
        pipeline.record_feedback(event_id, request.feedback)
    except ValueError as exc:
        status_code = 404 if "bulunamadi" in str(exc).lower() else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    return FeedbackResponse(
        event_id=event_id,
        feedback=request.feedback,
        message="Geri bildiriminiz kaydedildi.",
    )


@app.get("/history", response_model=List[HistoryListItem])
def list_history(limit: int = 50, offset: int = 0) -> List[HistoryListItem]:
    """Kalici analiz gecmisini en yeni ONCE (newest-first), sayfali olarak dondurur.

    Args:
        limit: Dondurulecek maksimum kayit (1-200 arasina kelepcelenir).
        offset: Atlanacak kayit sayisi (sayfalama).

    Returns:
        History listesi icin gerekli ozet alanlari iceren kayitlar.
    """
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    records = get_analysis_store().list(limit=limit, offset=offset)
    return [
        HistoryListItem(
            job_id=r.job_id,
            created_at=r.created_at,
            updated_at=r.updated_at,
            video_source=r.video_source,
            status=r.status,
            risk_level=r.risk_level,
            risk_score=r.risk_score,
            risk_status=r.risk_status,
            summary=r.summary,
        )
        for r in records
    ]


@app.get("/history/{job_id}", response_model=HistoryDetail)
def get_history_item(job_id: str) -> HistoryDetail:
    """Tek bir kalici analizi (metadata + gercek `SafirReport`) dondurur.

    Args:
        job_id: `/analyze/jobs` tarafindan uretilmis is kimligi.

    Returns:
        Metadata ve (tamamlanmissa) kalici `SafirReport`.

    Raises:
        HTTPException: `job_id` History'de bulunamazsa (404).
    """
    record = get_analysis_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Analiz bulunamadi: {job_id}")

    report: Optional[SafirReport] = None
    if record.report_json:
        try:
            report = SafirReport.model_validate_json(record.report_json)
        except Exception:  # noqa: BLE001 - bozuk/eski kayit UI'i kirmamali
            logger.exception("History: kayitli rapor JSON'u cozulemedi: job_id=%s", job_id)

    trace_events: List[Dict[str, Any]] = []
    if record.trace_json:
        try:
            trace_events = json.loads(record.trace_json)
        except Exception:  # noqa: BLE001 - bozuk/eski kayit UI'i kirmamali (bos liste ile devam)
            logger.exception("History: kayitli trace JSON'u cozulemedi: job_id=%s", job_id)

    return HistoryDetail(
        job_id=record.job_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        status=record.status,
        video_source=record.video_source,
        report=report,
        trace_events=trace_events,
    )


@app.get("/history/{job_id}/report.pdf")
def get_history_report_pdf(job_id: str) -> Response:
    """Bir analizin `SafirReport`'unu, kanit kareleri gomulu bicimli bir PDF olarak sunar.

    Rapor kaynagi (`get_job_frame` ile ayni oncelik sirasi): once CANLI is
    (`_jobs` icinde henuz tamamlanmis, disari henuz persist edilmis olsun/olmasin),
    sonra HISTORY (kalici `AnalysisStore` kaydi). PDF uretimi, Operator Panelinde
    (Streamlit) zaten kullanilan `ReportExporter.to_pdf()` (reportlab) ile
    YENIDEN KULLANILIR — yeni bir rapor-bicimlendirme mantigi YAZILMAZ.

    Args:
        job_id: Analiz isinin kimligi.

    Returns:
        `application/pdf` yaniti (indirilebilir dosya adi ile).

    Raises:
        HTTPException: `job_id` gecersizse (400), rapor bulunamazsa (404) veya
            PDF uretim bagimliligi (reportlab) kurulu degilse (503).
    """
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Gecersiz job_id.")

    report_dict: Optional[Dict[str, Any]] = None
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job and job.result:
            report_dict = job.result.model_dump(mode="json")
    if report_dict is None:
        record = get_analysis_store().get(job_id)
        if record and record.report_json:
            try:
                report_dict = json.loads(record.report_json)
            except Exception:  # noqa: BLE001 - bozuk kayit PDF uretimini kirmamali
                logger.exception("PDF export: kayitli rapor JSON'u cozulemedi: job_id=%s", job_id)

    if report_dict is None:
        raise HTTPException(status_code=404, detail=f"Bu analiz icin rapor bulunamadi: {job_id}")

    try:
        from src.ui.report_export import ReportExporter
    except ImportError:
        raise HTTPException(
            status_code=503, detail="PDF disa aktarma su anda kullanilamiyor (reportlab kurulu degil)."
        )

    try:
        pdf_bytes = ReportExporter(report_dict).to_pdf()
    except Exception as exc:  # noqa: BLE001 - PDF uretim hatasi 500 yerine anlasilir 500 detayi versin
        logger.exception("PDF export basarisiz: job_id=%s", job_id)
        raise HTTPException(status_code=500, detail=f"PDF uretilemedi: {exc}") from exc

    video_name = str(report_dict.get("video_source") or job_id)
    video_name = video_name.replace("/", "_").replace("\\", "_")
    filename = f"safir_report_{video_name}.pdf"
    # HTTP baslik degerleri Latin-1 ile SINIRLIDIR (Starlette/ASGI bunu
    # dogrudan encode eder) - video adi Turkce karakter icerirse (ör. "ı",
    # "ş", "ğ") bu ASLA Latin-1'e sigmaz ve `UnicodeEncodeError` ile 500
    # dondurur (indirme TAMAMEN basarisiz olur). RFC 6266/5987 cozumu: ASCII
    # bir yedek `filename` (Turkce karakterler duz ASCII'ye indirgenir) +
    # tam-sadakatli, yuzde-kodlanmis bir `filename*` birlikte verilir; Latin-1
    # sinirini asan hicbir karakter dogrudan baslikta YER ALMAZ.
    ascii_filename = filename.encode("ascii", "replace").decode("ascii")
    encoded_filename = urllib.parse.quote(filename)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{encoded_filename}'
            )
        },
    )


@app.post("/ask", response_model=AskResponse)
def ask_safir(request: AskRequest) -> AskResponse:
    """Ask SAFIR: soruyu (opsiyonel analiz baglami + RAG mevzuati ile) dayanakli yanitlar.

    - `job_id` verilirse: o analizin kalici `SafirReport`'u ONCELIKLI baglam olur.
    - Her durumda mevcut `EmbeddingRAGService` ile ilgili ISG mevzuati getirilir.
    - `use_video=True` verilirse (ve is/video hala diskte varsa) soru ONCE
      dogrudan videoya sorulur (EVREN prefix-cache avantaji, bkz.
      `AskService._try_video_answer`); basarisiz olursa SESSIZCE metin-tabanli
      akisa doner.
    - Cevap YALNIZCA guvenli baglamdan uretilir; raw_response/internal reasoning/
      secret/base64 modele gonderilmez ve yanitta yer almaz.

    Args:
        request: `question` (zorunlu) + `job_id` (opsiyonel).

    Returns:
        `answer`, `sources` (gercek retrieval sonuclari), `job_id`, `context_used`.

    Raises:
        HTTPException: `job_id` bilinmiyorsa (404), soru bossa (422),
            LLM/servis cevap uretemezse (503; stack trace UI'a donmez).
    """
    try:
        result = get_ask_service().answer(request.question, request.job_id, use_video=request.use_video)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail=f"Analiz bulunamadi: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - guvenli hata; ayrintilar loglarda
        logger.exception("Ask SAFIR cevap uretemedi")
        raise HTTPException(status_code=503, detail="SAFIR su anda cevap olusturamadi.") from exc

    return AskResponse(
        answer=result.answer,
        sources=[AskSource(type=s.type, text=s.text, score=s.score, label=s.label) for s in result.sources],
        job_id=result.job_id,
        context_used=result.context_used,
    )


@app.get("/ask/suggestions", response_model=AskSuggestionsResponse)
def ask_safir_suggestions(job_id: str) -> AskSuggestionsResponse:
    """Verilen analiz raporuna OZGU, dinamik takip sorusu onerileri (bkz. `suggestion_engine.py`).

    Mentor eleştirisi ("Jurinin Demo Videosu ve Beklentileri" - "inisiyatif
    alma ve dogru sorulari sorma"): sabit oneri listesi HER raporda AYNIdir,
    hicbir inisiyatif GOSTERMEZ. Bu uc nokta, raporun KENDI icerigine (siniflandirilamayan
    olay, insan incelemesine dusen eskalasyon, VLM'in kendi metninde belirttigi
    belirsizlik, kisi-iceren olay kategorisi) bakip O RAPORA OZGU sorular onerir.

    Args:
        job_id: Onerilerin cikarilacagi analizin kimligi.

    Returns:
        0-4 arasi oneri metni; hicbir sinyal yoksa BOS liste doner (UI statik/genel
        onerilere geri doner - bkz. frontend `SUGGESTIONS`).

    Raises:
        HTTPException: `job_id` bilinmiyorsa (404) veya raporu HENUZ yoksa (404).
    """
    record = get_analysis_store().get(job_id)
    if record is None or not record.report_json:
        raise HTTPException(status_code=404, detail=f"Analiz bulunamadi: {job_id}")
    report = SafirReport.model_validate_json(record.report_json)
    return AskSuggestionsResponse(suggestions=build_dynamic_suggestions(report))


_MAX_CONVERSATION_TITLE_LEN = 80


@app.post("/conversations", response_model=ConversationSummary)
def create_conversation(request: ConversationCreateRequest) -> ConversationSummary:
    """SAFIR Asistan icin yeni bir sohbet kaydi olusturur.

    Baslik icin HICBIR LLM cagrisi yapilmaz (basit kirpma; bkz. `request.title`,
    frontend'de genelde ilk kullanici sorusundan turetilir).

    Args:
        request: Opsiyonel baslik ve opsiyonel bagli analiz kimligi (`job_id`).

    Returns:
        Olusturulan sohbetin ozeti.
    """
    title = (request.title or "").strip()[:_MAX_CONVERSATION_TITLE_LEN] or None
    record = get_conversation_store().create(title=title, job_id=request.job_id)
    return ConversationSummary(
        conversation_id=record.conversation_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        title=record.title,
        job_id=record.job_id,
    )


@app.get("/conversations", response_model=List[ConversationSummary])
def list_conversations(limit: int = 50, offset: int = 0) -> List[ConversationSummary]:
    """Sohbetleri en son GUNCELLENEN once (updated_at DESC), sayfali dondurur.

    Args:
        limit: Dondurulecek maksimum kayit (1-200 arasina kelepcelenir).
        offset: Atlanacak kayit sayisi (sayfalama).
    """
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    records = get_conversation_store().list_with_message_counts(limit=limit, offset=offset)
    return [
        ConversationSummary(
            conversation_id=r.conversation_id,
            created_at=r.created_at,
            updated_at=r.updated_at,
            title=r.title,
            job_id=r.job_id,
            message_count=r.message_count,
        )
        for r in records
    ]


@app.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(conversation_id: str) -> ConversationDetailResponse:
    """Tek bir sohbeti, tam mesaj gecmisiyle (kronolojik) birlikte dondurur.

    Args:
        conversation_id: `POST /conversations` tarafindan uretilmis sohbet kimligi.

    Raises:
        HTTPException: `conversation_id` bulunamazsa (404).
    """
    store = get_conversation_store()
    record = store.get(conversation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Sohbet bulunamadi: {conversation_id}")

    messages = store.list_messages(conversation_id)
    context = store.list_context(conversation_id)
    documents = store.list_documents(conversation_id)
    return ConversationDetailResponse(
        conversation_id=record.conversation_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        title=record.title,
        job_id=record.job_id,
        messages=[
            ConversationMessageOut(id=m.id, role=m.role, content=m.content, created_at=m.created_at)
            for m in messages
        ],
        context=[
            ConversationContextOut(id=c.id, kind=c.kind, label=c.label, content=c.content, created_at=c.created_at)
            for c in context
        ],
        documents=[
            ConversationDocumentOut(
                document_id=d.document_id,
                filename=d.filename,
                file_ext=d.file_ext,
                file_size_bytes=d.file_size_bytes,
                page_count=d.page_count,
                status=d.status,
                error_message=d.error_message,
                created_at=d.created_at,
            )
            for d in documents
        ],
    )


@app.post("/conversations/{conversation_id}/messages", response_model=ConversationMessageOut)
def add_conversation_message(
    conversation_id: str, request: ConversationMessageCreateRequest
) -> ConversationMessageOut:
    """Bir sohbete yeni bir mesaj ekler.

    ONEMLI: Bu yalnizca UI/gecmis icindir — burada saklanan mesajlar `POST /ask`
    veya `GET /ask/stream`e HICBIR ZAMAN geri beslenmez (bkz.
    `src/assistant/ask_service.py` modul dokustringi); `AskService`in kendi
    (opsiyonel, sinirli) `history` mekanizmasi `GET /ask/stream` tarafindan
    AYRICA ve acikca yonetilir.

    Args:
        conversation_id: Hedef sohbetin kimligi.
        request: `role` (`"user"`/`"assistant"`) + `content`.

    Raises:
        HTTPException: `conversation_id` bulunamazsa (404) veya `role` gecersizse (422).
    """
    if request.role not in ("user", "assistant"):
        raise HTTPException(status_code=422, detail="role 'user' veya 'assistant' olmalidir.")

    store = get_conversation_store()
    if store.get(conversation_id) is None:
        raise HTTPException(status_code=404, detail=f"Sohbet bulunamadi: {conversation_id}")

    record = store.add_message(conversation_id, request.role, request.content)
    return ConversationMessageOut(id=record.id, role=record.role, content=record.content, created_at=record.created_at)


@app.post("/conversations/{conversation_id}/context", response_model=ConversationContextOut)
def add_conversation_context(
    conversation_id: str, request: ConversationContextCreateRequest
) -> ConversationContextOut:
    """Bir sohbete kullanicinin kendi ekledigi bir baglam notu ekler (Adim 3: yalnizca metin).

    ONEMLI (bkz. `AskService._prepare`): bu not `/ask/stream` prompt'una AYRI
    ve acikca "SISTEM TALIMATI DEGIL, ANALIZ KANITI DEGIL" etiketiyle eklenir;
    hicbir zaman sistem talimati veya analiz kaniti olarak islenmez. Yalnizca
    VERILEN sohbete baglidir — baska bir sohbete asla sizmaz (bkz.
    `ConversationStore.list_context`/`remove_context`, her zaman
    `conversation_id` ile filtrelenir).

    Args:
        conversation_id: Hedef sohbetin kimligi.
        request: Not metni (`content`, azami `_MAX_CONTEXT_CONTENT_LEN` karakter) + opsiyonel kisa `label`.

    Raises:
        HTTPException: `conversation_id` bulunamazsa (404).
    """
    store = get_conversation_store()
    if store.get(conversation_id) is None:
        raise HTTPException(status_code=404, detail=f"Sohbet bulunamadi: {conversation_id}")

    record = store.add_context(conversation_id, content=request.content.strip(), label=request.label)
    return ConversationContextOut(
        id=record.id, kind=record.kind, label=record.label, content=record.content, created_at=record.created_at
    )


@app.delete("/conversations/{conversation_id}/context/{context_id}")
def remove_conversation_context(conversation_id: str, context_id: int) -> Dict[str, bool]:
    """Bir sohbetten bir baglam notunu kaldirir; kaldirilan not bir daha AYNI sohbette bile ASK'a gonderilmez.

    Args:
        conversation_id: Hedef sohbetin kimligi.
        context_id: Kaldirilacak baglam kaydinin kimligi.

    Raises:
        HTTPException: Kayit bu sohbette bulunamazsa (404) — baska bir
            sohbete ait bir `context_id` verilirse de 404 doner (sizinti YOK).
    """
    removed = get_conversation_store().remove_context(conversation_id, context_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Baglam bulunamadi: {context_id}")
    return {"removed": True}


# -----------------------------------------------------------------------------
# Adim 4 - SAFIR Asistan: kullanici belge yukleme (PDF/TXT/DOCX).
#
# Akis: dosya -> (uzanti/boyut/sayi dogrulamasi) -> diske YAZ (ham bayt, DB'ye
# DEGIL) -> DB'de `status='processing'` kaydi olustur -> arka plan thread'inde
# metin cikar + chunk'la + DB'ye yaz -> `status='ready'|'error'`. Ikili veri
# hicbir zaman LLM'e gonderilmez (bkz. src/memory/document_extraction.py).
# -----------------------------------------------------------------------------


def _process_conversation_document(conversation_id: str, document_id: str, filename: str, raw: bytes) -> None:
    """Arka plan thread'i: metni cikarir, chunk'lar, kalici olarak saklar, durumu gunceller.

    ONEMLI: bu fonksiyon bir `threading.Thread` hedefi olarak calisir — hicbir
    exception'in sessizce is parcaciğini coktumesine izin verilmez; her hata
    `status='error'` + kullaniciya gosterilebilir kisa bir `error_message`
    olarak DB'ye yazilir (analiz pipeline'indaki degraded-rapor deseniyle
    tutarli: hata gizlenmez, ama sistem de cokmez).
    """
    store = get_conversation_store()
    try:
        page_texts, page_count = document_extraction.extract_text(filename, raw)
        chunks = document_extraction.chunk_page_texts(page_texts)
        if not chunks:
            raise document_extraction.ExtractionError("Belgeden hicbir metin parcasi cikarilamadi.")
        store.add_chunks(document_id, conversation_id, chunks)
        store.update_document_status(document_id, status="ready", page_count=page_count)
    except document_extraction.ExtractionError as exc:
        store.update_document_status(document_id, status="error", error_message=str(exc))
    except Exception:  # noqa: BLE001 - arka plan thread'i ASLA sessizce cokmemeli
        logger.exception("Belge isleme basarisiz: document_id=%s", document_id)
        store.update_document_status(
            document_id, status="error", error_message="Belge islenemedi (beklenmeyen hata)."
        )


@app.post("/conversations/{conversation_id}/documents", response_model=ConversationDocumentOut)
async def upload_conversation_document(
    conversation_id: str, file: UploadFile = File(...)
) -> ConversationDocumentOut:
    """Bir sohbete belge yukler (yalnizca PDF/TXT/DOCX); metin cikarimi ARKA PLANDA yapilir.

    Dogrulama sirasi: sohbet var mi (404) -> uzanti izinli mi (422) -> sohbet
    basina azami belge sinirina ulasilmis mi (422) -> boyut siniri (413,
    bellek-guvenli parca-parca okuma ile — tek seferde sinirsiz `read()` YOK).
    Basarili yanit `status="processing"` doner; istemci `GET /conversations/{id}`
    ile (zaten var olan uc nokta) durumu izler — ayrica bir "durum" uc noktasi
    GEREKMEZ.

    Args:
        conversation_id: Hedef sohbetin kimligi.
        file: Yuklenen dosya (multipart/form-data, alan adi `file`).

    Raises:
        HTTPException: Sohbet yoksa (404), uzanti desteklenmiyorsa/dosya
            bossa (422), belge sinirina ulasildiysa (422), dosya cok
            buyukse (413).
    """
    store = get_conversation_store()
    if store.get(conversation_id) is None:
        raise HTTPException(status_code=404, detail=f"Sohbet bulunamadi: {conversation_id}")

    filename = (file.filename or "").strip() or "belge"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in document_extraction.ALLOWED_EXTENSIONS:
        allowed = "/".join(sorted(document_extraction.ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=422, detail=f"Desteklenmeyen dosya turu: .{ext or '?'} (yalnizca {allowed})")

    if store.count_documents(conversation_id) >= document_extraction.MAX_FILES_PER_CONVERSATION:
        raise HTTPException(
            status_code=422,
            detail=f"Bu sohbette azami {document_extraction.MAX_FILES_PER_CONVERSATION} belge olabilir.",
        )

    # Bellek-guvenli parca-parca okuma: siniri asan bir yukleme, tamami
    # belleğe alinmadan erken reddedilir.
    max_bytes = document_extraction.MAX_FILE_SIZE_BYTES
    pieces: List[bytes] = []
    total = 0
    while True:
        piece = await file.read(1024 * 1024)
        if not piece:
            break
        total += len(piece)
        if total > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"Dosya cok buyuk (azami {max_bytes // (1024 * 1024)} MB)."
            )
        pieces.append(piece)
    raw = b"".join(pieces)
    if not raw:
        raise HTTPException(status_code=422, detail="Dosya bos.")

    record = store.create_document(conversation_id, filename=filename, file_ext=ext, file_size_bytes=len(raw))

    file_path = _conversation_file_path(conversation_id, record.document_id, ext)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(raw)
    except OSError:
        logger.exception("Belge diske yazilamadi: document_id=%s", record.document_id)
        store.update_document_status(record.document_id, status="error", error_message="Dosya diske yazilamadi.")
        return ConversationDocumentOut(
            document_id=record.document_id,
            filename=filename,
            file_ext=ext,
            file_size_bytes=len(raw),
            page_count=None,
            status="error",
            error_message="Dosya diske yazilamadi.",
            created_at=record.created_at,
        )

    threading.Thread(
        target=_process_conversation_document,
        args=(conversation_id, record.document_id, filename, raw),
        daemon=True,
    ).start()

    return ConversationDocumentOut(
        document_id=record.document_id,
        filename=filename,
        file_ext=ext,
        file_size_bytes=len(raw),
        page_count=None,
        status="processing",
        error_message=None,
        created_at=record.created_at,
    )


@app.delete("/conversations/{conversation_id}/documents/{document_id}")
def remove_conversation_document(conversation_id: str, document_id: str) -> Dict[str, bool]:
    """Bir belgeyi (ve tum chunk'larini + diskteki dosyasini) kaldirir; yalnizca VERILEN sohbete aitse.

    Raises:
        HTTPException: Belge bu sohbette bulunamazsa (404) — baska bir
            sohbete ait bir `document_id` verilirse de 404 doner (sizinti YOK).
    """
    store = get_conversation_store()
    record = store.get_document(conversation_id, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Belge bulunamadi: {document_id}")

    store.remove_document(conversation_id, document_id)
    try:
        _conversation_file_path(conversation_id, document_id, record.file_ext).unlink(missing_ok=True)
    except OSError:
        logger.exception("Belge dosyasi diskten silinemedi: document_id=%s", document_id)
    return {"removed": True}


# -----------------------------------------------------------------------------
# 11 - Sistem Verileri (Data Center): operatore, mevcut SQLite/disk depolarinda
# GERCEKTEN neyin saklandigini gosteren, TAMAMEN read-only bir ozet uc noktasi.
# Yeni bir depolama soyutlamasi/tablosu YOKTUR: mevcut AnalysisStore/
# ConversationStore uzerinden SAYIM yapar; representative frame sayisi da
# gercek diskteki dosyalarin (`_history_frame_path`) var olup olmadigina
# bakilarak hesaplanir (best-effort persistence ile TUTARLI - yalnizca
# GERCEKTEN yazilmis dosyalar sayilir). SQL calistirma/DELETE/UPDATE UC NOKTASI
# BULUNMAZ; yalnizca sayim + zaten var olan `/history`, `/history/{id}`,
# `/conversations*` uc noktalariyla birlikte kullanilmasi amaclanir (detay
# drill-down icin YENI bir endpoint GEREKMEZ).
# -----------------------------------------------------------------------------
_SYSTEM_OVERVIEW_SCAN_LIMIT = 500
"""Ozet sayimlar icin taranacak azami analiz/sohbet kaydi. Operator-tarafi
tanilama sayfasi oldugu icin (yuksek trafikli bir uc nokta degil) basit bir
sabit sinir yeterlidir; sinirsiz tarama/agir sayfalama gerektirmez."""


class SystemOverviewTotals(BaseModel):
    """Data Center ust KPI satiri icin gercek DB/disk sayimlari."""

    total_analyses: int
    completed_analyses: int
    failed_analyses: int
    running_or_queued_analyses: int
    total_conversations: int
    total_messages: int
    analyses_with_trace: int
    stored_representative_frame_count: int
    # --- RAG + Prompt Injection Guard telemetrisi (persisted trace_events uzerinden,
    # bkz. _aggregate_rag_security_totals) - hicbir alan UYDURULMAZ: veri yoksa
    # sayim alanlari 0, ortalama alanlari `None` ("N/A") kalir. ---
    total_events_detected: int
    critical_risk_analyses: int
    rag_query_count: int
    rag_zero_result_count: int
    avg_embedding_latency_ms: Optional[float] = None
    avg_rerank_latency_ms: Optional[float] = None
    avg_total_rag_latency_ms: Optional[float] = None
    guard_checks: int
    guard_allowed: int
    guard_quarantined: int
    guard_failures: int
    guard_fail_closed_blocks: int
    avg_guard_latency_ms: Optional[float] = None
    avg_guard_confidence: Optional[float] = None


class SystemOverviewResponse(BaseModel):
    """`GET /system/overview` yaniti."""

    totals: SystemOverviewTotals
    generated_at: str
    scan_limit: int = Field(description="Sayimlarin tarandigi azami kayit sayisi (bkz. _SYSTEM_OVERVIEW_SCAN_LIMIT).")


def _count_persisted_representative_frames(trace_json: Optional[str], job_id: str) -> int:
    """Bir analizin trace'inde referans verilen evidence frame'lerden DISKTE GERCEKTEN var olanlari sayar.

    `_persist_representative_frames_best_effort` ile ayni frame_id kaynagini
    (sampler asamasi `evidence_frames[]`) kullanir; diskte bulunmayan
    (best-effort yazma basarisiz olmus) kareler sayilmaz — boylece bu sayim
    HER ZAMAN gercekten erisebilecek kare sayisini yansitir.
    """
    if not trace_json:
        return 0
    try:
        events = json.loads(trace_json)
    except Exception:  # noqa: BLE001 - bozuk/eski kayit sayimi kirmamali
        return 0
    sampler_event = next((e for e in events if e.get("stage") == "sampler"), None)
    if sampler_event is None:
        return 0
    count = 0
    for ef in sampler_event.get("data", {}).get("evidence_frames", []):
        frame_id = ef.get("frame_id")
        if frame_id and _history_frame_path(job_id, frame_id).is_file():
            count += 1
    return count


@dataclass
class _RagSecurityAggregate:
    """`get_system_overview()` icin, tum kaydedilmis trace_events uzerinde biriken GERCEK toplamlar."""

    total_events_detected: int = 0
    critical_risk_analyses: int = 0
    rag_query_count: int = 0
    rag_zero_result_count: int = 0
    embedding_latencies: List[float] = field(default_factory=list)
    rerank_latencies: List[float] = field(default_factory=list)
    total_rag_latencies: List[float] = field(default_factory=list)
    guard_checks: int = 0
    guard_allowed: int = 0
    guard_quarantined: int = 0
    guard_failures: int = 0
    guard_fail_closed_blocks: int = 0
    guard_latencies: List[float] = field(default_factory=list)
    guard_confidences: List[float] = field(default_factory=list)


def _accumulate_rag_security_from_trace(trace_json: Optional[str], agg: _RagSecurityAggregate) -> None:
    """Bir analizin `trace_json`'undaki `events`/`report`/`rag_security` stage verilerini `agg`e ekler.

    Bozuk/eski (bu alanlari HENUZ icermeyen) kayitlar SESSIZCE atlanir -
    UYDURULMUS bir deger EKLENMEZ, yalnizca gercekten mevcut olan sayilir.
    """
    if not trace_json:
        return
    try:
        events = json.loads(trace_json)
    except Exception:  # noqa: BLE001 - bozuk/eski kayit sayimi kirmamali
        return

    events_event = next((e for e in events if e.get("stage") == "events"), None)
    if events_event is not None:
        agg.total_events_detected += len(events_event.get("data", {}).get("detected_events", []))

    report_event = next((e for e in events if e.get("stage") == "report"), None)
    if report_event is not None and report_event.get("data", {}).get("risk_level") == "kritik":
        agg.critical_risk_analyses += 1

    rag_security_event = next((e for e in events if e.get("stage") == "rag_security"), None)
    if rag_security_event is None:
        return
    data = rag_security_event.get("data", {})

    rag = data.get("rag")
    if rag is not None:
        agg.rag_query_count += 1
        if rag.get("zero_result"):
            agg.rag_zero_result_count += 1
        if rag.get("embedding_latency_ms") is not None:
            agg.embedding_latencies.append(rag["embedding_latency_ms"])
        if rag.get("rerank_latency_ms") is not None:
            agg.rerank_latencies.append(rag["rerank_latency_ms"])
        if rag.get("total_latency_ms") is not None:
            agg.total_rag_latencies.append(rag["total_latency_ms"])

    for check in data.get("security", []):
        agg.guard_checks += 1
        if check.get("action") == "quarantine":
            agg.guard_quarantined += 1
            if check.get("guard_failed"):
                agg.guard_fail_closed_blocks += 1
        else:
            agg.guard_allowed += 1
        if check.get("guard_failed"):
            agg.guard_failures += 1
        if check.get("latency_ms") is not None:
            agg.guard_latencies.append(check["latency_ms"])
        if check.get("confidence") is not None:
            agg.guard_confidences.append(check["confidence"])


def _avg_or_none(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 1) if values else None


@app.get("/system/overview", response_model=SystemOverviewResponse)
def get_system_overview() -> SystemOverviewResponse:
    """Operator icin 'Sistem Verileri' (Data Center) ozet KPI'lari — tamamen read-only.

    Hicbir SQL sorgusu kullaniciya gosterilmez/calistirilmaz; yalnizca mevcut
    `AnalysisStore`/`ConversationStore` uzerinden gercek sayimlar yapilir.

    Returns:
        Analiz/sohbet/mesaj/trace/representative-frame sayimlarini iceren ozet.
    """
    analysis_records = get_analysis_store().list(limit=_SYSTEM_OVERVIEW_SCAN_LIMIT, offset=0)
    completed = sum(1 for r in analysis_records if r.status == "completed")
    failed = sum(1 for r in analysis_records if r.status == "failed")
    with_trace = sum(1 for r in analysis_records if r.trace_json)
    stored_frames = sum(
        _count_persisted_representative_frames(r.trace_json, r.job_id) for r in analysis_records
    )

    rag_security_agg = _RagSecurityAggregate()
    for r in analysis_records:
        _accumulate_rag_security_from_trace(r.trace_json, rag_security_agg)

    conversation_records = get_conversation_store().list_with_message_counts(
        limit=_SYSTEM_OVERVIEW_SCAN_LIMIT, offset=0
    )
    total_messages = sum(c.message_count for c in conversation_records)

    return SystemOverviewResponse(
        totals=SystemOverviewTotals(
            total_analyses=len(analysis_records),
            completed_analyses=completed,
            failed_analyses=failed,
            running_or_queued_analyses=len(analysis_records) - completed - failed,
            total_conversations=len(conversation_records),
            total_messages=total_messages,
            analyses_with_trace=with_trace,
            stored_representative_frame_count=stored_frames,
            total_events_detected=rag_security_agg.total_events_detected,
            critical_risk_analyses=rag_security_agg.critical_risk_analyses,
            rag_query_count=rag_security_agg.rag_query_count,
            rag_zero_result_count=rag_security_agg.rag_zero_result_count,
            avg_embedding_latency_ms=_avg_or_none(rag_security_agg.embedding_latencies),
            avg_rerank_latency_ms=_avg_or_none(rag_security_agg.rerank_latencies),
            avg_total_rag_latency_ms=_avg_or_none(rag_security_agg.total_rag_latencies),
            guard_checks=rag_security_agg.guard_checks,
            guard_allowed=rag_security_agg.guard_allowed,
            guard_quarantined=rag_security_agg.guard_quarantined,
            guard_failures=rag_security_agg.guard_failures,
            guard_fail_closed_blocks=rag_security_agg.guard_fail_closed_blocks,
            avg_guard_latency_ms=_avg_or_none(rag_security_agg.guard_latencies),
            avg_guard_confidence=_avg_or_none(rag_security_agg.guard_confidences),
        ),
        generated_at=datetime.datetime.utcnow().isoformat() + "Z",
        scan_limit=_SYSTEM_OVERVIEW_SCAN_LIMIT,
    )


@app.get("/ask/stream")
async def ask_safir_stream(
    question: str,
    job_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    use_video: bool = False,
) -> StreamingResponse:
    """Ask SAFIR'i SSE ile parca parca (streaming) yanitlar.

    Mevcut `POST /ask` ile AYNI baglam-olusturma/guvenlik mantigini (`AskService`)
    kullanir; yalnizca LLM yanitini parca parca yayinlar. `/ask`in davranisi/
    response modeli DEGISMEZ — bu tamamen ADDITIVE, ayri bir uc noktadir.

    `conversation_id` verilirse, o sohbetin (bu soru HENUZ eklenmemis) onceki
    mesajlarindan en fazla son birkaci (bkz. `ask_service._MAX_HISTORY_MESSAGES`)
    baglam olarak eklenir — yalnizca takip sorularini anlamak icindir, KANIT
    olarak sunulmaz (bkz. `ASK_SYSTEM_PROMPT`). Mesaj YAZMA (persistence) burada
    YAPILMAZ; istemci, akis basariyla tamamlaninca kullanici sorusu + nihai
    cevabi AYRICA `POST /conversations/{id}/messages` ile kaydeder (bkz. o
    endpoint'in dokustringi) — boylece hicbir token/parca tek tek veritabanina
    yazilmaz, yalnizca tamamlanmis nihai mesajlar kalici olur.

    SSE bicimi (mevcut `/analyze/jobs/{job_id}/stream` ile ayni konvansiyon):
      - ilk olay: `data: {"sources": [...], "context_used": [...], "job_id": ...}`
      - sonraki olaylar: `data: {"delta": "..."}` (metin parcasi)
      - kapanis: `event: end` VEYA hata: `event: error`
    """

    async def event_generator():
        history: List[Tuple[str, str]] = []
        user_context: List[Tuple[Optional[str], str]] = []
        document_context: List[Tuple[str, Optional[int], str]] = []
        if conversation_id:
            try:
                history = [(m.role, m.content) for m in get_conversation_store().list_messages(conversation_id)]
            except Exception:  # noqa: BLE001 - gecmis okunamazsa baglamsiz (ama hatasiz) devam edilir
                logger.exception(
                    "Ask SAFIR stream: konusma gecmisi okunamadi: conversation_id=%s", conversation_id
                )
            try:
                user_context = [
                    (c.label, c.content) for c in get_conversation_store().list_context(conversation_id)
                ]
            except Exception:  # noqa: BLE001 - baglam okunamazsa baglamsiz (ama hatasiz) devam edilir
                logger.exception(
                    "Ask SAFIR stream: kullanici baglami okunamadi: conversation_id=%s", conversation_id
                )
            try:
                # Belgenin TAMAMI DEGIL — yalnizca bu soruyla lexical olarak
                # en ilgili birkac chunk (bkz. document_extraction.lexical_search).
                all_chunks = get_conversation_store().list_chunks_for_conversation(conversation_id)
                relevant_chunks = document_extraction.lexical_search(
                    all_chunks, question, top_k=document_extraction.MAX_RETRIEVED_CHUNKS
                )
                document_context = [(c.filename, c.page_number, c.content) for c in relevant_chunks]
            except Exception:  # noqa: BLE001 - belge baglami okunamazsa baglamsiz (ama hatasiz) devam edilir
                logger.exception(
                    "Ask SAFIR stream: belge baglami okunamadi: conversation_id=%s", conversation_id
                )

        try:
            meta, chunks = get_ask_service().stream_answer(
                question,
                job_id,
                history=history,
                user_context=user_context,
                document_context=document_context,
                use_video=use_video,
            )
        except JobNotFound as exc:
            yield f"event: error\ndata: {json.dumps({'detail': f'Analiz bulunamadi: {exc}'})}\n\n"
            return
        except ValueError as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return

        meta_payload = {
            "sources": [
                AskSource(type=s.type, text=s.text, score=s.score, label=s.label).model_dump() for s in meta.sources
            ],
            "job_id": meta.job_id,
            "context_used": meta.context_used,
        }
        yield f"data: {json.dumps(meta_payload, ensure_ascii=False)}\n\n"

        got_any = False
        try:
            for delta in chunks:
                if delta:
                    got_any = True
                    yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
        except Exception:  # noqa: BLE001 - guvenli hata; ayrintilar loglarda
            logger.exception("Ask SAFIR stream: LLM akisi basarisiz oldu")
            yield f"event: error\ndata: {json.dumps({'detail': 'SAFIR su anda cevap olusturamadi.'})}\n\n"
            return

        if not got_any:
            yield f"event: error\ndata: {json.dumps({'detail': 'SAFIR su anda cevap olusturamadi.'})}\n\n"
            return

        yield f"event: end\ndata: {json.dumps({'status': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(app, host=config.api.host, port=config.api.port)
