"""SAFIR uctan uca pipeline giris noktasi (FastAPI servisi).

Akis: Video Input -> AdaptiveFrameSampler (CPU) -> VLM (vLLM) -> ContextBuilder
(SQLite + FAISS RAG) -> SafirAgent (LangGraph, vLLM) -> SafirReport (JSON).

Operator paneli (`src/ui/dashboard.py`) icin canli ilerleme takibi
`/analyze/jobs` uzerinden asenkron (arka plan thread'i + durum sorgulama)
olarak saglanir; `/analyze` ise tek seferlik senkron kullanim icin korunur.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.observability.trace_serializer import MAX_FRAMES_PER_JOB, PipelineTraceCollector

from src.agent.langgraph_agent import SafirAgent
from src.agent.tools import RetrieverTool
from src.assistant.ask_service import AskService, JobNotFound
from src.decision.escalation import EscalationPolicy
from src.event_analysis.event_builder import EventBuilder
from src.event_analysis.event_engine import EventEngine
from src.event_analysis.event_history import EventHistory
from src.event_analysis.rule_engine import RuleEngine
from src.event_analysis.schemas import DetectedEvent, EventEngineInput, RuleMatch, TemporalEvent
from src.event_analysis.temporal_reasoner import DEFAULT_RELATION_WINDOW_SEC, TemporalReasoner
from src.memory.analysis_store import AnalysisStore
from src.memory.context_builder import ContextBuilder
from src.memory.embedding_rag_service import EmbeddingRAGService
from src.memory.event_store import EventStore
from src.sampler.adaptive_sampler import EventCluster, EvidenceFrame, sampler_from_config
from src.sampler.context.representative_frame_extractor import RepresentativeFrameExtractor
from src.schemas.report import EvidenceFrameOut, SafirReport, SamplerStats, TimelineEntry
from src.utils.config_loader import SafirConfig, load_config
from src.vlm.base_vlm import VLMResponse
from src.vlm.factory import get_llm_client, get_vlm_client

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
    temporal_events: List[TemporalEvent], latest_timestamp: float
) -> List[TemporalEvent]:
    """Bu pipeline cagrisinda uretilen/guncellenen TUM `TemporalEvent`leri secer.

    Bir VLM ciktisi ayni anda birden fazla kategori (orn. `kkd_ihlali` +
    `arac_yaya_yakinligi`) tetikleyebilir; bu durumda `TemporalReasoner`
    birden fazla grup uretir ve bunlarin HEPSİ bu cagriya aittir (hepsinin
    `end_timestamp`i `latest_timestamp`a esittir, cunku bu cagrinin
    `DetectedEvent`leri her zaman en yeni zaman damgasini tasir). Guven
    skoruna gore azalan sirali dondurulur; boylece cagiran taraf, ilk
    elemani "birincil" olay olarak kullanabilir (orn. `SafirReport.event_id`).

    Args:
        temporal_events: `TemporalReasoner.reason(...)` ciktisi (tum buffer
            uzerinden hesaplanmis, bu cagriya ozel olmayan tam liste).
        latest_timestamp: Bu pipeline cagrisinin gozlem zaman damgasi.

    Returns:
        `end_timestamp`i `latest_timestamp`a (tolerans dahilinde) esit olan
        `TemporalEvent`lerin, guvene gore azalan sirali listesi. Bos olabilir
        (teorik olarak; `EventEngine` her zaman en az bir `DetectedEvent`
        urettigi icin pratikte bos donmez).
    """
    current_call_events = [
        te
        for te in temporal_events
        if abs(te.end_timestamp - latest_timestamp) <= _CURRENT_CALL_TIMESTAMP_TOLERANCE
    ]
    current_call_events.sort(key=lambda te: te.confidence, reverse=True)
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


def is_live_source(video_source: str) -> bool:
    """`video_source` canli bir yayin URI'si (RTSP/HTTP) ise `True` dondurur.

    Temsili kare cikarici (seek tabanli) yalnizca kayitli (VOD) dosyalarda
    calisir; canli yayinlarda geriye seek yapilamayacagi icin bu ayrim gerekir.
    """
    return video_source.strip().lower().startswith(("rtsp://", "http://", "https://"))


class AnalyzeRequest(BaseModel):
    """`/analyze` ve `/analyze/jobs` uc noktalari icin ortak istek govdesi."""

    video_source: str
    user_prompt: str = "Sahnede riskli bir durum var mi degerlendir."
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
    video_source: Optional[str] = None
    status: str
    risk_level: Optional[str] = None
    risk_score: Optional[int] = None
    summary: Optional[str] = None


class HistoryDetail(BaseModel):
    """`GET /history/{job_id}` yaniti: metadata + kalici `SafirReport` (varsa)."""

    job_id: str
    created_at: str
    updated_at: str
    status: str
    video_source: Optional[str] = None
    report: Optional[SafirReport] = None


class AskRequest(BaseModel):
    """`POST /ask` istek govdesi."""

    question: str = Field(min_length=1, description="Kullanicinin dogal dil sorusu.")
    job_id: Optional[str] = Field(default=None, description="Opsiyonel: baglam alinacak analiz kimligi.")

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


class SafirPipeline:
    """Tum SAFIR katmanlarini tek bir uctan uca akista birlestiren orkestrator."""

    def __init__(self, config: SafirConfig) -> None:
        """Pipeline'i konfigurasyondan tum alt sistemleri kurarak baslatir.

        Args:
            config: `load_config()` ile uretilmis dogrulanmis `SafirConfig`.
        """
        self._config = config
        self._default_sample_fps = config.sampler.sample_fps
        self._vlm = get_vlm_client(config.vlm, use_mock=config.app.use_mock_vlm)
        self._event_store = EventStore(config.memory.sqlite)
        self._rag_service = EmbeddingRAGService(config.memory.embedding, config.memory.faiss)
        self._rag_service.seed_default_regulations()
        self._context_builder = ContextBuilder(self._event_store, self._rag_service)
        self._agent = SafirAgent(
            llm_config=config.llm,
            agent_config=config.agent,
            event_store=self._event_store,
            rag_service=self._rag_service,
            use_mock_llm=config.app.use_mock_llm,
        )

        # 07 - Olay Analizi Katmani (T008-T012): Context Builder ile LangGraph
        # Ajani arasindaki ara katman. `RetrieverTool`, `rag_service` None
        # olsa bile guvenlidir (mock veriye duser); `RuleEngine`in kendi
        # fallback'i de retriever hata verirse kisa mevzuat etiketine doner.
        self._event_engine = EventEngine()
        self._temporal_reasoner = TemporalReasoner(relation_window_sec=DEFAULT_RELATION_WINDOW_SEC)
        self._rule_engine = RuleEngine(retriever=RetrieverTool(self._rag_service))
        self._event_builder = EventBuilder()
        self._event_history = EventHistory(self._event_store)

        # 06 - Otomatik Eskalasyon (Human-on-the-Loop): risk skoruna gore aksiyon
        # kademesini KENDISI belirler ve yuksek/kritik durumda saha alarmini
        # operator onayi beklemeden OTOMATIK tetikler.
        self._escalation = EscalationPolicy(config.escalation)

        # Pipeline cagrilari arasinda SIFIRLANMAYAN, zaman bazli budanan
        # buffer (bkz. `_prune_stale_events`). `relation_window_sec`in 3
        # kati kadar (varsayilan 90s) gecmis her zaman korunur.
        self._event_buffer_retention_sec = DEFAULT_RELATION_WINDOW_SEC * 3
        self._event_history_buffer: Deque[DetectedEvent] = deque()

    def run(
        self,
        video_source: str,
        user_prompt: str,
        on_stage: Optional[OnStageCallback] = None,
        sample_fps_override: Optional[int] = None,
        min_change_threshold_override: Optional[float] = None,
        trace: Optional[TraceCallback] = None,
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

        Returns:
            Doga dil ozeti, risk skoru/seviyesi, kanit kareleri, ilgili
            mevzuat ve CPU suzgec istatistiklerini iceren rapor.

        Raises:
            RuntimeError: Video kaynagindan hic Evidence Frame/Olay Grubu uretilemezse.
        """
        pipeline_started_at = time.perf_counter()

        def _emit(stage: str, payload: Dict[str, object]) -> None:
            """Gozlem kancasini (varsa) bir asamanin gercek ciktisiyla cagirir (yan etkisiz)."""
            if trace is not None:
                trace(stage, payload)

        if on_stage:
            on_stage(*STAGE_SAMPLER)

        # Her asama, GERCEK production metoduna delege edilir; boylece hem run()
        # tek cagriyla calisir, hem de Jupyter demo bu metotlari tek tek cagirip
        # her asamanin gercek ciktisini gosterebilir (implementasyon kopyalanmaz).
        sampler, evidence_frames, clusters = self.stage_sample(
            video_source, sample_fps_override, min_change_threshold_override
        )
        _emit("sampler", {"evidence_frames": evidence_frames, "stats": sampler.last_run_stats})
        _emit("clusters", {"clusters": clusters})

        if on_stage:
            on_stage(*STAGE_VLM)

        vlm_response = self.stage_vlm(clusters, user_prompt)
        _emit("vlm", {"vlm_response": vlm_response, "clusters": clusters, "user_prompt": user_prompt})

        if on_stage:
            on_stage(*STAGE_AGENT)

        detected_events, temporal_events, rule_matches, latest_timestamp = self.stage_events(
            vlm_response, clusters
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
            vlm_response, user_prompt, latest_timestamp, rule_matches
        )
        _emit("agent_context", {"prompt_block": prompt_block})

        decision = self.stage_decide(prompt_block)
        _emit("decision", {"decision": decision})

        escalation = self.stage_escalate(decision, vlm_response)
        _emit("escalation", {"escalation": escalation})

        report = self.build_report(
            video_source=video_source,
            sampler=sampler,
            clusters=clusters,
            vlm_response=vlm_response,
            context=context,
            decision=decision,
            escalation=escalation,
            temporal_events=temporal_events,
            rule_matches=rule_matches,
            latest_timestamp=latest_timestamp,
        )
        logger.info(
            "SAFIR pipeline tamamlandi: video=%s risk=%d(%s) sure=%.3fs",
            video_source,
            decision.risk_score,
            decision.risk_level,
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
        """01-02: Adaptive Frame Sampler + Olay Kumeleme + Temsili Kare Cikarimi.

        Returns:
            `(sampler, evidence_frames, clusters)` — clusters, VLM'e gonderilecek
            pre/peak/post `representative_frames` ile doldurulmustur (VOD dosyalari).
        """
        sampler = sampler_from_config(self._config.sampler, min_change_threshold_override)
        sample_fps = sample_fps_override or self._default_sample_fps

        evidence_frames: List[EvidenceFrame] = sampler.process_video(video_source, sample_fps=sample_fps)
        if not evidence_frames:
            raise RuntimeError(f"Video kaynagindan kanit karesi uretilemedi: {video_source}")

        clusters: List[EventCluster] = sampler.cluster_events(evidence_frames)
        if not clusters:
            raise RuntimeError(f"Kanit karelerinden Olay Grubu uretilemedi: {video_source}")

        # 02b - Temsili Kare Cikarimi (seek tabanli -> yalnizca VOD dosyalari).
        if not is_live_source(video_source):
            rep_extractor = RepresentativeFrameExtractor(
                pre_event_sec=self._config.sampler.pre_peak_offset_sec,
                post_event_sec=self._config.sampler.post_peak_offset_sec,
            )
            for cluster in clusters:
                try:
                    cluster.representative_frames = rep_extractor.extract(video_source, cluster.peak_frame)
                except (ValueError, RuntimeError) as exc:
                    logger.warning(
                        "Temsili kare cikarilamadi (Olay #%d), tek kareye dusuluyor: %s", cluster.event_id, exc
                    )

        logger.info(
            "VLM oncesi katman ozeti: %d Kanit Karesi -> %d Olay Grubu -> %d temsili kare VLM'e gonderiliyor",
            len(evidence_frames),
            len(clusters),
            sum(len(c.representative_frames) for c in clusters) or len(clusters),
        )
        return sampler, evidence_frames, clusters

    def stage_vlm(self, clusters: List[EventCluster], user_prompt: str) -> VLMResponse:
        """03: VLM (Gemini) gorsel anlama. Hata halinde is'i cokertmez, degraded `VLMResponse` doner."""
        try:
            return self._vlm.describe_events(clusters, prompt=user_prompt)
        except Exception as exc:  # noqa: BLE001 - degraded rapora tasinir
            logger.exception("VLM analizi basarisiz; degraded raporla devam ediliyor.")
            return VLMResponse(
                description=f"[HATA] VLM analizi yapilamadi ({exc}). Manuel inceleme gerekli.",
                model_name=getattr(self._vlm, "model_name", "unknown"),
                frame_count=len(clusters),
                latency_ms=0.0,
                structured_events=[],
            )

    def stage_events(self, vlm_response: VLMResponse, clusters: List[EventCluster]):
        """07: EventEngine -> buffer/budama -> TemporalReasoner -> RuleEngine.

        Returns:
            `(detected_events, temporal_events, rule_matches, latest_timestamp)`.
            `temporal_events`/`rule_matches`, cagrilar arasi kalici buffer uzerinden hesaplanir.
        """
        latest_timestamp = clusters[-1].end_time
        engine_input = EventEngineInput.from_vlm_response(vlm_response, timestamp=latest_timestamp)
        detected_events = self._event_engine.detect(engine_input)
        self._event_history_buffer.extend(detected_events)
        _prune_stale_events(self._event_history_buffer, latest_timestamp, self._event_buffer_retention_sec)

        temporal_events = self._temporal_reasoner.reason(list(self._event_history_buffer))
        rule_matches = self._rule_engine.evaluate(temporal_events)
        return detected_events, temporal_events, rule_matches, latest_timestamp

    def stage_context(self, vlm_response: VLMResponse, user_prompt: str, latest_timestamp: float, rule_matches):
        """04: ContextBuilder (SQLite + FAISS RAG) + kural ozeti -> ajan istem blogu.

        Returns:
            `(prompt_block, context)`.
        """
        context = self._context_builder.build(
            vlm_description=vlm_response.description, user_prompt=user_prompt, timestamp=latest_timestamp
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

    def stage_escalate(self, decision, vlm_response: VLMResponse):
        """06: Otomatik eskalasyon -> `EscalationDecision` (yuksek/kritikte alarm otomatik tetiklenir)."""
        escalation = self._escalation.evaluate(
            risk_score=decision.risk_score,
            risk_level=decision.risk_level,
            recommended_action=decision.recommended_action,
            summary=decision.summary or vlm_response.description,
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
        clusters: List[EventCluster],
        vlm_response: VLMResponse,
        context,
        decision,
        escalation,
        temporal_events,
        rule_matches,
        latest_timestamp: float,
    ) -> SafirReport:
        """06-07: Olay kaydi (EventBuilder/History/Store) + nihai `SafirReport` insasi."""
        current_call_events = _select_current_call_events(temporal_events, latest_timestamp)
        detected_event_types = sorted({te.event_type for te in current_call_events})
        structured_events = self._event_builder.build_batch(current_call_events, rule_matches)
        recorded_event_ids = self._event_history.record_batch(
            structured_events,
            risk_scores=[decision.risk_score] * len(structured_events),
            risk_levels=[decision.risk_level] * len(structured_events),
        )
        logger.info(
            "Event Gecmisi: %d StructuredEvent kaydedildi (ids=%s)", len(recorded_event_ids), recorded_event_ids
        )
        event_id = recorded_event_ids[0] if recorded_event_ids else None
        timeline = self._event_store.get_timeline(start_ts=clusters[0].start_time, end_ts=latest_timestamp)

        return SafirReport(
            event_id=event_id,
            video_source=video_source,
            generated_at=datetime.datetime.utcnow().isoformat() + "Z",
            natural_language_summary=vlm_response.description,
            summary=decision.summary or vlm_response.description,
            risk_score=decision.risk_score,
            risk_level=decision.risk_level,
            recommended_action=decision.recommended_action,
            actions=decision.actions,
            escalation_tier=escalation.tier.value,
            auto_dispatched=escalation.auto_dispatched,
            alert_id=escalation.alert_id,
            detected_event_types=detected_event_types,
            timeline=[
                TimelineEntry(timestamp=e["timestamp"], description=e["description"]) for e in timeline
            ],
            evidence_frames=[
                EvidenceFrameOut(
                    event_id=cluster.event_id,
                    timestamp_sec=cluster.peak_frame.timestamp_sec,
                    timestamp_str=cluster.peak_frame.timestamp_str,
                    change_score=cluster.peak_frame.change_score,
                    base64_image=cluster.peak_frame.base64_image,
                    saved_path=cluster.peak_frame.saved_path,
                    is_fallback=cluster.peak_frame.is_fallback,
                )
                for cluster in clusters
            ],
            relevant_regulations=context.relevant_regulations,
            sampler_stats=(
                SamplerStats(
                    total_frames_scanned=sampler.last_run_stats.total_frames_scanned,
                    sampled_frames_evaluated=sampler.last_run_stats.sampled_frames_evaluated,
                    evidence_frame_count=sampler.last_run_stats.evidence_frame_count,
                    eliminated_frame_count=sampler.last_run_stats.eliminated_frame_count,
                    gpu_savings_ratio_pct=sampler.last_run_stats.eliminated_ratio_pct,
                    elapsed_sec=sampler.last_run_stats.elapsed_sec,
                )
                if sampler.last_run_stats
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
        _ask_service = AskService(
            analysis_store=get_analysis_store(),
            rag_service=get_pipeline()._rag_service,  # pipeline'in seed'lenmis RAG'i (yeniden kullanim)
            llm_client=llm_client,
            rag_top_k=cfg.memory.faiss.top_k,
        )
    return _ask_service


def _run_job(
    job_id: str,
    video_source: str,
    user_prompt: str,
    sample_fps_override: Optional[int] = None,
    min_change_threshold_override: Optional[float] = None,
) -> None:
    """Arka plan thread'inde pipeline'i calistirip `JobState`'i gunceller.

    Args:
        job_id: `_jobs` sozlugundeki hedef isin kimligi.
        video_source: Ham (henuz normalize edilmemis) video kaynagi.
        user_prompt: Ajanin odaklanmasi istenen kullanici istemi.
        sample_fps_override: Operator panelinden gelen ornekleme FPS override'i.
        min_change_threshold_override: Operator panelinden gelen hassasiyet esigi override'i.
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
        )
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "done"
            job.result = report
            job.step = job.total_steps
        # History: tamamlanmis analizi (gercek SafirReport ile) kalici olarak sakla.
        # Persistence hatasi is sonucunu ASLA etkilemez (best-effort).
        try:
            get_analysis_store().mark_completed(
                job_id,
                report_json=report.model_dump_json(),
                risk_level=report.risk_level,
                risk_score=report.risk_score,
                summary=report.summary or report.natural_language_summary,
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


# Frame kimligi yalnizca in-memory sozluk anahtaridir (dosya yolu DEGIL); path
# traversal riski yoktur. Yine de beklenmedik girdilere karsi bicimi kisitlariz.
_FRAME_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


@app.get("/analyze/jobs/{job_id}/frames/{frame_id}")
def get_job_frame(job_id: str, frame_id: str) -> Response:
    """Bir isin trace olaylarinda referans verilen kareyi (JPEG) sunar (base64 yerine).

    Kareler, `_run_job` sirasinda bellek-ici (`JobState.frames`) tutulur; bu uc
    nokta yalnizca o sozlukten okur (dosya sistemine erisim/yeni dosya kopyalama
    YOK).

    Args:
        job_id: Analiz isinin kimligi.
        frame_id: Trace olayindaki `frame_id`/`thumbnail_url` referansi.

    Returns:
        `image/jpeg` yaniti.

    Raises:
        HTTPException: Is/kare bulunamazsa (404) veya `frame_id` bicimi gecersizse (400).
    """
    if not _FRAME_ID_RE.match(frame_id):
        raise HTTPException(status_code=400, detail="Gecersiz frame_id.")
    with _jobs_lock:
        job = _jobs.get(job_id)
        raw = job.frames.get(frame_id) if job else None
    if job is None:
        raise HTTPException(status_code=404, detail=f"Is bulunamadi: {job_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Kare bulunamadi: {frame_id}")
    return Response(content=raw, media_type="image/jpeg")


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
            video_source=r.video_source,
            status=r.status,
            risk_level=r.risk_level,
            risk_score=r.risk_score,
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

    return HistoryDetail(
        job_id=record.job_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        status=record.status,
        video_source=record.video_source,
        report=report,
    )


@app.post("/ask", response_model=AskResponse)
def ask_safir(request: AskRequest) -> AskResponse:
    """Ask SAFIR: soruyu (opsiyonel analiz baglami + RAG mevzuati ile) dayanakli yanitlar.

    - `job_id` verilirse: o analizin kalici `SafirReport`'u ONCELIKLI baglam olur.
    - Her durumda mevcut `EmbeddingRAGService` ile ilgili ISG mevzuati getirilir.
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
        result = get_ask_service().answer(request.question, request.job_id)
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


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(app, host=config.api.host, port=config.api.port)
