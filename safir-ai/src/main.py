"""SAFIR uctan uca pipeline giris noktasi (FastAPI servisi).

Akis: Video Input -> AdaptiveFrameSampler (CPU) -> VLM (vLLM) -> ContextBuilder
(SQLite + FAISS RAG) -> SafirAgent (LangGraph, vLLM) -> SafirReport (JSON).

Operator paneli (`src/ui/dashboard.py`) icin canli ilerleme takibi
`/analyze/jobs` uzerinden asenkron (arka plan thread'i + durum sorgulama)
olarak saglanir; `/analyze` ise tek seferlik senkron kullanim icin korunur.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.langgraph_agent import SafirAgent
from src.memory.context_builder import ContextBuilder
from src.memory.embedding_rag_service import EmbeddingRAGService
from src.memory.event_store import EventStore
from src.sampler.adaptive_sampler import EventCluster, EvidenceFrame, sampler_from_config
from src.schemas.report import EvidenceFrameOut, SafirReport, SamplerStats, TimelineEntry
from src.utils.config_loader import SafirConfig, load_config
from src.vlm.vlm_factory import VLMFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAFIR", description="Saha Analiz ve Farkindalik Karar Sistemi API'si")

_DATA_DIR = "data"

# Operator panelinde gosterilecek 3 asamali ilerleme adlari (sira onemli).
STAGE_SAMPLER = ("VLM Oncesi Katman (Adaptive Frame Sampler)", 1, 3)
STAGE_VLM = ("Gorsel Dil Modeli (vLLM Gorsel Anlama)", 2, 3)
STAGE_AGENT = ("LangGraph Ajan (RAG & Karar)", 3, 3)

OnStageCallback = Callable[[str, int, int], None]


def normalize_video_source(video_source: str) -> str:
    """`video_source` degerini, canli yayin URI'lerini koruyarak yerel dosya yoluna normalize eder.

    Operator panelinden veya harici istemcilerden gelen Windows tam yollari
    (`C:\\Users\\...\\test.mp4`) veya bagil yollar, dosya adi ayiklanip
    her zaman `data/<dosya_adi>` seklinde yeniden yazilir; boylece konteyner
    icindeki `./data` bind mount'una gore calisir. RTSP/HTTP(S) canli yayin
    adresleri oldugu gibi birakilir.

    Args:
        video_source: `/analyze` istegindeki ham `video_source` degeri.

    Returns:
        Canli yayin URI'si ise degistirilmeden, aksi halde `data/<dosya_adi>`
        seklinde normalize edilmis yol.
    """
    lowered = video_source.strip().lower()
    if lowered.startswith(("rtsp://", "http://", "https://")):
        return video_source

    normalized_slashes = video_source.replace("\\", "/")
    filename = os.path.basename(normalized_slashes)
    return os.path.join(_DATA_DIR, filename)


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
    """`/alerts/trigger` icin istek govdesi (Human-in-the-Loop onayi)."""

    risk_score: int
    risk_level: str
    recommended_action: str
    operator_note: str = ""


class AlertTriggerResponse(BaseModel):
    """`/alerts/trigger` yaniti."""

    acknowledged: bool
    alert_id: str
    message: str


@dataclass
class JobState:
    """Bir `/analyze/jobs` isinin canli durumu (asama, sonuc, hata)."""

    status: str = "queued"                    # queued | running | done | error
    stage_name: str = ""
    step: int = 0
    total_steps: int = 3
    result: Optional[SafirReport] = None
    error: Optional[str] = None


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


class SafirPipeline:
    """Tum SAFIR katmanlarini tek bir uctan uca akista birlestiren orkestrator."""

    def __init__(self, config: SafirConfig) -> None:
        """Pipeline'i konfigurasyondan tum alt sistemleri kurarak baslatir.

        Args:
            config: `load_config()` ile uretilmis dogrulanmis `SafirConfig`.
        """
        self._config = config
        self._default_sample_fps = config.sampler.sample_fps
        self._vlm = VLMFactory.create(config.vlm)
        self._event_store = EventStore(config.memory.sqlite)
        self._rag_service = EmbeddingRAGService(config.memory.embedding, config.memory.faiss)
        self._rag_service.seed_default_regulations()
        self._context_builder = ContextBuilder(self._event_store, self._rag_service)
        self._agent = SafirAgent(
            llm_config=config.llm,
            agent_config=config.agent,
            event_store=self._event_store,
            rag_service=self._rag_service,
        )

    def run(
        self,
        video_source: str,
        user_prompt: str,
        on_stage: Optional[OnStageCallback] = None,
        sample_fps_override: Optional[int] = None,
        min_change_threshold_override: Optional[float] = None,
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

        if on_stage:
            on_stage(*STAGE_SAMPLER)

        sampler = sampler_from_config(self._config.sampler, min_change_threshold_override)
        sample_fps = sample_fps_override or self._default_sample_fps

        evidence_frames: List[EvidenceFrame] = sampler.process_video(
            video_source, sample_fps=sample_fps
        )
        if not evidence_frames:
            raise RuntimeError(f"Video kaynagindan kanit karesi uretilemedi: {video_source}")

        clusters: List[EventCluster] = sampler.cluster_events(evidence_frames)
        if not clusters:
            raise RuntimeError(f"Kanit karelerinden Olay Grubu uretilemedi: {video_source}")

        logger.info(
            "VLM oncesi katman ozeti: %d Kanit Karesi -> %d Olay Grubu (peak kareler VLM'e gonderiliyor)",
            len(evidence_frames),
            len(clusters),
        )

        if on_stage:
            on_stage(*STAGE_VLM)

        vlm_response = self._vlm.describe_events(clusters, prompt=user_prompt)

        if on_stage:
            on_stage(*STAGE_AGENT)

        latest_timestamp = clusters[-1].end_time
        context = self._context_builder.build(
            vlm_description=vlm_response.description,
            user_prompt=user_prompt,
            timestamp=latest_timestamp,
        )

        decision = self._agent.run(context.to_prompt_block())

        self._event_store.add_event(
            timestamp=latest_timestamp,
            description=vlm_response.description,
            risk_score=decision.risk_score,
            risk_level=decision.risk_level,
            source_model=vlm_response.model_name,
        )

        timeline = self._event_store.get_timeline(
            start_ts=clusters[0].start_time, end_ts=latest_timestamp
        )

        elapsed_sec = time.perf_counter() - pipeline_started_at
        logger.info(
            "SAFIR pipeline tamamlandi: video=%s risk=%d(%s) sure=%.3fs",
            video_source,
            decision.risk_score,
            decision.risk_level,
            elapsed_sec,
        )

        return SafirReport(
            video_source=video_source,
            generated_at=datetime.datetime.utcnow().isoformat() + "Z",
            natural_language_summary=vlm_response.description,
            risk_score=decision.risk_score,
            risk_level=decision.risk_level,
            recommended_action=decision.recommended_action,
            timeline=[
                TimelineEntry(timestamp=e["timestamp"], description=e["description"])
                for e in timeline
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
                )
                if sampler.last_run_stats
                else None
            ),
            vlm_model=vlm_response.model_name,
            llm_model=self._config.llm.active_endpoint().model_name,
        )


_pipeline: SafirPipeline | None = None
_jobs: Dict[str, JobState] = {}
_jobs_lock = threading.Lock()


def get_pipeline() -> SafirPipeline:
    """Uygulama omru boyunca tek bir `SafirPipeline` orneği olusturur/dondurur (lazy singleton).

    Returns:
        Ilklendirilmis `SafirPipeline` orneği.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = SafirPipeline(load_config())
    return _pipeline


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

    try:
        pipeline = get_pipeline()
        normalized_source = normalize_video_source(video_source)
        report = pipeline.run(
            normalized_source,
            user_prompt,
            on_stage=on_stage,
            sample_fps_override=sample_fps_override,
            min_change_threshold_override=min_change_threshold_override,
        )
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "done"
            job.result = report
            job.step = job.total_steps
    except Exception as exc:  # noqa: BLE001 - hata is durumuna tasinip UI'a iletilir
        logger.exception("Analiz isi basarisiz oldu: job_id=%s", job_id)
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "error"
            job.error = str(exc)


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
        _jobs[job_id] = JobState()

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


@app.post("/alerts/trigger", response_model=AlertTriggerResponse)
def trigger_alert(request: AlertTriggerRequest) -> AlertTriggerResponse:
    """Operatorun Human-in-the-Loop onayiyla tetikledigi saha alarmini isler (mock).

    Gercek bir saha entegrasyonunda bu uc nokta SMS/anons/SCADA sistemine
    baglanir; bu iskelette yalnizca alarmi loglar ve bir onay kimligi dondurur.

    Args:
        request: Onaylanan risk skoru/seviyesi ve operator notunu iceren istek.

    Returns:
        Alarmin kabul edildigini bildiren yanit.
    """
    alert_id = str(uuid.uuid4())
    logger.warning(
        "SAHA ALARMI TETIKLENDI (operator onayi): alert_id=%s risk=%d(%s) aksiyon=%s not=%s",
        alert_id,
        request.risk_score,
        request.risk_level,
        request.recommended_action,
        request.operator_note or "(yok)",
    )
    return AlertTriggerResponse(
        acknowledged=True,
        alert_id=alert_id,
        message="Saha alarmi operator onayiyla tetiklendi ve kayit altina alindi.",
    )


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(app, host=config.api.host, port=config.api.port)
