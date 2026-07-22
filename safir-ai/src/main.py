"""SAFIR uctan uca pipeline giris noktasi (FastAPI servisi).

Akis: Video Input -> AdaptiveSampler (CPU) -> VLM (vLLM) -> ContextBuilder
(SQLite + FAISS) -> SafirAgent (LangGraph, vLLM) -> SafirReport (JSON).
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent.langgraph_agent import SafirAgent
from src.memory.context_builder import ContextBuilder
from src.memory.event_store import EventStore
from src.memory.semantic_memory import SemanticMemory
from src.sampler.adaptive_sampler import EventCluster, EvidenceFrame, sampler_from_config
from src.schemas.report import SafirReport, TimelineEntry
from src.utils.config_loader import SafirConfig, load_config
from src.vlm.vlm_factory import VLMFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAFIR", description="Saha Analiz ve Farkindalik Karar Sistemi API'si")


class AnalyzeRequest(BaseModel):
    """`/analyze` uc noktasi icin istek govdesi."""

    video_source: str
    user_prompt: str = "Sahnede riskli bir durum var mi degerlendir."


class SafirPipeline:
    """Tum SAFIR katmanlarini tek bir uctan uca akista birlestiren orkestrator."""

    def __init__(self, config: SafirConfig) -> None:
        """Pipeline'i konfigurasyondan tum alt sistemleri kurarak baslatir.

        Args:
            config: `load_config()` ile uretilmis dogrulanmis `SafirConfig`.
        """
        self._config = config
        self._sampler = sampler_from_config(config.sampler)
        self._sample_fps = config.sampler.sample_fps
        self._vlm = VLMFactory.create(config.vlm)
        self._event_store = EventStore(config.memory.sqlite)
        self._semantic_memory = SemanticMemory(config.memory.faiss)
        self._context_builder = ContextBuilder(self._event_store, self._semantic_memory)
        self._agent = SafirAgent(
            llm_config=config.llm,
            agent_config=config.agent,
            event_store=self._event_store,
            semantic_memory=self._semantic_memory,
        )

    def run(self, video_source: str, user_prompt: str) -> SafirReport:
        """Video kaynagindan nihai `SafirReport`'a kadar tum pipeline'i calistirir.

        Args:
            video_source: `.mp4` dosya yolu veya RTSP URI'si.
            user_prompt: Ajanin odaklanmasi istenen kullanici istemi.

        Returns:
            Doga dil ozeti, risk skoru/seviyesi ve zaman cizelgesini iceren rapor.

        Raises:
            RuntimeError: Video kaynagindan hic Evidence Frame/Olay Grubu uretilemezse.
        """
        pipeline_started_at = time.perf_counter()

        evidence_frames: List[EvidenceFrame] = self._sampler.process_video(
            video_source, sample_fps=self._sample_fps
        )
        if not evidence_frames:
            raise RuntimeError(f"Video kaynagindan kanit karesi uretilemedi: {video_source}")

        clusters: List[EventCluster] = self._sampler.cluster_events(evidence_frames)
        if not clusters:
            raise RuntimeError(f"Kanit karelerinden Olay Grubu uretilemedi: {video_source}")

        logger.info(
            "VLM oncesi katman ozeti: %d Kanit Karesi -> %d Olay Grubu (peak kareler VLM'e gonderiliyor)",
            len(evidence_frames),
            len(clusters),
        )

        vlm_response = self._vlm.describe_events(clusters, prompt=user_prompt)

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
            vlm_model=vlm_response.model_name,
            llm_model=self._config.llm.active_endpoint().model_name,
        )


_pipeline: SafirPipeline | None = None


def get_pipeline() -> SafirPipeline:
    """Uygulama omru boyunca tek bir `SafirPipeline` orneği olusturur/dondurur (lazy singleton).

    Returns:
        Ilklendirilmis `SafirPipeline` orneği.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = SafirPipeline(load_config())
    return _pipeline


@app.get("/health")
def health() -> dict:
    """Servisin ayakta oldugunu bildiren basit saglik kontrolu uc noktasi."""
    return {"status": "ok", "system": "SAFIR"}


@app.post("/analyze", response_model=SafirReport)
def analyze(request: AnalyzeRequest) -> SafirReport:
    """Verilen video kaynagini uctan uca isleyip yapilandirilmis rapor uretir.

    Args:
        request: Video kaynagi ve kullanici istemini iceren istek govdesi.

    Returns:
        `SafirReport` JSON yaniti.

    Raises:
        HTTPException: Pipeline calistirilirken hata olusursa (400).
    """
    try:
        pipeline = get_pipeline()
        return pipeline.run(request.video_source, request.user_prompt)
    except Exception as exc:  # noqa: BLE001 - API tuketicisine anlamli hata donmek icin
        logger.exception("Analiz pipeline hatasi")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(app, host=config.api.host, port=config.api.port)
