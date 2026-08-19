"""T013: `src/main.py::SafirPipeline`nin `event_analysis/` (T008-T012) entegrasyonu icin uctan uca test.

Gercek `SafirPipeline.__init__`/`run()` kodunu, mock VLM/LLM (`app.use_mock_vlm`/
`app.use_mock_llm`) ve gercek (ama gecici, `tmp_path` altinda) bir SQLite
`EventStore` ile calistirir; boylece VLM ciktisi -> `EventEngine` ->
`TemporalReasoner` -> `RuleEngine` -> `EventBuilder` -> `EventHistory.record_batch`
zincirinin `SafirPipeline.run()` icinde GERCEKTEN cagrildigini dogrular.

Tek istisna: `src/memory/embedding_rag_service.py::EmbeddingRAGService`, gercek
bir `sentence-transformers` modeli indirmeyi gerektirdigi icin (agir + ag
bagimliligi, GPU'dan bagimsiz ama bu testin kapsami disinda) `src.main`
icindeki referansi, `RegulationRetriever`/`ContextBuilder` ile ayni
`.query(question, top_k)` sozlesmesine uyan hafif bir sahte servisle
degistirilir (`monkeypatch`). Video girdisi, `tests/test_sampler.py`daki
gibi tamamen sentetik (cv2 ile uretilmis) bir `.mp4` dosyasidir; gercek bir
GPU, Docker veya vLLM servisi gerekmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pytest

from src.main import SafirPipeline
from src.utils.config_loader import SafirConfig


@dataclass
class _FakeRetrievedDocument:
    text: str
    score: float = 1.0


class _FakeRagService:
    """`ContextBuilder`/`RetrieverTool`nin bekledigi `.query(question, top_k)` sozlesmesine uyan sahte servis.

    Gercek `EmbeddingRAGService`nin `sentence-transformers` model indirme
    gereksinimini (ag bagimliligi) atlamak icin `src.main.EmbeddingRAGService`
    yerine gecirilir; `seed_default_regulations()` no-op'tur.
    """

    def __init__(self) -> None:
        self.queries: List[str] = []

    def seed_default_regulations(self) -> None:
        return None

    def query(self, question: str, top_k: Optional[int] = None) -> List[_FakeRetrievedDocument]:
        self.queries.append(question)
        return [_FakeRetrievedDocument(text=f"[FAKE-RAG] {question}")]


def _write_video(path: Path, frames: list) -> None:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.fixture
def motion_video(tmp_path: Path) -> str:
    """`tests/test_sampler.py`daki ile ayni desende, 20-30. kareler arasi hareket iceren sentetik video."""
    frames = []
    for i in range(60):
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        if 20 <= i < 30:
            cv2.rectangle(frame, (5, 5), (58, 42), (255, 255, 255), -1)
        frames.append(frame)
    path = tmp_path / "motion.mp4"
    _write_video(path, frames)
    return str(path)


def _build_test_config(tmp_path: Path) -> SafirConfig:
    """Gercek `configs/config.yaml`e dokunmadan, mock-mod acik, gecici yollara sahip bir `SafirConfig` uretir."""
    raw = {
        "app": {"name": "SAFIR-TEST", "version": "test", "use_mock_vlm": True, "use_mock_llm": True},
        "system": {
            "name": "SAFIR-TEST",
            "environment": "development",
            "device": "cpu",
            "cuda_device_index": 0,
            "log_level": "INFO",
            "random_seed": 42,
        },
        "sampler": {
            "min_change_threshold": 0.001,
            "blur_kernel_size": [21, 21],
            "history_window": 30,
            "min_event_interval_sec": 2.0,
            "sample_fps": 5,
            "idle_interval_sec": 2.0,
            "active_fps": 5,
            "noise_floor": 0.011,
            "motion_threshold": 0.02,
            "scene_change_threshold": 0.15,
            "resize_width": 640,
            "warmup_frames": 30,
        },
        "vlm": {
            "active_model": "qwen",
            "models": {
                "qwen": {
                    "model_name": "test-vlm",
                    "vllm_host": "localhost",
                    "vllm_port": 1,
                    "max_new_tokens": 64,
                    "temperature": 0.2,
                }
            },
        },
        "memory": {
            "sqlite": {"db_path": str(tmp_path / "events.db")},
            "embedding": {
                "provider": "sentence-transformers",
                "model_name": "test-embedding-model",
                "device": "cpu",
            },
            "faiss": {
                "index_path": str(tmp_path / "faiss_index"),
                "embedding_model": "test-embedding-model",
                "top_k": 3,
            },
        },
        "llm": {
            "active_model": "qwen3",
            "models": {
                "qwen3": {
                    "model_name": "test-llm",
                    "vllm_host": "localhost",
                    "vllm_port": 2,
                    "max_new_tokens": 64,
                    "temperature": 0.1,
                }
            },
        },
        "agent": {
            "max_iterations": 6,
            "risk_thresholds": {"low": 25, "medium": 50, "high": 75, "critical": 100},
            "tools": {
                "sql_tool_enabled": True,
                "rag_tool_enabled": True,
                "retriever_tool_enabled": True,
                "timeline_tool_enabled": True,
                "verification_tool_enabled": True,
            },
        },
        "api": {"host": "0.0.0.0", "port": 8000, "reload": False, "cors_origins": ["*"]},
        "output": {
            "language": "tr",
            "json_report_dir": str(tmp_path / "reports"),
            "timeline_export_dir": str(tmp_path / "timelines"),
            "pdf_report_dir": str(tmp_path / "pdf"),
            "streamlit_port": 8501,
        },
    }
    return SafirConfig(**raw)


@pytest.fixture
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SafirPipeline:
    """Gercek `SafirPipeline.__init__`i, yalnizca `EmbeddingRAGService`yi sahteyle degistirerek calistirir."""
    fake_rag_service = _FakeRagService()
    monkeypatch.setattr("src.main.EmbeddingRAGService", lambda *args, **kwargs: fake_rag_service)

    config = _build_test_config(tmp_path)
    return SafirPipeline(config)


def test_pipeline_wires_event_analysis_dependencies(pipeline: SafirPipeline) -> None:
    """`__init__`in T013'te eklenen tum event_analysis bagimliliklarini kurdugunu dogrular."""
    assert pipeline._event_engine is not None
    assert pipeline._temporal_reasoner is not None
    assert pipeline._rule_engine is not None
    assert pipeline._event_builder is not None
    assert pipeline._event_history is not None
    assert pipeline._event_history_buffer is not None
    assert len(pipeline._event_history_buffer) == 0


def test_run_writes_structured_event_to_real_event_store(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """Tek bir `run()` cagrisi: VLM -> EventEngine -> TemporalReasoner -> RuleEngine -> EventBuilder -> EventHistory zincirinin gercekten calistigini, SQLite'a yazilan satirla dogrular.

    MockVLMClient'in sabit aciklamasi "forklift" (`arac_yaya_yakinligi`)
    kelimesini icerir; bu SQLite'a yazilan aciklamada OK-07 kural ozetiyle
    gorunmelidir. Aciklama ayrica "Herhangi bir duman veya yangin belirtisi
    tespit edilmedi." cumlesini de icerir - T014'un olumsuzlama tespiti
    sayesinde bu artik YANLIS POZITIF bir `yangin_duman` tespiti URETMEZ
    (T013 sirasinda bu testin ilk hali tam da bu hatayi yakalamisti; bkz.
    T014 commit'i).
    """
    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    assert report.event_id is not None
    assert len(pipeline._event_history_buffer) >= 1

    rows = pipeline._event_store.query_recent(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == report.event_id
    assert row["risk_score"] == report.risk_score
    assert row["risk_level"] == report.risk_level
    assert "OK-07" in row["description"]
    assert "YG-03" not in row["description"]


def test_event_buffer_persists_across_pipeline_calls(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """Buffer, pipeline cagrilari arasinda SIFIRLANMAZ: ayni videoyu iki kez analiz etmek tekrar-tespitini uretmeli.

    T014 (olumsuzlama tespiti) sayesinde `MockVLMClient`in gercek sabit
    metni artik TEK kategori (`arac_yaya_yakinligi`, "forklift") tetikliyor
    (bkz. `test_run_writes_structured_event_to_real_event_store`); bu
    yuzden ek bir sahte VLM istemcisine gerek kalmadan gercek `pipeline`
    kullanilabilir. Ayni video ikinci kez analiz edildiginde ayni tip +
    ayni zaman damgasi uretilir, T009 bunu ilk cagrinin olayiyla
    BIRLESTIRIR (`occurrence_count=2`) ve EventBuilder bunu "ardisik
    gozlemde" notuyla ikinci cagrinin SQLite satirina yazar; ilk cagrinin
    satiri degismeden kalir (EventStore'da UPDATE yok - T012'nin bilinen
    kisiti).
    """
    first_report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")
    first_row_ids = {row["id"] for row in pipeline._event_store.query_recent(limit=10)}

    second_report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    assert first_report.event_id != second_report.event_id

    all_rows = pipeline._event_store.query_recent(limit=10)
    assert len(all_rows) == 2

    first_call_rows = [row for row in all_rows if row["id"] in first_row_ids]
    second_call_rows = [row for row in all_rows if row["id"] not in first_row_ids]
    assert len(first_call_rows) == 1
    assert len(second_call_rows) == 1

    assert "ardisik gozlemde" not in first_call_rows[0]["description"]
    assert "ardisik gozlemde" in second_call_rows[0]["description"]


def test_event_buffer_does_not_persist_across_different_video_sources(
    pipeline: SafirPipeline, motion_video: str, tmp_path: Path
) -> None:
    """Kok neden Hata #3'un dogrudan regresyon testi (_event_history_buffer contamination).

    `test_event_buffer_persists_across_pipeline_calls` AYNI video icin buffer'in
    KORUNMASI gerektigini dogrular (surekli kamera/stream izleme). Bu test ise
    tam tersini: video_source DEGISTIGINDE (bagimsiz, ilgisiz bir video analizi)
    buffer'in temizlenmesi gerektigini dogrular - aksi halde ikinci videonun
    TemporalReasoner/RuleEngine degerlendirmesine ilk videonun DetectedEvent'lari
    sizar ve yanlislikla "ardisik gozlemde" (occurrence_count=2) birlesmesi olusur.
    """
    second_video_path = tmp_path / "motion_other.mp4"
    frames = []
    for i in range(60):
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        if 20 <= i < 30:
            cv2.rectangle(frame, (5, 5), (58, 42), (255, 255, 255), -1)
        frames.append(frame)
    _write_video(second_video_path, frames)

    pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")
    assert len(pipeline._event_history_buffer) >= 1

    pipeline.run(str(second_video_path), "Sahnede riskli bir durum var mi degerlendir.")

    second_video_rows = [
        row
        for row in pipeline._event_store.query_recent(limit=10)
        if row["video_source"] == str(second_video_path)
    ]
    assert len(second_video_rows) == 1
    assert "ardisik gozlemde" not in second_video_rows[0]["description"]


def test_report_timeline_isolates_overlapping_timestamp_videos(
    pipeline: SafirPipeline, motion_video: str, tmp_path: Path
) -> None:
    """Kok neden Hata #2'nin uctan uca regresyon testi (EventStore.get_timeline contamination).

    Iki FARKLI video (ayni sentetik hareket deseni -> ayni sure -> ORTUSEN
    zaman damgasi araligi, gercek kisa test klipleri gibi) art arda analiz
    edilir; ikinci videonun `report.timeline`i, ilk videonun eventlerini
    ICERMEMELI.
    """
    second_video_path = tmp_path / "motion_overlap.mp4"
    frames = []
    for i in range(60):
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        if 20 <= i < 30:
            cv2.rectangle(frame, (5, 5), (58, 42), (255, 255, 255), -1)
        frames.append(frame)
    _write_video(second_video_path, frames)

    pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")
    second_report = pipeline.run(str(second_video_path), "Sahnede riskli bir durum var mi degerlendir.")

    assert len(second_report.timeline) == 1

    timeline_a = pipeline._event_store.get_timeline(start_ts=0, end_ts=10_000, video_source=motion_video)
    timeline_b = pipeline._event_store.get_timeline(
        start_ts=0, end_ts=10_000, video_source=str(second_video_path)
    )
    assert len(timeline_a) == 1
    assert len(timeline_b) == 1


def test_run_populates_rule_engine_retriever_through_real_rag_service(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """`RuleEngine`e enjekte edilen `RetrieverTool`nin, pipeline'daki gercek `rag_service`yi (sahte) kullandigini dogrular."""
    fake_rag_service: _FakeRagService = pipeline._rag_service  # type: ignore[assignment]
    assert fake_rag_service.queries == []

    pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    # RuleEngine, tekli kural aciklamasini zenginlestirmek icin retriever'i
    # cagirir (bkz. RuleEngine._describe_regulation); bu da ayni rag_service'i
    # kullanan RetrieverTool uzerinden fake_rag_service.query'ye duser.
    assert len(fake_rag_service.queries) >= 1


def test_run_produces_schema_complete_report_with_escalation(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """`run()` sartname-uyumlu, sema-eksiksiz bir rapor uretmeli (summary/actions/eskalasyon).

    Mock LLM orta risk (35) doner; bu yuzden otomatik eskalasyon kademesi
    'notify' olmali ve alarm OTOMATIK tetiklenMEmeli (auto_dispatched=False).
    Ayrica sartname-uyumlu ozet JSON'un beklenen anahtarlari tasidigi dogrulanir.
    """
    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    # Yeni sema alanlari dolu olmali.
    assert report.summary
    assert isinstance(report.actions, list) and report.actions
    assert report.recommended_action == report.actions[0]

    # Otomatik eskalasyon (Human-on-the-Loop): orta risk -> notify, alarm yok.
    assert report.escalation_tier == "notify"
    assert report.auto_dispatched is False
    assert report.alert_id is None

    # Sartname-uyumlu ozet JSON beklenen sekilde olmali.
    sartname = report.to_sartname_json()
    assert set(sartname) >= {"summary", "events", "risk", "actions"}
    assert sartname["risk"] == report.risk_level
    assert all("time" in e and "event" in e for e in sartname["events"])


def test_high_risk_auto_dispatches_alarm_in_pipeline(
    pipeline: SafirPipeline, motion_video: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Yuksek risk skorunda pipeline, operator onayi beklemeden alarmi OTOMATIK tetiklemeli."""
    from src.agent.langgraph_agent import AgentDecision

    high_decision = AgentDecision(
        risk_score=90,
        risk_level="kritik",
        recommended_action="Saglik ekibini cagir",
        raw_response="{}",
        summary="Kritik durum",
        actions=["Saglik ekibini cagir", "Alani guvenlik altina al"],
        events=[],
    )
    monkeypatch.setattr(pipeline._agent, "run", lambda _prompt: high_decision)

    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    assert report.risk_score == 90
    assert report.escalation_tier == "alarm"
    assert report.auto_dispatched is True
    assert report.alert_id is not None

    # Operator, otomatik tetiklenen alarmi sonradan onaylayabilmeli (Human-on-the-Loop).
    record = pipeline.acknowledge_alert(report.alert_id, "operator denetledi")
    assert record.acknowledged is True


def test_pipeline_sends_all_evidence_frames_to_vlm_with_no_positional_role(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """Pipeline, VLM'e sampler'in urettigi TUM evidence karelerini (kumelenmeden) gondermeli.

    Sampler artik olay kumelemesi yapmaz; VLM'e giden evidence karelerinin
    hicbirine kalici bir 'pre'/'peak'/'post' konumsal ROL verilmez. VLM'e
    gecirilen kareler yakalanarak dogrulanir.
    """
    captured = {}
    original_analyze = pipeline._vlm.analyze_evidence

    def _capture(evidence_frames, prompt):
        captured["evidence_frames"] = evidence_frames
        return original_analyze(evidence_frames, prompt)

    pipeline._vlm.analyze_evidence = _capture  # type: ignore[assignment]
    pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    evidence_frames = captured["evidence_frames"]
    assert evidence_frames, "VLM'e en az bir evidence karesi gitmeli."
    # Hicbir evidence karesi 'label' gibi konumsal bir alan TASIMIYOR.
    # `selection_reason` (threshold_exceeded/temporal_coverage/fallback) bir
    # konumsal ROL DEGILDIR - yalnizca "bu kare neden secildi" bilgisidir.
    for ef in evidence_frames:
        field_names = set(type(ef).model_fields.keys())
        assert "label" not in field_names
        assert ef.selection_reason in {"threshold_exceeded", "temporal_coverage", "fallback"}


def test_pipeline_produces_degraded_report_when_vlm_fails(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """VLM analizi patlarsa pipeline cokmemeli; degraded (hata notlu) bir rapor uretmeli."""

    def _boom(evidence_frames, prompt):
        raise RuntimeError("VLM servisi erisilemez")

    pipeline._vlm.analyze_evidence = _boom  # type: ignore[assignment]
    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    # Is "error" ile cokmedi; rapor uretildi ve hata operatore gorunur.
    assert report is not None
    assert "[HATA]" in report.natural_language_summary
    assert report.escalation_tier in {"monitor", "notify", "alarm"}


def test_record_feedback_delegates_to_event_history(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """`SafirPipeline.record_feedback`, `EventHistory.mark_feedback` uzerinden `EventStore`e yazmali (T013-D)."""
    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    pipeline.record_feedback(report.event_id, "true_positive")

    rows = pipeline._event_store.query_recent(limit=10)
    row = next(r for r in rows if r["id"] == report.event_id)
    assert row["feedback"] == "true_positive"
