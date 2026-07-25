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
from src.vlm.base_vlm import VLMResponse


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


class _SingleCategoryVLMClient:
    """Sabit, TEK kategori (`arac_yaya_yakinligi`, 'forklift') tetikleyen sahte VLM istemcisi.

    `MockVLMClient`in gercek sabit metni ayni anda 2 kategori tetikler
    (bkz. `test_run_writes_structured_event_to_real_event_store`); bu, cok
    kategorili tek-cagri davranisini test etmek icin dogru, ama cagrilar
    arasi AYNI TIPTEKI birlesmeyi (T009) izole test etmek icin karisiklik
    yaratir (2 kategori de her cagrida tekrarlanir ve buffer'da ic ice
    girer). Bu sahte istemci, `BaseVLM` sozlesmesine (`describe_events`,
    `health_check`) uyar ve yalnizca `_event_buffer_persistence` testinde
    `pipeline._vlm` yerine gecirilir.
    """

    def describe_events(self, clusters: list, prompt: str) -> VLMResponse:
        return VLMResponse(
            description="Forklift yaya gecidine yaklasiyor.",
            model_name="fake-single-category-vlm",
            frame_count=len(clusters),
            latency_ms=0.0,
        )

    def health_check(self) -> bool:
        return True


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
            "max_evidence_buffer": 100,
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
    """Tek bir `run()` cagrisi: VLM -> EventEngine -> TemporalReasoner -> RuleEngine -> EventBuilder -> EventHistory zincirinin gercekten calistigini, SQLite'a yazilan satirlarla dogrular.

    MockVLMClient'in sabit aciklamasi hem "forklift" (`arac_yaya_yakinligi`)
    hem de "duman"/"yangin" (`yangin_duman` - cumle aslinda bunlarin
    OLMADIGINI soylese de, T008'in kural-tabanli eslestirmesi olumsuzlamayi
    anlamaz; bu, o katmanin bilinen bir sinirlamasidir) kelimelerini icerir;
    bu yuzden TEK bir VLM cikisi 2 ayri kategoriyi tetikler ve `run()` 2
    ayri `StructuredEvent` kaydetmelidir (T013'un C-adimi geregi, sadece
    "birincil" olani degil).
    """
    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    assert report.event_id is not None
    assert len(pipeline._event_history_buffer) >= 1

    rows = pipeline._event_store.query_recent(limit=10)
    assert len(rows) == 2
    row_ids = {row["id"] for row in rows}
    assert report.event_id in row_ids
    for row in rows:
        assert row["risk_score"] == report.risk_score
        assert row["risk_level"] == report.risk_level

    descriptions = " ".join(row["description"] for row in rows)
    assert "OK-07" in descriptions
    assert "YG-03" in descriptions


def test_event_buffer_persists_across_pipeline_calls(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """Buffer, pipeline cagrilari arasinda SIFIRLANMAZ: ayni videoyu iki kez analiz etmek tekrar-tespitini uretmeli.

    Tek-kategori sahte VLM istemcisi (`_SingleCategoryVLMClient`) kullanilir;
    boylece ayni video ikinci kez analiz edildiginde ayni tip + ayni zaman
    damgasi uretilir, T009 bunu ilk cagrinin olayiyla BIRLESTIRIR
    (`occurrence_count=2`) ve EventBuilder bunu "ardisik gozlemde" notuyla
    ikinci cagrinin SQLite satirina yazar; ilk cagrinin satiri degismeden
    kalir (EventStore'da UPDATE yok - T012'nin bilinen kisiti).

    Not: `MockVLMClient`in gercek sabit metniyle (2 kategori, bkz.
    `test_run_writes_structured_event_to_real_event_store`) bu senaryo
    tekrarlandiginda, T009'un "yalnizca dogrudan ARDISIK ayni-tip olaylari
    birlestirme" kurali, buffer'da ic ice giren farkli kategoriler yuzunden
    birlesmeyi engelliyor (COMBO/merge, iki kategori sirayla degil,
    kategoriler arasinda gecis yaparak eklendigi icin "ardisik" sayilmiyor).
    Bu, T009'un bilinen bir sinirlamasi olarak ayrica raporlandi; bu test
    onu degil, T013'un buffer-kalicilik davranisini izole test eder.
    """
    pipeline._vlm = _SingleCategoryVLMClient()

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


def test_record_feedback_delegates_to_event_history(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """`SafirPipeline.record_feedback`, `EventHistory.mark_feedback` uzerinden `EventStore`e yazmali (T013-D)."""
    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    pipeline.record_feedback(report.event_id, "true_positive")

    rows = pipeline._event_store.query_recent(limit=10)
    row = next(r for r in rows if r["id"] == report.event_id)
    assert row["feedback"] == "true_positive"
