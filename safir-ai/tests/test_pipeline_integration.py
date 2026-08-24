"""T013: `src/main.py::SafirPipeline`nin `event_analysis/` (T008-T012) entegrasyonu icin uctan uca test.

Gercek `SafirPipeline.__init__`/`run()` kodunu, mock VLM/LLM (`app.use_mock_vlm`/
`app.use_mock_llm`) ve gercek (ama gecici, `tmp_path` altinda) bir SQLite
`EventStore` ile calistirir; boylece VLM ciktisi -> `EventEngine` ->
`TemporalReasoner` -> `RuleEngine` -> `EventBuilder` -> `EventHistory.record_batch`
zincirinin `SafirPipeline.run()` icinde GERCEKTEN cagrildigini dogrular.

Tek istisna: `src/rag/embedding_rag_service.py::EmbeddingRAGService`, gercek
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

from src.event_analysis.schemas import TemporalEvent
from src.main import SafirPipeline
from src.utils.config_loader import SafirConfig
from src.vlm.base_vlm import VLMResponse


@dataclass
class _FakeRetrievedDocument:
    text: str
    score: float = 1.0
    # RAG entegrasyon dogrulama turu (2026-08-24): provenance alanlari - `getattr(...,
    # None)` KASITLI kullanilan yerlerde (context_builder.py, tools.py, main.py::build_report)
    # bu alanlar OLMADAN da (varsayilan None) hicbir yer PATLAMAMALI; burada VAR OLMALARI,
    # provenance'in gercekten uctan uca tasindigini test edebilmek icindir.
    embedding_score: Optional[float] = None
    relevance_score: Optional[float] = None
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    article_number: Optional[str] = None
    source_url: Optional[str] = None


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

    def query(self, question: str, top_k: Optional[int] = None, keywords: Optional[List[str]] = None) -> List[_FakeRetrievedDocument]:
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

    T016'dan beri nihai risk, mock LLM'in "orta" (35) kararindan DEGIL,
    deterministik RuleEngine'den gelir (bkz. `stage_finalize_risk`):
    MockVLMClient'in aciklamasi "forklift" icerir -> `arac_yaya_yakinligi`
    tespit edilir -> OK-07 kurali "yuksek" siddet uretir (score=63) -> bu,
    LLM'in kararini gezer ve otomatik alarmi tetikler. Ayrica sartname-uyumlu
    ozet JSON'un beklenen anahtarlari tasidigi dogrulanir.
    """
    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    # Yeni sema alanlari dolu olmali.
    assert report.summary
    assert isinstance(report.actions, list) and report.actions
    assert report.recommended_action == report.actions[0]

    # Otomatik eskalasyon (Human-on-the-Loop): RuleEngine-turevli yuksek risk -> alarm.
    assert report.escalation_tier == "alarm"
    assert report.auto_dispatched is True
    assert report.alert_id is not None

    # Sartname-uyumlu ozet JSON beklenen sekilde olmali.
    sartname = report.to_sartname_json()
    assert set(sartname) >= {"summary", "events", "risk", "actions"}
    assert sartname["risk"] == report.risk_level
    assert all("time" in e and "event" in e for e in sartname["events"])


def test_high_risk_auto_dispatches_alarm_in_pipeline(
    pipeline: SafirPipeline, motion_video: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Yuksek risk skorunda pipeline, operator onayi beklemeden alarmi OTOMATIK tetiklemeli.

    T016 (risk_resolver): nihai risk_score/risk_level artik LLM Agent'in
    kararindan DEGIL, `RuleEngine`in bu cagriya ait deterministik
    `RuleMatch`lerinden (`stage_finalize_risk`) turetiliyor. Bu videonun
    MockVLMClient aciklamasi "forklift" iceriyor -> `arac_yaya_yakinligi`
    tespit edilir -> OK-07 kurali "yuksek" siddet uretir (score=63); bu,
    LLM Agent'in (burada kasitli olarak dusuk/celisen bir deger doner)
    kararini GEZER ve yine de otomatik alarmi tetikler.
    """
    from src.agent.langgraph_agent import AgentDecision

    low_decision = AgentDecision(
        risk_score=5,
        risk_level="dusuk",
        recommended_action="Rutin izleme",
        raw_response="{}",
        summary="Rutin durum",
        actions=["Rutin izleme"],
        events=[],
    )
    monkeypatch.setattr(pipeline._agent, "run", lambda _prompt: low_decision)

    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    # Deterministik rule engine sonucu (OK-07 -> yuksek/63), LLM'in dusuk
    # kararini gezip nihai raporu belirler.
    assert report.risk_score == 63
    assert report.risk_level == "yuksek"
    assert report.escalation_tier == "alarm"
    assert report.auto_dispatched is True
    assert report.alert_id is not None

    # Operator, otomatik tetiklenen alarmi sonradan onaylayabilmeli (Human-on-the-Loop).
    record = pipeline.acknowledge_alert(report.alert_id, "operator denetledi")
    assert record.acknowledged is True


def test_rule_engine_has_no_matches_falls_back_to_llm_agent_decision(
    pipeline: SafirPipeline, motion_video: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hicbir kural eslesmediginde (RuleEngine sinyal uretmediginde), sistem CRASH OLMAZ

    ve nihai risk, LLM Agent'in kararina (varsa) duser - deterministik risk
    UYDURULMAZ."""
    from src.agent.langgraph_agent import AgentDecision

    monkeypatch.setattr(pipeline, "_rule_engine", type("_Empty", (), {"evaluate": staticmethod(lambda _events: [])})())

    llm_decision = AgentDecision(
        risk_score=42,
        risk_level="orta",
        recommended_action="Sahayi izleyin",
        raw_response="{}",
        summary="Belirsiz durum",
        actions=["Sahayi izleyin"],
        events=[],
    )
    monkeypatch.setattr(pipeline._agent, "run", lambda _prompt: llm_decision)

    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    assert report.risk_score == 42
    assert report.risk_level == "orta"


# ------------------------------------------------------------------
# T017: RAG/LangGraph mevzuat eslestirme - forced-match duzeltmesi
# ------------------------------------------------------------------


def test_report_regulations_reflect_rule_engine_match_not_raw_text_similarity(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """En onemli regresyon: `_FakeRagService`, HANGI soru sorulursa sorulsun HER ZAMAN
    `"[FAKE-RAG] {question}"` seklinde bir "belge" doner (yani her sorguya ilgisiz/
    kosulsuz bir eslesme verir - tam olarak "ilgisiz bir ISG dokumaninin getirilip
    zorla eslestirilmesi" senaryosu). Eger `ContextBuilder` hala eskisi gibi
    `vlm_description` uzerinde bagimsiz bir RAG sorgusu yapsaydi, rapor bu HAM
    metni ("[FAKE-RAG] <tum VLM aciklamasi>") "ilgili mevzuat" olarak gosterirdi.

    T017 sonrasi, mevzuat listesi YALNIZCA RuleEngine'in deterministik
    event_type -> mevzuat eslemesinden (bu videoda: forklift -> arac_yaya_yakinligi
    -> OK-07) gelmeli; ham VLM aciklama metninin RAG'e sorulup donen sonucu
    DEGIL.
    """
    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    assert report.relevant_regulations, "OK-07 deterministik olarak eslesmis olmali"
    for regulation in report.relevant_regulations:
        # RuleEngine._describe_regulation, retriever'i mevzuat ETIKETIYLE (orn.
        # "Operasyonel Kural OK-07") sorgular - ham VLM aciklama metniyle DEGIL;
        # `_FakeRagService`, sorulan soruyu aynen "[FAKE-RAG] <soru>" olarak
        # yansitir, bu yuzden etiketin kendisi metinde gorunmeli.
        assert "[FAKE-RAG] Operasyonel Kural OK-07" in regulation or "[FAKE-RAG] ISG" in regulation, (
            f"Beklenmedik mevzuat metni (ham VLM aciklamasindan mi geldi?): {regulation!r}"
        )
        assert report.natural_language_summary not in regulation


def test_no_rule_match_produces_empty_regulation_list_not_a_random_fake_regulation(
    pipeline: SafirPipeline, motion_video: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RuleEngine hicbir eslesme uretmediginde (`evaluate` bos donerse), rapor
    BOS bir `relevant_regulations` uretmeli - `_FakeRagService`nin (ilgisiz de
    olsa) DAIMA bir "belge" dondurmesine ragmen. Kullanici, RAG retriever'in
    kosulsuz dondurdugu rastgele bir mevzuati asla GORMEMELI."""
    monkeypatch.setattr(
        pipeline, "_rule_engine", type("_Empty", (), {"evaluate": staticmethod(lambda _events: [])})()
    )

    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    assert report.relevant_regulations == []


def test_rag_service_failure_does_not_crash_the_pipeline(
    pipeline: SafirPipeline, motion_video: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RAG servisi (retriever) patlarsa (agdan erisilemedi/hata), pipeline COKMEMELI.

    `ContextBuilder` artik RAG'e hic dokunmuyor (bkz. T017); tek kalan RAG
    cagrisi `RuleEngine._describe_regulation` icinde ZATEN try/except ile
    korunuyor - bu test o dayanikliligi uctan uca dogrular."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("RAG servisi erisilemez durumda (simulasyon)")

    monkeypatch.setattr(pipeline._rag_service, "query", _boom)

    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    assert report is not None
    assert report.risk_score is not None


def test_regulation_match_presence_does_not_alter_deterministic_risk(
    pipeline: SafirPipeline, motion_video: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bir mevzuat eslesmesinin var/yok olmasi, RuleEngine-turevli deterministik
    risk_score/risk_level'i DEGISTIRMEMELI (RAG/mevzuat, gizli bir ikinci risk
    motoru DEGILDIR - bkz. risk_resolver.py, bu PR'da degistirilmedi)."""
    report_with_match = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")
    assert report_with_match.relevant_regulations  # OK-07 eslesmis olmali

    # Ayni riski (OK-07 -> yuksek/63) uretecek RuleMatch'i KORU, ama mevzuat
    # metnini bos dondurecek sekilde retriever'i degistir: risk ayni kalmali.
    monkeypatch.setattr(pipeline._rag_service, "query", lambda *a, **k: [])
    report_without_enriched_text = pipeline.run(
        motion_video, "Sahnede riskli bir durum var mi degerlendir."
    )

    assert report_without_enriched_text.risk_score == report_with_match.risk_score
    assert report_without_enriched_text.risk_level == report_with_match.risk_level


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
    # `selection_reason` (threshold_exceeded/temporal_coverage/early_change/
    # significant_change/fallback) bir konumsal ROL DEGILDIR - yalnizca "bu
    # kare neden secildi" bilgisidir.
    for ef in evidence_frames:
        field_names = set(type(ef).model_fields.keys())
        assert "label" not in field_names
        assert ef.selection_reason in {
            "threshold_exceeded",
            "temporal_coverage",
            "early_change",
            "significant_change",
            "fallback",
        }


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


# ------------------------------------------------------------------
# T019: VLM-uretimi serbest-bicimli event keywords - uctan uca (VLM -> rapor)
# ------------------------------------------------------------------


def test_report_event_keywords_preserve_exact_non_taxonomy_vlm_strings(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """EN ONEMLI regresyon: VLM'in urettigi, ONCEDEN TANIMLI `_KEYWORD_RULES`
    taksonomisinde HICBIR SEKILDE bulunmayan serbest ifadeler (orn. "yogun
    siyah duman", "dumanin tavana dogru yukselmesi"), pipeline'in TAMAMINDAN
    (VLM -> EVENTS_JSON -> EventEngine -> TemporalEvent -> StructuredEvent ->
    SafirReport -> API payload/`model_dump()`) BIREBIR AYNI sekilde gecmeli -
    "duman"/"alev" gibi taksonomi kelimelerine INDIRGENMEMELI."""
    from src.vlm.base_vlm import VLMResponse

    deliberately_non_taxonomy_keywords = [
        "yoğun siyah duman",
        "alevlenme",
        "dumanın tavana doğru yükselmesi",
        "yanma belirtisi",
    ]

    def _fake_analyze(evidence_frames, prompt):
        return VLMResponse(
            description="Sahada gozlemlenen olay.",
            model_name="fake-vlm",
            frame_count=len(evidence_frames),
            latency_ms=1.0,
            structured_events=[
                {
                    "event_id": "e1",
                    "event_name": "yangin_duman",
                    "canonical_event_type": "yangin_duman",
                    "start_time": evidence_frames[-1].timestamp_sec if evidence_frames else 0.0,
                    "end_time": evidence_frames[-1].timestamp_sec if evidence_frames else 0.0,
                    "evidence_ids": [ef.evidence_id for ef in evidence_frames],
                    "description": "Dumanla ilgili gozlem.",
                    "keywords": deliberately_non_taxonomy_keywords,
                    "risk_score": 70,
                    "confidence": 0.85,
                }
            ],
        )

    pipeline._vlm.analyze_evidence = _fake_analyze  # type: ignore[assignment]

    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    assert report.events, "En az bir EventSummary girisi olmali"
    yangin_entry = next(ev for ev in report.events if ev.event_name == "yangin_duman")

    # 1) Tam olarak VLM'in urettigi ifadeler - EKSIKSIZ ve DEGISTIRILMEDEN.
    assert yangin_entry.keywords == deliberately_non_taxonomy_keywords

    # 2) Taksonomiye INDIRGENMEMIS: sabit "duman"/"alev" kelimeleri TEK BASINA
    #    (ayri elemanlar olarak) listede YOK - yalnizca VLM'in kendi ifadeleri var.
    assert "duman" not in yangin_entry.keywords
    assert "alev" not in yangin_entry.keywords
    assert "yangin" not in yangin_entry.keywords

    # 3) API payload'i simule eden model_dump() ciktisinda da AYNEN korunmali.
    payload = report.model_dump(mode="json")
    payload_entry = next(ev for ev in payload["events"] if ev["event_name"] == "yangin_duman")
    assert payload_entry["keywords"] == deliberately_non_taxonomy_keywords


def test_report_event_keywords_empty_when_vlm_provides_none(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """VLM `keywords` uretmezse (eski/degrade cikti), taksonomi-fallback devreye girer
    (EventEngine seviyesinde beklenen); rapor CRASH olmaz, bos liste UYDURULMAZ."""
    from src.vlm.base_vlm import VLMResponse

    def _fake_analyze(evidence_frames, prompt):
        return VLMResponse(
            description="Sahada gozlemlenen olay.",
            model_name="fake-vlm",
            frame_count=len(evidence_frames),
            latency_ms=1.0,
            structured_events=[
                {
                    "event_id": "e1",
                    "event_name": "genel_gozlem",
                    "canonical_event_type": "genel_gozlem",
                    "start_time": evidence_frames[-1].timestamp_sec if evidence_frames else 0.0,
                    "end_time": evidence_frames[-1].timestamp_sec if evidence_frames else 0.0,
                    "evidence_ids": [ef.evidence_id for ef in evidence_frames],
                    "description": "Rutin gozlem, belirgin bulgu yok.",
                    "risk_score": 5,
                    "confidence": 0.5,
                }
            ],
        )

    pipeline._vlm.analyze_evidence = _fake_analyze  # type: ignore[assignment]

    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    assert report is not None
    assert report.events == [] or all(isinstance(ev.keywords, list) for ev in report.events)


# ------------------------------------------------------------------
# T020: VLM olay kimligini KENDI belirler - taksonomiye ZORLAMA yok (EN ONEMLI TEST)
# ------------------------------------------------------------------


def test_vlm_event_impossible_to_match_existing_taxonomy_survives_as_first_class_event(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """EN ONEMLI test (T020): VLM, ZATEN BILINEN 11 kategoriden HICBIRINE
    oturmayan bir olay uretir ("yerde_hareketsiz_kisi", `canonical_event_type`
    vermez). Bu olay:
    - event_name DEGISMEDEN hayatta kalmali,
    - hicbir EventType'a (orn. 'dusme_riski', 'genel_gozlem') ZORLA
      OTURTULMAMALI (`event_type` None kalmali),
    - RuleEngine bu olay icin HICBIR RuleMatch URETMEMELI,
    - risk UYDURULMAMALI (bu olayin risk_level/risk_score'u None kalmali),
    - UI/API payload'inda ("yerde_hareketsiz_kisi" STRING'i) aynen gorunmeli,
    - keywords DEGISTIRILMEDEN korunmali."""
    from src.vlm.base_vlm import VLMResponse

    keywords = ["yerde yatan kişi", "hareketsiz kişi", "olası yaralanma"]

    def _fake_analyze(evidence_frames, prompt):
        return VLMResponse(
            description="Sahada bir kisi yerde hareketsiz yatiyor.",
            model_name="fake-vlm",
            frame_count=len(evidence_frames),
            latency_ms=1.0,
            structured_events=[
                {
                    "event_id": "e1",
                    "event_name": "yerde_hareketsiz_kisi",
                    # canonical_event_type BILEREK VERILMEDI (VLM emin degil).
                    "start_time": evidence_frames[-1].timestamp_sec if evidence_frames else 0.0,
                    "end_time": evidence_frames[-1].timestamp_sec if evidence_frames else 0.0,
                    "evidence_ids": [ef.evidence_id for ef in evidence_frames],
                    "description": "Bir kisi yerde hareketsiz yatiyor.",
                    "keywords": keywords,
                    "confidence": 0.8,
                    # risk_score de KASITLI olarak verilmedi.
                }
            ],
        )

    pipeline._vlm.analyze_evidence = _fake_analyze  # type: ignore[assignment]

    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    assert report is not None  # sistem COKMEDI

    entry = next(ev for ev in report.events if ev.event_name == "yerde_hareketsiz_kisi")

    # event_name DEGISMEDEN hayatta kaldi.
    assert entry.event_name == "yerde_hareketsiz_kisi"
    # Hicbir EventType'a ZORLA OTURTULMADI.
    assert entry.event_type is None
    # Risk UYDURULMADI (bu olay icin RuleEngine eslesmesi yok).
    assert entry.risk_level is None
    assert entry.risk_score is None
    # keywords DEGISTIRILMEDEN korundu.
    assert entry.keywords == keywords

    # detected_event_names, canonical'i olmasa bile bu olayi ICERIR.
    assert "yerde_hareketsiz_kisi" in report.detected_event_names
    # detected_event_types (yalnizca GERCEKTEN eslesenler) bu olayi ICERMEZ.
    assert "yerde_hareketsiz_kisi" not in report.detected_event_types

    # API payload'inda (model_dump) STRING aynen gorunuyor.
    payload = report.model_dump(mode="json")
    assert "yerde_hareketsiz_kisi" in payload["detected_event_names"]
    payload_entry = next(ev for ev in payload["events"] if ev["event_name"] == "yerde_hareketsiz_kisi")
    assert payload_entry["event_type"] is None
    assert payload_entry["risk_level"] is None
    assert payload_entry["keywords"] == keywords


def test_vlm_event_with_null_canonical_type_remains_first_class_alongside_others(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """Ikinci ornek (T020): "dengesiz_malzeme_istifi", `canonical_event_type=null`.
    Bu olay da (baska taksonomi-uyumlu bir olayla birlikte gelse bile) birinci
    sinif bir olay olarak kalmali - gizlenmemeli/atlanmamali."""
    from src.vlm.base_vlm import VLMResponse

    def _fake_analyze(evidence_frames, prompt):
        return VLMResponse(
            description="Sahada dengesiz istiflenmis malzeme var.",
            model_name="fake-vlm",
            frame_count=len(evidence_frames),
            latency_ms=1.0,
            structured_events=[
                {
                    "event_id": "e1",
                    "event_name": "dengesiz_malzeme_istifi",
                    "canonical_event_type": None,
                    "start_time": evidence_frames[-1].timestamp_sec if evidence_frames else 0.0,
                    "end_time": evidence_frames[-1].timestamp_sec if evidence_frames else 0.0,
                    "evidence_ids": [ef.evidence_id for ef in evidence_frames],
                    "description": "Dengesiz istiflenmis malzeme gozlemlendi.",
                    "keywords": ["dengesiz istif", "devrilme riski"],
                    "confidence": 0.7,
                }
            ],
        )

    pipeline._vlm.analyze_evidence = _fake_analyze  # type: ignore[assignment]

    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    entry = next(ev for ev in report.events if ev.event_name == "dengesiz_malzeme_istifi")
    assert entry.event_type is None
    assert entry.keywords == ["dengesiz istif", "devrilme riski"]


# ------------------------------------------------------------------
# T021: KRITIK regresyon - gercek VLM olaylari (farkli start/end zamanlari
# olan, cok-olayli cikti) SafirReport.events'te SESSIZCE KAYBOLMAMALI.
#
# Kok neden: `TemporalReasoner._build_temporal_event`, `TemporalEvent.
# end_timestamp`i yanlislikla `DetectedEvent.timestamp` (BASLANGIC zamani)
# ile dolduruyordu; gercek bir olay icin (start_time != end_time) bu,
# `main.py::_select_current_call_events`in 1e-6 toleransli karsilastirmasini
# HER ZAMAN basarisiz kiliyor, `structured_events` bos donuyor, dolayisiyla
# `SafirReport.events`/`detected_event_names`/`detected_event_types`
# SESSIZCE BOS kaliyordu - VLM gercekte 2 olay uretmis olsa bile.
# ------------------------------------------------------------------


def test_multiple_real_vlm_events_with_distinct_start_end_times_all_survive_to_report(
    pipeline: SafirPipeline, motion_video: str
) -> None:
    """Gercek Gemini VLM ciktisina benzer (start_time != end_time olan, iki
    farkli serbest-bicimli olay iceren) bir yanit simule edilir. Her iki
    olay da SafirReport.events'e ULASMALI - hicbiri sessizce kaybolmamali."""
    from src.vlm.base_vlm import VLMResponse

    def _fake_analyze(evidence_frames, prompt):
        return VLMResponse(
            description="Sahada iki ayri olay gozlemlendi.",
            model_name="fake-gemini",
            frame_count=len(evidence_frames),
            latency_ms=1.0,
            structured_events=[
                {
                    "event_id": "e1",
                    "event_name": "personelin_alanı_terk_etmesi",
                    "start_time": 6.0,
                    "end_time": 22.0,
                    "evidence_ids": [ef.evidence_id for ef in evidence_frames[:3]],
                    "description": "Personel alani hizla terk etti.",
                    "keywords": ["personel yok", "alanin bosaltilmasi", "aceleci hareket"],
                    "confidence": 0.8,
                },
                {
                    "event_id": "e2",
                    "event_name": "kovada_alev_baslangici",
                    "canonical_event_type": "yangin_duman",
                    "start_time": 38.0,
                    "end_time": 75.0,
                    "evidence_ids": [ef.evidence_id for ef in evidence_frames[3:]] or [],
                    "description": "Kovada kucuk bir alev basladi.",
                    "keywords": ["kovada alev", "kucuk yangin baslangici", "duman izi"],
                    "risk_score": 65,
                    "confidence": 0.75,
                },
            ],
        )

    pipeline._vlm.analyze_evidence = _fake_analyze  # type: ignore[assignment]

    report = pipeline.run(motion_video, "Sahnede riskli bir durum var mi degerlendir.")

    # Hicbir olay sessizce kaybolmadi.
    assert len(report.events) == 2, f"Beklenen 2 olay, gelen: {report.events}"

    by_name = {ev.event_name: ev for ev in report.events}
    assert "personelin_alanı_terk_etmesi" in by_name
    assert "kovada_alev_baslangici" in by_name

    # detected_event_names HER IKISINI de icerir.
    assert "personelin_alanı_terk_etmesi" in report.detected_event_names
    assert "kovada_alev_baslangici" in report.detected_event_names

    # Olay 1: canonical_event_type verilmedi -> event_type None kalmali (ZORLANMADI).
    ev1 = by_name["personelin_alanı_terk_etmesi"]
    assert ev1.event_type is None
    assert ev1.keywords == ["personel yok", "alanin bosaltilmasi", "aceleci hareket"]

    # Olay 2: canonical_event_type GERCEKTEN verildi (yangin_duman) -> korunur.
    ev2 = by_name["kovada_alev_baslangici"]
    assert ev2.event_type == "yangin_duman"
    assert ev2.keywords == ["kovada alev", "kucuk yangin baslangici", "duman izi"]
    # event_name ASLA "yangin_duman"a donusturulmedi (RuleEngine eslesmesi
    # olsa bile birincil kimlik event_name'dir).
    assert ev2.event_name == "kovada_alev_baslangici"

    # API payload'inda (model_dump) da ayni sekilde goruluyor.
    payload = report.model_dump(mode="json")
    assert len(payload["events"]) == 2
    assert set(payload["detected_event_names"]) == {"personelin_alanı_terk_etmesi", "kovada_alev_baslangici"}


# ---------------------------------------------------------------------------
# RAG entegrasyon dogrulama turu (2026-08-24): semantik RAG'in secilen chunk'inin
# (metin + provenance), GERCEK `stage_context`/`build_report` kodundan gecerek
# hem Agent'in aldigi mesaja hem NIHAI, KALICI rapora ulastigini uctan uca dogrular.
# ---------------------------------------------------------------------------


def test_semantic_rag_chunk_reaches_agent_prompt_and_report_provenance(
    pipeline: SafirPipeline,
) -> None:
    """HEDEF 1/3/4: RAG query -> selected chunk -> Agent context -> rapor provenance zincirini mock ile uctan uca dogrular."""
    distinctive_chunk = _FakeRetrievedDocument(
        text="FORKLIFT-UNIQUE-MARKER-XYZ: is ekipmani kullaniminda risk mevcuttur.",
        score=0.77,
        embedding_score=0.77,
        chunk_id="test_dok__madde_1",
        document_id="test_dok",
        document_title="Test ISG Yonetmeligi",
        article_number="1",
        source_url="https://example.org/test-yonetmelik",
    )
    fake_rag_service = pipeline._rag_service  # type: ignore[assignment]
    fake_rag_service.query = lambda question, top_k=None, keywords=None: (  # noqa: E731 - test-only stub
        fake_rag_service.queries.append(question),
        [distinctive_chunk],
    )[1]

    # Agent'in GERCEKTEN aldigi mesajlari yakalamak icin mock LLM'in invoke'unu sarmalar
    # (davranisi DEGISTIRMEZ, yalnizca gozlemler).
    captured_message_batches: List = []
    original_invoke = pipeline._agent._llm.invoke

    def _capturing_invoke(messages):
        captured_message_batches.append(list(messages))
        return original_invoke(messages)

    pipeline._agent._llm.invoke = _capturing_invoke

    temporal_event = TemporalEvent(
        event_id="te-1",
        event_name="forklift_yakinlik",
        event_type="arac_yaya_yakinligi",
        description="Forklift yaya yakininda calisiyor.",
        start_timestamp=1.0,
        end_timestamp=1.0,
        duration=0.0,
        confidence=0.8,
        occurrence_count=1,
        matched_keywords=["forklift", "yaya"],
        source_model="test-vlm",
        related_events=[],
    )
    vlm_response = VLMResponse(
        description="Forklift yaya yakininda calisiyor.", model_name="test-vlm", frame_count=1, latency_ms=1.0
    )

    prompt_block, context = pipeline.stage_context(
        vlm_response, "Risk durumu nedir?", latest_timestamp=1.0, rule_matches=[], temporal_events=[temporal_event]
    )

    # A) RAG GERCEKTEN sorgulandi (matched_keywords sayesinde semantic_query bos degildi).
    assert fake_rag_service.queries, "semantik RAG sorgusu hic yapilmamis (matched_keywords eksik mi?)"

    # B) chunk metni GERCEKTEN ContextBuilder'in urettigi, Agent'e giden prompt_block'ta.
    assert "FORKLIFT-UNIQUE-MARKER-XYZ" in prompt_block
    assert context.semantically_related_chunks[0].chunk_id == "test_dok__madde_1"

    decision = pipeline.stage_decide(prompt_block)

    # C) Agent GERCEKTEN bu metni iceren mesajlarla cagrildi (LangGraph -> LLM sinirinda).
    assert captured_message_batches, "Agent hic cagrilmamis"
    full_text = " ".join(str(m.content) for batch in captured_message_batches for m in batch)
    assert "FORKLIFT-UNIQUE-MARKER-XYZ" in full_text

    decision = pipeline.stage_finalize_risk(decision, [])
    report = pipeline.build_report(
        video_source="test-video",
        sampler=_NullSampler(),
        evidence_frames=[],
        vlm_response=vlm_response,
        context=context,
        decision=decision,
        escalation=_NullEscalation(),
        temporal_events=[temporal_event],
        rule_matches=[],
        latest_timestamp=1.0,
    )

    # D) provenance (chunk_id/document_id/article_number/source_url) rapora KALICI olarak tasindi.
    assert report.semantic_rag_sources, "semantic_rag_sources bos - provenance rapora ulasmadi"
    source = report.semantic_rag_sources[0]
    assert source.chunk_id == "test_dok__madde_1"
    assert source.document_id == "test_dok"
    assert source.article_number == "1"
    assert source.source_url == "https://example.org/test-yonetmelik"
    assert "FORKLIFT-UNIQUE-MARKER-XYZ" in source.content


class _NullSampler:
    """`sampler.last_run_stats`e erisen `build_report` icin minimal sahte - bu testte sampler'in KENDISI test EDILMIYOR."""

    class _Stats:
        total_frames_scanned = 0
        sampled_frames_evaluated = 0
        evidence_frame_count = 0
        eliminated_frame_count = 0
        eliminated_ratio_pct = 0.0
        elapsed_sec = 0.0

    last_run_stats = _Stats()


class _NullEscalation:
    """`escalation.tier/auto_dispatched/alert_id`e erisen `build_report` icin minimal sahte."""

    class _Tier:
        value = "monitor"

    tier = _Tier()
    auto_dispatched = False
    alert_id = None
