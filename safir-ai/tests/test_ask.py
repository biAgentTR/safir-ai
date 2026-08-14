"""ADIM 5B: Ask SAFIR (/ask) testleri.

Mevcut RAG + LLM soyutlamalari (fake RAG + MockLLMClient) ile; pipeline/history
davranisi degismeden, soru -> (opsiyonel analiz baglami) + RAG -> LLM -> dayanakli
cevap + gercek kaynaklar zincirini ve sizinti-yoklugunu dogrular.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.main as main
from src.main import app
from src.assistant.ask_service import AskService
from src.memory.analysis_store import AnalysisStore
from src.vlm.llm_client import MockLLMClient


# --------------------------- fakes / fixtures ---------------------------


class _RagDoc:
    def __init__(self, text: str, score: float) -> None:
        self.text = text
        self.score = score


class _FakeAskRAG:
    """Ask icin sabit iki ISG maddesi (gercek text+score sozlesmesi)."""

    def query(self, question: str, top_k=None):
        return [
            _RagDoc("ISG Yonetmeligi Madde 12: Yuksekte calismada KKD zorunludur.", 0.82),
            _RagDoc("Operasyonel Kural OK-07: Forklift-yaya ayrimi saglanmalidir.", 0.71),
        ]


class _FakeDoc:
    def __init__(self, text: str) -> None:
        self.text = text
        self.score = 1.0


class _FakePipelineRAG:
    def seed_default_regulations(self):
        return None

    def query(self, question, top_k=None):
        return [_FakeDoc(f"[FAKE] {question}")]


@pytest.fixture
def ask_env(monkeypatch, tmp_path):
    """Mock pipeline + izole AnalysisStore + fake-RAG/mock-LLM AskService."""
    monkeypatch.setattr(main, "EmbeddingRAGService", lambda *a, **k: _FakePipelineRAG())
    base = main.load_config()
    cfg = base.model_copy(update={"app": base.app.model_copy(update={"use_mock_vlm": True, "use_mock_llm": True})})
    monkeypatch.setattr(main, "_pipeline", main.SafirPipeline(cfg))

    store = AnalysisStore(tmp_path / "analyses.db")
    monkeypatch.setattr(main, "_analysis_store", store)

    llm = MockLLMClient()
    ask = AskService(analysis_store=store, rag_service=_FakeAskRAG(), llm_client=llm, rag_top_k=5)
    monkeypatch.setattr(main, "_ask_service", ask)
    return {"store": store, "ask": ask, "llm": llm}


@pytest.fixture
def video_in_data():
    Path("data").mkdir(exist_ok=True)
    path = Path("data") / "_ask_test.mp4"
    frames = [np.full((120, 160, 3), 30, np.uint8) for _ in range(60)]
    for i in range(20, 40):
        cv2.rectangle(frames[i], (20, 20), (140, 100), (210, 210, 210), -1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
    for f in frames:
        writer.write(f)
    writer.release()
    yield str(path)
    path.unlink(missing_ok=True)


def _run_job(client: TestClient, video: str) -> str:
    job_id = client.post("/analyze/jobs", json={"video_source": video, "user_prompt": "risk?"}).json()["job_id"]
    deadline = time.time() + 30
    while time.time() < deadline:
        if client.get(f"/analyze/jobs/{job_id}").json()["status"] in ("done", "error"):
            return job_id
        time.sleep(0.1)
    raise AssertionError("Is tamamlanmadi.")


# --------------------------- general ask ---------------------------


def test_general_ask_uses_rag_sources(ask_env):
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "Yuksekte calismada nelere dikkat edilmeli?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] and isinstance(body["answer"], str)
    assert body["job_id"] is None
    # sources GERCEK retrieval sonucundan (fake RAG'in text+score'u)
    regs = [s for s in body["sources"] if s["type"] == "regulation"]
    assert len(regs) == 2
    assert regs[0]["text"].startswith("ISG Yonetmeligi Madde 12") and regs[0]["score"] == 0.82
    assert "RAG mevzuat (2)" in body["context_used"]


# --------------------------- job-scoped ask ---------------------------


def test_job_scoped_ask_uses_analysis_context(ask_env, video_in_data):
    client = TestClient(app)
    job_id = _run_job(client, video_in_data)

    resp = client.post("/ask", json={"question": "Bu analiz neden bu risk seviyesinde?", "job_id": job_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert any("analiz raporu" in c for c in body["context_used"])
    # analiz kaynagi + mevzuat kaynaklari birlikte
    types = {s["type"] for s in body["sources"]}
    assert "analysis" in types and "regulation" in types
    analysis_src = next(s for s in body["sources"] if s["type"] == "analysis")
    assert analysis_src["label"] == "SAFIR analiz raporu"


def test_unknown_job_id_404(ask_env):
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "neden?", "job_id": "yok-boyle-job"})
    assert resp.status_code == 404


def test_empty_question_422(ask_env):
    client = TestClient(app)
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422


# --------------------------- safety / leakage ---------------------------


def test_no_internal_or_secret_leakage(ask_env, video_in_data):
    client = TestClient(app)
    job_id = _run_job(client, video_in_data)
    resp = client.post("/ask", json={"question": "Bu kararin dayanagi ne?", "job_id": job_id})
    blob = json.dumps(resp.json(), ensure_ascii=False)
    for forbidden in ("raw_response", "api_key", "API_KEY", "base64", "image_bytes", "system_prompt", "Traceback"):
        assert forbidden not in blob, f"sizinti: {forbidden}"


def test_llm_unavailable_returns_safe_503(ask_env, monkeypatch):
    client = TestClient(app)

    class _BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("gizli stack detay: SECRET")

    monkeypatch.setattr(main._ask_service, "_llm", _BoomLLM())
    resp = client.post("/ask", json={"question": "test"})
    assert resp.status_code == 503
    assert "SECRET" not in resp.text and "Traceback" not in resp.text
    assert resp.json()["detail"] == "SAFIR su anda cevap olusturamadi."


def test_ask_system_prompt_enforces_grounding():
    from src.assistant.ask_service import ASK_SYSTEM_PROMPT

    low = ASK_SYSTEM_PROMPT.lower()
    assert "uydurma" in low and "baglam" in low  # sadece baglamdan, uydurma yok


# --------------------------- _prepare / history (conversation-aware) ---------------------------


def test_prepare_without_history_has_no_history_block(ask_env):
    """`history` verilmezse (varsayilan, `/ask`in kullandigi yol) davranis oncekiyle BIREBIR AYNI."""
    ask = ask_env["ask"]
    messages, sources, context_used, used_job = ask._prepare("Genel bir soru", None, None)
    human = messages[-1].content
    assert "KONUSMA GECMISI" not in human
    assert not any("konusma gecmisi" in c for c in context_used)


def test_prepare_with_history_adds_labeled_non_evidence_block(ask_env):
    ask = ask_env["ask"]
    history = [("user", "Bu analizde en kritik olay ne?"), ("assistant", "En kritik olay X.")]
    messages, sources, context_used, used_job = ask._prepare("Peki neden?", None, history)
    human = messages[-1].content
    assert "KONUSMA GECMISI" in human
    assert "KANIT DEGILDIR" in human
    assert "Bu analizde en kritik olay ne?" in human
    assert any("konusma gecmisi" in c for c in context_used)


def test_prepare_history_is_bounded_to_max_messages(ask_env):
    from src.assistant.ask_service import _MAX_HISTORY_MESSAGES

    ask = ask_env["ask"]
    history = [("user", f"soru {i}") for i in range(_MAX_HISTORY_MESSAGES + 10)]
    messages, sources, context_used, used_job = ask._prepare("son soru", None, history)
    human = messages[-1].content
    assert "soru 0" not in human  # eski mesajlar kirpilmis olmali
    assert f"soru {_MAX_HISTORY_MESSAGES + 9}" in human  # en yeni mesaj kalmali


# --------------------------- /ask/stream (SSE) ---------------------------


def _parse_sse_blocks(text: str) -> list[tuple[str, dict]]:
    blocks: list[tuple[str, dict]] = []
    for raw in text.split("\n\n"):
        raw = raw.strip("\n")
        if not raw:
            continue
        event = "message"
        data = None
        for line in raw.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        if data is not None:
            blocks.append((event, json.loads(data)))
    return blocks


def test_ask_stream_returns_meta_then_deltas_then_end(ask_env):
    from src.vlm.llm_client import _MOCK_DECISION_TEXT

    client = TestClient(app)
    resp = client.get("/ask/stream", params={"question": "Yuksekte calismada nelere dikkat edilmeli?"})
    assert resp.status_code == 200
    blocks = _parse_sse_blocks(resp.text)

    assert len(blocks) >= 3  # meta + en az bir delta + end
    first_event, first_data = blocks[0]
    assert first_event == "message"
    assert "sources" in first_data and "context_used" in first_data and "job_id" in first_data

    delta_blocks = [b for b in blocks[1:-1]]
    assert all(ev == "message" and "delta" in data for ev, data in delta_blocks)

    last_event, last_data = blocks[-1]
    assert last_event == "end"
    assert last_data["status"] == "done"

    # parcalar dogru sirada birlestiginde MockLLMClient'in TAM sabit metnini vermeli
    reassembled = "".join(data["delta"] for _, data in delta_blocks)
    assert reassembled == _MOCK_DECISION_TEXT


def test_ask_stream_job_scoped_meta_matches_ask(ask_env, video_in_data):
    client = TestClient(app)
    job_id = _run_job(client, video_in_data)

    resp = client.get("/ask/stream", params={"question": "Bu analiz neden bu risk seviyesinde?", "job_id": job_id})
    blocks = _parse_sse_blocks(resp.text)
    _, meta = blocks[0]
    assert meta["job_id"] == job_id
    types = {s["type"] for s in meta["sources"]}
    assert "analysis" in types and "regulation" in types


def test_ask_stream_unknown_job_emits_error_event(ask_env):
    client = TestClient(app)
    resp = client.get("/ask/stream", params={"question": "neden?", "job_id": "yok-boyle-job"})
    assert resp.status_code == 200  # SSE: HTTP durumu 200, hata olay olarak gelir
    blocks = _parse_sse_blocks(resp.text)
    assert len(blocks) == 1
    event, data = blocks[0]
    assert event == "error"
    assert "bulunamadi" in data["detail"].lower()


def test_ask_stream_blank_question_emits_error_event(ask_env):
    client = TestClient(app)
    resp = client.get("/ask/stream", params={"question": "   "})
    blocks = _parse_sse_blocks(resp.text)
    assert len(blocks) == 1
    assert blocks[0][0] == "error"


def test_ask_stream_llm_failure_emits_safe_error_no_leak(ask_env):
    class _BoomStreamLLM:
        def invoke(self, messages):
            raise RuntimeError("gizli detay: SECRET")

        def stream(self, messages):
            raise RuntimeError("gizli stack detay: SECRET_STREAM")

    ask_env["ask"]._llm = _BoomStreamLLM()
    client = TestClient(app)
    resp = client.get("/ask/stream", params={"question": "test"})
    blocks = _parse_sse_blocks(resp.text)

    # ilk blok (meta) her zaman gelir - LLM cagrisi ondan SONRA yapilir
    assert blocks[0][0] == "message"
    error_blocks = [b for b in blocks if b[0] == "error"]
    assert len(error_blocks) == 1
    assert error_blocks[0][1]["detail"] == "SAFIR su anda cevap olusturamadi."
    assert "SECRET_STREAM" not in resp.text and "Traceback" not in resp.text


def test_ask_stream_empty_llm_response_emits_error_event(ask_env):
    class _EmptyStreamLLM:
        def invoke(self, messages):
            from langchain_core.messages import AIMessage

            return AIMessage(content="")

        def stream(self, messages):
            return iter(())  # hic parca yok

    ask_env["ask"]._llm = _EmptyStreamLLM()
    client = TestClient(app)
    resp = client.get("/ask/stream", params={"question": "test"})
    blocks = _parse_sse_blocks(resp.text)
    assert blocks[-1][0] == "error"
    assert blocks[-1][1]["detail"] == "SAFIR su anda cevap olusturamadi."


def test_ask_stream_uses_conversation_history_end_to_end(monkeypatch, ask_env):
    """`conversation_id` verilirse gecmis okunur ve akis yine de basariyla tamamlanir (crash yok)."""
    from src.memory.conversation_store import ConversationStore

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(Path(tmp) / "conversations.db")
        monkeypatch.setattr(main, "_conversation_store", store)
        conv = store.create(title="t")
        store.add_message(conv.conversation_id, "user", "Bu analizde en kritik olay ne?")
        store.add_message(conv.conversation_id, "assistant", "[MOCK] cevap")

        client = TestClient(app)
        resp = client.get(
            "/ask/stream", params={"question": "Peki neden?", "conversation_id": conv.conversation_id}
        )
        blocks = _parse_sse_blocks(resp.text)
        assert blocks[-1][0] == "end"


def test_ask_stream_unknown_conversation_id_does_not_crash(ask_env):
    """Bilinmeyen `conversation_id` sessizce baglamsiz devam etmeli (cokme yok)."""
    client = TestClient(app)
    resp = client.get("/ask/stream", params={"question": "test", "conversation_id": "yok-boyle-sohbet"})
    assert resp.status_code == 200
    blocks = _parse_sse_blocks(resp.text)
    assert blocks[-1][0] == "end"
