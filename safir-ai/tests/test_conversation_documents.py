"""Adim 4 - belge yukleme uc noktalari (`POST`/`DELETE /conversations/{id}/documents*`) testleri.

Gercek PDF/TXT/DOCX baytlariyla, gercek `TestClient` HTTP istekleriyle,
arka-plan thread'inin GERCEKTEN tamamlanmasini bekleyerek (poll) calisir.
"""

from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient

import src.main as main
from src.main import app
from src.memory.conversation_store import ConversationStore
from src.memory import document_extraction as de


@pytest.fixture
def conv_env(monkeypatch, tmp_path):
    """Izole (tmp) ConversationStore + izole (tmp) belge diski, main'e enjekte edilir."""
    store = ConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr(main, "_conversation_store", store)
    monkeypatch.setattr(main, "_DATA_DIR", str(tmp_path / "data"))
    return store


def _make_pdf_bytes(pages: list[str]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        c.drawString(50, 800, text)
        c.showPage()
    c.save()
    return buf.getvalue()


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    import docx

    buf = io.BytesIO()
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    d.save(buf)
    return buf.getvalue()


def _wait_for_document(client: TestClient, conv_id: str, document_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        detail = client.get(f"/conversations/{conv_id}").json()
        doc = next((d for d in detail["documents"] if d["document_id"] == document_id), None)
        if doc is not None and doc["status"] != "processing":
            return doc
        time.sleep(0.05)
    raise AssertionError("Belge zamaninda islenmedi.")


def _new_conversation(client: TestClient) -> str:
    return client.post("/conversations", json={}).json()["conversation_id"]


# --------------------------- upload: her format ---------------------------


def test_upload_txt_succeeds_and_becomes_ready(conv_env):
    client = TestClient(app)
    conv_id = _new_conversation(client)

    resp = client.post(
        f"/conversations/{conv_id}/documents",
        files={"file": ("notlar.txt", b"Bu tesiste CO2 bazli yangin sondurme sistemi kullanilmaktadir.", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "processing"
    assert body["filename"] == "notlar.txt"
    assert body["file_ext"] == "txt"

    doc = _wait_for_document(client, conv_id, body["document_id"])
    assert doc["status"] == "ready"
    assert doc["error_message"] is None


def test_upload_pdf_succeeds_and_becomes_ready(conv_env):
    client = TestClient(app)
    conv_id = _new_conversation(client)
    pdf_bytes = _make_pdf_bytes(["Is Guvenligi Yonetmeligi sayfa bir icerigi."])

    resp = client.post(
        f"/conversations/{conv_id}/documents",
        files={"file": ("yonetmelik.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200
    document_id = resp.json()["document_id"]

    doc = _wait_for_document(client, conv_id, document_id)
    assert doc["status"] == "ready"
    assert doc["page_count"] == 1


def test_upload_docx_succeeds_and_becomes_ready(conv_env):
    client = TestClient(app)
    conv_id = _new_conversation(client)
    docx_bytes = _make_docx_bytes(["Prosedur maddesi bir.", "Prosedur maddesi iki."])

    resp = client.post(
        f"/conversations/{conv_id}/documents",
        files={"file": ("prosedur.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert resp.status_code == 200
    document_id = resp.json()["document_id"]

    doc = _wait_for_document(client, conv_id, document_id)
    assert doc["status"] == "ready"


# --------------------------- reddedilen durumlar ---------------------------


def test_upload_unsupported_extension_rejected_422(conv_env):
    client = TestClient(app)
    conv_id = _new_conversation(client)
    resp = client.post(
        f"/conversations/{conv_id}/documents",
        files={"file": ("virus.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert resp.status_code == 422


def test_upload_oversized_file_rejected_413(conv_env, monkeypatch):
    monkeypatch.setattr(de, "MAX_FILE_SIZE_BYTES", 50)  # test hizi icin kucuk sinir
    client = TestClient(app)
    conv_id = _new_conversation(client)
    resp = client.post(
        f"/conversations/{conv_id}/documents",
        files={"file": ("buyuk.txt", b"a" * 1000, "text/plain")},
    )
    assert resp.status_code == 413


def test_upload_too_many_files_rejected_422(conv_env, monkeypatch):
    monkeypatch.setattr(de, "MAX_FILES_PER_CONVERSATION", 2)
    client = TestClient(app)
    conv_id = _new_conversation(client)
    for i in range(2):
        resp = client.post(
            f"/conversations/{conv_id}/documents",
            files={"file": (f"f{i}.txt", b"icerik", "text/plain")},
        )
        assert resp.status_code == 200
    resp = client.post(
        f"/conversations/{conv_id}/documents",
        files={"file": ("uc.txt", b"icerik", "text/plain")},
    )
    assert resp.status_code == 422


def test_upload_to_unknown_conversation_404(conv_env):
    client = TestClient(app)
    resp = client.post(
        "/conversations/yok-boyle-sohbet/documents",
        files={"file": ("x.txt", b"icerik", "text/plain")},
    )
    assert resp.status_code == 404


def test_upload_empty_file_rejected_422(conv_env):
    client = TestClient(app)
    conv_id = _new_conversation(client)
    resp = client.post(
        f"/conversations/{conv_id}/documents",
        files={"file": ("bos.txt", b"", "text/plain")},
    )
    assert resp.status_code == 422


# --------------------------- izolasyon + silme + kalicilik ---------------------------


def test_document_isolated_between_conversations(conv_env):
    client = TestClient(app)
    conv_a = _new_conversation(client)
    conv_b = _new_conversation(client)

    resp = client.post(
        f"/conversations/{conv_a}/documents",
        files={"file": ("yalnizca-a.txt", b"A'ya ait icerik", "text/plain")},
    )
    document_id = resp.json()["document_id"]
    _wait_for_document(client, conv_a, document_id)

    detail_b = client.get(f"/conversations/{conv_b}").json()
    assert detail_b["documents"] == []

    # B uzerinden A'nin belgesini silmeye calismak -> 404, A'da hala duruyor olmali
    del_resp = client.delete(f"/conversations/{conv_b}/documents/{document_id}")
    assert del_resp.status_code == 404
    assert len(client.get(f"/conversations/{conv_a}").json()["documents"]) == 1


def test_remove_document_deletes_db_row_and_disk_file(conv_env):
    client = TestClient(app)
    conv_id = _new_conversation(client)
    resp = client.post(
        f"/conversations/{conv_id}/documents",
        files={"file": ("silinecek.txt", b"silinecek icerik", "text/plain")},
    )
    document_id = resp.json()["document_id"]
    _wait_for_document(client, conv_id, document_id)

    file_path = main._conversation_file_path(conv_id, document_id, "txt")
    assert file_path.is_file()

    del_resp = client.delete(f"/conversations/{conv_id}/documents/{document_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"removed": True}

    assert not file_path.exists()
    assert client.get(f"/conversations/{conv_id}").json()["documents"] == []


def test_remove_unknown_document_404(conv_env):
    client = TestClient(app)
    conv_id = _new_conversation(client)
    resp = client.delete(f"/conversations/{conv_id}/documents/yok-boyle-belge")
    assert resp.status_code == 404


def test_document_and_chunks_persist_across_store_reopen(tmp_path):
    """Sayfa yenilense/sunucu yeniden baslasa bile yuklenen belge + chunk'lari kaybolmamali."""
    db = tmp_path / "c.db"
    s1 = ConversationStore(db)
    conv = s1.create(title="t")
    record = s1.create_document(conv.conversation_id, filename="kalici.txt", file_ext="txt", file_size_bytes=10)
    chunks = de.chunk_page_texts([(None, "kalici chunk icerigi")])
    s1.add_chunks(record.document_id, conv.conversation_id, chunks)
    s1.update_document_status(record.document_id, status="ready")
    s1.close()

    s2 = ConversationStore(db)
    docs = s2.list_documents(conv.conversation_id)
    assert len(docs) == 1 and docs[0].status == "ready"
    stored_chunks = s2.list_chunks_for_conversation(conv.conversation_id)
    assert len(stored_chunks) == 1
    assert stored_chunks[0].content == "kalici chunk icerigi"
    assert stored_chunks[0].filename == "kalici.txt"
    s2.close()


# --------------------------- gercekten cikarim/chunk oluyor mu ---------------------------


def test_extraction_actually_produces_real_text(conv_env):
    client = TestClient(app)
    conv_id = _new_conversation(client)
    resp = client.post(
        f"/conversations/{conv_id}/documents",
        files={"file": ("gercek.txt", b"Forklift-yaya carpisma riskine dikkat edilmelidir.", "text/plain")},
    )
    document_id = resp.json()["document_id"]
    _wait_for_document(client, conv_id, document_id)

    store: ConversationStore = conv_env
    chunks = store.list_chunks_for_conversation(conv_id)
    assert len(chunks) >= 1
    assert "forklift" in chunks[0].content.lower()


def test_multi_page_pdf_produces_chunks_with_correct_page_numbers(conv_env):
    client = TestClient(app)
    conv_id = _new_conversation(client)
    pdf_bytes = _make_pdf_bytes(["Birinci sayfa forklift bilgisi.", "Ikinci sayfa yangin CO2 bilgisi."])
    resp = client.post(
        f"/conversations/{conv_id}/documents",
        files={"file": ("cok-sayfali.pdf", pdf_bytes, "application/pdf")},
    )
    document_id = resp.json()["document_id"]
    _wait_for_document(client, conv_id, document_id)

    store: ConversationStore = conv_env
    chunks = store.list_chunks_for_conversation(conv_id)
    pages = {c.page_number for c in chunks}
    assert pages == {1, 2}
    page1 = next(c for c in chunks if c.page_number == 1)
    page2 = next(c for c in chunks if c.page_number == 2)
    assert "forklift" in page1.content.lower()
    assert "yangin" in page2.content.lower() or "co2" in page2.content.lower()
