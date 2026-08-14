"""SAFIR Asistan sohbet gecmisi (Conversation) persistence + API testleri.

`ConversationStore` (src/memory/conversation_store.py) ve `/conversations*`
uc noktalarinin CRUD davranisini dogrular. Bu mesajlar hicbir zaman LLM'e
geri beslenmez (bkz. ask_service.py) - burada yalnizca UI/kalicilik
davranisi test edilir.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.main as main
from src.main import app
from src.memory.conversation_store import ConversationStore


@pytest.fixture
def conv_env(monkeypatch, tmp_path):
    """Izole (tmp) ConversationStore, main'e enjekte edilir."""
    store = ConversationStore(tmp_path / "conversations.db")
    monkeypatch.setattr(main, "_conversation_store", store)
    return store


# --------------------------- store unit ---------------------------


def test_store_create_and_get(tmp_path):
    store = ConversationStore(tmp_path / "c.db")
    conv = store.create(title="Forklift neden riskli?", job_id="job-1")
    assert conv.conversation_id
    fetched = store.get(conv.conversation_id)
    assert fetched is not None
    assert fetched.title == "Forklift neden riskli?"
    assert fetched.job_id == "job-1"


def test_store_add_message_updates_conversation_timestamp(tmp_path):
    store = ConversationStore(tmp_path / "c.db")
    conv = store.create(title="t")
    before = store.get(conv.conversation_id).updated_at
    msg = store.add_message(conv.conversation_id, "user", "Merhaba")
    after = store.get(conv.conversation_id).updated_at
    assert msg.role == "user" and msg.content == "Merhaba"
    assert after >= before


def test_store_list_messages_chronological(tmp_path):
    store = ConversationStore(tmp_path / "c.db")
    conv = store.create()
    store.add_message(conv.conversation_id, "user", "soru 1")
    store.add_message(conv.conversation_id, "assistant", "cevap 1")
    store.add_message(conv.conversation_id, "user", "soru 2")
    msgs = store.list_messages(conv.conversation_id)
    assert [m.content for m in msgs] == ["soru 1", "cevap 1", "soru 2"]
    assert [m.role for m in msgs] == ["user", "assistant", "user"]


def test_store_list_newest_updated_first(tmp_path):
    import time

    store = ConversationStore(tmp_path / "c.db")
    c1 = store.create(title="ilk")
    time.sleep(0.005)
    c2 = store.create(title="ikinci")
    ids = [c.conversation_id for c in store.list(limit=10)]
    assert ids == [c2.conversation_id, c1.conversation_id]

    # c1'e mesaj eklenince en guncel o olmali (updated_at DESC)
    time.sleep(0.005)
    store.add_message(c1.conversation_id, "user", "merhaba")
    ids_after = [c.conversation_id for c in store.list(limit=10)]
    assert ids_after[0] == c1.conversation_id


def test_store_persists_across_reopen(tmp_path):
    db = tmp_path / "c.db"
    s1 = ConversationStore(db)
    conv = s1.create(title="kalici")
    s1.add_message(conv.conversation_id, "user", "soru")
    s1.close()

    s2 = ConversationStore(db)
    fetched = s2.get(conv.conversation_id)
    assert fetched is not None and fetched.title == "kalici"
    assert len(s2.list_messages(conv.conversation_id)) == 1
    s2.close()


# --------------------------- API: CRUD ---------------------------


def test_create_conversation_via_api(conv_env):
    client = TestClient(app)
    resp = client.post("/conversations", json={"title": "Depo analizi", "job_id": "job-abc"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Depo analizi"
    assert body["job_id"] == "job-abc"
    assert body["conversation_id"]


def test_create_conversation_without_title_or_job(conv_env):
    client = TestClient(app)
    resp = client.post("/conversations", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] is None
    assert body["job_id"] is None


def test_list_conversations_newest_first(conv_env):
    client = TestClient(app)
    c1 = client.post("/conversations", json={"title": "birinci"}).json()["conversation_id"]
    c2 = client.post("/conversations", json={"title": "ikinci"}).json()["conversation_id"]
    lst = client.get("/conversations").json()
    ids = [c["conversation_id"] for c in lst]
    assert ids[0] == c2 and c1 in ids


def test_get_conversation_detail_includes_messages(conv_env):
    client = TestClient(app)
    conv_id = client.post("/conversations", json={"title": "t"}).json()["conversation_id"]
    client.post(f"/conversations/{conv_id}/messages", json={"role": "user", "content": "Bu analizde en kritik olay ne?"})
    client.post(f"/conversations/{conv_id}/messages", json={"role": "assistant", "content": "[MOCK] cevap"})

    detail = client.get(f"/conversations/{conv_id}").json()
    assert detail["conversation_id"] == conv_id
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"


def test_get_unknown_conversation_404(conv_env):
    client = TestClient(app)
    assert client.get("/conversations/yok-boyle-sohbet").status_code == 404


def test_add_message_to_unknown_conversation_404(conv_env):
    client = TestClient(app)
    resp = client.post("/conversations/yok-boyle-sohbet/messages", json={"role": "user", "content": "x"})
    assert resp.status_code == 404


def test_add_message_invalid_role_422(conv_env):
    client = TestClient(app)
    conv_id = client.post("/conversations", json={}).json()["conversation_id"]
    resp = client.post(f"/conversations/{conv_id}/messages", json={"role": "system", "content": "x"})
    assert resp.status_code == 422


def test_conversation_title_is_truncated_server_side(conv_env):
    client = TestClient(app)
    long_title = "a" * 500
    resp = client.post("/conversations", json={"title": long_title})
    assert len(resp.json()["title"]) <= 80
