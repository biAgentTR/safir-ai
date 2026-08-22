"""Modul 3 (src/memory/) icin GPU/aga bagimlilik gerektirmeyen birim testleri.

`EventStore` testleri stdlib `sqlite3` uzerinde calisir (hicbir agir
bagimlilik gerekmez). `EmbeddingRAGService`/`FAISSRagService` testleri, gercek
bir embedding modelini internetten indirmemek icin `SentenceTransformer`'i
deterministik, sahte bir uygulama ile degistirir (monkeypatch); boylece bu
testler de agsiz ve GPU'suz calisir.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from src.memory import embedding_rag_service as rag_module
from src.memory.embedding_rag_service import (
    DEFAULT_ISG_REGULATIONS,
    EmbeddingRAGService,
    FAISSRagService,
    _load_kb_chunk_documents,
)
from src.memory.event_store import EventStore, SQLiteEventStore
from src.utils.config_loader import EmbeddingConfig, FaissMemoryConfig, SQLiteMemoryConfig

# ---------------------------------------------------------------------------
# EventStore
# ---------------------------------------------------------------------------


@pytest.fixture
def event_store(tmp_path):
    store = SQLiteEventStore(SQLiteMemoryConfig(db_path=str(tmp_path / "events.db")))
    yield store
    store.close()


def test_event_store_alias_points_to_same_class() -> None:
    assert SQLiteEventStore is EventStore


def test_add_event_and_query_recent_orders_by_newest_first(event_store) -> None:
    now = time.time()
    event_store.add_event(timestamp=now, description="Olay A", risk_score=50, risk_level="orta")
    event_store.add_event(timestamp=now + 1, description="Olay B", risk_score=80, risk_level="kritik")

    recent = event_store.query_recent(limit=5)
    assert len(recent) == 2
    assert recent[0]["description"] == "Olay B"


def test_query_by_risk_level_filters_correctly(event_store) -> None:
    now = time.time()
    event_store.add_event(timestamp=now, description="dusuk olay", risk_level="dusuk")
    event_store.add_event(timestamp=now + 1, description="kritik olay", risk_level="kritik")

    kritik_events = event_store.query_by_risk_level("kritik")
    assert len(kritik_events) == 1
    assert kritik_events[0]["description"] == "kritik olay"


def test_get_timeline_orders_chronologically(event_store) -> None:
    now = time.time()
    event_store.add_event(timestamp=now, description="ilk")
    event_store.add_event(timestamp=now + 5, description="ikinci")

    timeline = event_store.get_timeline(start_ts=now, end_ts=now + 5)
    assert [entry["description"] for entry in timeline] == ["ilk", "ikinci"]


def test_record_feedback_updates_row(event_store) -> None:
    event_id = event_store.add_event(timestamp=time.time(), description="test", risk_score=90, risk_level="kritik")
    event_store.record_feedback(event_id, "false_positive")

    row = event_store.query_recent(limit=1)[0]
    assert row["feedback"] == "false_positive"


def test_record_feedback_rejects_invalid_value(event_store) -> None:
    event_id = event_store.add_event(timestamp=time.time(), description="test")
    with pytest.raises(ValueError):
        event_store.record_feedback(event_id, "bogus")


def test_record_feedback_raises_for_unknown_event(event_store) -> None:
    with pytest.raises(ValueError):
        event_store.record_feedback(99999, "true_positive")


# ---------------------------------------------------------------------------
# EventStore - video_source izolasyonu (Hata #2 duzeltmesi: get_timeline
# cross-video contamination). Bkz. VLM Pipeline Kok Neden Analiz Raporu.
# ---------------------------------------------------------------------------


def test_migration_adds_video_source_column_to_legacy_database(tmp_path) -> None:
    """`video_source` kolonu olmayan ESKI bir veritabani, acilista bozulmadan migrate edilmeli."""
    db_path = tmp_path / "legacy_events.db"

    # ESKI semayi (video_source YOK) elle olusturarak "gecmisten kalan" bir DB simule et.
    import sqlite3

    legacy_conn = sqlite3.connect(db_path)
    legacy_conn.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            description TEXT NOT NULL,
            risk_score INTEGER,
            risk_level TEXT,
            source_model TEXT
        );
        """
    )
    legacy_conn.execute(
        "INSERT INTO events (timestamp, description) VALUES (?, ?)", (5.0, "eski kayit")
    )
    legacy_conn.commit()
    legacy_conn.close()

    # SAFIR'in EventStore'u ayni DB'yi acinca migration idempotent calismali; eski satir kaybolmamali.
    store = SQLiteEventStore(SQLiteMemoryConfig(db_path=str(db_path)))
    try:
        rows = store.query_recent(limit=10)
        assert len(rows) == 1
        assert rows[0]["description"] == "eski kayit"
        assert rows[0]["video_source"] is None
    finally:
        store.close()


def test_get_timeline_without_video_source_filter_returns_all_rows(event_store) -> None:
    """`video_source` verilmezse eski davranis (tum kaynaklar) korunur."""
    now = time.time()
    event_store.add_event(timestamp=now, description="A videosu", video_source="/data/a.mp4")
    event_store.add_event(timestamp=now + 1, description="B videosu", video_source="/data/b.mp4")

    timeline = event_store.get_timeline(start_ts=now, end_ts=now + 1)
    assert len(timeline) == 2


def test_get_timeline_with_video_source_filter_excludes_other_videos(event_store) -> None:
    """Kok neden Hata #2'nin dogrudan regresyon testi: ayni zaman araligina dusen
    IKI FARKLI videonun eventleri, video_source filtresiyle birbirine karismamali."""
    now = time.time()
    event_store.add_event(timestamp=now, description="A videosunda olay", video_source="/data/video_a.mp4")
    event_store.add_event(timestamp=now, description="B videosunda olay", video_source="/data/video_b.mp4")

    timeline_a = event_store.get_timeline(start_ts=now, end_ts=now, video_source="/data/video_a.mp4")
    timeline_b = event_store.get_timeline(start_ts=now, end_ts=now, video_source="/data/video_b.mp4")

    assert len(timeline_a) == 1
    assert timeline_a[0]["description"] == "A videosunda olay"
    assert len(timeline_b) == 1
    assert timeline_b[0]["description"] == "B videosunda olay"


def test_get_timeline_with_video_source_filter_excludes_legacy_null_rows(event_store) -> None:
    """`video_source`'u bilinmeyen (NULL) eski kayitlar, yeni bir video'nun
    scoped timeline'ina KESINLIKLE dahil edilmemeli."""
    now = time.time()
    event_store.add_event(timestamp=now, description="video_source'suz eski kayit")
    event_store.add_event(timestamp=now, description="yeni video kaydi", video_source="/data/new.mp4")

    timeline = event_store.get_timeline(start_ts=now, end_ts=now, video_source="/data/new.mp4")

    assert len(timeline) == 1
    assert timeline[0]["description"] == "yeni video kaydi"


def test_add_event_persists_video_source(event_store) -> None:
    event_id = event_store.add_event(timestamp=time.time(), description="test", video_source="/data/x.mp4")
    row = next(r for r in event_store.query_recent(limit=10) if r["id"] == event_id)
    assert row["video_source"] == "/data/x.mp4"


# ---------------------------------------------------------------------------
# EmbeddingRAGService / FAISSRagService (agsiz, sahte SentenceTransformer ile)
# ---------------------------------------------------------------------------


class _FakeSentenceTransformer:
    """Gercek indirme yapmayan, deterministik sahte embedding modeli."""

    _DIMENSION = 16

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return self._DIMENSION

    def encode(self, texts, normalize_embeddings: bool = True, convert_to_numpy: bool = True):
        vectors = []
        for text in texts:
            vector = np.zeros(self._DIMENSION, dtype="float32")
            for token in text.lower().split():
                # DETERMINISTIK hash: Python'un yerlesik hash()'i sureç basina
                # rastgelelestirildiginden (PYTHONHASHSEED) test sonucu run'dan
                # run'a degisirdi; stabil bir hash ile bu flakiness giderilir.
                token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vector[token_hash % self._DIMENSION] += 1.0
            norm = np.linalg.norm(vector)
            if normalize_embeddings and norm > 0:
                vector = vector / norm
            vectors.append(vector)
        return np.array(vectors, dtype="float32")


@pytest.fixture(autouse=True)
def _patch_sentence_transformer(monkeypatch):
    """`SentenceTransformer`'i tum bu dosyadaki testler icin sahte, agsiz surumle degistirir."""
    monkeypatch.setattr(rag_module, "SentenceTransformer", _FakeSentenceTransformer)


@pytest.fixture
def rag_service(tmp_path) -> EmbeddingRAGService:
    embedding_config = EmbeddingConfig(provider="sentence-transformers", model_name="fake-model", device="cpu")
    faiss_config = FaissMemoryConfig(
        index_path=str(tmp_path / "index.faiss"), embedding_model="fake-model", top_k=3
    )
    return EmbeddingRAGService(embedding_config, faiss_config)


def test_rag_service_alias_points_to_same_class() -> None:
    assert FAISSRagService is EmbeddingRAGService


def test_rag_service_rejects_unsupported_provider(tmp_path) -> None:
    embedding_config = EmbeddingConfig(provider="unknown-provider", model_name="x", device="cpu")
    faiss_config = FaissMemoryConfig(index_path=str(tmp_path / "index.faiss"), embedding_model="x", top_k=3)

    with pytest.raises(ValueError):
        EmbeddingRAGService(embedding_config, faiss_config)


def test_query_on_empty_index_returns_empty_list(rag_service) -> None:
    assert rag_service.document_count() == 0
    assert rag_service.query("herhangi bir soru") == []


def test_seed_default_regulations_is_idempotent(rag_service) -> None:
    """`seed_default_regulations()` gercek KB chunk'larini (varsa) veya `DEFAULT_ISG_REGULATIONS`
    fallback'ini yukler (bkz. `_load_kb_chunk_documents`/modul dokustringi); hangisi olursa olsun
    tekrar cagirma ayni sayida dokumani KORUR (idempotent)."""
    kb_documents = _load_kb_chunk_documents()
    expected_count = len(kb_documents) if kb_documents else len(DEFAULT_ISG_REGULATIONS)

    rag_service.seed_default_regulations()
    count_after_first = rag_service.document_count()

    assert count_after_first == expected_count
    assert count_after_first > 0

    rag_service.seed_default_regulations()
    assert rag_service.document_count() == count_after_first


def test_search_laws_finds_relevant_document(rag_service) -> None:
    rag_service.add_documents(
        [
            "Yuksekte calisirken emniyet kemeri takilmalidir.",
            "Forklift kullaniminda egitim sarttir.",
        ]
    )

    results = rag_service.search_laws("emniyet kemeri", top_k=1)

    assert len(results) == 1
    assert "emniyet kemeri" in results[0].text
