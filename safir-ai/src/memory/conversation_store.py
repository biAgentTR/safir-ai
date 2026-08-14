"""10 - SAFIR Asistan sohbet gecmisi (Conversation) kalici deposu.

`AnalysisStore`den (`src/memory/analysis_store.py`) BAGIMSIZ, AYRI bir SQLite
dosyasi/semasi kullanir; ayni hafif desene (dataclass kayitlar, dosya-tabanli
SQLite, yazim kilidi) uyar. Yalnizca sohbet GECMISINI (konusma basligi, hangi
analize bagli oldugu, mesaj listesi) saklar — `POST /ask`in kendi muhakeme
davranisini DEGISTIRMEZ ve ona hicbir gecmis mesaji geri BESLEMEZ (bkz.
`src/assistant/ask_service.py`in prompt yapisi; bu modul ona dokunmaz).
"""

from __future__ import annotations

import datetime
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    title           TEXT,
    job_id          TEXT
);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations (updated_at);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_conv ON conversation_messages (conversation_id, id);
"""


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


@dataclass
class ConversationRecord:
    """`conversations` tablosundaki tek bir satirin tipli temsili."""

    conversation_id: str
    created_at: str
    updated_at: str
    title: Optional[str]
    job_id: Optional[str]


@dataclass
class ConversationMessageRecord:
    """`conversation_messages` tablosundaki tek bir satirin tipli temsili."""

    id: int
    conversation_id: str
    role: str  # "user" | "assistant"
    content: str
    created_at: str


@dataclass
class ConversationDetail:
    """Tek bir sohbetin, mesajlariyla birlikte tam gorunumu."""

    conversation: ConversationRecord
    messages: List[ConversationMessageRecord] = field(default_factory=list)


class ConversationStore:
    """SAFIR Asistan sohbetlerini ve mesajlarini saklayan hafif kalici depo."""

    def __init__(self, db_path: str | Path) -> None:
        """Depoyu verilen SQLite dosya yolu ile baslatir ve (yoksa) semayi olusturur.

        Args:
            db_path: `conversations`/`conversation_messages` tablolarinin tutulacagi SQLite dosyasinin yolu.
        """
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()
        self._lock = threading.Lock()

    def create(self, title: Optional[str] = None, job_id: Optional[str] = None) -> ConversationRecord:
        """Yeni bir sohbet kaydi olusturur ve dondurur.

        Args:
            title: Sohbetin kisa basligi (genelde ilk kullanici sorusundan kirpilir); opsiyonel.
            job_id: Bu sohbetin bagli oldugu analiz kimligi (varsa); opsiyonel.

        Returns:
            Olusturulan sohbetin kaydi.
        """
        conversation_id = str(uuid.uuid4())
        now = _now_iso()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO conversations (conversation_id, created_at, updated_at, title, job_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, now, now, title, job_id),
            )
            self._connection.commit()
        return ConversationRecord(conversation_id, now, now, title, job_id)

    def list(self, limit: int = 50, offset: int = 0) -> List[ConversationRecord]:
        """Sohbetleri en son GUNCELLENEN once (updated_at DESC) sayfali dondurur."""
        rows = self._connection.execute(
            """
            SELECT * FROM conversations
             ORDER BY updated_at DESC, rowid DESC
             LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [self._to_record(r) for r in rows]

    def get(self, conversation_id: str) -> Optional[ConversationRecord]:
        """Tek bir sohbet kaydini dondurur; yoksa None."""
        row = self._connection.execute(
            "SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def add_message(self, conversation_id: str, role: str, content: str) -> ConversationMessageRecord:
        """Sohbete yeni bir mesaj ekler ve sohbetin `updated_at`ini gunceller.

        Args:
            conversation_id: Hedef sohbetin kimligi (cagiran taraf varligini onceden dogrulamali).
            role: `"user"` veya `"assistant"`.
            content: Mesaj metni.

        Returns:
            Eklenen mesajin kaydi.
        """
        now = _now_iso()
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO conversation_messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, role, content, now),
            )
            self._connection.execute(
                "UPDATE conversations SET updated_at=? WHERE conversation_id=?", (now, conversation_id)
            )
            self._connection.commit()
        return ConversationMessageRecord(
            id=int(cursor.lastrowid), conversation_id=conversation_id, role=role, content=content, created_at=now
        )

    def list_messages(self, conversation_id: str) -> List[ConversationMessageRecord]:
        """Bir sohbetin tum mesajlarini kronolojik sirada (id ASC) dondurur."""
        rows = self._connection.execute(
            "SELECT * FROM conversation_messages WHERE conversation_id=? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [self._to_message(r) for r in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ConversationRecord:
        return ConversationRecord(
            conversation_id=row["conversation_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            title=row["title"],
            job_id=row["job_id"],
        )

    @staticmethod
    def _to_message(row: sqlite3.Row) -> ConversationMessageRecord:
        return ConversationMessageRecord(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
        )

    def close(self) -> None:
        """Veritabani baglantisini kapatir."""
        self._connection.close()


if __name__ == "__main__":
    # Bagimsiz calistirilabilirlik testi:  python -m src.memory.conversation_store
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(f"{tmp}/conversations.db")
        conv = store.create(title="Forklift neden riskli?", job_id="demo-job-1")
        store.add_message(conv.conversation_id, "user", "Forklift neden riskli?")
        store.add_message(conv.conversation_id, "assistant", "[MOCK] Yaya gecidine yakinligi risk olusturuyor.")
        print("conversations:", store.list())
        print("messages:", store.list_messages(conv.conversation_id))
        store.close()
