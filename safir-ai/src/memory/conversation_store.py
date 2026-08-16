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

-- Operatorun bir sohbete manuel ekledigi kisa metin/not (Adim 3). Yuklenen
-- BELGELER (Adim 4) buraya DEGIL, asagidaki conversation_documents/
-- conversation_document_chunks tablolarina gider (chunk-bazli retrieval
-- gerektirdigi icin ayri bir yapi). Bu tablo `conversations` ile ayni dosya/
-- sema icindedir (yeni bir depo/DB SOYUTLAMASI DEGIL) ve DAIMA conversation_id
-- ile filtrelenir -> bir sohbetin baglami baska bir sohbete ASLA sizmaz.
CREATE TABLE IF NOT EXISTS conversation_context (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'note',
    label           TEXT,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_context_conv ON conversation_context (conversation_id, id);

-- Adim 4: kullanicinin bir sohbete yukledigi belgeler (PDF/TXT/DOCX) - yalnizca
-- METADATA + durum burada tutulur; belgenin kendisi (ham baytlari) diskte
-- `data/conversation_files/{conversation_id}/{document_id}.{ext}` altinda
-- saklanir (bkz. src/main.py). Ikili veri SQLite'a HIC yazilmaz.
CREATE TABLE IF NOT EXISTS conversation_documents (
    document_id     TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    filename        TEXT NOT NULL,
    file_ext        TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    page_count      INTEGER,
    status          TEXT NOT NULL DEFAULT 'processing',
    error_message   TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_documents_conv ON conversation_documents (conversation_id, created_at);

-- Adim 4: her belgenin cikarilan metninden uretilen chunk'lar (duz metin
-- YALNIZCA - `src/memory/document_extraction.py::chunk_page_texts`). `/ask/stream`,
-- bir soru geldiginde TUM belgeyi degil, yalnizca bu chunk'lardan lexical
-- olarak en ilgili birkacini prompt'a ekler (bkz. document_extraction.lexical_search).
-- `conversation_id` burada DA tutulur (document_id uzerinden JOIN yerine
-- dogrudan filtre) -> retrieval sorgusu tek tabloda, izolasyon garantili.
CREATE TABLE IF NOT EXISTS conversation_document_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    page_number     INTEGER,
    content         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_document_chunks_doc ON conversation_document_chunks (document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_conversation_document_chunks_conv ON conversation_document_chunks (conversation_id);
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
    message_count: int = 0
    """Yalnizca `list_with_message_counts` tarafindan doldurulur (Data Center
    icin); `create`/`get`/`list` cagrilarinda varsayilan 0'da kalir (ek sorgu
    YAPILMAZ, mevcut davranis DEGISMEZ)."""


@dataclass
class ConversationMessageRecord:
    """`conversation_messages` tablosundaki tek bir satirin tipli temsili."""

    id: int
    conversation_id: str
    role: str  # "user" | "assistant"
    content: str
    created_at: str


@dataclass
class ConversationContextRecord:
    """`conversation_context` tablosundaki tek bir satirin tipli temsili.

    `kind`: su an yalnizca `"note"` (Adim 3). Adim 4'te `"file"` eklenecek;
    `content` o zaman yuklenen belgeden cikarilan METIN olacak (dosyanin
    kendisi/binary DEGIL) - ayni tablo/sema yeniden kullanilacak.
    """

    id: int
    conversation_id: str
    kind: str
    label: Optional[str]
    content: str
    created_at: str


@dataclass
class ConversationDocumentRecord:
    """`conversation_documents` tablosundaki tek bir satirin tipli temsili (Adim 4)."""

    document_id: str
    conversation_id: str
    filename: str
    file_ext: str
    file_size_bytes: int
    page_count: Optional[int]
    status: str  # "processing" | "ready" | "error"
    error_message: Optional[str]
    created_at: str


@dataclass
class ConversationDocumentChunkRecord:
    """`conversation_document_chunks` tablosundaki tek bir satirin tipli temsili (Adim 4).

    `content: str` alani, `document_extraction.lexical_search`in bekledigi
    sozlesmeyi (`.content` erisimi olan nesne) saglar. `filename`, retrieval
    sonrasi kaynak gosterimi icin `conversation_documents` ile JOIN edilerek
    doldurulur (bkz. `list_chunks_for_conversation`).
    """

    id: int
    document_id: str
    conversation_id: str
    chunk_index: int
    page_number: Optional[int]
    content: str
    filename: str


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

    def list_with_message_counts(self, limit: int = 50, offset: int = 0) -> List[ConversationRecord]:
        """Sohbetleri, her birinin mesaj sayisiyla birlikte (tek sorguda, N+1 YOK) dondurur.

        Data Center ("Sistem Verileri") sohbet listesi icin; `list()`in
        davranisini DEGISTIRMEZ, ayri bir metottur.
        """
        rows = self._connection.execute(
            """
            SELECT c.conversation_id, c.created_at, c.updated_at, c.title, c.job_id,
                   COUNT(m.id) AS message_count
              FROM conversations c
              LEFT JOIN conversation_messages m ON m.conversation_id = c.conversation_id
             GROUP BY c.conversation_id
             ORDER BY c.updated_at DESC, c.rowid DESC
             LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [
            ConversationRecord(
                conversation_id=row["conversation_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                title=row["title"],
                job_id=row["job_id"],
                message_count=row["message_count"],
            )
            for row in rows
        ]

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

    def add_context(
        self, conversation_id: str, content: str, label: Optional[str] = None, kind: str = "note"
    ) -> ConversationContextRecord:
        """Bir sohbete kullanici tarafindan eklenen bir baglam kaydi (not) ekler.

        Args:
            conversation_id: Hedef sohbetin kimligi (cagiran taraf varligini onceden dogrulamali).
            content: Baglamin metni (Adim 3: kullanicinin yazdigi not).
            label: Kisa goruntuleme etiketi (opsiyonel, ör. "Tesis bilgisi").
            kind: `"note"` (Adim 3) | `"file"` (Adim 4, henuz kullanilmiyor).

        Returns:
            Eklenen baglamin kaydi.
        """
        now = _now_iso()
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO conversation_context (conversation_id, kind, label, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, kind, label, content, now),
            )
            self._connection.commit()
        return ConversationContextRecord(
            id=int(cursor.lastrowid),
            conversation_id=conversation_id,
            kind=kind,
            label=label,
            content=content,
            created_at=now,
        )

    def list_context(self, conversation_id: str) -> List[ConversationContextRecord]:
        """Bir sohbetin tum ek baglam kayitlarini kronolojik sirada (id ASC) dondurur."""
        rows = self._connection.execute(
            "SELECT * FROM conversation_context WHERE conversation_id=? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [self._to_context(r) for r in rows]

    def remove_context(self, conversation_id: str, context_id: int) -> bool:
        """Bir baglam kaydini siler; yalnizca VERILEN sohbete aitse (baska sohbete sizmayi engeller).

        Returns:
            Gercekten bir satir silindiyse `True`; kayit yoksa veya baska bir
            sohbete aitse `False`.
        """
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM conversation_context WHERE id=? AND conversation_id=?",
                (context_id, conversation_id),
            )
            self._connection.commit()
        return cursor.rowcount > 0

    # --------------------- Adim 4: yuklenen belgeler + chunk'lar ---------------------

    def create_document(
        self, conversation_id: str, filename: str, file_ext: str, file_size_bytes: int
    ) -> ConversationDocumentRecord:
        """Yeni bir belge kaydi olusturur (durum `processing` ile baslar; ham bayt SAKLANMAZ).

        Args:
            conversation_id: Hedef sohbetin kimligi (cagiran taraf varligini onceden dogrulamali).
            filename: Orijinal dosya adi (yalnizca goruntuleme icin; dosya YOLU icin KULLANILMAZ).
            file_ext: Kucuk harfli uzanti (`pdf`/`txt`/`docx`).
            file_size_bytes: Ham dosyanin bayt cinsinden boyutu.

        Returns:
            Olusturulan belgenin kaydi (`status="processing"`).
        """
        document_id = str(uuid.uuid4())
        now = _now_iso()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO conversation_documents
                    (document_id, conversation_id, filename, file_ext, file_size_bytes, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'processing', ?)
                """,
                (document_id, conversation_id, filename, file_ext, file_size_bytes, now),
            )
            self._connection.commit()
        return ConversationDocumentRecord(
            document_id=document_id,
            conversation_id=conversation_id,
            filename=filename,
            file_ext=file_ext,
            file_size_bytes=file_size_bytes,
            page_count=None,
            status="processing",
            error_message=None,
            created_at=now,
        )

    def update_document_status(
        self,
        document_id: str,
        status: str,
        page_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Bir belgenin isleme durumunu gunceller (`processing` -> `ready`/`error`).

        Args:
            document_id: Guncellenecek belgenin kimligi.
            status: `"ready"` veya `"error"`.
            page_count: Bilinen sayfa sayisi (yalnizca PDF icin; aksi halde `None`).
            error_message: `status="error"` ise kullaniciya gosterilebilir kisa hata metni.
        """
        with self._lock:
            self._connection.execute(
                "UPDATE conversation_documents SET status=?, page_count=?, error_message=? WHERE document_id=?",
                (status, page_count, error_message, document_id),
            )
            self._connection.commit()

    def add_chunks(self, document_id: str, conversation_id: str, chunks) -> int:
        """Bir belgenin cikarilan chunk'larini kalici olarak saklar.

        Args:
            document_id: Bu chunk'larin ait oldugu belge.
            conversation_id: Denormalize edilmis sohbet kimligi (izolasyonlu tek-tablo sorgu icin).
            chunks: `document_extraction.ChunkInput` (veya ayni `.page_number`/`.content`
                sozlesmesine sahip) nesnelerin sirali listesi.

        Returns:
            Eklenen chunk sayisi.
        """
        rows = [(document_id, conversation_id, i, c.page_number, c.content) for i, c in enumerate(chunks)]
        if not rows:
            return 0
        with self._lock:
            self._connection.executemany(
                """
                INSERT INTO conversation_document_chunks
                    (document_id, conversation_id, chunk_index, page_number, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._connection.commit()
        return len(rows)

    def list_documents(self, conversation_id: str) -> List[ConversationDocumentRecord]:
        """Bir sohbetin tum belgelerini yukleme sirasinda (created_at ASC) dondurur."""
        rows = self._connection.execute(
            "SELECT * FROM conversation_documents WHERE conversation_id=? ORDER BY created_at ASC, rowid ASC",
            (conversation_id,),
        ).fetchall()
        return [self._to_document(r) for r in rows]

    def get_document(self, conversation_id: str, document_id: str) -> Optional[ConversationDocumentRecord]:
        """Tek bir belgeyi dondurur; yalnizca VERILEN sohbete aitse (izolasyon)."""
        row = self._connection.execute(
            "SELECT * FROM conversation_documents WHERE document_id=? AND conversation_id=?",
            (document_id, conversation_id),
        ).fetchone()
        return self._to_document(row) if row else None

    def count_documents(self, conversation_id: str) -> int:
        """Bir sohbetteki toplam belge sayisini dondurur (azami sinir kontrolu icin)."""
        row = self._connection.execute(
            "SELECT COUNT(*) AS n FROM conversation_documents WHERE conversation_id=?", (conversation_id,)
        ).fetchone()
        return int(row["n"])

    def remove_document(self, conversation_id: str, document_id: str) -> bool:
        """Bir belgeyi VE tum chunk'larini siler; yalnizca VERILEN sohbete aitse.

        Diskteki dosyanin silinmesi bu metodun SORUMLULUGUNDA DEGILDIR (bkz.
        `src/main.py` cagiran uc nokta - depo yalnizca DB, dosya sistemine
        dokunmaz; ayni ayrim `_persist_representative_frames_best_effort` ile
        History representative-frame'lerinde de kullanilir).

        Returns:
            Belge gercekten bulunup silindiyse `True`; yoksa/baska sohbete aitse `False`.
        """
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM conversation_documents WHERE document_id=? AND conversation_id=?",
                (document_id, conversation_id),
            )
            removed = cursor.rowcount > 0
            if removed:
                self._connection.execute(
                    "DELETE FROM conversation_document_chunks WHERE document_id=? AND conversation_id=?",
                    (document_id, conversation_id),
                )
            self._connection.commit()
        return removed

    def list_chunks_for_conversation(self, conversation_id: str) -> List[ConversationDocumentChunkRecord]:
        """Bir sohbetteki, YALNIZCA `status='ready'` belgelere ait tum chunk'lari dondurur.

        `/ask/stream` retrieval'i icindir (bkz. `document_extraction.lexical_search`).
        Hala islenmekte olan (`processing`) veya basarisiz (`error`) belgelerin
        chunk'lari (varsa) HIC dondurulmez - boylece yarim/bozuk cikarim
        prompt'a asla sizmaz.
        """
        rows = self._connection.execute(
            """
            SELECT ch.*, d.filename AS filename
              FROM conversation_document_chunks ch
              JOIN conversation_documents d ON d.document_id = ch.document_id
             WHERE ch.conversation_id = ? AND d.status = 'ready'
             ORDER BY ch.document_id, ch.chunk_index
            """,
            (conversation_id,),
        ).fetchall()
        return [
            ConversationDocumentChunkRecord(
                id=row["id"],
                document_id=row["document_id"],
                conversation_id=row["conversation_id"],
                chunk_index=row["chunk_index"],
                page_number=row["page_number"],
                content=row["content"],
                filename=row["filename"],
            )
            for row in rows
        ]

    @staticmethod
    def _to_document(row: sqlite3.Row) -> ConversationDocumentRecord:
        return ConversationDocumentRecord(
            document_id=row["document_id"],
            conversation_id=row["conversation_id"],
            filename=row["filename"],
            file_ext=row["file_ext"],
            file_size_bytes=row["file_size_bytes"],
            page_count=row["page_count"],
            status=row["status"],
            error_message=row["error_message"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_context(row: sqlite3.Row) -> ConversationContextRecord:
        return ConversationContextRecord(
            id=row["id"],
            conversation_id=row["conversation_id"],
            kind=row["kind"],
            label=row["label"],
            content=row["content"],
            created_at=row["created_at"],
        )

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
