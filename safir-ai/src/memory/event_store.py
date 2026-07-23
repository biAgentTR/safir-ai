"""04 - Yapilandirilmis Olay Bellegi: SQLite tabanli olay/timeline/istatistik deposu."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config_loader import SQLiteMemoryConfig

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    description TEXT NOT NULL,
    risk_score INTEGER,
    risk_level TEXT,
    source_model TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp);
"""

_VALID_FEEDBACK_VALUES = ("true_positive", "false_positive")


@dataclass
class EventRecord:
    """`events` tablosundaki tek bir satirin tipli temsili."""

    id: int
    timestamp: float
    description: str
    risk_score: Optional[int]
    risk_level: Optional[str]
    source_model: Optional[str]
    feedback: Optional[str] = None


class EventStore:
    """Olaylari, zaman cizelgesini ve istatistikleri iliskisel olarak saklayan hafif veritabani.

    `Dynamic Tool Router` icindeki SQL Tool tarafindan gecmis olay sorgulamalari
    icin kullanilir.
    """

    def __init__(self, config: SQLiteMemoryConfig) -> None:
        """EventStore'u verilen SQLite konfigurasyonu ile baslatir ve semayi olusturur.

        Args:
            config: `configs/config.yaml` icindeki `memory.sqlite` blogu.
        """
        db_path = Path(config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._migrate_add_feedback_column()
        self._connection.commit()

    def _migrate_add_feedback_column(self) -> None:
        """`events` tablosuna, daha eski veritabanlarinda eksik olabilecek `feedback` kolonunu ekler.

        Idempotenttir: kolon zaten varsa hicbir sey yapmaz. Bu, Human-in-the-Loop
        operator dogrulamasini (`true_positive`/`false_positive`) saklamak icin
        sonradan eklenen bir alandir.
        """
        columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(events)").fetchall()}
        if "feedback" not in columns:
            self._connection.execute("ALTER TABLE events ADD COLUMN feedback TEXT")
            logger.info("EventStore semasi guncellendi: 'feedback' kolonu eklendi.")

    def add_event(
        self,
        timestamp: float,
        description: str,
        risk_score: Optional[int] = None,
        risk_level: Optional[str] = None,
        source_model: Optional[str] = None,
    ) -> int:
        """Yeni bir olay kaydi ekler.

        Args:
            timestamp: Olayin gerceklestigi saniye cinsinden zaman damgasi.
            description: Olayin dogal dil aciklamasi (VLM/Agent ciktisi).
            risk_score: Hesaplanmis 0-100 risk skoru (varsa).
            risk_level: Risk seviyesi etiketi (dusuk/orta/yuksek/kritik).
            source_model: Aciklamayi ureten model adi.

        Returns:
            Eklenen kaydin veritabani ID'si.
        """
        cursor = self._connection.execute(
            """
            INSERT INTO events (timestamp, description, risk_score, risk_level, source_model)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, description, risk_score, risk_level, source_model),
        )
        self._connection.commit()
        logger.debug("Olay kaydedildi: id=%d ts=%.2f", cursor.lastrowid, timestamp)
        return int(cursor.lastrowid)

    def query_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """En son kaydedilen olaylari zaman damgasina gore azalan sirada dondurur.

        Args:
            limit: Dondurulecek maksimum kayit sayisi.

        Returns:
            Her biri bir olayi temsil eden sozlukler listesi.
        """
        rows = self._connection.execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def query_by_risk_level(self, risk_level: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Belirli bir risk seviyesindeki olaylari getirir.

        Args:
            risk_level: Filtrelenecek risk seviyesi (orn. "kritik").
            limit: Dondurulecek maksimum kayit sayisi.

        Returns:
            Esesen risk seviyesindeki olaylarin sozluk listesi.
        """
        rows = self._connection.execute(
            "SELECT * FROM events WHERE risk_level = ? ORDER BY timestamp DESC LIMIT ?",
            (risk_level, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_timeline(self, start_ts: float, end_ts: float) -> List[Dict[str, Any]]:
        """Belirtilen zaman araligindaki olaylari kronolojik sirada dondurur.

        Args:
            start_ts: Aralik baslangici (saniye).
            end_ts: Aralik bitisi (saniye).

        Returns:
            Zaman damgasina gore artan sirali olay sozlukleri.
        """
        rows = self._connection.execute(
            """
            SELECT * FROM events
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
            """,
            (start_ts, end_ts),
        ).fetchall()
        return [dict(row) for row in rows]

    def record_feedback(self, event_id: int, feedback: str) -> None:
        """Operatorun Human-in-the-Loop dogrulamasini bir olay kaydina isler.

        Bu, bir aktif-ogrenme/RLHF dongusunu otomatik tetiklemez; yalnizca
        `true_positive`/`false_positive` etiketini kalici olarak saklar,
        boylece gelecekteki analiz/ince-ayar calismalari bu etiketli veriyi
        kullanabilir.

        Args:
            event_id: `add_event` tarafindan donen veritabani kaydi kimligi.
            feedback: `"true_positive"` veya `"false_positive"`.

        Raises:
            ValueError: `feedback` gecersiz bir deger olursa veya `event_id`
                bulunamazsa.
        """
        if feedback not in _VALID_FEEDBACK_VALUES:
            raise ValueError(
                f"Gecersiz feedback degeri: '{feedback}'. Kabul edilenler: {_VALID_FEEDBACK_VALUES}"
            )

        cursor = self._connection.execute(
            "UPDATE events SET feedback = ? WHERE id = ?", (feedback, event_id)
        )
        self._connection.commit()

        if cursor.rowcount == 0:
            raise ValueError(f"Olay bulunamadi: id={event_id}")

        logger.info("Operator geri bildirimi kaydedildi: id=%d feedback=%s", event_id, feedback)

    def close(self) -> None:
        """Veritabani baglantisini kapatir."""
        self._connection.close()


# Modul 3 spesifikasyonundaki isim: ayni sinifa isaret eden alias (geriye
# donuk uyumluluk icin `EventStore` adi da tum cagiran kodda aynen kullanilmaya
# devam eder).
SQLiteEventStore = EventStore


if __name__ == "__main__":
    # Modul 3'un bagimsiz calistirilabilirlik testi:
    #   python -m src.memory.event_store
    import tempfile
    import time

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmp_dir:
        demo_store = SQLiteEventStore(SQLiteMemoryConfig(db_path=f"{tmp_dir}/demo_events.db"))

        now = time.time()
        first_id = demo_store.add_event(timestamp=now, description="Personel korumasiz alanda tespit edildi.", risk_score=60, risk_level="yuksek")
        demo_store.add_event(timestamp=now + 5, description="Forklift yaya gecidine yaklasti.", risk_score=75, risk_level="yuksek")
        demo_store.add_event(timestamp=now + 10, description="Sahada normal calisma gozlemlendi.", risk_score=10, risk_level="dusuk")

        demo_store.record_feedback(first_id, "false_positive")
        print(f"Geri bildirim kaydedildi: id={first_id} -> false_positive")

        print("\nEn son 5 olay:")
        for event in demo_store.query_recent(limit=5):
            print(f"  [{event['timestamp']:.1f}] {event['description']} (risk={event['risk_level']}, feedback={event['feedback']})")

        print("\nZaman cizelgesi (get_timeline):")
        for event in demo_store.get_timeline(start_ts=now, end_ts=now + 10):
            print(f"  [{event['timestamp']:.1f}] {event['description']}")

        demo_store.close()
