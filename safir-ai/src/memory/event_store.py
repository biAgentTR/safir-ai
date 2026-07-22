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


@dataclass
class EventRecord:
    """`events` tablosundaki tek bir satirin tipli temsili."""

    id: int
    timestamp: float
    description: str
    risk_score: Optional[int]
    risk_level: Optional[str]
    source_model: Optional[str]


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
        self._connection.commit()

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

    def close(self) -> None:
        """Veritabani baglantisini kapatir."""
        self._connection.close()
