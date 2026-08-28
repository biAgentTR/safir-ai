"""04 - Yapilandirilmis Olay Bellegi: SQLite tabanli olay/timeline/istatistik deposu."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    risk_score INTEGER NOT NULL,
    risk_level TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    summary TEXT NOT NULL,
    auto INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    operator_note TEXT NOT NULL DEFAULT ''
);
"""
# `alerts` tablosu: audit'te P0 olarak isaretlenen bulgu - `FieldAlarmDispatcher`
# oncesinde YALNIZCA bellek-ici bir dict'te tutuluyordu; surec yeniden
# baslarsa (crash/restart) TUM tetiklenmis alarmlar ve operatorun onceki
# `acknowledge` kayitlari SESSIZCE kayboluyordu - guvenlik-kritik bir sistem
# icin kabul edilemez bir kalicilik bosluguydu. Bu tablo, `EventStore`in
# zaten sahip oldugu SQLite baglantisini yeniden kullanarak (yeni bir
# persistence mimarisi KURMADAN) bu bosluğu kapatir.

# T012 duzeltmesi (2026-08-23, audit izlenebilirlik bulgusu): `StructuredEvent`
# ile `EventStore` arasindaki alan kaybini kapatir. Onceden `StructuredEvent`in
# 14 alanindan yalnizca 5'i (`timestamp`/`description`/`risk_score`/
# `risk_level`/`source_model`) SQLite'a yaziliyordu; `evidence_ids` (bir olayin
# HANGI kanit karelerine dayandigi), `event_id`/`event_type`/`event_name`,
# `confidence`, `occurrence_count`, `duration` ve `keywords` kaydedilir
# kaydedilmez KAYBOLUYORDU - gecmis bir olay icin "bu karara hangi kareler
# sebep oldu" sorusu SONRADAN asla yeniden kurulamiyordu. Bu kolonlar bu
# kaybi kapatir; `evidence_ids`/`keywords` JSON-serilestirilmis TEXT olarak
# tutulur (SQLite'ta yerlesik bir dizi tipi yoktur).
_TRACEABILITY_COLUMNS: Dict[str, str] = {
    "temporal_event_id": "TEXT",
    "event_name": "TEXT",
    "event_type": "TEXT",
    "confidence": "REAL",
    "occurrence_count": "INTEGER",
    "duration": "REAL",
    "evidence_ids": "TEXT",
    "keywords": "TEXT",
}

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
    temporal_event_id: Optional[str] = None
    event_name: Optional[str] = None
    event_type: Optional[str] = None
    confidence: Optional[float] = None
    occurrence_count: Optional[int] = None
    duration: Optional[float] = None
    evidence_ids: List[str] = None  # type: ignore[assignment]
    keywords: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.evidence_ids is None:
            self.evidence_ids = []
        if self.keywords is None:
            self.keywords = []


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
        self._migrate_add_video_source_column()
        self._migrate_add_traceability_columns()
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

    def _migrate_add_video_source_column(self) -> None:
        """`events` tablosuna, daha eski veritabanlarinda eksik olabilecek `video_source` kolonunu ekler.

        Idempotenttir: kolon zaten varsa hicbir sey yapmaz. Bu kolon, `get_timeline`
        sorgularinin FARKLI video/analiz calismalarina ait olaylari birbirine
        karistirmasini (zaman damgasi araliklarinin ortusmesi durumunda olusan
        cross-video contamination) onlemek icin eklenmistir. Eski (bu migration
        oncesi yazilmis) satirlarda deger NULL kalir ve `get_timeline`e bir
        `video_source` filtresi verildiginde bu eski satirlar KASITLI olarak
        sonuca dahil edilmez (NULL = 'x' SQL'de hicbir zaman dogru degildir).
        """
        columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(events)").fetchall()}
        if "video_source" not in columns:
            self._connection.execute("ALTER TABLE events ADD COLUMN video_source TEXT")
            logger.info("EventStore semasi guncellendi: 'video_source' kolonu eklendi.")

    def _migrate_add_traceability_columns(self) -> None:
        """`events` tablosuna izlenebilirlik icin eksik olabilecek kolonlari ekler (bkz. `_TRACEABILITY_COLUMNS`).

        Idempotenttir. Bu kolonlar eklenmeden once yazilmis eski satirlarda
        deger NULL kalir (geriye donuk uyumlu, veri UYDURULMAZ).
        """
        columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(events)").fetchall()}
        for column, sql_type in _TRACEABILITY_COLUMNS.items():
            if column not in columns:
                self._connection.execute(f"ALTER TABLE events ADD COLUMN {column} {sql_type}")
                logger.info("EventStore semasi guncellendi: '%s' kolonu eklendi.", column)

    def add_event(
        self,
        timestamp: float,
        description: str,
        risk_score: Optional[int] = None,
        risk_level: Optional[str] = None,
        source_model: Optional[str] = None,
        video_source: Optional[str] = None,
        temporal_event_id: Optional[str] = None,
        event_name: Optional[str] = None,
        event_type: Optional[str] = None,
        confidence: Optional[float] = None,
        occurrence_count: Optional[int] = None,
        duration: Optional[float] = None,
        evidence_ids: Optional[Sequence[str]] = None,
        keywords: Optional[Sequence[str]] = None,
    ) -> int:
        """Yeni bir olay kaydi ekler.

        Args:
            timestamp: Olayin gerceklestigi saniye cinsinden zaman damgasi.
            description: Olayin dogal dil aciklamasi (VLM/Agent ciktisi).
            risk_score: Hesaplanmis 0-100 risk skoru (varsa).
            risk_level: Risk seviyesi etiketi (dusuk/orta/yuksek/kritik).
            source_model: Aciklamayi ureten model adi.
            video_source: Bu olayin uretildigi video/analiz kaynagi (normalize
                edilmis dosya yolu veya canli yayin URI'si). `get_timeline`
                tarafindan analiz-bazli izolasyon icin kullanilir; `None`
                birakilirsa satir kaynaksiz (eski davranisla uyumlu) yazilir.
            temporal_event_id: `TemporalEvent.event_id` (bkz. `StructuredEvent.
                temporal_event_id`); `None` birakilirsa izlenebilirlik kolonu
                bos kalir (eski davranisla uyumlu).
            event_name: Olayin birincil, serbest-bicimli kimligi.
            event_type: Opsiyonel canonical baglanti (`EventType` degeri).
            confidence: `TemporalEvent.confidence` (0.0-1.0).
            occurrence_count: Birlesen tespit sayisi.
            duration: Olayin saniye cinsinden suresi.
            evidence_ids: Bu olaya ait kanit karesi kimlikleri (JSON olarak saklanir).
            keywords: VLM-uretimi serbest-bicimli kanit ifadeleri (JSON olarak saklanir).

        Returns:
            Eklenen kaydin veritabani ID'si.
        """
        cursor = self._connection.execute(
            """
            INSERT INTO events (
                timestamp, description, risk_score, risk_level, source_model, video_source,
                temporal_event_id, event_name, event_type, confidence, occurrence_count, duration,
                evidence_ids, keywords
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                description,
                risk_score,
                risk_level,
                source_model,
                video_source,
                temporal_event_id,
                event_name,
                event_type,
                confidence,
                occurrence_count,
                duration,
                json.dumps(list(evidence_ids)) if evidence_ids is not None else None,
                json.dumps(list(keywords)) if keywords is not None else None,
            ),
        )
        self._connection.commit()
        logger.debug("Olay kaydedildi: id=%d ts=%.2f", cursor.lastrowid, timestamp)
        return int(cursor.lastrowid)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Bir satiri sozluge cevirir; `evidence_ids`/`keywords` JSON-TEXT kolonlarini listeye geri coder.

        Eski (izlenebilirlik migration'indan once yazilmis) satirlarda bu
        kolonlar NULL'dur - bu durumda bos liste dondurulur (UYDURULMAZ).
        """
        record = dict(row)
        for column in ("evidence_ids", "keywords"):
            raw_value = record.get(column)
            record[column] = json.loads(raw_value) if raw_value else []
        return record

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
        return [self._row_to_dict(row) for row in rows]

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
        return [self._row_to_dict(row) for row in rows]

    def get_timeline(
        self, start_ts: float, end_ts: float, video_source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Belirtilen zaman araligindaki olaylari kronolojik sirada dondurur.

        Args:
            start_ts: Aralik baslangici (saniye).
            end_ts: Aralik bitisi (saniye).
            video_source: Verilirse, yalnizca TAM OLARAK bu video/analiz
                kaynagina ait satirlar dondurulur; farkli videolarin zaman
                damgasi araliklari ortusse bile birbirine karismaz. Bu
                kolon eklenmeden ONCE yazilmis eski satirlar `video_source`
                degeri NULL oldugu icin (SQL'de `NULL = ?` hicbir zaman
                dogru olmadigindan) KASITLI olarak sonuca dahil edilmez.
                `None` (varsayilan) birakilirsa filtre uygulanmaz ve eski
                davranis (tum kaynaklar) korunur.

        Returns:
            Zaman damgasina gore artan sirali olay sozlukleri.
        """
        if video_source is not None:
            rows = self._connection.execute(
                """
                SELECT * FROM events
                WHERE timestamp BETWEEN ? AND ? AND video_source = ?
                ORDER BY timestamp ASC
                """,
                (start_ts, end_ts, video_source),
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT * FROM events
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
                """,
                (start_ts, end_ts),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

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

    def record_alert(
        self,
        alert_id: str,
        risk_score: int,
        risk_level: str,
        recommended_action: str,
        summary: str,
        auto: bool,
        created_at: str,
    ) -> None:
        """Tetiklenen bir saha alarmini kalici olarak kaydeder (bkz. `_SCHEMA::alerts`).

        `src.decision.escalation.FieldAlarmDispatcher` tarafindan cagirilir;
        bellek-ici kaydin YANI SIRA (onun yerine DEGIL) kalicilik saglar -
        surec yeniden baslasa bile alarm kaybolmaz.

        Args:
            alert_id: Alarmin benzersiz kimligi (`uuid4`).
            risk_score: 0-100 risk skoru.
            risk_level: `dusuk|orta|yuksek|kritik`.
            recommended_action: Alarma iliskin birincil aksiyon onerisi.
            summary: Kisa durum ozeti.
            auto: Otomatik mi (True) yoksa operator-tetikli mi (False).
            created_at: ISO-8601 zaman damgasi.
        """
        self._connection.execute(
            """
            INSERT INTO alerts (alert_id, risk_score, risk_level, recommended_action, summary, auto, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (alert_id, risk_score, risk_level, recommended_action, summary, int(auto), created_at),
        )
        self._connection.commit()
        logger.debug("Alarm kalici olarak kaydedildi: alert_id=%s", alert_id)

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Kalici olarak kaydedilmis bir alarmi getirir; yoksa `None`.

        Args:
            alert_id: `record_alert` ile kaydedilmis alarm kimligi.

        Returns:
            Alarmi temsil eden sozluk (`acknowledged` bool'a cevrilmis) veya `None`.
        """
        row = self._connection.execute(
            "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
        ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["auto"] = bool(record["auto"])
        record["acknowledged"] = bool(record["acknowledged"])
        return record

    def acknowledge_alert(self, alert_id: str, operator_note: str = "") -> bool:
        """Kalici alarm kaydini operator-onaylandi olarak isaretler.

        Args:
            alert_id: `record_alert` ile kaydedilmis alarm kimligi.
            operator_note: Operatorun opsiyonel notu.

        Returns:
            Kayit bulunup guncellendiyse `True`; boyle bir kalici kayit
            yoksa (orn. yalnizca bellek-ici olarak tetiklenmis eski bir
            alarmsa) `False` - cagiran bu durumda bellek-ici kaydini
            koruyabilir.
        """
        cursor = self._connection.execute(
            "UPDATE alerts SET acknowledged = 1, operator_note = ? WHERE alert_id = ?",
            (operator_note, alert_id),
        )
        self._connection.commit()
        return cursor.rowcount > 0

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
