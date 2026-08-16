"""09 - Analiz Gecmisi (History) kalici deposu.

Mevcut `EventStore`'dan (olay-satiri deposu) BAGIMSIZ, AYRI bir SQLite tablosu/
dosyasi kullanir; boylece analiz pipeline'i ve `EventStore` semasi HIC
degismeden, her `/analyze/jobs` isinin nihai `SafirReport`'u kalici olarak
saklanip History'de yeniden sunulabilir.

Kalicilik: dosya-tabanli SQLite oldugundan process/sunucu yeniden baslasa da
tamamlanmis analizler kaybolmaz. Yazimlar arka plan thread'lerinden (`_run_job`)
ve endpoint'lerden gelebildigi icin `check_same_thread=False` + bir yazim kilidi
kullanilir.
"""

from __future__ import annotations

import datetime
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# queued | running | completed | failed
_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    job_id       TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    status       TEXT NOT NULL,
    video_source TEXT,
    risk_level   TEXT,
    risk_score   INTEGER,
    summary      TEXT,
    report_json  TEXT,
    risk_status  TEXT DEFAULT 'assessed'
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at);
"""


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


@dataclass
class AnalysisRecord:
    """`analyses` tablosundaki tek bir satirin tipli temsili."""

    job_id: str
    created_at: str
    updated_at: str
    status: str
    video_source: Optional[str]
    risk_level: Optional[str]
    risk_score: Optional[int]
    summary: Optional[str]
    report_json: Optional[str]
    trace_json: Optional[str] = None
    risk_status: str = "assessed"


class AnalysisStore:
    """Analiz isi yasam dongusunu ve nihai `SafirReport`'u saklayan hafif kalici depo."""

    def __init__(self, db_path: str | Path) -> None:
        """Depoyu verilen SQLite dosya yolu ile baslatir ve (yoksa) semayi olusturur.

        Args:
            db_path: `analyses` tablosunun tutulacagi SQLite dosyasinin yolu.
        """
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._migrate_add_trace_json_column()
        self._migrate_add_risk_status_column()
        self._connection.commit()
        self._lock = threading.Lock()

    def _migrate_add_trace_json_column(self) -> None:
        """`analyses` tablosuna, daha eski veritabanlarinda eksik olabilecek `trace_json` kolonunu ekler.

        Idempotenttir: kolon zaten varsa hicbir sey yapmaz. `report_json` ile
        ayni amaca hizmet eder (kalici, JSON-serilestirilmis veri); yalnizca
        pipeline'in tam trace olay listesini (`job.trace_events`) tasir, boylece
        History'de gecmis bir analizin PipelineTimeline/StageCard gorunumu de
        (canli analizle ayni veri yapisiyla) yeniden kurulabilir.
        """
        columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(analyses)").fetchall()}
        if "trace_json" not in columns:
            self._connection.execute("ALTER TABLE analyses ADD COLUMN trace_json TEXT")
            logger.info("AnalysisStore semasi guncellendi: 'trace_json' kolonu eklendi.")

    def _migrate_add_risk_status_column(self) -> None:
        """`analyses` tablosuna eksik olabilecek `risk_status` kolonunu ekler (P0 fix).

        Idempotenttir. Eski kayitlar `DEFAULT 'assessed'` alir — bu, geriye
        donuk uyumlu ve guvenli bir varsayilandir: eski kayitlarin tumu zaten
        (eski davranista) sayisal bir risk_score ile tamamlanmisti.
        """
        columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(analyses)").fetchall()}
        if "risk_status" not in columns:
            self._connection.execute("ALTER TABLE analyses ADD COLUMN risk_status TEXT DEFAULT 'assessed'")
            logger.info("AnalysisStore semasi guncellendi: 'risk_status' kolonu eklendi.")

    # ---- yazim (lifecycle) ----

    def create(self, job_id: str, video_source: Optional[str], status: str = "queued") -> None:
        """Yeni bir analiz kaydini (created_at ile) ekler; ayni job_id varsa dokunmaz."""
        now = _now_iso()
        with self._lock:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO analyses
                    (job_id, created_at, updated_at, status, video_source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, now, now, status, video_source),
            )
            self._connection.commit()

    def mark_completed(
        self,
        job_id: str,
        *,
        report_json: str,
        risk_level: Optional[str],
        risk_score: Optional[int],
        summary: Optional[str],
        trace_json: Optional[str] = None,
        risk_status: str = "assessed",
    ) -> None:
        """Isi `completed` olarak isaretler ve nihai `SafirReport` JSON'unu (+ varsa trace_json) saklar.

        Args:
            trace_json: Pipeline'in tam trace olay listesinin (`job.trace_events`)
                JSON-serilestirilmis hali; History'de PipelineTimeline/StageCard
                gorunumunu yeniden kurmak icin kullanilir. Opsiyoneldir (`None`
                birakilirsa kolon dokunulmadan `NULL` kalir) — geriye donuk
                uyumluluk icin mevcut cagiranlarin hicbiri bozulmaz.
            risk_status: `assessed` | `unknown` (bkz. `AgentDecision.risk_status`).
                Varsayilan `assessed` geriye donuk uyumluluk icindir.
        """
        with self._lock:
            self._connection.execute(
                """
                UPDATE analyses
                   SET status='completed', updated_at=?, report_json=?,
                       risk_level=?, risk_score=?, summary=?, trace_json=?, risk_status=?
                 WHERE job_id=?
                """,
                (_now_iso(), report_json, risk_level, risk_score, summary, trace_json, risk_status, job_id),
            )
            self._connection.commit()

    def mark_failed(self, job_id: str) -> None:
        """Isi `failed` olarak isaretler (rapor JSON'u yok)."""
        with self._lock:
            self._connection.execute(
                "UPDATE analyses SET status='failed', updated_at=? WHERE job_id=?",
                (_now_iso(), job_id),
            )
            self._connection.commit()

    def set_status(self, job_id: str, status: str) -> None:
        """Ara durum guncellemesi (orn. running)."""
        with self._lock:
            self._connection.execute(
                "UPDATE analyses SET status=?, updated_at=? WHERE job_id=?",
                (status, _now_iso(), job_id),
            )
            self._connection.commit()

    # ---- okuma (history) ----

    def list(self, limit: int = 50, offset: int = 0) -> List[AnalysisRecord]:
        """Analizleri en yeni ONCE olacak sekilde (created_at DESC) sayfali dondurur."""
        rows = self._connection.execute(
            """
            SELECT * FROM analyses
             ORDER BY created_at DESC, rowid DESC
             LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [self._to_record(r) for r in rows]

    def get(self, job_id: str) -> Optional[AnalysisRecord]:
        """Tek bir analiz kaydini dondurur; yoksa None."""
        row = self._connection.execute(
            "SELECT * FROM analyses WHERE job_id=?", (job_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def count(self) -> int:
        """Toplam kayit sayisi (sayfalama/istatistik icin)."""
        return int(self._connection.execute("SELECT COUNT(*) AS c FROM analyses").fetchone()["c"])

    @staticmethod
    def _to_record(row: sqlite3.Row) -> AnalysisRecord:
        d: Dict[str, Any] = dict(row)
        return AnalysisRecord(
            job_id=d["job_id"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            status=d["status"],
            video_source=d.get("video_source"),
            risk_level=d.get("risk_level"),
            risk_score=d.get("risk_score"),
            summary=d.get("summary"),
            report_json=d.get("report_json"),
            trace_json=d.get("trace_json"),
            risk_status=d.get("risk_status") or "assessed",
        )

    def close(self) -> None:
        """Veritabani baglantisini kapatir."""
        self._connection.close()


if __name__ == "__main__":
    # Bagimsiz calistirilabilirlik testi:  python -m src.memory.analysis_store
    import tempfile

    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as tmp:
        store = AnalysisStore(f"{tmp}/analyses.db")
        store.create("job-1", "data/a.mp4")
        store.mark_completed(
            "job-1", report_json=json.dumps({"risk_score": 80}), risk_level="yuksek", risk_score=80, summary="ozet"
        )
        store.create("job-2", "data/b.mp4")
        store.mark_failed("job-2")
        print("count:", store.count())
        for rec in store.list():
            print(" ", rec.job_id, rec.status, rec.risk_level, rec.risk_score)
        # restart: yeni baglanti ayni dosyayi gorur mu
        store.close()
        store2 = AnalysisStore(f"{tmp}/analyses.db")
        print("after 'restart' count:", store2.count())
        store2.close()
