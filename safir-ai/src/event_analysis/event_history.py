"""T012 - Event Gecmisi: `StructuredEvent`i `EventStore`a yazan orkestrasyon katmani.

Onemli kisit: `src/memory/event_store.py::EventStore`'un bir guncelleme
(UPDATE) metodu YOKTUR (yalnizca `add_event`/`query_recent`/
`query_by_risk_level`/`get_timeline`/`record_feedback`/`close`). Bu yuzden
`EventHistory.record()`, risk skoru/seviyesini `StructuredEvent`in kendi
(T011'den gelen, henuz `None` olan) degerleri yerine gecebilecek opsiyonel
parametreler olarak alir; boylece `05 LangGraph Ajani`nin karari
CIKTIKTAN SONRA bu metod risk bilgisiyle BIRLIKTE, TEK SEFERDE cagrilmalidir.

Erken cagri kisiti
-------------------
`record()`, agent kararindan ONCE (`risk_score`/`risk_level` verilmeden,
yani `StructuredEvent`in kendi `None` degerleriyle) cagrilirsa, SQLite'a
risk bilgisi olmadan yazar. `EventStore`'da update metodu olmadigi icin bu
satir SONRADAN risk bilgisiyle guncellenemez. Bu bilinen ve kasitli bir
kisittir: dogru kullanim, `record()`i yalnizca agent karari elde
edildikten SONRA, `risk_score`/`risk_level` parametreleriyle BIRLIKTE
cagirmaktir (bkz. modul sonundaki `__main__` demo ve entegrasyon onerisi
icin proje sohbetindeki not).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

from src.event_analysis.schemas import StructuredEvent

if TYPE_CHECKING:
    from src.memory.event_store import EventStore

logger = logging.getLogger(__name__)


@runtime_checkable
class EventStoreLike(Protocol):
    """`src.memory.event_store.EventStore` ile ayni imzaya sahip herhangi bir nesne icin duck-typing sozlesmesi.

    `EventHistory`, gercek SQLite'a bagimli olmadan testte calisabilmesi
    icin bu Protocol'e uyan herhangi bir nesneyi (gercek `EventStore` veya
    basit bir in-memory test double) constructor'inda kabul eder.
    `src/memory/event_store.py`'a hicbir degisiklik yapilmaz, yalnizca
    import edilip kullanilir.
    """

    def add_event(
        self,
        timestamp: float,
        description: str,
        risk_score: Optional[int] = None,
        risk_level: Optional[str] = None,
        source_model: Optional[str] = None,
    ) -> int:
        """Bkz. `EventStore.add_event`. Eklenen kaydin veritabani ID'sini dondurur."""
        ...

    def record_feedback(self, event_id: int, feedback: str) -> None:
        """Bkz. `EventStore.record_feedback`. Operator HITL geri bildirimini kalici olarak isler."""
        ...


class EventHistory:
    """`StructuredEvent`leri `EventStoreLike` bir depoya yazan ince orkestrasyon katmani.

    Bkz. modul dokustringi icin `EventStore`'da update metodu olmamasinin
    `record()` tasarimina etkisi ve erken-cagri kisiti.
    """

    def __init__(self, event_store: "EventStore | EventStoreLike") -> None:
        """EventHistory'yi bir `EventStoreLike` bagimliligiyla baslatir.

        Args:
            event_store: Gercek `src.memory.event_store.EventStore` orneği
                veya ayni `add_event`/`record_feedback` imzalarina sahip
                herhangi bir nesne (testte in-memory bir sahte depo).
        """
        self._event_store = event_store

    def record(
        self,
        event: StructuredEvent,
        risk_score: Optional[int] = None,
        risk_level: Optional[str] = None,
    ) -> int:
        """Bir `StructuredEvent`i, opsiyonel risk bilgisiyle birlikte `EventStore.add_event`e yazar.

        `event.to_event_store_kwargs()` temel kwargs'i uretir (risk alanlari
        `StructuredEvent`in kendi degeriyle, T011'den geldigi haliyle
        genelde `None`); bu metoda verilen `risk_score`/`risk_level`
        `None` DEGILSE, temel degerlerin yerine gecer. `EventStore`'da
        update metodu olmadigi icin bu cagri, o satirin risk bilgisiyle
        yazilacagi TEK firsattir (bkz. modul dokustringi).

        Args:
            event: `EventBuilder.build(...)` ciktisi.
            risk_score: `05 LangGraph Ajani`nin urettigi risk skoru (0-100);
                `None` ise `event.risk_score` (genelde `None`) kullanilir.
            risk_level: Ajanin urettigi risk seviyesi etiketi; `None` ise
                `event.risk_level` (genelde `None`) kullanilir.

        Returns:
            `EventStore.add_event(...)`in dondurdugu veritabani kaydi kimligi.
        """
        kwargs = event.to_event_store_kwargs()
        if risk_score is not None:
            kwargs["risk_score"] = risk_score
        if risk_level is not None:
            kwargs["risk_level"] = risk_level

        event_id = self._event_store.add_event(**kwargs)
        logger.debug(
            "EventHistory: olay kaydedildi id=%s event_type=%s risk_score=%s risk_level=%s",
            event_id,
            event.event_type,
            kwargs["risk_score"],
            kwargs["risk_level"],
        )
        return event_id

    def record_batch(
        self,
        events: List[StructuredEvent],
        risk_scores: Optional[List[Optional[int]]] = None,
        risk_levels: Optional[List[Optional[str]]] = None,
    ) -> List[int]:
        """Birden fazla `StructuredEvent`i, `events` ile ayni sirada `EventStore`a kaydeder.

        Args:
            events: Kaydedilecek `StructuredEvent` listesi.
            risk_scores: `events` ile ayni uzunlukta, her olay icin
                (varsa) risk skoru override listesi. `None` (varsayilan)
                ise hicbir olaya override uygulanmaz. Listenin kendi
                elemanlari da `None` olabilir (o olay icin override yok,
                digerleri icin var).
            risk_levels: `risk_scores` ile ayni mantik, risk seviyesi icin.

        Returns:
            `events` ile ayni sirada, uretilen veritabani kimliklerinin listesi.

        Raises:
            ValueError: `risk_scores`/`risk_levels` verilmis ama uzunlugu
                `events` ile eslesmiyorsa.
        """
        if risk_scores is not None and len(risk_scores) != len(events):
            raise ValueError("risk_scores uzunlugu events ile eslesmeli.")
        if risk_levels is not None and len(risk_levels) != len(events):
            raise ValueError("risk_levels uzunlugu events ile eslesmeli.")

        event_ids: List[int] = []
        for index, event in enumerate(events):
            score = risk_scores[index] if risk_scores is not None else None
            level = risk_levels[index] if risk_levels is not None else None
            event_ids.append(self.record(event, risk_score=score, risk_level=level))

        logger.debug("EventHistory: %d olay toplu kaydedildi.", len(event_ids))
        return event_ids

    def mark_feedback(self, event_id: int, feedback: str) -> None:
        """Operatorun HITL geri bildirimini `EventStore.record_feedback`e ince bir sarmalayiciyla iletir.

        Ekstra dogrulama/mantik icermez; `feedback` degerinin gecerliligi
        (`"true_positive"`/`"false_positive"`) `EventStore.record_feedback`
        tarafinda kontrol edilir.

        Args:
            event_id: `record()`/`record_batch()` tarafindan donen veritabani kimligi.
            feedback: `"true_positive"` veya `"false_positive"`.
        """
        self._event_store.record_feedback(event_id, feedback)


if __name__ == "__main__":
    # T012'nin bagimsiz calistirilabilirlik testi (gercek SQLite ile):
    #   python -m src.event_analysis.event_history
    import tempfile

    from src.memory.event_store import EventStore
    from src.utils.config_loader import SQLiteMemoryConfig

    logging.basicConfig(level=logging.INFO)

    demo_event = StructuredEvent(
        timestamp=12.4,
        description="Personel baretsiz calisiyor.",
        source_model="demo-vlm",
        event_type="kkd_ihlali",
        confidence=0.6,
        temporal_event_id="evt_0",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        demo_store = EventStore(SQLiteMemoryConfig(db_path=f"{tmp_dir}/demo_events.db"))
        demo_history = EventHistory(demo_store)

        # Erken cagri (agent karari yok): risk bilgisi None yazilir.
        early_id = demo_history.record(demo_event)
        print(f"Erken kayit (risk yok): id={early_id}")

        # Agent karari sonrasi cagri: risk bilgisiyle birlikte yazilir.
        decided_id = demo_history.record(demo_event, risk_score=45, risk_level="orta")
        print(f"Karar sonrasi kayit: id={decided_id}")

        demo_history.mark_feedback(decided_id, "true_positive")
        print("Geri bildirim islendi.")

        demo_store.close()
