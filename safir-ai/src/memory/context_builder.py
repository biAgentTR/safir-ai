"""04 - Context Builder: VLM ciktisini, istemleri ve zaman damgalarini birlestirir.

Mevzuat (regulation) alani - T017 duzeltmesi
---------------------------------------------
Onceden bu katman, `EmbeddingRAGService.query(vlm_description)` ile SERBEST
METIN uzerinde bagimsiz bir FAISS benzerlik aramasi yapip top-k sonucu,
hicbir uygulanabilirlik kontrolu olmadan "ilgili mevzuat" olarak
`relevant_regulations`e yaziyordu. Vektor benzerligi TEK BASINA bir
mevzuatin GERCEKTEN uygulanabilir oldugunun kaniti DEGILDIR (bkz.
`src/event_analysis/regulation_matcher.py` modul dokustringi icin tam
gerekce ve somut yanlis-pozitif ornekleri).

Bu katman artik KENDI RAG sorgusunu yapmiyor. `relevant_regulations`,
cagiran tarafindan (bkz. `src/main.py::SafirPipeline.stage_context`) ZATEN
`RuleEngine.evaluate(...)` -> `resolve_regulation_matches(...)` ile
deterministik olarak DOGRULANMIS bir mevzuat basligi listesi olarak
verilir; `build()` bu listeyi OLDUGU GIBI tasir, kendi basina bir
uygulanabilirlik karari VERMEZ. Boylece bu modul artik `EmbeddingRAGService`e
bagimli degildir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.memory.event_store import EventStore

logger = logging.getLogger(__name__)


@dataclass
class EnrichedContext:
    """Ajan/Muhakeme Katmanina iletilecek zenginlestirilmis baglam paketi."""

    vlm_description: str
    user_prompt: str
    timestamp: float
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    relevant_regulations: List[str] = field(default_factory=list)
    """ZATEN dogrulanmis (RuleEngine-turevli, deterministik) mevzuat basliklari;
    bkz. modul dokustringi. Bos liste = "Mevzuat eslestirilemedi" (GECERLI sonuc)."""

    def to_prompt_block(self) -> str:
        """Bu baglami LangGraph ajanina verilecek tek bir metin bloguna cevirir.

        Returns:
            Ajanin sistem/kullanici istemine dogrudan eklenebilecek metin.
        """
        recent = "\n".join(
            f"- [{event.get('timestamp'):.1f}s] {event.get('description')}"
            for event in self.recent_events
        ) or "(gecmis olay bulunamadi)"

        regulations = "\n".join(f"- {reg}" for reg in self.relevant_regulations) or (
            "Mevzuat eslestirilemedi: bu olay/gozlem icin guvenilir, dogrulanmis bir ISG "
            "mevzuat eslesmesi bulunamadi. Bu durum GECERLI ve BEKLENEN bir sonuctur - bir "
            "mevzuat UYDURMA. NOT: mevzuat eslesmesinin olmamasi risk seviyesini DUSURMEZ/"
            "YUKSELTMEZ; risk ayrica, tamamen bagimsiz olarak RuleEngine tarafindan belirlenir."
        )

        return (
            f"## Guncel Gozlem (t={self.timestamp:.1f}s)\n{self.vlm_description}\n\n"
            f"## Kullanici Istemi\n{self.user_prompt}\n\n"
            f"## Yakin Gecmis Olaylar\n{recent}\n\n"
            f"## Ilgili Operasyonel Mevzuat (RuleEngine-dogrulanmis)\n{regulations}"
        )


class ContextBuilder:
    """VLM metnini, kullanici istemini ve gecmis olay bellegini birlestiren katman.

    Bkz. modul dokustringi: mevzuat listesi artik bu katmanin kendi RAG
    sorgusundan DEGIL, cagiranin verdigi (RuleEngine-dogrulanmis) `build(...,
    relevant_regulations=...)` argumanindan gelir.
    """

    def __init__(self, event_store: EventStore) -> None:
        """ContextBuilder'i olay bellegi bagimliligiyla baslatir.

        Args:
            event_store: Gecmis olaylari sorgulamak icin SQLite tabanli depo.
        """
        self._event_store = event_store

    def build(
        self,
        vlm_description: str,
        user_prompt: str,
        timestamp: float,
        recent_event_limit: int = 5,
        relevant_regulations: Optional[List[str]] = None,
    ) -> EnrichedContext:
        """VLM ciktisini gecmis olaylar ve (varsa) dogrulanmis mevzuat basliklariyla zenginlestirir.

        Args:
            vlm_description: VLM katmanindan gelen dogal dil olay aciklamasi.
            user_prompt: Operatorden veya sistemden gelen kullanici istemi.
            timestamp: Mevcut gozlemin zaman damgasi.
            recent_event_limit: Getirilecek yakin gecmis olay sayisi.
            relevant_regulations: Cagiranin ONCEDEN dogruladigi (bkz.
                `resolve_regulation_matches`) mevzuat basligi listesi;
                `None`/bos liste ise "Mevzuat eslestirilemedi" GECERLI
                sonucu olarak ele alinir (bu katman kendi basina bir RAG
                sorgusu YAPMAZ, bkz. modul dokustringi).

        Returns:
            Ajan katmanina iletilmeye hazir `EnrichedContext` nesnesi.
        """
        recent_events = self._event_store.query_recent(limit=recent_event_limit)

        context = EnrichedContext(
            vlm_description=vlm_description,
            user_prompt=user_prompt,
            timestamp=timestamp,
            recent_events=recent_events,
            relevant_regulations=list(relevant_regulations or []),
        )
        logger.debug(
            "Baglam olusturuldu: %d gecmis olay, %d dogrulanmis mevzuat",
            len(recent_events),
            len(context.relevant_regulations),
        )
        return context
