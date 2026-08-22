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
uygulanabilirlik karari VERMEZ.

`semantically_related_chunks` alani (2026-08-22, RAG 2. asama) - AYRI ve
FARKLI bir kavram
-------------------------------------------------------------------------
Bu alan, VLM'in urettigi dinamik risk keyword'lerinden (`TemporalEvent.
matched_keywords`) kurulan bir sorguyla `EmbeddingRAGService.query()`
uzerinden gelen, OLASILIKSAL/semantik arama sonuclaridir - `relevant_
regulations` (RuleEngine-dogrulanmis, DETERMINISTIK) ile KARISTIRILMAMASI
icin BILEREK ayri bir alanda, ayri bir prompt basligi ("Semantik Olarak
Ilgili Kaynaklar", ASLA "Ilgili Mevzuat" DEGIL) altinda tutulur. Bu
kaynaklarin varligi/yoklugu risk_score/risk_level'i ETKILEMEZ (bkz.
`src/event_analysis/risk_resolver.py`, bu modulden hic cagrilmaz).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.memory.embedding_rag_service import RetrievedDocument
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
    semantically_related_chunks: List[RetrievedDocument] = field(default_factory=list)
    """VLM dinamik risk keyword'lerinden kurulan sorguyla gelen, OLASILIKSAL
    semantik arama sonuclari (bkz. modul dokustringi). `relevant_regulations`den
    TAMAMEN AYRI bir kavramdir; risk kararini ETKILEMEZ. Bos liste = esik-uzeri
    hicbir sonuc bulunamadi (GECERLI sonuc, rastgele sonuc UYDURULMAZ)."""

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

        semantic_block = self._format_semantic_chunks()

        return (
            f"## Guncel Gozlem (t={self.timestamp:.1f}s)\n{self.vlm_description}\n\n"
            f"## Kullanici Istemi\n{self.user_prompt}\n\n"
            f"## Yakin Gecmis Olaylar\n{recent}\n\n"
            f"## Ilgili Operasyonel Mevzuat (RuleEngine-dogrulanmis)\n{regulations}\n\n"
            f"## Semantik Olarak Ilgili Kaynaklar (deterministik DEGIL)\n{semantic_block}"
        )

    def _format_semantic_chunks(self) -> str:
        """`semantically_related_chunks`i, acik bir uyari/disclaimer ile birlikte metne cevirir."""
        disclaimer = (
            "Bu kaynaklar semantik arama sonucudur; tek basina mevzuatin olaya "
            "uygulanabilirliginin KANITI DEGILDIR. Yukaridaki 'Ilgili Operasyonel "
            "Mevzuat' bolumunden BAGIMSIZDIR ve risk kararini ETKILEMEZ."
        )
        if not self.semantically_related_chunks:
            return f"{disclaimer}\nBu sorgu icin esik-uzeri semantik olarak ilgili kaynak bulunamadi."

        lines = [disclaimer, ""]
        for chunk in self.semantically_related_chunks:
            # `getattr(..., None)` KASITLI: `RetrievedDocument` disinda, yalnizca
            # `text`/`score` tasiyan eski/duck-typed nesneler (orn. bazi testlerin
            # sahte RAG fixture'lari) icin de GUVENLI CALISIR - eksik alan acikca
            # `None`/fallback olarak ele alinir, UYDURULMAZ (bkz. gorev tanimi 3. bolum).
            title = getattr(chunk, "document_title", None) or getattr(chunk, "document_id", None) or "(bilinmeyen kaynak)"
            article_number = getattr(chunk, "article_number", None)
            article = f"Madde {article_number}" if article_number else ""
            rerank_score = getattr(chunk, "rerank_score", None)
            embedding_score = getattr(chunk, "embedding_score", None)
            if rerank_score is not None:
                score = f"{rerank_score:.3f}"
            elif embedding_score is not None:
                score = f"{embedding_score:.3f}"
            else:
                score = f"{getattr(chunk, 'score', 0.0):.3f}"
            text = getattr(chunk, "text", "")
            lines.append(f"- [{title}{(' - ' + article) if article else ''}] (skor={score})\n  {text[:400]}")
        return "\n".join(lines)


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
        semantically_related_chunks: Optional[List[RetrievedDocument]] = None,
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
            semantically_related_chunks=list(semantically_related_chunks or []),
        )
        logger.debug(
            "Baglam olusturuldu: %d gecmis olay, %d dogrulanmis mevzuat",
            len(recent_events),
            len(context.relevant_regulations),
        )
        return context
