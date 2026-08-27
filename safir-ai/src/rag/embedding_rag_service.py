"""04 - Embedding & RAG Katmani: EVREN embedding + Qdrant + deterministik relevance skorlama tabanli anlamsal bellek.

Operasyonel kurallari ve ISG mevzuatini EVREN'in (TEKNOFEST yarisma cikarim
servisi) OpenAI-uyumlu embedding ucuyla (`bge-m3-embed`, bkz.
`embedding_providers.py::EvrenEmbeddingProvider`) vektorlestirip, takima
tahsis edilmis izole bir Qdrant orneğinde saklayan/arayan servistir.
LangGraph ajaninin `retriever_tool` araci ve `RuleEngine._describe_regulation()`
bu servis uzerinden calisir.

Knowledge base kaynagi (2026-08-23 -> 2026-08-25, tarihce)
--------------------------------------------------------------
1. asama: `seed_default_regulations()` `data/knowledge_base/chunks/*.json`
   altindaki GERCEK, resmi mevzuat metinlerinden turetilmis madde-bazli
   chunk'lari (bkz. `scripts/build_kb_chunks.py`) yukledi.
2-3. asama: embedding saglayicisi once Gemini Embedding API'ye, sonra TAMAMEN
   LOKAL `sentence-transformers`e tasindi; `RetrievedDocument` yapilandirilmis
   metadata (document_title, article_number, source_url, ...) tasimaya basladi
   (`sources.yaml` ile join edilmis - bkz. `_load_kb_chunk_records`); iki-
   asamali retrieval (candidate_k -> rerank -> score_threshold) eklendi.
4. asama (2026-08-24, RAG RERANKER DETERMINIZATION): ikinci-asama relevance
   skorlamasi (eskiden LLM-as-judge Gemini/Groq) TAMAMEN yerel/matematiksel
   bir algoritmaya (`deterministic_reranker.py`) tasindi.
5. asama (2026-08-24, RAG+RISK PRODUCTION KAPANIS): deterministic relevance
   gate'ten GECMIS ("accepted") adaylar, opsiyonel bir UCUNCU asamada
   (`self._cross_encoder`) YENIDEN siralanir.
6. asama (bu dosya, 2026-08-25, EVREN MIGRASYONU): LOKAL embedding
   (`sentence-transformers`) VE FAISS TAMAMEN KALDIRILDI - embedding artik
   `EvrenEmbeddingProvider` (EVREN `/v1/embeddings`, `bge-m3-embed`, 1024
   boyut), vektor deposu artik EVREN'e tahsis edilmis izole Qdrant orneği
   (bkz. `_build_qdrant_client`, `QdrantMemoryConfig`). Ucuncu asama
   (`self._cross_encoder`) artik VARSAYILAN olarak `LocalCrossEncoderReranker`
   DEGIL, EVREN'in LLM ucunu "LLM-as-judge" kullanan `EvrenReranker` (bkz.
   `src/rag/reranker.py`) - `LocalCrossEncoderReranker` sinifi KALDIRILMADI
   (standalone/benchmark icin durur) ama production'da ARTIK VARSAYILAN
   DEGIL (bkz. `src/main.py::SafirPipeline.__init__`). Deterministik
   relevance gate (`deterministic_reranker.py`) VE bu ucuncu-asama extension
   point'inin KENDI ARAYUZU (`CrossEncoderReranker.score()`) DEGISMEDI -
   yalnizca hangi implementasyonun VARSAYILAN olarak baglandigi degisti.

ONEMLI (davranis degisikligi, 2026-08-23, HALA GECERLI): Qdrant koleksiyonu
GUNCEL degilse/yoksa (bkz. `_try_load_qdrant_collection`), `seed_default_regulations()`
`DEFAULT_ISG_REGULATIONS` (8 ornek madde) placeholder'ina SESSIZCE DUSMEZ -
acikca `KnowledgeBaseNotBuiltError` firlatir. `DEFAULT_ISG_REGULATIONS` sabiti
yalnizca DIGER modullerin (orn. `src/agent/agent_workflow.py`nin mock
ornekleri) DOGRUDAN, bu servisten BAGIMSIZ kullanimi icin KALIR - bu
servisin otomatik fallback'i DEGILDIR.

ONEMLI (bilinen, KASITLI sinirlama): `RuleEngine._describe_regulation()`,
sorguladigi sabit kisa etiketin (orn. "ISG Yonetmeligi Madde 12" -
`EVENT_TYPE_REGULATION_MAP`'ten gelir) donen metinde GERCEKTEN GECIP
GECMEDIGINI kontrol eder. Bu etiketler ESKI/sahte 8-madde listesiyle EL
YORDAMIYLA eslesecek sekilde uydurulmustu; GERCEK mevzuat metinlerinde bu
TAM ETIKETLER GECMEZ. Sonuc olarak `_describe_regulation` COGUNLUKLA
enrichment'i REDDEDECEK ve kisa etikete DONECEKTIR - bu GUVENLI bir
davranistir (yanlis bir metin ASLA rapora sizmaz, yalnizca zenginlestirme
kaybolur) ve BILEREK bu degisikligin kapsami DISINDA birakildi;
`EVENT_TYPE_REGULATION_MAP`in etiketleri GUNCELLENMEDI - RuleEngine'in
deterministik event_type->mevzuat eslemesi ayri, gelecekteki bir adimdir.

MIMARI AYRIM (ONEMLI): Bu serviste uretilen `embedding_score`/`relevance_score`
DEGERLERI, `RuleEngine`in deterministik risk_score/risk_level/escalation
kararina ASLA girdi OLMAZ (bkz. `src/event_analysis/risk_resolver.py`,
degistirilmedi). Bu servis yalnizca (A) `RuleEngine._describe_regulation`
icin kisa-etiket lookup'i ve (B) ajanin opsiyonel `retriever_tool` cagrisi
VE (C) `ContextBuilder`in `semantically_related_chunks` alani icin
KULLANILIR - hicbiri risk kararini degistirmez.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.rag.deterministic_reranker import RelevanceBreakdown, RelevanceWeights, score_candidate
from src.rag.embedding_providers import (
    ConfigurationError,
    EmbeddingProvider,
    _DEFAULT_SAFE_TOKEN_BUDGET,
    _estimate_tokens,
    build_embedding_provider,
    split_oversized_text,
)
from src.rag.local_cross_encoder_reranker import CrossEncoderReranker, CrossEncoderUnavailableError
from src.utils.config_loader import EmbeddingConfig, QdrantMemoryConfig, RerankerConfig

logger = logging.getLogger(__name__)

DEFAULT_ISG_REGULATIONS: List[str] = [
    "ISG Yonetmeligi Madde 12: Yuksekte calisma alanlarinda dusme onleyici ekipman (emniyet kemeri, "
    "yasam hatti) zorunludur; 2 metre ve uzerindeki calismalarda korkuluk veya guvenlik agi bulunmalidir.",
    "ISG Yonetmeligi Madde 24: Kisisel Koruyucu Donanim (baret, is ayakkabisi, yansitici yelek) sahaya "
    "giren tum personel ve ziyaretciler icin zorunludur; eksik KKD ile sahaya giris yasaktir.",
    "Operasyonel Kural OK-07: Forklift ve is makinesi trafiginde yaya gecitleri her zaman acik "
    "tutulmali, arac-yaya ayrimi sarrafiye/bariyerle saglanmalidir.",
    "ISG Yonetmeligi Madde 31: Yanici/patlayici madde bulunan alanlarda ates, kivilcim ve sicak yuzey "
    "kaynakli islemler icin sicak calisma izni (hot work permit) alinmadan calisma baslatilamaz.",
    "Yangin Guvenligi Talimati YG-03: Duman veya alev tespit edilen alanlarda calisma derhal "
    "durdurulmali, en yakin tahliye noktasindan sahadan cikilmali ve yangin ekibi bilgilendirilmelidir.",
    "ISG Yonetmeligi Madde 45: Kapali/dar alan calismalarinda gaz olcumu yapilmadan ve gozetmen "
    "atanmadan alana giris yasaktir.",
    "Operasyonel Kural OK-15: Elektrik pano ve hatlarina yakin calismalarda enerji kesme/kilitleme "
    "(LOTO - Lockout/Tagout) prosedurune uyulmadan mudahale edilemez.",
    "ISG Yonetmeligi Madde 52: Agir yuk kaldirma ekipmanlari (vinc, kren) calisma alaninda, operatorun "
    "gorus alani disindaki bolgelerde sinyalman gorevlendirilmesi zorunludur.",
]
"""SADECE diger modullerin (orn. `src/agent/agent_workflow.py`nin mock ornekleri)
DOGRUDAN kullanimi icin duran sabit bir metin listesi - `EmbeddingRAGService`nin
KENDISI bunu ARTIK OTOMATIK FALLBACK olarak kullanmaz (bkz. modul dokustringi,
`seed_default_regulations`: persisted index yoksa SESSIZCE bu listeye DUSMEZ,
acik bir hatayla FAIL-FAST eder)."""


class KnowledgeBaseNotBuiltError(RuntimeError):
    """Gercek, doldurulmus bir KB Qdrant koleksiyonu bulunamadi/guncel degil ve SESSIZCE placeholder'a DUSULMEDI.

    `seed_default_regulations()` tarafindan firlatilir - operator/gelistirici
    `python -m src.rag.build_knowledge_index` calistirmadan sistemi
    BASLATAMAZ (bkz. gorev tanimi: "acik sekilde fail-fast et").
    """

_KB_ROOT = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"
_KB_CHUNKS_DIR = _KB_ROOT / "chunks"
_KB_SOURCES_YAML = _KB_ROOT / "metadata" / "sources.yaml"

_QDRANT_META_POINT_ID = 1
"""Meta-koleksiyondaki (bkz. `_meta_collection_name`) TEK noktanin sabit ID'si -
model/dimension/kb_hash manifest'i bu noktanin payload'unda tutulur (eski
`index_meta.json`in Qdrant karsiligi); asil koleksiyonun ICINE KARISTIRILMAZ
(arama/count sonuclarini KIRLETMEMESI icin AYRI bir koleksiyonda tutulur)."""

_INDEX_NORMALIZATION = "l2"
"""Manifest'te persist edilen normalization semasi (bkz. `embedding_providers.py::_l2_normalize`) -
hem indeksleme hem sorgu vektorleri AYNI (L2) normalizasyondan gecer."""
_INDEX_METRIC = "cosine"
"""Qdrant `Distance.COSINE` - L2-normalize vektorler uzerinde eski FAISS `IndexFlatIP`
(inner product) ile MATEMATIKSEL OLARAK ES DEGER siralama sonucu uretir."""

_LEVEL_LABELS = {
    "madde": "MADDE",
    "gecici_madde": "GEÇİCİ MADDE",
    "ek_alt_madde": "EK",
    "ek_liste": "EK",
}


def _load_sources_metadata(sources_yaml: Path = _KB_SOURCES_YAML) -> Dict[str, Dict[str, Any]]:
    """`data/knowledge_base/metadata/sources.yaml`i `document_id -> {title, institution, source_url, publication_date}` sozlugune cevirir.

    Dosya yoksa/gecersizse BOS SOZLUK doner - hicbir metadata UYDURULMAZ,
    cagiran taraf eksik alanlari `None` olarak tasir.
    """
    if not sources_yaml.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(sources_yaml.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - metadata join en kotu ihtimalle bos kalir
        logger.warning("sources.yaml okunamadi/gecersiz: %s", sources_yaml)
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for doc in data.get("documents", []) or []:
        doc_id = doc.get("id")
        if not doc_id:
            continue
        result[doc_id] = {
            "document_title": doc.get("title"),
            "institution": doc.get("institution"),
            "source_url": doc.get("source_url"),
            "publication_date": doc.get("publication_date"),
        }
    return result


def _load_kb_chunk_records(chunks_dir: Path = _KB_CHUNKS_DIR, sources_yaml: Path = _KB_SOURCES_YAML) -> List[Dict[str, Any]]:
    """`data/knowledge_base/chunks/*.json` chunk'larini `sources.yaml` metadata'siyla JOIN ederek yapilandirilmis kayit listesine cevirir.

    Her kayit, `RetrievedDocument` alanlarinin buyuk kismini (text HARIC
    tumunu) doldurur: `chunk_id`, `document_id`, `document_title`, `level`,
    `article_number`, `article_title`, `is_annex`, `page_start`, `page_end`,
    `source_url`, `institution`, `publication_date`, `text`.

    `article_title` bu surumde HENUZ cikarilamiyor (chunk metninde ayri bir
    alan olarak yok) - acikca `None` birakilir, UYDURULMAZ. `sources.yaml`de
    karsiligi olmayan bir `document_id` icin `document_title`/`institution`/
    `source_url`/`publication_date` de acikca `None` kalir.

    Args:
        chunks_dir: `*.json` chunk dosyalarinin bulundugu klasor.
        sources_yaml: Doküman-seviyesi metadata dosyasi.

    Returns:
        Kayit listesi (dosya adina, sonra dosya icindeki sıraya gore
        deterministik siralanir). Klasor yoksa/bossa BOS LISTE doner.
    """
    if not chunks_dir.exists():
        return []

    sources = _load_sources_metadata(sources_yaml)
    records: List[Dict[str, Any]] = []

    for chunk_file in sorted(chunks_dir.glob("*.json")):
        try:
            items = json.loads(chunk_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("KB chunk dosyasi okunamadi/gecersiz JSON, atlaniyor: %s", chunk_file)
            continue
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            document_id = str(item.get("document_id") or chunk_file.stem)
            meta = sources.get(document_id, {})
            records.append(
                {
                    "chunk_id": item.get("chunk_id"),
                    "document_id": document_id,
                    "document_title": meta.get("document_title"),
                    "level": item.get("level"),
                    "article_number": item.get("number"),
                    "article_title": None,  # bkz. docstring: bu surumde cikarilamiyor, UYDURULMAZ
                    "is_annex": item.get("is_annex"),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "source_url": meta.get("source_url"),
                    "institution": meta.get("institution"),
                    "publication_date": meta.get("publication_date"),
                    "text": text,
                }
            )

    return records


def _split_oversized_records(
    records: List[Dict[str, Any]], safe_token_budget: int
) -> List[Dict[str, Any]]:
    """Metni `safe_token_budget`i TEK BASINA asan kayitlari, TUM metadata'yi KORUYARAK birden fazla alt-kayida boler.

    2026-08-25 guncellemesi (GUVENLI CHUNK BOLME): `_load_kb_chunk_records()`
    urettigi 748 kaydin URETIM MANTIGI (chunk stratejisi, `scripts/
    build_kb_chunks.py`) DEGISTIRILMEZ - bu fonksiyon, embed'lemeden HEMEN
    ONCE, yalnizca GUVENLI token butcesini (bkz. `embedding_providers.py::
    _DEFAULT_SAFE_TOKEN_BUDGET`) TEK BASINA asan (nadir) kayitlari `split_
    oversized_text()` ile semantik/metin sinirlarina (paragraf->cumle->
    kelime) gore ALT-PARCALARA boler - SESSIZCE KIRPMAZ (bkz. o fonksiyonun
    dokustringi, veri kaybi YOK).

    `document_id`/`document_title`/`level`/`article_number`/`article_title`/
    `is_annex`/`page_start`/`page_end`/`source_url`/`institution`/
    `publication_date` alanlarinin TUMU, olusan HER alt-kayitta AYNEN
    KORUNUR (ayni maddeden/kaynaktan geldikleri icin) - yalnizca `chunk_id`
    (`__partN` eki ile benzersizlestirilir) ve `text` (alt-parca) degisir.
    Bu, bir kaynak kaydin BIRDEN FAZLA nihai (embed'lenmis/Qdrant'a
    yazilmis) dokumana karsilik gelebilecegi, dolayisiyla `document_count()`
    ozgun kaynak kayit sayisini (orn. 748) ASABILECEGI anlamina gelir - bu
    BEKLENEN ve DOGRU bir davranistir.

    Args:
        records: `_load_kb_chunk_records()` (veya esdegeri) ciktisi.
        safe_token_budget: Bir kaydin TEK BASINA asmamasi gereken tahmini token sayisi.

    Returns:
        Butceyi asmayan kayitlar oldugu gibi, asanlar ise alt-parcalara
        bolunmus olarak GENISLETILMIS kayit listesi (orijinal sira KORUNUR).
    """
    expanded: List[Dict[str, Any]] = []
    for record in records:
        text = str(record.get("text") or "")
        pieces = split_oversized_text(text, safe_token_budget)
        if len(pieces) <= 1:
            expanded.append(record)
            continue

        base_chunk_id = record.get("chunk_id") or "chunk"
        logger.warning(
            "KB kaydi '%s' (document_id=%s) guvenli token butcesini asiyor (tahmini %d token > %d); "
            "%d alt-parcaya bolundu (semantik/metin sinirlarina gore, metadata KORUNDU, veri kaybi YOK).",
            base_chunk_id,
            record.get("document_id"),
            _estimate_tokens(text),
            safe_token_budget,
            len(pieces),
        )
        for part_index, piece in enumerate(pieces, start=1):
            sub_record = dict(record)
            sub_record["chunk_id"] = f"{base_chunk_id}__part{part_index}"
            sub_record["text"] = piece
            expanded.append(sub_record)

    return expanded


def _compute_kb_hash(chunks_dir: Path = _KB_CHUNKS_DIR) -> str:
    """Chunk klasorunun icerik hash'ini hesaplar (persisted index'in GUNCEL olup olmadigini anlamak icin).

    2026-08-24 duzeltmesi: satir sonu (LF/CRLF) temsili NORMALIZE edilir
    (CRLF'e - bkz. asagida) - aksi halde AYNI chunk ICERIGI, farkli isletim
    sistemi/git `core.autocrlf` ayariyla checkout edilmis iki ortamda
    FARKLI hash uretiyordu (gercek bir kopukluk: Windows'ta uretilen bir
    persisted index, Linux'ta - veya tam tersi - SADECE satir sonu farki
    yuzunden "guncel degil" sayilip fail-fast ile REDDEDILIYORDU, halbuki
    metin icerigi TAMAMEN AYNIYDI). Normalizasyon CRLF'e yapilir (LF'e
    DEGIL) - zaten persist edilmis `index_meta.json`lardaki `kb_hash`
    degerleri boyle uretilmisti; bu index'lerin YENIDEN insa EDILMESINI
    gerektirmeden GUNCEL sayilabilmesi icin bu yon secildi.
    """
    if not chunks_dir.exists():
        return "no-chunks-dir"
    hasher = hashlib.sha256()
    for chunk_file in sorted(chunks_dir.glob("*.json")):
        hasher.update(chunk_file.name.encode("utf-8"))
        normalized = chunk_file.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        hasher.update(normalized)
    return hasher.hexdigest()


@dataclass
class RetrievedDocument:
    """CANONICAL retrieved-evidence nesnesi: FAISS(+opsiyonel deterministik relevance skorlama)dan gelen tek bir chunk'i, yapilandirilmis kaynak metadata'si + karar-izlenebilirligi ile birlikte tasir.

    `embedding_score` (FAISS/cosine benzerligi) ile `relevance_score` (yerel,
    deterministik agirlikli-toplam skoru - bkz. `deterministic_reranker.py`,
    skorlama devre disiysa `None`) KASITLI olarak AYRI alanlardir - birbirine
    KARISTIRILMAZ (bkz. modul dokustringi).

    2026-08-24 (RAG PIPELINE RECONSTRUCTION): retrieval'dan Agent'e/rapora
    kadar TEK canonical nesne olarak korunmasi icin `retrieval_rank`/
    `relevance_status`/`relevance_reason`/`source_verified` eklendi - "neden
    secildi/elendi?" sorusu artik BU nesnenin uzerinde, ayri bir yapida
    YENIDEN URETILMEDEN cevaplanabilir.
    """

    text: str
    embedding_score: float
    relevance_score: Optional[float] = None
    semantic_score: Optional[float] = None
    """`deterministic_reranker.RelevanceBreakdown.semantic_score` - `relevance_score`e
    giden bes bilesenden biri (bkz. `deterministic_reranker.py` modul dokustringi).
    Relevance skorlama devre disiysa/hesaplanmadiysa `None` (UYDURULMAZ)."""
    lexical_score: Optional[float] = None
    """`RelevanceBreakdown.lexical_score` - ayni gerekce."""
    keyword_score: Optional[float] = None
    """`RelevanceBreakdown.keyword_score` - ayni gerekce."""
    metadata_score: Optional[float] = None
    """`RelevanceBreakdown.metadata_score` - ayni gerekce."""
    phrase_score: Optional[float] = None
    """`RelevanceBreakdown.phrase_score` - ayni gerekce."""
    cross_encoder_score: Optional[float] = None
    """LOKAL Cross-Encoder'in (varsa, bkz. `local_cross_encoder_reranker.py`)
    (query, chunk) cift skoru - `embedding_score`/`relevance_score`den AYRI,
    `risk_score`/`confidence`/`probability` OLARAK ADLANDIRILMAZ. Cross-Encoder
    bu cagrida devreye GIRMEDIYSE (varsayilan - bkz. `EmbeddingRAGService.__init__`
    `cross_encoder` parametresi) HER ZAMAN `None`."""
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    level: Optional[str] = None
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    is_annex: Optional[bool] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_url: Optional[str] = None
    institution: Optional[str] = None
    publication_date: Optional[str] = None
    retrieval_rank: Optional[int] = None
    """FAISS aday siralamasindaki 1-index'li sira (embedding_score'a gore azalan)."""
    final_rank: Optional[int] = None
    """NIHAI sonuc kumesindeki (final_docs, threshold+Cross-Encoder SONRASI) 1-index'li
    sira. Cross-Encoder devredeyse `cross_encoder_score`e gore, degilse
    `relevance_score`e (o da yoksa `embedding_score`e) gore atanir - hangi
    siralamanin GERCEKTEN kullanildigini `RagQueryTelemetry.cross_encoder_status`
    uzerinden izlenebilir kilar (bkz. gorev tanimi 4/6. bolum)."""
    relevance_status: Optional[str] = None
    """'accepted' | 'rejected' - bu adayin NEDEN final sonuca girip girmedigini
    gosteren, `query()` icinde atanan deterministik karar (bkz.
    `deterministic_reranker.score_candidate`). ARTIK bir 'unavailable'
    degeri YOK - relevance skorlama TAMAMEN yerel/matematiksel oldugu icin
    bir API/LLM basarisizlik modu MUMKUN DEGIL (bkz. gorev tanimi)."""
    relevance_reason: Optional[str] = None
    """`relevance_status`un insan-okunur, hangi skor/esikten geldigini
    gosteren gerekcesi (orn. 'relevance_score (0.850) >= threshold (0.100)')."""
    source_verified: bool = True
    """Bu chunk, persisted KB index'inden (gercek corpus) GERCEKTEN geldigi
    icin HER ZAMAN `True` - retrieval sonucu, tanimi geregi corpus-kokenlidir
    (bkz. gorev tanimi 10. bolum: bu, Agent'in SERBEST METIN ciktisindaki
    olasi UYDURULMUS referanslarla KARISTIRILMAMALIDIR - onlar icin bkz.
    `SafirReport.unverified_references`)."""

    @property
    def score(self) -> float:
        """Geriye-uyumluluk: eski `{text, score}` sozlesmesine bagli cagiran kod icin.

        Rerank yapildiysa `relevance_score`, yapilmadiysa `embedding_score`
        dondurur - "en guvenilir elimizdeki skor" anlaminda, ama bu ikisini
        BIRBIRINE KARISTIRMAZ (asil kod `embedding_score`/`relevance_score`e
        AYRI AYRI erismelidir).
        """
        return self.relevance_score if self.relevance_score is not None else self.embedding_score


def _avg(values: List[float]) -> Optional[float]:
    """Bos listede `None` dondurur (0.0 DEGIL - "hic sonuc yok" ile "ortalama 0" KARISTIRILMAZ)."""
    return round(sum(values) / len(values), 4) if values else None


def _result_telemetry(doc: "RetrievedDocument", selected: bool) -> "RagResultTelemetry":
    return RagResultTelemetry(
        chunk_id=doc.chunk_id,
        document_id=doc.document_id,
        document_title=doc.document_title,
        article_number=doc.article_number,
        source_url=doc.source_url,
        embedding_score=round(doc.embedding_score, 4),
        relevance_score=round(doc.relevance_score, 4) if doc.relevance_score is not None else None,
        semantic_score=round(doc.semantic_score, 4) if doc.semantic_score is not None else None,
        lexical_score=round(doc.lexical_score, 4) if doc.lexical_score is not None else None,
        keyword_score=round(doc.keyword_score, 4) if doc.keyword_score is not None else None,
        metadata_score=round(doc.metadata_score, 4) if doc.metadata_score is not None else None,
        phrase_score=round(doc.phrase_score, 4) if doc.phrase_score is not None else None,
        cross_encoder_score=round(doc.cross_encoder_score, 4) if doc.cross_encoder_score is not None else None,
        selected=selected,
        rank=doc.retrieval_rank,
        final_rank=doc.final_rank,
        relevance_status=doc.relevance_status,
        relevance_reason=doc.relevance_reason,
        text=doc.text,
    )


@dataclass
class RagResultTelemetry:
    """Tek bir RAG adayinin/sonucunun TAM (metin DAHIL) telemetri kaydi.

    2026-08-24 (RAG PIPELINE RECONSTRUCTION, gorev tanimi 13. bolum): onceki
    surumde `text` KASITLI disarida birakilmisti (dashboard'a yalnizca
    metadata/skor gitsin diye); bu turun ACIK talebiyle (operator "bu madde
    neden secildi?" sorusunu chunk'in GERCEK metnini gormeden cevaplayamaz)
    `text` da eklendi. Cagiran taraf (trace/dashboard) bunu ARTIK
    gosterebilir/gizleyebilir - serializer KENDISI veri KAYBETMEZ.
    """

    document_id: Optional[str]
    document_title: Optional[str]
    article_number: Optional[str]
    source_url: Optional[str]
    embedding_score: float
    relevance_score: Optional[float]
    selected: bool
    """Threshold/top_k sonrasi NIHAI sonuc kumesine girdi mi (final_docs icinde mi)."""
    chunk_id: Optional[str] = None
    """Kaynak chunk'in kimligi (bkz. `RetrievedDocument.chunk_id`) - hangi
    dokumanin HANGI maddesinden/parcasindan geldigini izlenebilir kilar."""
    rank: Optional[int] = None
    """FAISS aday siralamasindaki 1-index'li sira (bkz. `RetrievedDocument.retrieval_rank`)."""
    final_rank: Optional[int] = None
    """NIHAI (Cross-Encoder SONRASI, devredeyse) sira (bkz. `RetrievedDocument.final_rank`)."""
    cross_encoder_score: Optional[float] = None
    """LOKAL Cross-Encoder skoru (bkz. `RetrievedDocument.cross_encoder_score`) - devrede
    degilse `None`. `risk_score`/`confidence`/`probability` DEGILDIR, yalnizca bir
    siralama sinyalidir."""
    relevance_status: Optional[str] = None
    """'accepted' | 'rejected' (bkz. `RetrievedDocument.relevance_status`)."""
    relevance_reason: Optional[str] = None
    """`relevance_status`un gerekcesi (bkz. `RetrievedDocument.relevance_reason`)."""
    semantic_score: Optional[float] = None
    """`relevance_score`e giden bes bilesenden biri (bkz. `RetrievedDocument.semantic_score`) -
    explainability (2026-08-24) icin trace telemetrisine de tasinir."""
    lexical_score: Optional[float] = None
    keyword_score: Optional[float] = None
    metadata_score: Optional[float] = None
    phrase_score: Optional[float] = None
    text: str = ""
    """Chunk'in GERCEK metni (bkz. sinif dokustringi - onceki surumde YOKTU)."""


@dataclass
class RagQueryTelemetry:
    """Bir `EmbeddingRAGService.query()` cagrisinin GERCEK, yapilandirilmis telemetri kaydi.

    Ham/tam mevzuat metni TASIMAZ (yalnizca `RagResultTelemetry`, metadata +
    skor). Dashboard/trace katmani, bu nesneyi UYDURMADAN, DOGRUDAN sunar -
    hicbir alan burada yoksa frontend'de "N/A" gosterilmelidir (rastgele
    deger UYDURULMAZ).
    """

    query: str
    candidate_count: int
    final_count: int
    zero_result: bool
    retrieval_status: str  # "relevance_scored" | "embedding_only" | "insufficient_evidence" | "empty_index"
    threshold: Optional[float]
    embedding_latency_ms: float
    rerank_latency_ms: Optional[float]
    total_latency_ms: float
    avg_embedding_score: Optional[float]
    avg_relevance_score: Optional[float]
    corpus_source: str = "unseeded"
    """'qdrant_collection' | 'fallback_placeholder' | 'chunks_rebuild' | 'unseeded'
    (bkz. `EmbeddingRAGService.corpus_source`). 'fallback_placeholder' ise bu
    sonuclar GERCEK mevzuat corpus'undan DEGIL, 8 maddelik placeholder
    listeden geliyor - retrieval mekanizmasi calisiyor olsa da bu, RAG'in
    "gercekte" calismadigi anlamina gelir (bkz. gorev tanimi P0)."""
    cross_encoder_status: str = "disabled"
    """'used' | 'unavailable' | 'disabled' - bu cagrida LOKAL Cross-Encoder'in
    GERCEKTEN calisip calismadigi (bkz. `EmbeddingRAGService.__init__`
    `cross_encoder` parametresi + `local_cross_encoder_reranker.py`).
    'disabled': cagiran taraf hic Cross-Encoder GECMEDI. 'unavailable': bir
    Cross-Encoder verildi ama model agirligi yuklenemedi (bkz.
    `CrossEncoderUnavailableError`) - bu durumda sonuclar deterministik
    relevance siralamasina SESSIZCE DEGIL, ACIKCA bu alanla isaretlenerek
    duser (kontrollu degradasyon, harici bir API'ye ASLA dusulmez). 'used':
    Cross-Encoder GERCEKTEN calisti ve final siralamayi belirledi."""
    results: List[RagResultTelemetry] = field(default_factory=list)


def _build_qdrant_client(qdrant_config: QdrantMemoryConfig) -> QdrantClient:
    """`QdrantMemoryConfig`den bir `QdrantClient` kurar (HICBIR AG CAGRISI YAPMAZ - istemci lazy baglanir).

    `url=":memory:"` ise tamamen bellek-ici, agsiz bir Qdrant orneği
    kullanilir (test/offline kullanim - GERCEK Qdrant istemci KODU calisir,
    yalnizca ag baglantisi yoktur; her `QdrantClient(location=":memory:")`
    cagrisi BAGIMSIZ/izole bir orneği baslatir - istemciler arasi durum
    PAYLASILMAZ). `url`, `http(s)://` ile BASLAMAYAN bir deger ise (orn.
    `tmp_path` altinda bir klasor) YEREL DISK-tabanli bir Qdrant orneği
    kullanilir (birden fazla `QdrantClient` cagrisi arasinda DURUM
    PAYLASILIR - persist/reload round-trip testleri icin). Aksi halde EVREN
    dokumantasyonu SS 11'in ZORUNLU kildigi gibi `port=443` ACIKCA belirtilir
    (aksi halde istemci kendi varsayilanina yonelip "Connection refused"
    verir) ve takim koduna ozel yol on-eki (`prefix`) ile REST protokolu
    (gRPC KULLANILMAZ) uzerinden baglanilir.
    """
    if qdrant_config.url == ":memory:":
        return QdrantClient(location=":memory:")
    if not qdrant_config.url.startswith(("http://", "https://")):
        return QdrantClient(path=qdrant_config.url)

    api_key = os.environ.get(qdrant_config.api_key_env, "").strip()
    if not api_key:
        raise ConfigurationError(
            f"Qdrant icin '{qdrant_config.api_key_env}' ortam degiskeni tanimli degil."
        )
    prefix = os.environ.get(qdrant_config.prefix_env, "").strip()
    if not prefix:
        raise ConfigurationError(
            f"Qdrant icin '{qdrant_config.prefix_env}' ortam degiskeni (takim kodu) tanimli degil."
        )
    return QdrantClient(
        url=qdrant_config.url,
        port=443,  # ZORUNLU (bkz. dokustring) - yoksa istemci kendi varsayilanina duser
        prefix=prefix,
        api_key=api_key,
        prefer_grpc=False,  # yalnizca 443 acik; gRPC bu yol on-eki yonlendirmesini desteklemez
        timeout=600,
    )


class EmbeddingRAGService:
    """ISG mevzuati ve operasyonel kurallari EVREN embedding ile vektorlestirip Qdrant'ta arayan, deterministik agirlikli skorlama ile yeniden siralayan servis.

    `Dynamic Tool Router` icindeki `retriever_tool` tarafindan mevzuat/kural
    sorgulari icin kullanilir. Bkz. modul dokustringi icin tam mimari.
    """

    def __init__(
        self,
        embedding_config: EmbeddingConfig,
        qdrant_config: QdrantMemoryConfig,
        reranker_config: Optional[RerankerConfig] = None,
        cross_encoder: Optional[CrossEncoderReranker] = None,
    ) -> None:
        """EmbeddingRAGService'i konfigurasyondan kurar (HICBIR AG CAGRISI YAPMAZ).

        Args:
            embedding_config: `configs/config.yaml` icindeki `memory.embedding` blogu.
            qdrant_config: `configs/config.yaml` icindeki `memory.qdrant` blogu.
            reranker_config: `configs/config.yaml` icindeki `memory.reranker` blogu; `None`/`enabled=False` ise rerank atlanir.
            cross_encoder: (bkz. `local_cross_encoder_reranker.py` modul
                dokustringi + `src/rag/reranker.py::EvrenReranker`) `None`
                ise ucuncu-asama YENIDEN siralama TAMAMEN ATLANIR (yalnizca
                deterministic relevance siralamasi kullanilir - test/izole
                kullanim icin). 2026-08-25 EVREN MIGRASYONU itibariyle
                `src/main.py::SafirPipeline.__init__` VARSAYILAN olarak
                GERCEK bir `EvrenReranker` GECER - EVREN cagrisi basarisiz
                olursa `query()` KONTROLLU sekilde deterministic relevance'a
                duser (bkz. `RagQueryTelemetry.cross_encoder_status`).

        Raises:
            ConfigurationError: `embedding_config.provider` desteklenmiyorsa,
                `output_dimensionality`/`base_url`/`api_key_env` eksikse veya
                Qdrant baglanti bilgisi (API anahtari/takim kodu) eksikse.
        """
        self._embedding_config = embedding_config
        self._qdrant_config = qdrant_config
        self._reranker_config = reranker_config
        self._cross_encoder = cross_encoder

        self._provider: EmbeddingProvider = build_embedding_provider(
            provider=embedding_config.provider,
            model_name=embedding_config.model_name,
            output_dimensionality=embedding_config.output_dimensionality,
            base_url=embedding_config.base_url,
            api_key_env=embedding_config.api_key_env,
            max_batch_tokens=embedding_config.max_batch_tokens,
        )
        self._dimension = self._provider.dimension

        # 2026-08-24 (RAG RERANKER DETERMINIZATION): ikinci-asama relevance
        # skorlamasi bir LLM'e SORULMUYOR - `deterministic_reranker.score_candidate()`
        # TAMAMEN yerel/matematiksel calisir (bkz. modul dokustringi).
        # `reranker_config.enabled=False` ise bu asama TAMAMEN atlanir
        # (embedding_only yol - `_score_and_gate_embedding_only`).
        self._relevance_weights = (
            RelevanceWeights(
                semantic=reranker_config.weights.semantic,
                lexical=reranker_config.weights.lexical,
                keyword=reranker_config.weights.keyword,
                metadata=reranker_config.weights.metadata,
                phrase=reranker_config.weights.phrase,
            )
            if reranker_config is not None
            else RelevanceWeights()
        )

        self._qdrant = _build_qdrant_client(qdrant_config)
        self._collection_name = qdrant_config.collection_name
        self._meta_collection_name = f"{qdrant_config.collection_name}__meta"
        self._last_query_telemetry: Optional[RagQueryTelemetry] = None
        self._corpus_source: str = "unseeded"
        """RAG'in P0 aciklanabilirlik bulgusu: "chunks/metadata diskte var ama
        retrieval sonucu yanlis gorunuyor" semptomunun kok nedeni, retrieval
        KODUNUN kirik olmasi DEGIL - Qdrant koleksiyonu hic olusturulmamissa
        (bkz. `seed_default_regulations`), sistem SESSIZCE yalnizca 8 maddelik
        `DEFAULT_ISG_REGULATIONS` PLACEHOLDER corpus'una duser; gercek
        yuzlerce chunk'lik mevzuat indeksi retrieval'a HIC girmez. Bu alan,
        HANGI corpus'un aktif oldugunu (qdrant_collection | fallback_placeholder
        | chunks_rebuild | unseeded) her `query()` telemetrisine tasiyarak bu
        durumu operator icin GORUNUR kilar (bkz. `RagQueryTelemetry.corpus_source`,
        `trace_serializer.serialize_rag_security`)."""

        logger.info(
            "EmbeddingRAGService baslatildi: embedding_provider=%s model=%s dim=%d qdrant_collection=%s reranker=%s relevance_method=%s",
            embedding_config.provider,
            embedding_config.model_name,
            self._dimension,
            self._collection_name,
            "deterministic" if (reranker_config and reranker_config.enabled) else "devre-disi",
            "weighted_hybrid" if (reranker_config and reranker_config.enabled) else "embedding_only",
        )

    @property
    def dimension(self) -> int:
        """Embedding modelinin urettigi vektor boyutunu dondurur."""
        return self._dimension

    def document_count(self) -> int:
        """Qdrant koleksiyonuna su ana kadar eklenmis dokuman sayisini dondurur (koleksiyon yoksa 0)."""
        try:
            return self._qdrant.count(self._collection_name, exact=True).count
        except Exception:  # noqa: BLE001 - koleksiyon henuz olusturulmamis olabilir (unseeded)
            return 0

    @property
    def relevance_weights(self) -> RelevanceWeights:
        """Bu servisin `score_candidate()`e GERCEKTEN gecirdigi (config'ten okunmus, HARD-CODE
        DEGIL) `RelevanceWeights` - explainability/UI katmaninin, formulun agirliklarini KENDI
        varsayimindan DEGIL, calisan koddan okuyabilmesi icin (bkz. gorev tanimi 8. bolum)."""
        return self._relevance_weights

    @property
    def corpus_source(self) -> str:
        """Aktif corpus'un kaynagi: 'qdrant_collection' | 'fallback_placeholder' | 'chunks_rebuild' | 'unseeded'.

        'fallback_placeholder' degeri geriye-uyum/telemetri semasi icin
        KORUNUR ama `seed_default_regulations()` ARTIK bunu URETMEZ (bkz.
        modul dokustringi) - persisted index yoksa SESSIZCE bu duruma
        DUSULMEZ, `KnowledgeBaseNotBuiltError` firlatilir (fail-fast).
        """
        return self._corpus_source

    def get_last_query_telemetry(self) -> Optional[RagQueryTelemetry]:
        """En son `query()` cagrisinin yapilandirilmis telemetrisini dondurur; hic sorgu yapilmadiysa `None`.

        Dashboard/trace katmani (bkz. `src/main.py::stage_context`) bunu
        DOGRUDAN kullanir - burada olmayan bir alan UYDURULMAZ.
        """
        return self._last_query_telemetry

    def seed_default_regulations(self) -> None:
        """Koleksiyon bossa, YALNIZCA gercek doldurulmus Qdrant koleksiyonunu dogrular - BASKA HICBIR KAYNAGA DUSMEZ.

        Koleksiyon zaten dokuman iceriyorsa hicbir sey yapmaz (idempotent).
        Kaynak (ONEMLI - taze chunk embedding'i BURADA OTOMATIK TETIKLENMEZ,
        cunku bu her pipeline baslangicinda ag/gecikme maliyeti demektir;
        taze embedding yalnizca `build_knowledge_index.py` CLI'inin BILEREK
        cagirdigi `build_index_from_chunks()` ile olur):
          - Qdrant'ta GUNCEL (model/dimension/kb_hash eslesen) bir koleksiyon
            varsa DOGRULANIR (bkz. `_try_load_qdrant_collection`).
          - Yoksa/guncel degilse: `DEFAULT_ISG_REGULATIONS` gibi bir placeholder'a
            SESSIZCE DUSULMEZ - acik, yakalan(mayan)abilir bir
            `KnowledgeBaseNotBuiltError` FIRLATILIR (fail-fast). Cagiran taraf
            (orn. `SafirPipeline.__init__`) bunu yutmaz; sistem gercek bir KB
            index'i olmadan CALISMAZ.

        Raises:
            KnowledgeBaseNotBuiltError: Qdrant'ta GUNCEL bir koleksiyon bulunamadi.
        """
        if self.document_count() > 0:
            logger.info("EmbeddingRAGService zaten %d dokuman iceriyor; seed atlandi.", self.document_count())
            return

        if self._try_load_qdrant_collection():
            return

        raise KnowledgeBaseNotBuiltError(
            f"Qdrant koleksiyonu '{self._collection_name}' icinde GUNCEL bir KB bulunamadi. "
            "Gercek 748 mevzuat chunk'ini EVREN embedding modeliyle indekslemek icin "
            "'python -m src.rag.build_knowledge_index' calistirin. Sessiz bir placeholder'a "
            "DUSULMEDI - bu, RAG'in operator/gelistiriciye GORUNMEDEN 'sahte' verilerle "
            "calismasini ONLER (fail-fast)."
        )

    def _try_load_qdrant_collection(self) -> bool:
        """Qdrant'taki meta-noktayi (bkz. `_QDRANT_META_POINT_ID`), GUNCEL (model/dimension/kb_hash eslesen) ise dogrular.

        Returns:
            Koleksiyon GUNCEL VE dolu ise `True`; meta-koleksiyon/nokta
            yoksa veya GUNCEL DEGILSE (model/dimension/normalization/metric/
            kb_hash uyusmuyorsa) `False` (hicbir sey degistirilmez, cagiran
            taraf `seed_default_regulations()` uzerinden FAIL-FAST eder -
            bkz. `KnowledgeBaseNotBuiltError`, artik SESSIZCE bir
            placeholder'a DUSULMEZ).

        NOT (gorev tanimi 7. madde): model/dimension/normalization/metric
        uyusmazligi HICBIR ZAMAN sessizce kabul EDILMEZ - her biri ayri ayri
        kontrol edilir, herhangi biri uyusmuyorsa koleksiyon GUNCEL DEGIL sayilir.
        """
        try:
            points = self._qdrant.retrieve(self._meta_collection_name, ids=[_QDRANT_META_POINT_ID])
        except Exception:  # noqa: BLE001 - meta-koleksiyon henuz olusturulmamis (unseeded)
            return False
        if not points:
            return False
        meta = points[0].payload or {}

        if meta.get("model_name") != self._embedding_config.model_name:
            logger.warning(
                "Qdrant KB koleksiyonu farkli bir embedding modeliyle uretilmis (%r != %r); index rebuild gerekiyor.",
                meta.get("model_name"),
                self._embedding_config.model_name,
            )
            return False
        if meta.get("dimension") != self._dimension:
            logger.warning(
                "Qdrant KB koleksiyonu farkli boyutta (%r != %r); index rebuild gerekiyor.",
                meta.get("dimension"),
                self._dimension,
            )
            return False
        if meta.get("normalization") != _INDEX_NORMALIZATION:
            logger.warning(
                "Qdrant KB koleksiyonu farkli normalization semasiyla uretilmis (%r != %r); index rebuild gerekiyor.",
                meta.get("normalization"),
                _INDEX_NORMALIZATION,
            )
            return False
        if meta.get("metric") != _INDEX_METRIC:
            logger.warning(
                "Qdrant KB koleksiyonu farkli metric ile uretilmis (%r != %r); index rebuild gerekiyor.",
                meta.get("metric"),
                _INDEX_METRIC,
            )
            return False
        current_hash = _compute_kb_hash()
        if meta.get("kb_hash") != current_hash:
            logger.warning(
                "Qdrant KB koleksiyonu guncel chunk'larla uyusmuyor (kb_hash farkli); "
                "'python -m src.rag.build_knowledge_index' ile yeniden olusturun."
            )
            return False

        if self.document_count() == 0:
            logger.warning("Qdrant KB meta kaydi var ama ana koleksiyon bos; index rebuild gerekiyor.")
            return False

        self._corpus_source = "qdrant_collection"
        logger.info(
            "EmbeddingRAGService: Qdrant KB koleksiyonu dogrulandi (%d dokuman, model=%s, kb_version=%s).",
            self.document_count(),
            meta.get("model_name"),
            str(meta.get("kb_hash", ""))[:12],
        )
        return True

    def build_index_from_chunks(self) -> int:
        """TUM `data/knowledge_base/chunks/*.json` kayitlarini `sources.yaml` ile join edip embed eder (rebuild CLI'i icin).

        Bu metod idempotency GUARD'INI ATLAR (mevcut dokumanlarin ustune
        EKLER/GUNCELLER) - yalnizca `scripts`/`build_knowledge_index.py` gibi
        BILEREK taze bir `EmbeddingRAGService` orneginde cagrilmasi
        beklenir. KB chunk URETIM MANTIGI (748 kaynak kayit,
        `scripts/build_kb_chunks.py`) DEGISMEZ; `_add_structured_documents`
        icinde, embed'lemeden ONCE, GUVENLI token butcesini TEK BASINA asan
        (nadir) kayitlar semantik/metin sinirlarina gore alt-parcalara
        bolunur (bkz. `_split_oversized_records`) - bu yuzden donen
        deger/`document_count()`, kaynak 748 kayit sayisini ASABILIR; bu
        BEKLENEN bir davranistir.

        Returns:
            Qdrant'a GERCEKTEN yazilan (bolme SONRASI) nihai dokuman sayisi.

        Raises:
            RuntimeError: `data/knowledge_base/chunks/` altinda hicbir chunk bulunamazsa.
        """
        records = _load_kb_chunk_records()
        if not records:
            raise RuntimeError(
                f"{_KB_CHUNKS_DIR} altinda hicbir chunk bulunamadi. Once "
                "'python scripts/build_kb_chunks.py' calistirip chunk'lari uretin."
            )
        self._ensure_collection_exists()
        added = self._add_structured_documents(records)
        self._corpus_source = "chunks_rebuild"
        return added

    def _ensure_collection_exists(self) -> None:
        """Ana Qdrant koleksiyonunu (yoksa) `self._dimension`/`cosine` ile olusturur (idempotent)."""
        if not self._qdrant.collection_exists(self._collection_name):
            self._qdrant.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
            )

    @staticmethod
    def _point_id_for(document_id: str, chunk_id: Optional[str], index: int) -> str:
        """`(document_id, chunk_id, index)` ucundan DETERMINISTIK bir Qdrant nokta ID'si (UUID5) turetir.

        Ayni chunk'in tekrar indekslenmesi (rebuild) AYNI ID'yi uretir -
        boylece `upsert` DUPLICATE nokta biriktirmez, mevcut noktayi GUNCELLER.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"safir-kb:{document_id}:{chunk_id}:{index}"))

    def _add_structured_documents(self, records: List[Dict[str, Any]]) -> int:
        """Yapilandirilmis kayitlari (metadata + text) embed edip Qdrant koleksiyonuna upsert eder.

        Embed'lemeden ONCE, GUVENLI token butcesini TEK BASINA asan
        (nadir) kayitlar semantik/metin sinirlarina gore alt-parcalara
        bolunur (bkz. `_split_oversized_records`) - metadata/source/
        document/chunk iliskileri KORUNUR, hicbir metin SESSIZCE
        KIRPILMAZ. Bu yuzden Qdrant'a yazilan nihai nokta sayisi,
        `records`in ozgun uzunlugunu ASABILIR.

        Returns:
            Qdrant'a GERCEKTEN yazilan (bolme SONRASI) nihai kayit sayisi.
        """
        if not records:
            return 0
        self._ensure_collection_exists()
        safe_token_budget = getattr(self._provider, "safe_token_budget", _DEFAULT_SAFE_TOKEN_BUDGET)
        expanded_records = _split_oversized_records(records, safe_token_budget)
        texts = [str(r.get("text") or "") for r in expanded_records]
        vectors = self._provider.embed_documents(texts)
        points = [
            PointStruct(
                id=self._point_id_for(str(r.get("document_id") or ""), r.get("chunk_id"), i),
                vector=vectors[i].tolist(),
                payload=r,
            )
            for i, r in enumerate(expanded_records)
        ]
        self._qdrant.upsert(collection_name=self._collection_name, points=points)
        logger.info(
            "EmbeddingRAGService: %d kaynak kayit -> %d dokuman indekslendi (toplam=%d)",
            len(records),
            len(expanded_records),
            self.document_count(),
        )
        return len(expanded_records)

    def add_document(self, text: str) -> None:
        """Bir kural/mevzuat metnini (metadata'siz, DUZ METIN) anlamsal bellege ekler.

        Args:
            text: Eklenecek dokuman icerigi (orn. bir ISG maddesi).
        """
        self.add_documents([text])

    def add_documents(self, documents: List[str]) -> None:
        """Birden fazla DUZ METIN dokumani (metadata'siz - geriye-uyum/fallback yolu) toplu olarak embed'leyip FAISS indeksine ekler.

        Yapilandirilmis (metadata'li) chunk eklemek icin `build_index_from_chunks()`/
        `_add_structured_documents()` kullanin.

        Args:
            documents: Eklenecek dokuman metinleri listesi.
        """
        self._add_structured_documents([{"text": d} for d in documents])

    def query(
        self, question: str, top_k: Optional[int] = None, keywords: Optional[List[str]] = None
    ) -> List[RetrievedDocument]:
        """Verilen soruya en yakin dokumanlari, iki-asamali retrieval (Qdrant candidate_k -> DETERMINISTIK relevance skorlama -> threshold) ile dondurur.

        2026-08-24 (RAG RERANKER DETERMINIZATION): ikinci asama bir LLM'e
        SORULMUYOR - `deterministic_reranker.score_candidate()` ile TAMAMEN
        yerel/matematiksel hesaplanir (bkz. o modulun dokustringi). Bu,
        HICBIR AG CAGRISI/API anahtari/kota GEREKTIRMEZ.

        Akis:
          1. Qdrant'tan `candidate_k` aday cekilir (embedding benzerligi).
          2. Relevance skorlama AKTIFSE: her adayin `relevance_score`u
             (semantic+lexical+keyword+metadata+phrase agirlikli toplami)
             hesaplanir; `score_threshold`in ALTINDA kalanlar ELENIR (bkz.
             `RerankerConfig.score_threshold`) - eslesme YOKSA BOS LISTE
             doner (`retrieval_status="insufficient_evidence"`), bu GECERLI
             bir sonuctur, rastgele/dusuk-alakali sonuc UYDURULMAZ.
          3. Relevance skorlama DEVRE DISIYSA: adaylar yalnizca embedding
             skoruna gore siralanir (`similarity_threshold` varsa uygulanir),
             ilk `top_k` dondurulur (`retrieval_status="embedding_only"`).

        NOT: gercek bir TEKNIK retrieval hatasi (orn. embedding modeli/Qdrant
        baglantisi basarisiz) BU METODUN sorumlulugunda DEGILDIR - o durumda
        istisna oldugu gibi YUKARI firlatilir (cagiran taraf ele alir); bu,
        relevance skorlamasinin "basarisizligi" ile KARISTIRILMAZ.

        Args:
            question: Dogal dil sorgusu.
            top_k: NIHAI (skorlama/filtre SONRASI) sonuc sayisi; verilmezse
                `qdrant_config.top_k`/`reranker_config.top_k` kullanilir.
            keywords: VLM'in dinamik risk keyword'leri (varsa) - keyword_score
                sinyaline girdi olur; `None`/bos ise bu sinyal SESSIZCE 0
                katkı verir (sistem BOZULMAZ).

        Returns:
            `relevance_score`e gore azalan sirali `RetrievedDocument` listesi;
            hicbir esik-uzeri sonuc yoksa BOS LISTE.
        """
        query_started = time.perf_counter()
        document_count = self.document_count()
        if document_count == 0:
            logger.warning("EmbeddingRAGService bos; sorgu icin dokuman bulunamadi.")
            self._last_query_telemetry = RagQueryTelemetry(
                query=question,
                candidate_count=0,
                final_count=0,
                zero_result=True,
                retrieval_status="empty_index",
                threshold=self._reranker_config.score_threshold if self._reranker_config else None,
                embedding_latency_ms=0.0,
                rerank_latency_ms=None,
                total_latency_ms=round((time.perf_counter() - query_started) * 1000.0, 1),
                avg_embedding_score=None,
                avg_relevance_score=None,
                corpus_source=self._corpus_source,
                results=[],
            )
            return []

        final_k = top_k or (self._reranker_config.top_k if self._reranker_config else None) or self._qdrant_config.top_k
        candidate_k = max(self._qdrant_config.candidate_k, final_k)
        candidate_k = min(candidate_k, document_count)

        embedding_started = time.perf_counter()
        query_vector = self._provider.embed_query(question)
        hits = self._qdrant.query_points(
            collection_name=self._collection_name, query=query_vector.tolist(), limit=candidate_k
        ).points
        embedding_latency_ms = (time.perf_counter() - embedding_started) * 1000.0

        candidates: List[RetrievedDocument] = []
        for rank, point in enumerate(hits, start=1):
            record = point.payload or {}
            candidates.append(
                RetrievedDocument(
                    text=record.get("text", ""),
                    embedding_score=float(point.score),
                    chunk_id=record.get("chunk_id"),
                    document_id=record.get("document_id"),
                    document_title=record.get("document_title"),
                    level=record.get("level"),
                    article_number=record.get("article_number"),
                    article_title=record.get("article_title"),
                    is_annex=record.get("is_annex"),
                    page_start=record.get("page_start"),
                    page_end=record.get("page_end"),
                    source_url=record.get("source_url"),
                    institution=record.get("institution"),
                    publication_date=record.get("publication_date"),
                    retrieval_rank=rank,  # Qdrant zaten embedding_score'a gore AZALAN sirada doner
                )
            )

        threshold = self._reranker_config.score_threshold if self._reranker_config else None
        scoring_started = time.perf_counter()

        if self._reranker_config is not None and self._reranker_config.enabled:
            # Deterministik relevance skorlama - HICBIR AG/LLM CAGRISI YOK
            # (bkz. `deterministic_reranker.py`). Her adaya (final sonuca
            # girsin girmesin) `relevance_score`/`relevance_status`/
            # `relevance_reason` damgalanir - "neden secildi/elendi?"
            # sorusu HER ZAMAN, AYNI girdide AYNI cevapla yanitlanabilir.
            for doc in candidates:
                breakdown = score_candidate(
                    query=question,
                    chunk_text=doc.text,
                    embedding_score=doc.embedding_score,
                    document_title=doc.document_title,
                    article_number=doc.article_number,
                    keywords=keywords,
                    weights=self._relevance_weights,
                )
                doc.relevance_score = breakdown.relevance_score
                doc.semantic_score = breakdown.semantic_score
                doc.lexical_score = breakdown.lexical_score
                doc.keyword_score = breakdown.keyword_score
                doc.metadata_score = breakdown.metadata_score
                doc.phrase_score = breakdown.phrase_score
                if threshold is not None and breakdown.relevance_score < threshold:
                    doc.relevance_status = "rejected"
                    doc.relevance_reason = f"{breakdown.reason()} | threshold={threshold:.3f} (ALTINDA)"
                else:
                    doc.relevance_status = "accepted"
                    doc.relevance_reason = (
                        f"{breakdown.reason()} | threshold={threshold:.3f} (UZERINDE/ESIT)"
                        if threshold is not None
                        else f"{breakdown.reason()} | threshold tanimli degil"
                    )

            final_docs = sorted(
                (d for d in candidates if d.relevance_status == "accepted"),
                key=lambda d: d.relevance_score,
                reverse=True,
            )
            retrieval_status = "relevance_scored" if final_docs else "insufficient_evidence"
        else:
            # Relevance skorlama DEVRE DISI: adaylar yalnizca embedding skoruna
            # gore siralanir (FAISS zaten azalan sirada dondurmustu). ONCE
            # `similarity_threshold` (varsa) BASIT bir gate olarak uygulanir -
            # bu ONCEDEN dokumante edilmis ama HIC UYGULANMAMIS bir davranistiydi
            # ("corpus disi" bir sorgu FAISS'ten HER ZAMAN "en az kotu" adaylari
            # dondurur - bu esik OLMADAN, skorlama devre disiyken tamamen
            # alakasiz bir sorgu bile SESSIZCE "accepted" sayilirdi).
            similarity_threshold = self._qdrant_config.similarity_threshold
            embedding_only_final: List[RetrievedDocument] = []
            for doc in candidates:
                if similarity_threshold is not None and doc.embedding_score < similarity_threshold:
                    doc.relevance_status = "rejected"
                    doc.relevance_reason = (
                        f"embedding_score ({doc.embedding_score:.3f}) < similarity_threshold ({similarity_threshold:.3f})"
                    )
                    continue
                if len(embedding_only_final) < final_k:
                    doc.relevance_status = "accepted"
                    doc.relevance_reason = f"embedding top-{final_k} icinde (relevance skorlama devre disi)"
                    embedding_only_final.append(doc)
                else:
                    doc.relevance_status = "rejected"
                    doc.relevance_reason = f"embedding top-{final_k} disinda (relevance skorlama devre disi)"
            final_docs = embedding_only_final
            retrieval_status = "embedding_only"

        # [LOCAL CROSS-ENCODER EXTENSION POINT] (bkz. `local_cross_encoder_reranker.py`
        # modul dokustringi) - `self._cross_encoder` yalnizca cagiran taraf BILEREK bir
        # `LocalCrossEncoderReranker` GECTIYSE calisir. Deterministic relevance/evidence
        # gate'ten GECMIS `final_docs` (top-N, zaten "accepted") uzerinde calisir,
        # final_k'ya kirpilmadan ONCE - gate'i BYPASS ETMEZ, yalnizca ZATEN kabul
        # edilmis adaylari YENIDEN SIRALAR (gorev tanimi 5. bolum).
        cross_encoder_status = "disabled"
        if self._cross_encoder is not None and final_docs:
            try:
                ce_scores = self._cross_encoder.score(question, [d.text for d in final_docs])
                for doc, ce_score in zip(final_docs, ce_scores):
                    doc.cross_encoder_score = float(ce_score)
                final_docs = sorted(final_docs, key=lambda d: d.cross_encoder_score, reverse=True)
                cross_encoder_status = "used"
            except CrossEncoderUnavailableError:
                # KONTROLLU DEGRADASYON (gorev tanimi 12. bolum): lokal model
                # agirligi yuklenemedi (paket eksik/indirilemedi) - HARICI BIR
                # API'YE ASLA DUSULMEZ, sessizce de degil: durum ACIKCA
                # telemetriye yazilir, siralama deterministic relevance'ta KALIR.
                logger.warning(
                    "Lokal Cross-Encoder kullanilamiyor (model agirligi yuklenemedi); "
                    "siralama deterministic relevance skoruna gore devam ediyor.",
                    exc_info=True,
                )
                cross_encoder_status = "unavailable"
        elif self._cross_encoder is not None:
            cross_encoder_status = "used"  # Cross-Encoder verildi ama sirlanacak aday yoktu (bos final_docs) - hata degil.

        for rank, doc in enumerate(final_docs, start=1):
            doc.final_rank = rank

        scoring_latency_ms = (time.perf_counter() - scoring_started) * 1000.0
        final_docs = final_docs[:final_k]
        self._log_retrieval_trace(question, candidates, final_docs, retrieval_status=retrieval_status, final_k=final_k)

        final_ids = {id(d) for d in final_docs}
        self._last_query_telemetry = RagQueryTelemetry(
            query=question,
            candidate_count=len(candidates),
            final_count=len(final_docs),
            zero_result=len(final_docs) == 0,
            retrieval_status=retrieval_status,
            threshold=threshold if retrieval_status in ("relevance_scored", "insufficient_evidence") else None,
            embedding_latency_ms=round(embedding_latency_ms, 1),
            rerank_latency_ms=round(scoring_latency_ms, 1),
            total_latency_ms=round((time.perf_counter() - query_started) * 1000.0, 1),
            avg_embedding_score=_avg([c.embedding_score for c in candidates]),
            avg_relevance_score=_avg([d.relevance_score for d in final_docs if d.relevance_score is not None]),
            corpus_source=self._corpus_source,
            cross_encoder_status=cross_encoder_status,
            results=[_result_telemetry(c, selected=id(c) in final_ids) for c in candidates],
        )
        return final_docs

    def _log_retrieval_trace(
        self,
        question: str,
        candidates: List[RetrievedDocument],
        final_docs: List[RetrievedDocument],
        retrieval_status: str,
        final_k: int,
    ) -> None:
        """RAG cagrisinin tani/izleme (trace) bilgisini yapilandirilmis sekilde loglar (API anahtari/secret ASLA loglanmaz)."""
        if self._corpus_source == "fallback_placeholder":
            logger.warning(
                "RAG retrieval: query=%r corpus_source=fallback_placeholder - sonuclar GERCEK mevzuat "
                "corpus'undan DEGIL, 8 maddelik placeholder listeden geliyor ('python -m "
                "src.rag.build_knowledge_index' hic calistirilmamis/GUNCEL DEGIL).",
                question,
            )
        logger.info(
            "RAG retrieval: query=%r corpus_source=%s embedding_model=%s candidate_count=%d "
            "reranker=deterministic relevance_method=weighted_hybrid "
            "final_count=%d threshold=%s retrieval_status=%s "
            "embedding_scores=%s relevance_scores=%s sources=%s",
            question,
            self._corpus_source,
            self._embedding_config.model_name,
            len(candidates),
            len(final_docs),
            self._reranker_config.score_threshold if self._reranker_config else None,
            retrieval_status,
            [round(c.embedding_score, 4) for c in candidates],
            [round(d.relevance_score, 4) for d in final_docs if d.relevance_score is not None],
            [
                {"chunk_id": d.chunk_id, "document_id": d.document_id, "article_number": d.article_number, "source_url": d.source_url}
                for d in final_docs
            ],
        )

    def search_laws(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Modul 3 spesifikasyonundaki isim: `query()` metoduna dogrudan devreder.

        Args:
            query: Dogal dil sorgusu (orn. bir ISG mevzuat sorusu).
            top_k: Dondurulecek maksimum sonuc sayisi; verilmezse config degeri kullanilir.

        Returns:
            `query(query, top_k)` ile aynı sonuc.
        """
        return self.query(query, top_k)

    def persist(self) -> None:
        """Index manifest'ini (model/dimension/normalization/metric/corpus fingerprint) Qdrant meta-koleksiyonuna yazar.

        Qdrant koleksiyonunun kendisi (`_add_structured_documents`/`upsert`
        cagrilarinda) zaten UZAKTAN kalicidir - bu metod yalnizca manifest
        noktasini (bkz. `_QDRANT_META_POINT_ID`) yazar; runtime'da
        `_try_load_qdrant_collection()` tarafindan model/dimension/
        normalization/metric/kb_hash UYUSMAZLIGINI SESSIZCE KABUL ETMEMEK
        icin kullanilir (bkz. gorev tanimi 6/7. madde).
        """
        kb_hash = _compute_kb_hash()
        chunk_count = self.document_count()
        meta = {
            "model_name": self._embedding_config.model_name,
            "dimension": self._dimension,
            "normalization": _INDEX_NORMALIZATION,
            "metric": _INDEX_METRIC,
            "kb_hash": kb_hash,
            "corpus_fingerprint": kb_hash,  # kb_hash ile AYNI deger - manifest'te ACIK isimle de bulunsun diye
            "chunk_count": chunk_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if not self._qdrant.collection_exists(self._meta_collection_name):
            self._qdrant.create_collection(
                collection_name=self._meta_collection_name,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )
        self._qdrant.upsert(
            collection_name=self._meta_collection_name,
            points=[PointStruct(id=_QDRANT_META_POINT_ID, vector=[0.0], payload=meta)],
        )
        logger.info("KB index manifest persisted: %s (%d dokuman)", self._meta_collection_name, chunk_count)


class MockEmbeddingRAGService:
    """Ag/GPU-bagimsiz sahte RAG servisi - `app.use_mock_vlm` + `app.use_mock_llm`
    ikisi de `true` iken `SafirPipeline.__init__` tarafindan gercek
    `EmbeddingRAGService` YERINE kullanilir (bkz. `src/main.py`).

    KOK NEDEN (duzeltilen bug): gercek `EmbeddingRAGService`, KURULUSTA
    (`__init__`) her zaman gercek bir Qdrant istemcisi kurar - bu,
    `use_mock_vlm`/`use_mock_llm` `true` olsa BILE degismiyordu, cunku RAG
    hicbir mock bayragiyla GECILMIYORDU. Sonuc: `configs/config.yaml`'in
    GELISTIRME/TEST yorumundaki ("harici hicbir sey gerekmez") vaadin aksine,
    yalnizca mock modda calistirmak bile `EVREN_QDRANT_KEY` olmadan
    `ConfigurationError` ile pipeline kurulusunda PATLIYORDU.

    Bu sinif, cagiran kodun (ContextBuilder/tools.py/main.py) beklegi asgari
    sozlesmeyi (`seed_default_regulations`, `query`, `get_last_query_telemetry`,
    `relevance_weights`) HICBIR ag/Qdrant/embedding modeli olmadan, basit bir
    alt-dizge (substring) eslesmesiyle `DEFAULT_ISG_REGULATIONS` (8 maddelik
    placeholder) uzerinde karsilar - gercek servisin retrieval KALITESINI
    TAKLIT ETMEYE calismaz, yalnizca ajanin/rapor akisinin GPU'suz/anahtarsiz
    uctan uca calismasini saglar.
    """

    def __init__(self) -> None:
        self._relevance_weights = RelevanceWeights()
        self._last_query_telemetry: Optional[RagQueryTelemetry] = None

    @property
    def relevance_weights(self) -> RelevanceWeights:
        return self._relevance_weights

    def seed_default_regulations(self) -> None:
        return None

    def query(
        self, question: str, top_k: Optional[int] = None, keywords: Optional[List[str]] = None
    ) -> List[RetrievedDocument]:
        """Basit alt-dizge eslesmesiyle `DEFAULT_ISG_REGULATIONS`den en alakali maddeleri dondurur.

        Gercek servisin embedding/Qdrant/deterministic-reranker zincirinin
        YERINE GECMEZ - yalnizca (soru + keywords) icindeki 4+ karakterlik
        kelimelerden kacinin ilgili madde metninde GECTIGINI sayar; hic
        eslesme yoksa BOS LISTE doner (gercek serviste oldugu gibi, rastgele
        bir sonuc UYDURULMAZ).
        """
        started = time.perf_counter()
        limit = top_k or 3
        needle_words = {
            w for term in ([question] + list(keywords or [])) if term for w in term.lower().split() if len(w) > 3
        }
        scored = []
        for idx, text in enumerate(DEFAULT_ISG_REGULATIONS):
            text_lower = text.lower()
            hits = sum(1 for w in needle_words if w in text_lower)
            if hits:
                scored.append((hits, idx, text))
        scored.sort(key=lambda t: (-t[0], t[1]))

        docs = [
            RetrievedDocument(
                text=text,
                embedding_score=1.0,
                relevance_score=1.0,
                document_title="[MOCK] ISG Mevzuati (placeholder - use_mock_vlm+use_mock_llm)",
                source_verified=False,
                relevance_status="accepted",
                relevance_reason="mock alt-dizge eslesmesi (gercek retrieval degil)",
            )
            for _hits, _idx, text in scored[:limit]
        ]
        total_latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        self._last_query_telemetry = RagQueryTelemetry(
            query=question,
            candidate_count=len(DEFAULT_ISG_REGULATIONS),
            final_count=len(docs),
            zero_result=len(docs) == 0,
            retrieval_status="relevance_scored" if docs else "insufficient_evidence",
            threshold=None,
            embedding_latency_ms=0.0,
            rerank_latency_ms=0.0,
            total_latency_ms=total_latency_ms,
            avg_embedding_score=1.0 if docs else None,
            avg_relevance_score=1.0 if docs else None,
            corpus_source="fallback_placeholder",
            cross_encoder_status="disabled",
            results=[],
        )
        return docs

    def search_laws(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        return self.query(query, top_k)

    def get_last_query_telemetry(self) -> Optional[RagQueryTelemetry]:
        return self._last_query_telemetry


# Modul 3 spesifikasyonundaki isim: ayni sinifa isaret eden alias (geriye
# donuk uyumluluk icin `EmbeddingRAGService` adi da tum cagiran kodda aynen
# kullanilmaya devam eder).
FAISSRagService = EmbeddingRAGService


if __name__ == "__main__":
    # Modul 3'un bagimsiz calistirilabilirlik testi:
    #   python -m src.rag.embedding_rag_service
    from src.utils.config_loader import load_config

    logging.basicConfig(level=logging.INFO)

    demo_config = load_config()
    demo_service = FAISSRagService(demo_config.memory.embedding, demo_config.memory.qdrant, demo_config.memory.reranker)
    demo_service.seed_default_regulations()

    demo_query = "Yuksekte calisirken hangi ekipmanlar zorunlu?"
    print(f"Sorgu: {demo_query}\n")
    for i, result in enumerate(demo_service.search_laws(demo_query, top_k=3), start=1):
        print(f"{i}. (skor={result.score:.4f}) {result.text}")
