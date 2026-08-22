"""04 - Embedding & RAG Katmani: Gemini embedding + FAISS + Gemini rerank tabanli anlamsal bellek.

Operasyonel kurallari ve ISG mevzuatini Google Gemini Embedding API ile
vektorlestirip FAISS uzerinde saklayan/arayan servistir. LangGraph ajaninin
`retriever_tool` araci ve `RuleEngine._describe_regulation()` bu servis
uzerinden calisir.

Knowledge base kaynagi (2026-08-22, iki asamali guncelleme)
--------------------------------------------------------------
1. asama: `seed_default_regulations()` `data/knowledge_base/chunks/*.json`
   altindaki GERCEK, resmi mevzuat metinlerinden turetilmis madde-bazli
   chunk'lari (bkz. `scripts/build_kb_chunks.py`) yukledi.
2. asama (bu dosya): Embedding saglayicisi `sentence-transformers`den
   Gemini Embedding API'ye (bkz. `embedding_providers.py`) tasindi;
   `RetrievedDocument` artik yapilandirilmis metadata (document_title,
   article_number, source_url, ...) tasiyor (`sources.yaml` ile join
   edilmis - bkz. `_load_kb_chunk_records`); iki-asamali retrieval
   (FAISS candidate_k -> Gemini rerank -> score_threshold) eklendi; index
   artik DISKE KALICI olarak yaziliyor (`data/knowledge_base/index/`) ve
   pipeline her baslangicta 748 embedding'i YENIDEN URETMEK ZORUNDA
   DEGIL - bkz. `build_knowledge_index.py` (rebuild CLI'i) ve
   `_try_load_persisted_index()`.

`DEFAULT_ISG_REGULATIONS` (8 ORNEK/placeholder madde), persisted index de
gercek KB chunk'lari da bulunamazsa GERIYE-DONUK UYUMLULUK icin fallback
olarak KALIR - tamamen SILINMEDI (bkz. gorev tanimi).

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

MIMARI AYRIM (ONEMLI): Bu serviste uretilen `embedding_score`/`rerank_score`
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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from src.memory.embedding_providers import ConfigurationError, EmbeddingProvider, build_embedding_provider
from src.memory.reranker import GeminiReranker, Reranker, RerankerUnavailableError
from src.utils.config_loader import EmbeddingConfig, FaissMemoryConfig, RerankerConfig

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
"""Persisted index de gercek KB chunk'lari da bulunamazsa kullanilan GERIYE-DONUK
UYUMLULUK fallback'i (orn. `src/agent/agent_workflow.py`nin mock ornekleri hala
bunu kullanir). Canli sistemde artik BIRINCIL kaynak DEGILDIR - bkz. modul dokustringi."""

_KB_ROOT = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"
_KB_CHUNKS_DIR = _KB_ROOT / "chunks"
_KB_SOURCES_YAML = _KB_ROOT / "metadata" / "sources.yaml"
_KB_INDEX_DIR = _KB_ROOT / "index"
_INDEX_FILE = _KB_INDEX_DIR / "faiss.index"
_DOCUMENTS_FILE = _KB_INDEX_DIR / "documents.json"
_INDEX_META_FILE = _KB_INDEX_DIR / "index_meta.json"

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


def _compute_kb_hash(chunks_dir: Path = _KB_CHUNKS_DIR) -> str:
    """Chunk klasorunun icerik hash'ini hesaplar (persisted index'in GUNCEL olup olmadigini anlamak icin)."""
    if not chunks_dir.exists():
        return "no-chunks-dir"
    hasher = hashlib.sha256()
    for chunk_file in sorted(chunks_dir.glob("*.json")):
        hasher.update(chunk_file.name.encode("utf-8"))
        hasher.update(chunk_file.read_bytes())
    return hasher.hexdigest()


@dataclass
class RetrievedDocument:
    """FAISS(+opsiyonel Gemini rerank)ten geri getirilen tek bir chunk'i, yapilandirilmis kaynak metadata'siyla birlikte tasir.

    `embedding_score` (FAISS/cosine benzerligi) ile `rerank_score` (Gemini
    relevance_score, reranker devre disiysa `None`) KASITLI olarak AYRI
    alanlardir - birbirine KARISTIRILMAZ (bkz. modul dokustringi).
    """

    text: str
    embedding_score: float
    rerank_score: Optional[float] = None
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

    @property
    def score(self) -> float:
        """Geriye-uyumluluk: eski `{text, score}` sozlesmesine bagli cagiran kod icin.

        Rerank yapildiysa `rerank_score`, yapilmadiysa `embedding_score`
        dondurur - "en guvenilir elimizdeki skor" anlaminda, ama bu ikisini
        BIRBIRINE KARISTIRMAZ (asil kod `embedding_score`/`rerank_score`e
        AYRI AYRI erismelidir).
        """
        return self.rerank_score if self.rerank_score is not None else self.embedding_score


def _avg(values: List[float]) -> Optional[float]:
    """Bos listede `None` dondurur (0.0 DEGIL - "hic sonuc yok" ile "ortalama 0" KARISTIRILMAZ)."""
    return round(sum(values) / len(values), 4) if values else None


def _result_telemetry(doc: "RetrievedDocument", selected: bool) -> "RagResultTelemetry":
    return RagResultTelemetry(
        document_id=doc.document_id,
        document_title=doc.document_title,
        article_number=doc.article_number,
        source_url=doc.source_url,
        embedding_score=round(doc.embedding_score, 4),
        rerank_score=round(doc.rerank_score, 4) if doc.rerank_score is not None else None,
        selected=selected,
    )


@dataclass
class RagResultTelemetry:
    """Tek bir RAG adayinin/sonucunun DASHBOARD-guvenli (metin ICERMEYEN) telemetri kaydi.

    `text` KASITLI OLARAK burada YOKTUR - dashboard'a yalnizca yapilandirilmis
    kaynak metadata'si ve skorlar gider, tam mevzuat metni degil (bkz. RAG +
    Security Observability gorev tanimi, bolum 3/14).
    """

    document_id: Optional[str]
    document_title: Optional[str]
    article_number: Optional[str]
    source_url: Optional[str]
    embedding_score: float
    rerank_score: Optional[float]
    selected: bool
    """Threshold/top_k sonrasi NIHAI sonuc kumesine girdi mi (final_docs icinde mi)."""


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
    retrieval_status: str  # "reranked" | "embedding_only" | "reranker_unavailable"
    threshold: Optional[float]
    embedding_latency_ms: float
    rerank_latency_ms: Optional[float]
    total_latency_ms: float
    avg_embedding_score: Optional[float]
    avg_rerank_score: Optional[float]
    results: List[RagResultTelemetry] = field(default_factory=list)


class EmbeddingRAGService:
    """ISG mevzuati ve operasyonel kurallari Gemini embedding ile vektorlestirip FAISS'te arayan, Gemini ile yeniden siralayan servis.

    `Dynamic Tool Router` icindeki `retriever_tool` tarafindan mevzuat/kural
    sorgulari icin kullanilir. Bkz. modul dokustringi icin tam mimari.
    """

    def __init__(
        self,
        embedding_config: EmbeddingConfig,
        faiss_config: FaissMemoryConfig,
        reranker_config: Optional[RerankerConfig] = None,
    ) -> None:
        """EmbeddingRAGService'i konfigurasyondan kurar (HICBIR AG CAGRISI YAPMAZ).

        Args:
            embedding_config: `configs/config.yaml` icindeki `memory.embedding` blogu.
            faiss_config: `configs/config.yaml` icindeki `memory.faiss` blogu.
            reranker_config: `configs/config.yaml` icindeki `memory.reranker` blogu; `None`/`enabled=False` ise rerank atlanir.

        Raises:
            ConfigurationError: `embedding_config.provider` desteklenmiyorsa
                veya `output_dimensionality` eksikse. API ANAHTARI eksikligi
                BURADA degil, ilk gercek embed/rerank cagrisinda firlatilir.
        """
        self._embedding_config = embedding_config
        self._faiss_config = faiss_config
        self._reranker_config = reranker_config

        self._provider: EmbeddingProvider = build_embedding_provider(
            provider=embedding_config.provider,
            model_name=embedding_config.model_name,
            output_dimensionality=embedding_config.output_dimensionality,
            api_key_env=embedding_config.api_key_env,
        )
        self._dimension = self._provider.dimension

        self._reranker: Optional[Reranker] = None
        if reranker_config is not None and reranker_config.enabled:
            if reranker_config.provider != "gemini":
                raise ConfigurationError(
                    f"Desteklenmeyen reranker saglayicisi: '{reranker_config.provider}'. Su an yalnizca 'gemini' destekleniyor."
                )
            self._reranker = GeminiReranker(
                model_name=reranker_config.model_name, api_key_env=reranker_config.api_key_env
            )

        self._index = faiss.IndexFlatIP(self._dimension)
        self._documents: List[Dict[str, Any]] = []
        self._last_query_telemetry: Optional[RagQueryTelemetry] = None

        logger.info(
            "EmbeddingRAGService baslatildi: embedding_provider=%s model=%s dim=%d reranker=%s",
            embedding_config.provider,
            embedding_config.model_name,
            self._dimension,
            reranker_config.model_name if (reranker_config and reranker_config.enabled) else "devre-disi",
        )

    @property
    def dimension(self) -> int:
        """Embedding modelinin urettigi vektor boyutunu dondurur."""
        return self._dimension

    def document_count(self) -> int:
        """FAISS indeksine su ana kadar eklenmis dokuman sayisini dondurur."""
        return len(self._documents)

    def get_last_query_telemetry(self) -> Optional[RagQueryTelemetry]:
        """En son `query()` cagrisinin yapilandirilmis telemetrisini dondurur; hic sorgu yapilmadiysa `None`.

        Dashboard/trace katmani (bkz. `src/main.py::stage_context`) bunu
        DOGRUDAN kullanir - burada olmayan bir alan UYDURULMAZ.
        """
        return self._last_query_telemetry

    def seed_default_regulations(self) -> None:
        """Indeks bossa knowledge base'i yukler: ONCELIKLE persisted index, sonra taze chunk embedding'i, son care DEFAULT_ISG_REGULATIONS.

        Indeks zaten dokuman iceriyorsa hicbir sey yapmaz (idempotent).
        Kaynak sirasi (ONEMLI - maliyet/performans icin):
          1. `data/knowledge_base/index/` altinda GUNCEL (model/dimension/kb_hash
             eslesen) bir persisted index varsa, hicbir API cagrisi yapmadan
             YUKLENIR (bkz. `_try_load_persisted_index`).
          2. Yoksa/guncel degilse: acik bir uyari loglanir ("python -m
             src.memory.build_knowledge_index calistirin") ve GERIYE-DONUK
             UYUMLULUK icin yalnizca `DEFAULT_ISG_REGULATIONS` (8 kisa
             placeholder madde) embed edilir - 748 chunk'in TAMAMI HER
             pipeline baslangicinda YENIDEN embed EDILMEZ (bkz. gorev
             tanimi 15. bolum).
        """
        if self.document_count() > 0:
            logger.info("EmbeddingRAGService zaten %d dokuman iceriyor; seed atlandi.", self.document_count())
            return

        if self._try_load_persisted_index():
            return

        logger.warning(
            "EmbeddingRAGService: %s altinda GUNCEL bir persisted KB index'i bulunamadi. "
            "Gercek 748 mevzuat chunk'ini kullanmak icin 'python -m src.memory.build_knowledge_index' "
            "calistirin. Su an icin GERIYE-DONUK UYUMLULUK amacli %d placeholder ISG maddesine dusuluyor.",
            _KB_INDEX_DIR,
            len(DEFAULT_ISG_REGULATIONS),
        )
        self.add_documents(DEFAULT_ISG_REGULATIONS)

    def _try_load_persisted_index(self) -> bool:
        """Diskteki persisted FAISS index + metadata'yi, GUNCEL (model/dimension/kb_hash eslesen) ise yukler.

        Returns:
            Basariyla yuklendiyse `True`; dosyalar yoksa, bozuksa veya
            GUNCEL DEGILSE (model/dimension/kb_hash uyusmuyorsa) `False`
            (hicbir sey degistirilmez, cagiran taraf fallback'e duser).
        """
        if not (_INDEX_FILE.exists() and _DOCUMENTS_FILE.exists() and _INDEX_META_FILE.exists()):
            return False

        try:
            meta = json.loads(_INDEX_META_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Persisted KB index_meta.json okunamadi/gecersiz: %s", _INDEX_META_FILE)
            return False

        if meta.get("model_name") != self._embedding_config.model_name:
            logger.warning(
                "Persisted KB index farkli bir embedding modeliyle uretilmis (%r != %r); index rebuild gerekiyor.",
                meta.get("model_name"),
                self._embedding_config.model_name,
            )
            return False
        if meta.get("dimension") != self._dimension:
            logger.warning(
                "Persisted KB index farkli boyutta (%r != %r); index rebuild gerekiyor.",
                meta.get("dimension"),
                self._dimension,
            )
            return False
        current_hash = _compute_kb_hash()
        if meta.get("kb_hash") != current_hash:
            logger.warning(
                "Persisted KB index guncel chunk'larla uyusmuyor (kb_hash farkli); "
                "'python -m src.memory.build_knowledge_index' ile yeniden olusturun."
            )
            return False

        try:
            index = faiss.read_index(str(_INDEX_FILE))
            documents = json.loads(_DOCUMENTS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - bozuk persisted dosya, guvenli sekilde fallback'e dus
            logger.warning("Persisted KB index yuklenemedi: %s", exc)
            return False

        self._index = index
        self._documents = documents
        logger.info(
            "EmbeddingRAGService: persisted KB index yuklendi (%d dokuman, model=%s, kb_version=%s).",
            len(documents),
            meta.get("model_name"),
            meta.get("kb_hash", "")[:12],
        )
        return True

    def build_index_from_chunks(self) -> int:
        """TUM `data/knowledge_base/chunks/*.json` kayitlarini `sources.yaml` ile join edip embed eder (rebuild CLI'i icin).

        Bu metod idempotency GUARD'INI ATLAR (mevcut dokumanlarin ustune
        EKLER) - yalnizca `scripts`/`build_knowledge_index.py` gibi
        BILEREK taze bir `EmbeddingRAGService` orneginde cagrilmasi
        beklenir.

        Returns:
            Eklenen dokuman (chunk) sayisi.

        Raises:
            RuntimeError: `data/knowledge_base/chunks/` altinda hicbir chunk bulunamazsa.
        """
        records = _load_kb_chunk_records()
        if not records:
            raise RuntimeError(
                f"{_KB_CHUNKS_DIR} altinda hicbir chunk bulunamadi. Once "
                "'python scripts/build_kb_chunks.py' calistirip chunk'lari uretin."
            )
        self._add_structured_documents(records)
        return len(records)

    def _add_structured_documents(self, records: List[Dict[str, Any]]) -> None:
        """Yapilandirilmis kayitlari (metadata + text) embed edip FAISS'e ve `self._documents`e ekler."""
        if not records:
            return
        texts = [str(r.get("text") or "") for r in records]
        vectors = self._provider.embed_documents(texts)
        self._index.add(vectors)
        self._documents.extend(records)
        logger.info("EmbeddingRAGService: %d dokuman indekslendi (toplam=%d)", len(records), len(self._documents))

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

    def query(self, question: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Verilen soruya en yakin dokumanlari, iki-asamali retrieval (FAISS candidate_k -> Gemini rerank -> threshold) ile dondurur.

        Akis:
          1. FAISS'ten `candidate_k` aday cekilir (embedding benzerligi).
          2. Reranker AKTIFSE: adaylar Gemini (LLM-as-judge) ile YENIDEN siralanir;
             `score_threshold`in ALTINDA kalan sonuclar ELENIR (bkz.
             `RerankerConfig.score_threshold`) - eslesme YOKSA BOS LISTE
             doner, bu GECERLI bir sonuctur, rastgele/dusuk-alakali
             sonuc UYDURULMAZ.
          3. Reranker cagrisi BASARISIZ olursa (`RerankerUnavailableError`):
             embedding-sirali adaylar SESSIZCE "final sonuc" gibi
             SUNULMAZ - retrieval "unavailable" sayilir ve BOS LISTE doner.
          4. Reranker DEVRE DISIYSA: adaylar yalnizca embedding skoruna
             gore siralanir (`similarity_threshold` varsa uygulanir),
             ilk `top_k` dondurulur.

        Args:
            question: Dogal dil sorgusu.
            top_k: NIHAI (rerank/filtre SONRASI) sonuc sayisi; verilmezse
                `faiss_config.top_k`/`reranker_config.top_k` kullanilir.

        Returns:
            En alakali (veya rerank aktifse alaka skoruna gore siralanmis)
            `RetrievedDocument` listesi; hicbir esik-uzeri sonuc yoksa BOS LISTE.
        """
        query_started = time.perf_counter()
        if self._index.ntotal == 0:
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
                avg_rerank_score=None,
                results=[],
            )
            return []

        final_k = top_k or (self._reranker_config.top_k if self._reranker_config else None) or self._faiss_config.top_k
        candidate_k = max(self._faiss_config.candidate_k, final_k)
        candidate_k = min(candidate_k, self._index.ntotal)

        embedding_started = time.perf_counter()
        query_vector = self._provider.embed_query(question)
        scores, indices = self._index.search(np.expand_dims(query_vector, axis=0), candidate_k)
        embedding_latency_ms = (time.perf_counter() - embedding_started) * 1000.0

        candidates: List[RetrievedDocument] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = self._documents[idx]
            candidates.append(
                RetrievedDocument(
                    text=record.get("text", ""),
                    embedding_score=float(score),
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
                )
            )

        retrieval_status = "embedding_only"
        final_docs: List[RetrievedDocument] = candidates
        rerank_latency_ms: Optional[float] = None
        threshold = self._reranker_config.score_threshold if self._reranker_config else None

        if self._reranker is not None:
            rerank_started = time.perf_counter()
            try:
                ranked = self._reranker.rerank(question, [c.text for c in candidates])
            except RerankerUnavailableError as exc:
                rerank_latency_ms = (time.perf_counter() - rerank_started) * 1000.0
                logger.error(
                    "EmbeddingRAGService: reranker basarisiz, retrieval UNAVAILABLE sayiliyor "
                    "(embedding top-k SESSIZCE final sonuc olarak DONDURULMUYOR): %s",
                    exc,
                )
                self._log_retrieval_trace(
                    question, candidates, [], retrieval_status="reranker_unavailable", final_k=final_k
                )
                self._last_query_telemetry = RagQueryTelemetry(
                    query=question,
                    candidate_count=len(candidates),
                    final_count=0,
                    zero_result=True,
                    retrieval_status="reranker_unavailable",
                    threshold=threshold,
                    embedding_latency_ms=round(embedding_latency_ms, 1),
                    rerank_latency_ms=round(rerank_latency_ms, 1),
                    total_latency_ms=round((time.perf_counter() - query_started) * 1000.0, 1),
                    avg_embedding_score=_avg([c.embedding_score for c in candidates]),
                    avg_rerank_score=None,
                    results=[_result_telemetry(c, selected=False) for c in candidates],
                )
                return []
            rerank_latency_ms = (time.perf_counter() - rerank_started) * 1000.0

            reranked_docs: List[RetrievedDocument] = []
            for idx, rerank_score in ranked:
                if threshold is not None and rerank_score < threshold:
                    continue
                doc = candidates[idx]
                doc.rerank_score = rerank_score
                reranked_docs.append(doc)
            final_docs = reranked_docs
            retrieval_status = "reranked"

        final_docs = final_docs[:final_k]
        self._log_retrieval_trace(question, candidates, final_docs, retrieval_status=retrieval_status, final_k=final_k)

        final_ids = {id(d) for d in final_docs}
        self._last_query_telemetry = RagQueryTelemetry(
            query=question,
            candidate_count=len(candidates),
            final_count=len(final_docs),
            zero_result=len(final_docs) == 0,
            retrieval_status=retrieval_status,
            threshold=threshold if retrieval_status == "reranked" else None,
            embedding_latency_ms=round(embedding_latency_ms, 1),
            rerank_latency_ms=round(rerank_latency_ms, 1) if rerank_latency_ms is not None else None,
            total_latency_ms=round((time.perf_counter() - query_started) * 1000.0, 1),
            avg_embedding_score=_avg([c.embedding_score for c in candidates]),
            avg_rerank_score=_avg([d.rerank_score for d in final_docs if d.rerank_score is not None]),
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
        logger.info(
            "RAG retrieval: query=%r embedding_model=%s candidate_count=%d reranker_model=%s "
            "reranked_count=%d final_count=%d threshold=%s retrieval_status=%s "
            "embedding_scores=%s rerank_scores=%s sources=%s",
            question,
            self._embedding_config.model_name,
            len(candidates),
            self._reranker_config.model_name if self._reranker is not None else None,
            len(final_docs) if retrieval_status == "reranked" else 0,
            len(final_docs),
            self._reranker_config.score_threshold if self._reranker_config else None,
            retrieval_status,
            [round(c.embedding_score, 4) for c in candidates],
            [round(d.rerank_score, 4) for d in final_docs if d.rerank_score is not None],
            [
                {"document_id": d.document_id, "article_number": d.article_number, "source_url": d.source_url}
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
        """FAISS indeksini, dokuman metadata'sini ve index meta bilgisini (model/dimension/kb_hash) diske yazar."""
        _KB_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(_INDEX_FILE))
        _DOCUMENTS_FILE.write_text(json.dumps(self._documents, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = {
            "model_name": self._embedding_config.model_name,
            "dimension": self._dimension,
            "kb_hash": _compute_kb_hash(),
            "chunk_count": len(self._documents),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _INDEX_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("KB index persisted: %s (%d dokuman)", _KB_INDEX_DIR, len(self._documents))


# Modul 3 spesifikasyonundaki isim: ayni sinifa isaret eden alias (geriye
# donuk uyumluluk icin `EmbeddingRAGService` adi da tum cagiran kodda aynen
# kullanilmaya devam eder).
FAISSRagService = EmbeddingRAGService


if __name__ == "__main__":
    # Modul 3'un bagimsiz calistirilabilirlik testi:
    #   python -m src.memory.embedding_rag_service
    from src.utils.config_loader import load_config

    logging.basicConfig(level=logging.INFO)

    demo_config = load_config()
    demo_service = FAISSRagService(demo_config.memory.embedding, demo_config.memory.faiss, demo_config.memory.reranker)
    demo_service.seed_default_regulations()

    demo_query = "Yuksekte calisirken hangi ekipmanlar zorunlu?"
    print(f"Sorgu: {demo_query}\n")
    for i, result in enumerate(demo_service.search_laws(demo_query, top_k=3), start=1):
        print(f"{i}. (skor={result.score:.4f}) {result.text}")
