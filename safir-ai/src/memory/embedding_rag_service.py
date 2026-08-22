"""04 - Embedding & RAG Katmani: sentence-transformers + FAISS tabanli anlamsal bellek.

Operasyonel kurallari ve ISG mevzuatini gercek bir embedding modeliyle
(varsayilan `BAAI/bge-m3`, alternatif `Qwen/Qwen3-VL-Embedding`) vektorlestirip
FAISS uzerinde saklayan/arayan servistir. LangGraph ajaninin `retriever_tool`
araci bu servis uzerinden calisir.

Knowledge base kaynagi (2026-08-22 guncellemesi)
--------------------------------------------------
`seed_default_regulations()` artik ONCELIKLE `data/knowledge_base/chunks/*.json`
altindaki GERCEK, resmi mevzuat metinlerinden turetilmis madde-bazli chunk'lari
yukler (bkz. `scripts/build_kb_chunks.py` ve `data/knowledge_base/metadata/
sources.yaml` - hangi resmi kaynaktan geldikleri, dogrulama durumlari orada
kayitlidir). Bu chunk'lar bulunamazsa (orn. `data/knowledge_base/chunks/`
klasoru yoksa/bossa), eskiden beri var olan `DEFAULT_ISG_REGULATIONS` (8
ORNEK/placeholder madde) GERIYE-DONUK UYUMLULUK icin fallback olarak kullanilir
- bu, chunk klasoru olmayan bir ortamda (orn. bazi test/CI kurulumlari)
`EmbeddingRAGService`in tamamen bos kalmasini onler.

ONEMLI (bilinen sonuc): `RuleEngine._describe_regulation()`, sorguladigi
sabit kisa etiketin (orn. "ISG Yonetmeligi Madde 12" - `EVENT_TYPE_
REGULATION_MAP`'ten gelir) donen metinde GERCEKTEN GECIP GECMEDIGINI kontrol
eder (bkz. `src/event_analysis/rule_engine.py`). Bu etiketler, ESKI/sahte
8-madde listesiyle EL YORDAMIYLA eslesecek sekilde uydurulmustu; GERCEK
mevzuat metinlerinde bu TAM ETIKETLER GECMEZ. Sonuc olarak bu degisiklikten
sonra `_describe_regulation` cogunlukla enrichment'i REDDEDECEK ve kisa
etikete DONECEKTIR (guvenli davranis - YANLIS bir metin asla rapora
sizmaz, sadece zenginlestirme kaybolur). Bu, KASITLI olarak bu degisikligin
kapsami DISINDA birakildi (yalnizca `embedding_rag_service.py` degistirildi,
`EVENT_TYPE_REGULATION_MAP`in etiketleri GUNCELLENMEDI) - RuleEngine'in
deterministik event_type->mevzuat eslemesi ayri bir sonraki adimdir.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config_loader import EmbeddingConfig, FaissMemoryConfig

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
"""Chunk klasoru bulunamazsa kullanilan GERIYE-DONUK UYUMLULUK fallback'i
(orn. `src/agent/agent_workflow.py`nin mock ornekleri hala bunu kullanir).
Canli sistemde artik BIRINCIL kaynak DEGILDIR - bkz. modul dokustringi."""

_KB_CHUNKS_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base" / "chunks"

_LEVEL_LABELS = {
    "madde": "MADDE",
    "gecici_madde": "GEÇİCİ MADDE",
    "ek_alt_madde": "EK",
    "ek_liste": "EK",
}


def _load_kb_chunk_documents(chunks_dir: Path = _KB_CHUNKS_DIR) -> List[str]:
    """`data/knowledge_base/chunks/*.json` altindaki madde-bazli chunk'lari, her biri kendi kaynak baslikli tek bir metin olarak dondurur.

    Her `*.json` dosyasi `scripts/build_kb_chunks.py::Chunk` alanlarina
    sahip bir sozluk listesidir (`chunk_id`, `document_id`, `level`,
    `number`, `page_start`, `page_end`, `text`). Bu fonksiyon her chunk'i,
    hangi dokuman/madde/sayfadan geldigini belirten kisa bir baslikla
    (orn. "[6331 Sayılı İş Sağlığı ve Güvenliği Kanunu - MADDE 12]")
    onceleyerek FAISS'e eklenecek tek bir metne cevirir; boylece kisa/parca
    bir chunk metni bile hangi kaynaga ait oldugunu kendi icinde tasir
    (`RetrievedDocument` yalnizca duz `text` tasidigi icin ayri bir
    metadata alani yoktur).

    Args:
        chunks_dir: `*.json` chunk dosyalarinin bulundugu klasor.

    Returns:
        Chunk basina bir metin dizesi listesi (dosya adina, sonra dosya
        icindeki sıraya gore deterministik siralanir). Klasor yoksa/bossa/
        hicbir gecerli chunk yoksa BOS LISTE doner - hicbir metin UYDURULMAZ.
    """
    if not chunks_dir.exists():
        return []

    documents: List[str] = []
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
            level = str(item.get("level") or "")
            number = str(item.get("number") or "")
            document_id = str(item.get("document_id") or chunk_file.stem)
            label = _LEVEL_LABELS.get(level, level.upper() or "MADDE")
            header = f"[{document_id} - {label} {number}]".strip()
            documents.append(f"{header}\n{text}")

    return documents


@dataclass
class RetrievedDocument:
    """FAISS'ten geri getirilen tek bir belgeyi ve benzerlik skorunu tasir."""

    text: str
    score: float


class EmbeddingRAGService:
    """ISG mevzuati ve operasyonel kurallari embedding modeliyle vektorlestirip FAISS'te arayan servis.

    `Dynamic Tool Router` icindeki `retriever_tool` tarafindan mevzuat/kural
    sorgulari icin kullanilir. Embedding modeli config uzerinden
    (`memory.embedding.model_name`) degistirilebilir; FAISS indeks boyutu,
    model tarafindan uretilen vektor boyutundan otomatik cikarilir.
    """

    def __init__(self, embedding_config: EmbeddingConfig, faiss_config: FaissMemoryConfig) -> None:
        """EmbeddingRAGService'i embedding modeli ve FAISS konfigurasyonuyla baslatir.

        Args:
            embedding_config: `configs/config.yaml` icindeki `memory.embedding` blogu.
            faiss_config: `configs/config.yaml` icindeki `memory.faiss` blogu.

        Raises:
            ValueError: `embedding_config.provider` desteklenmeyen bir deger olursa.
        """
        if embedding_config.provider != "sentence-transformers":
            raise ValueError(
                f"Desteklenmeyen embedding saglayicisi: '{embedding_config.provider}'. "
                "Su an yalnizca 'sentence-transformers' destekleniyor."
            )

        self._embedding_config = embedding_config
        self._faiss_config = faiss_config
        self._model = SentenceTransformer(embedding_config.model_name, device=embedding_config.device)
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        self._index = faiss.IndexFlatIP(self._dimension)
        self._documents: List[str] = []
        self._index_path = Path(faiss_config.index_path)

        logger.info(
            "EmbeddingRAGService baslatildi: model=%s device=%s dim=%d",
            embedding_config.model_name,
            embedding_config.device,
            self._dimension,
        )

    @property
    def dimension(self) -> int:
        """Embedding modelinin urettigi vektor boyutunu dondurur."""
        return self._dimension

    def document_count(self) -> int:
        """FAISS indeksine su ana kadar eklenmis dokuman sayisini dondurur."""
        return len(self._documents)

    def seed_default_regulations(self) -> None:
        """Indeks bossa knowledge base'i yukler: ONCELIKLE gercek KB chunk'lari, yoksa `DEFAULT_ISG_REGULATIONS` fallback'i.

        Indeks zaten dokuman iceriyorsa hicbir sey yapmaz (idempotent); boylece
        uygulama her yeniden baslatildiginda ayni maddeler tekrar tekrar
        eklenmez. Kaynak sirasi icin bkz. modul dokustringi.
        """
        if self.document_count() > 0:
            logger.info(
                "EmbeddingRAGService zaten %d dokuman iceriyor; seed atlandi.",
                self.document_count(),
            )
            return

        kb_documents = _load_kb_chunk_documents()
        if kb_documents:
            self.add_documents(kb_documents)
            logger.info(
                "EmbeddingRAGService: %d gercek KB chunk'i yuklendi (kaynak: %s).",
                len(kb_documents),
                _KB_CHUNKS_DIR,
            )
            return

        logger.warning(
            "EmbeddingRAGService: %s altinda KB chunk'i bulunamadi; "
            "geriye-donuk uyumluluk icin %d placeholder ISG maddesine (DEFAULT_ISG_REGULATIONS) dusuluyor.",
            _KB_CHUNKS_DIR,
            len(DEFAULT_ISG_REGULATIONS),
        )
        self.add_documents(DEFAULT_ISG_REGULATIONS)

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Metin listesini embedding modeliyle vektorlere cevirir.

        Args:
            texts: Vektore cevrilecek metinler.

        Returns:
            `(len(texts), dimension)` boyutunda float32 vektor dizisi.
        """
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self._embedding_config.normalize_embeddings,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")

    def add_document(self, text: str) -> None:
        """Bir kural/mevzuat metnini anlamsal bellege ekler.

        Args:
            text: Eklenecek dokuman icerigi (orn. bir ISG maddesi).
        """
        self.add_documents([text])

    def add_documents(self, documents: List[str]) -> None:
        """Birden fazla dokumani toplu olarak embedding'leyip FAISS indeksine ekler.

        Args:
            documents: Eklenecek dokuman metinleri listesi.
        """
        if not documents:
            return

        vectors = self._embed(documents)
        self._index.add(vectors)
        self._documents.extend(documents)
        logger.info(
            "EmbeddingRAGService: %d dokuman indekslendi (toplam=%d)",
            len(documents),
            len(self._documents),
        )

    def query(self, question: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Verilen soruya en yakin dokumanlari benzerlik skoruyla birlikte dondurur.

        Args:
            question: Dogal dil sorgusu (orn. "yuksekte calisma kurallari nedir?").
            top_k: Dondurulecek maksimum sonuc sayisi; verilmezse config degeri kullanilir.

        Returns:
            Benzerlik skoruna gore azalan sirali `RetrievedDocument` listesi.
        """
        if self._index.ntotal == 0:
            logger.warning("EmbeddingRAGService bos; sorgu icin dokuman bulunamadi.")
            return []

        k = min(top_k or self._faiss_config.top_k, self._index.ntotal)
        query_vector = self._embed([question])
        scores, indices = self._index.search(query_vector, k)

        return [
            RetrievedDocument(text=self._documents[idx], score=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

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
        """FAISS indeksini diske yazar (kalicilik icin)."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        logger.info("FAISS indeksi kaydedildi: %s", self._index_path)


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
    demo_service = FAISSRagService(demo_config.memory.embedding, demo_config.memory.faiss)
    demo_service.seed_default_regulations()

    demo_query = "Yuksekte calisirken hangi ekipmanlar zorunlu?"
    print(f"Sorgu: {demo_query}\n")
    for i, result in enumerate(demo_service.search_laws(demo_query, top_k=3), start=1):
        print(f"{i}. (skor={result.score:.4f}) {result.text}")
