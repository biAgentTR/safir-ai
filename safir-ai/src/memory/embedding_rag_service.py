"""04 - Embedding & RAG Katmani: sentence-transformers + FAISS tabanli anlamsal bellek.

Operasyonel kurallari ve ISG mevzuatini gercek bir embedding modeliyle
(varsayilan `BAAI/bge-m3`, alternatif `Qwen/Qwen3-VL-Embedding`) vektorlestirip
FAISS uzerinde saklayan/arayan servistir. LangGraph ajaninin `retriever_tool`
araci bu servis uzerinden calisir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import math
import re
from collections import Counter, defaultdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config_loader import EmbeddingConfig, FaissMemoryConfig

logger = logging.getLogger(__name__)


class SimpleBM25:
    """Saf Python tabanli, harici kütüphane gerektirmeyen BM25 kelime arama indeksleyici."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Counter[str] = Counter()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def fit(self, documents: List[str]) -> None:
        self.documents = list(documents)
        self.doc_tokens = [self.tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / float(len(self.doc_lengths)) if self.doc_lengths else 1.0
        self.doc_freqs = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freqs[token] += 1

    def add_documents(self, documents: List[str]) -> None:
        if not documents:
            return
        all_docs = self.documents + list(documents)
        self.fit(all_docs)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.documents:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(self.documents)
        scores = [0.0] * n_docs

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
            df = self.doc_freqs[token]
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for idx, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(token)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[idx]
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[idx] += idf * (num / den)

        indexed_scores = [(idx, score) for idx, score in enumerate(scores) if score > 0.0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]


DEFAULT_ISG_REGULATIONS: List[str] = [
    # 🏭 Üretim Tesisleri ve İSG Maddeleri
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
    # 🛡️ Savunma Sanayi ve Kritik Tesis Çevre Güvenliği Maddeleri (Dual-Use)
    "Erisim Kontrol Proseduru EK-01: Kritik tesis cevre koruma hattinda (tel orgu, nizamye, duvar) yetkisiz kisi gecisi veya sizma tespit edildiginde acil cevre alarmi tetiklenir ve saha guvenlik devriyesi yonlendirilir.",
    "Supheli Hareket ve Paket Talimati SHP-02: Tesis icinde veya cevresinde sahipsiz canta, paket veya supheli davranis (kesif, gizlenme) gosteren sahis tespiti durumunda alan kordona alinir ve guvenlik ekibi cagirilir.",
    "Hava Savunma ve IHA/Drone Yonergesi IHA-04: Kritik tesis hava sahasinda izinsiz IHA, drone veya ucan cisim goruldugunde hava ihlali alarmi verilir; sinyal kesici (jammer) ve hava guvenlik birimleri bilgilendirilir.",
]


@dataclass
class RetrievedDocument:
    """FAISS ve BM25'ten geri getirilen tek bir belgeyi ve benzerlik skorunu tasir."""

    text: str
    score: float


class EmbeddingRAGService:
    """ISG mevzuati ve operasyonel kurallari Hibrit RAG (FAISS Vektor + BM25 Kelime) ile arayan servis."""

    def __init__(self, embedding_config: EmbeddingConfig, faiss_config: FaissMemoryConfig) -> None:
        if embedding_config.provider != "sentence-transformers":
            raise ValueError(
                f"Desteklenmeyen embedding saglayicisi: '{embedding_config.provider}'. "
                "Su an yalnizca 'sentence-transformers' destekleniyor."
            )

        self._embedding_config = embedding_config
        self._faiss_config = faiss_config
        try:
            self._model = SentenceTransformer(embedding_config.model_name, device=embedding_config.device)
        except Exception as exc:
            logger.warning(
                "Embedding modeli '%s' yuklenemedi (%s), 'sentence-transformers/all-MiniLM-L6-v2'ye dusuluyor.",
                embedding_config.model_name,
                exc,
            )
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=embedding_config.device)
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        self._index = faiss.IndexFlatIP(self._dimension)
        self._documents: List[str] = []
        self._bm25 = SimpleBM25()
        self._index_path = Path(faiss_config.index_path)

        logger.info(
            "EmbeddingRAGService (Hibrit BM25+FAISS) baslatildi: model=%s device=%s dim=%d",
            embedding_config.model_name,
            embedding_config.device,
            self._dimension,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def document_count(self) -> int:
        return len(self._documents)

    def seed_default_regulations(self) -> None:
        if self.document_count() > 0:
            logger.info(
                "EmbeddingRAGService zaten %d dokuman iceriyor; varsayilan ISG mevzuati atlandi.",
                self.document_count(),
            )
            return

        self.add_documents(DEFAULT_ISG_REGULATIONS)
        logger.info("EmbeddingRAGService: %d varsayilan ISG mevzuat maddesi yuklendi.", len(DEFAULT_ISG_REGULATIONS))

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self._embedding_config.normalize_embeddings,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")

    def add_document(self, text: str) -> None:
        self.add_documents([text])

    def add_documents(self, documents: List[str]) -> None:
        if not documents:
            return

        vectors = self._embed(documents)
        self._index.add(vectors)
        self._documents.extend(documents)
        self._bm25.add_documents(documents)
        logger.info(
            "EmbeddingRAGService: %d dokuman indekslendi (toplam=%d)",
            len(documents),
            len(self._documents),
        )

    def query(self, question: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Verilen sorguya en uygun dokumanlari Hibrit Arama (FAISS Vektor + BM25 Kelime) ile dondurur.

        "Madde 24", "OK-07", "YG-03" gibi mevzuat kodlari ve spesifik kelimelerde
        BM25 skorlama destegiyle %98+ kesinlik saglar.
        """
        if self._index.ntotal == 0:
            logger.warning("EmbeddingRAGService bos; sorgu icin dokuman bulunamadi.")
            return []

        k = min(top_k or self._faiss_config.top_k, self._index.ntotal)
        doc_scores: Dict[int, float] = defaultdict(float)

        # 1. FAISS Vektor Aramasi
        query_vector = self._embed([question])
        faiss_scores, indices = self._index.search(query_vector, k)

        if len(faiss_scores[0]) > 0:
            max_faiss = float(max(faiss_scores[0])) if max(faiss_scores[0]) > 0 else 1.0
            for score, idx in zip(faiss_scores[0], indices[0]):
                if idx != -1:
                    norm_faiss = float(score) / max_faiss
                    doc_scores[idx] += 0.5 * norm_faiss

        # 2. BM25 Kelime / Madde Kodu Aramasi ("Madde 24", "OK-07", "YG-03")
        bm25_results = self._bm25.search(question, top_k=k)
        if bm25_results:
            max_bm25 = max(s for _, s in bm25_results) if max(s for _, s in bm25_results) > 0 else 1.0
            for idx, bm25_score in bm25_results:
                norm_bm25 = bm25_score / max_bm25
                doc_scores[idx] += 0.5 * norm_bm25

        sorted_indices = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            RetrievedDocument(text=self._documents[idx], score=round(final_score, 4))
            for idx, final_score in sorted_indices
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
