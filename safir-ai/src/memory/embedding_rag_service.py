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

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config_loader import EmbeddingConfig, FaissMemoryConfig

logger = logging.getLogger(__name__)


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

    def persist(self) -> None:
        """FAISS indeksini diske yazar (kalicilik icin)."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        logger.info("FAISS indeksi kaydedildi: %s", self._index_path)
