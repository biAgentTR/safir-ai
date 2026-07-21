"""04 - Anlamsal Bellek: FAISS tabanli RAG deposu (operasyonel kurallar / ISG mevzuati).

Not: Bu boilerplate icinde gercek bir embedding modeli indirmeden calisabilmesi
icin basit, deterministik bir "hash embedding" implementasyonu varsayilan
olarak kullanilir. Production ortaminda `_embed` metodu, config'te tanimli
`sentence-transformers` modeliyle degistirilmelidir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from src.utils.config_loader import FaissMemoryConfig

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 384


@dataclass
class RetrievedDocument:
    """FAISS'ten geri getirilen tek bir belgeyi ve benzerlik skorunu tasir."""

    text: str
    score: float


class SemanticMemory:
    """Operasyonel kurallari ve ISG mevzuatini vektor uzayinda saklayan RAG bellegi.

    `Dynamic Tool Router` icindeki RAG Tool tarafindan mevzuat/kural sorgulari
    icin kullanilir.
    """

    def __init__(self, config: FaissMemoryConfig) -> None:
        """SemanticMemory'yi verilen FAISS konfigurasyonu ile baslatir.

        Args:
            config: `configs/config.yaml` icindeki `memory.faiss` blogu.
        """
        self._config = config
        self._index = faiss.IndexFlatIP(_EMBEDDING_DIM)
        self._documents: List[str] = []
        self._index_path = Path(config.index_path)

    def _embed(self, text: str) -> np.ndarray:
        """Metni sabit boyutlu, normalize edilmis bir vektore cevirir.

        Not:
            Bu, harici model indirmeden calisabilen deterministik bir yer
            tutucudur (karakter n-gram hash'lerine dayali). Gercek dagitimda
            `config.embedding_model` (orn. multilingual MiniLM) ile
            degistirilmelidir.

        Args:
            text: Vektore cevrilecek metin.

        Returns:
            L2-normalize edilmis, `_EMBEDDING_DIM` boyutunda float32 vektor.
        """
        vector = np.zeros(_EMBEDDING_DIM, dtype="float32")
        for i, token in enumerate(text.lower().split()):
            bucket = hash(token) % _EMBEDDING_DIM
            vector[bucket] += 1.0 + (i * 0.0)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def add_document(self, text: str) -> None:
        """Bir kural/mevzuat metnini anlamsal bellege ekler.

        Args:
            text: Eklenecek dokuman icerigi (orn. bir ISG maddesi).
        """
        vector = self._embed(text).reshape(1, -1)
        self._index.add(vector)
        self._documents.append(text)

    def bulk_add(self, documents: List[str]) -> None:
        """Birden fazla dokumani toplu olarak bellege ekler.

        Args:
            documents: Eklenecek dokuman metinleri listesi.
        """
        for document in documents:
            self.add_document(document)

    def query(self, question: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Verilen soruya en yakin dokumanlari benzerlik skoruyla birlikte dondurur.

        Args:
            question: Dogal dil sorgusu (orn. "yuksekte calisma kurallari nedir?").
            top_k: Dondurulecek maksimum sonuc sayisi; verilmezse config degeri kullanilir.

        Returns:
            Benzerlik skoruna gore azalan sirali `RetrievedDocument` listesi.
        """
        if self._index.ntotal == 0:
            logger.warning("SemanticMemory bos; sorgu icin dokuman bulunamadi.")
            return []

        k = top_k or self._config.top_k
        k = min(k, self._index.ntotal)
        query_vector = self._embed(question).reshape(1, -1)
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
