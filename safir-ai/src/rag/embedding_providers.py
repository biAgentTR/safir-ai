"""04 - Embedding & RAG Katmani: embedding saglayici soyutlamasi (`EmbeddingProvider`).

`EmbeddingRAGService`, hangi embedding backend'inin kullanildigini bilmeden
`embed_documents(texts)`/`embed_query(text)` cagirabilsin diye bu soyutlama
tanimlanmistir.

2026-08-25 guncellemesi (LOKAL embedding TAMAMEN KALDIRILDI): embedding artik
TAMAMEN EVREN'in (TEKNOFEST yarisma cikarim servisi) OpenAI-uyumlu
`/v1/embeddings` ucu uzerinden calisir (`model="bge-m3-embed"`, 1024 boyut -
bkz. katilimci dokumantasyonu SS 5/10). Tek gercek implementasyon
`EvrenEmbeddingProvider`dir; lokal `sentence-transformers` yoluna HICBIR
KOD YOLU KALMADI (ne birincil yol ne fallback).

Istemci (`openai` SDK) YALNIZCA ilk gercek embed cagrisinda (lazy) kurulur -
boylece bu modulun import edilmesi veya `EvrenEmbeddingProvider` orneklenmesi
ASLA bir ag baglantisi TETIKLEMEZ.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class ConfigurationError(Exception):
    """Embedding saglayicisi icin gerekli konfigurasyon eksik/gecersiz (orn. desteklenmeyen provider, API anahtari eksik)."""


class EmbeddingProvider(ABC):
    """Tum embedding saglayicilari icin ortak sozlesme."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Bu saglayicinin urettigi vektorlerin boyutu."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Dokuman (indeksleme) amacli metinleri vektorlere cevirir.

        Args:
            texts: Vektore cevrilecek dokuman metinleri.

        Returns:
            `(len(texts), dimension)` boyutunda, L2-normalize edilmis float32 dizi.
        """

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Tek bir sorgu metnini vektore cevirir.

        Args:
            text: Sorgu metni.

        Returns:
            `(dimension,)` boyutunda, L2-normalize edilmis float32 dizi.
        """


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class EvrenEmbeddingProvider(EmbeddingProvider):
    """EVREN'in OpenAI-uyumlu `/v1/embeddings` ucu uzerinden CALISAN embedding saglayicisi.

    Dokumantasyon SS 5/10: `bge-m3-embed` yogun (dense) gomme modeli, 1024
    boyut, en yuksek ilk-isabet (R@1) dogrulugunu vermektedir (yeniden
    siralama ONERILMEMEKTEDIR - bkz. `src/rag/evren_reranker.py` modul
    dokustringi, bu servis dedike bir rerank endpoint'i KULLANMAZ). Kimlik
    dogrulama, standart OpenAI istemcisiyle AYNI sekilde `Authorization:
    Bearer <EVREN_API_KEY>` uzerinden yapilir.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key_env: str,
        output_dimensionality: Optional[int] = None,
    ) -> None:
        """EvrenEmbeddingProvider'i model/uc nokta bilgisiyle kurar (HICBIR AG CAGRISI YAPMAZ).

        Args:
            model_name: EVREN embedding model takma adi (orn. "bge-m3-embed").
            base_url: EVREN'in OpenAI-uyumlu taban adresi (orn.
                "https://evren-llmapi.ssyz.org.tr/v1").
            api_key_env: API anahtarinin okunacagi ortam degiskeni adi
                (orn. "EVREN_API_KEY").
            output_dimensionality: Config'ten gelen BEKLENEN vektor boyutu;
                `None` ise `bge-m3-embed` icin dokumantasyondaki bilinen
                deger (1024) kullanilir.
        """
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._configured_dimension = output_dimensionality
        self._client = None  # lazy - ilk gercek embed cagrisina kadar YUKLENMEZ

    @property
    def dimension(self) -> int:
        return self._configured_dimension or 1024

    def _get_client(self):
        """`openai` istemcisini (lazy) kurar; API anahtari eksikse acik `ConfigurationError` firlatir."""
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self._api_key_env, "").strip()
        if not api_key:
            raise ConfigurationError(
                f"EVREN embedding icin '{self._api_key_env}' ortam degiskeni tanimli degil."
            )

        from openai import OpenAI

        self._client = OpenAI(base_url=self._base_url, api_key=api_key)
        return self._client

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        client = self._get_client()
        response = client.embeddings.create(model=self._model_name, input=texts)
        vectors = np.array([item.embedding for item in response.data], dtype="float32")
        return _l2_normalize(vectors)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


def build_embedding_provider(
    provider: str,
    model_name: str,
    output_dimensionality: Optional[int],
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    **_ignored: object,
) -> EmbeddingProvider:
    """Config'teki `provider` degerine gore uygun `EmbeddingProvider`i uretir.

    Args:
        provider: `configs/config.yaml` -> `memory.embedding.provider` (su an YALNIZCA "evren").
        model_name: EVREN embedding model takma adi (orn. "bge-m3-embed").
        output_dimensionality: Beklenen vektor boyutu; `None` ise EVREN'in bilinen degeri (1024) kullanilir.
        base_url: EVREN taban adresi.
        api_key_env: API anahtarinin okunacagi ortam degiskeni adi.

    Returns:
        Kurulmus (ama henuz hicbir ag baglantisi ACMAMIS) `EmbeddingProvider`.

    Raises:
        ConfigurationError: `provider` desteklenmiyorsa veya `base_url`/`api_key_env` eksikse.
            "local" DAHIL, "evren" DISINDAKI HICBIR deger kabul EDILMEZ.
    """
    if provider != "evren":
        raise ConfigurationError(
            f"Desteklenmeyen embedding saglayicisi: '{provider}'. Su an YALNIZCA 'evren' destekleniyor "
            "(lokal sentence-transformers embedding kaldirildi)."
        )
    if not base_url or not api_key_env:
        raise ConfigurationError(
            "memory.embedding.base_url ve memory.embedding.api_key_env config'te tanimli olmalidir."
        )
    return EvrenEmbeddingProvider(
        model_name=model_name,
        base_url=base_url,
        api_key_env=api_key_env,
        output_dimensionality=output_dimensionality,
    )
