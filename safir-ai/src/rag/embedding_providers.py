"""04 - Embedding & RAG Katmani: embedding saglayici soyutlamasi (`EmbeddingProvider`).

`EmbeddingRAGService`, hangi embedding backend'inin kullanildigini bilmeden
`embed_documents(texts)`/`embed_query(text)` cagirabilsin diye bu soyutlama
tanimlanmistir. Su an TEK gercek implementasyon `GeminiEmbeddingProvider`
(Google Gemini Embedding API, `google-genai` SDK); eskiden dogrudan
`SentenceTransformer` kullanan yerel/CPU yol KALDIRILDI (bkz.
`embedding_rag_service.py` modul dokustringi).

API anahtari (`GEMINI_API_KEY`) YALNIZCA ilk gercek embed cagrisinda
okunur (lazy client) - boylece bu modulun import edilmesi veya
`GeminiEmbeddingProvider` orneklenmesi, anahtar tanimli olmasa bile
ASLA patlamaz; anahtar eksikse `ConfigurationError` yalnizca gercekten
bir embed cagrisi yapilmaya calisildiginda, acik bir mesajla firlatilir.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 100
"""Gemini embed_content'e tek istekte gonderilecek azami metin sayisi."""

_MAX_RATE_LIMIT_RETRIES = 5
"""429 (RESOURCE_EXHAUSTED) alindiginda, batch basina denenecek azami yeniden-deneme sayisi.

`google-genai` SDK'sinin kendi (tenacity tabanli) yeniden-denemesi kisa
araliklidir (saniyeler); dakikalik RPM kotasi asildiginda bu YETERSIZ
kalabilir (bkz. gercek 429 hatasi). Bu yuzden burada AYRICA, katlanarak
artan (exponential backoff) bir bekleme uygulanir - yalnizca GERCEKTEN
429 alindiginda devreye girer, normal akisi ETKILEMEZ."""

_RATE_LIMIT_BACKOFF_BASE_SEC = 20.0
"""Ilk 429 sonrasi bekleme suresi (saniye); her tekrarda ikiye katlanir."""

_INTER_BATCH_DELAY_SEC = 3.0
"""BASARILI her batch cagrisi arasinda beklenecek sabit sure (saniye).

429'a REAKTIF olan yukaridaki backoff'tan farkli olarak bu, rate limit'e
HIC CARPMADAN once RPM (dakika basi istek) yukunu azaltmak icin PROAKTIF
bir onlemdir - Gemini embedding API'sinin ucretsiz katmaninin RPM kotasi
dusuk oldugunda (748 chunk / 100'luk batch = 8 istek bile arka arkaya
gonderilirse kotayi asabilir), bu araya giren bekleme sorunu BASTAN
onler. `build_index_from_chunks()` gibi tek-seferlik, toplu islemler
disinda (tekil `embed_query` cagrilarinda) etkisi yoktur - yalnizca
`_embed()`in KENDI ic dongusunde, birden fazla batch varsa uygulanir."""


def _is_rate_limit_error(exc: Exception) -> bool:
    """Verilen istisnanin Gemini API'sinin 429 (RESOURCE_EXHAUSTED) hatasi olup olmadigini kontrol eder."""
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "")
    return code == 429 or "RESOURCE_EXHAUSTED" in status.upper()


class ConfigurationError(Exception):
    """Embedding saglayicisi icin gerekli konfigurasyon (orn. API anahtari) eksik/gecersiz."""


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


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini Embedding API (`google-genai` SDK) uzerinden calisan `EmbeddingProvider`.

    Dokuman embedding'i icin `task_type="RETRIEVAL_DOCUMENT"`, sorgu
    embedding'i icin `task_type="RETRIEVAL_QUERY"` kullanilir (Gemini'nin
    onerdigi asimetrik retrieval semasi - ayni metin farkli rollerde farkli
    vektorlere karsilik gelebilir).
    """

    def __init__(
        self,
        model_name: str,
        output_dimensionality: int,
        api_key_env: str = "GEMINI_API_KEY",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        """GeminiEmbeddingProvider'i model adi ve vektor boyutuyla kurar (AG CAGRISI YAPMAZ).

        Args:
            model_name: Gemini embedding model adi (orn. "gemini-embedding-001").
            output_dimensionality: Istenen vektor boyutu (config'ten gelir, HARD-CODE degildir).
            api_key_env: API anahtarinin okunacagi ortam degiskeni adi.
            batch_size: Tek `embed_content` cagrisina gonderilecek azami metin sayisi.
        """
        self._model_name = model_name
        self._dimension = output_dimensionality
        self._api_key_env = api_key_env
        self._batch_size = max(1, batch_size)
        self._client = None  # lazy - ilk gercek cagriya kadar olusturulmaz

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self):
        """Gemini istemcisini (lazy) olusturur; API anahtari yoksa acik `ConfigurationError` firlatir."""
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self._api_key_env, "").strip()
        if not api_key:
            raise ConfigurationError(
                f"Gemini embedding icin '{self._api_key_env}' ortam degiskeni tanimli degil. "
                "Gercek bir RAG retrieval cagrisi yapabilmek icin bu anahtari tanimlayin."
            )

        from google import genai  # gec import: paket kurulu degilse bile modul import'u patlamasin

        self._client = genai.Client(api_key=api_key)
        return self._client

    def _embed(self, texts: List[str], task_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype="float32")

        from google.genai import types

        client = self._get_client()
        vectors: List[List[float]] = []
        batch_starts = list(range(0, len(texts), self._batch_size))
        for i, start in enumerate(batch_starts):
            batch = texts[start : start + self._batch_size]
            response = self._embed_batch_with_rate_limit_retry(client, types, batch, task_type)
            vectors.extend(embedding.values for embedding in response.embeddings)
            if i < len(batch_starts) - 1:
                time.sleep(_INTER_BATCH_DELAY_SEC)

        array = np.asarray(vectors, dtype="float32")
        return _l2_normalize(array)

    def _embed_batch_with_rate_limit_retry(self, client, types, batch: List[str], task_type: str):
        """Tek bir `embed_content` batch cagrisini yapar; 429 (RESOURCE_EXHAUSTED) alinirsa katlanarak-artan bekleme ile yeniden dener.

        Yalnizca RATE-LIMIT hatasinda (429) yeniden dener - baska bir hata
        (gecersiz istek, auth vb.) OLDUGU GIBI yukari firlatilir (sessizce
        yutulmaz/mock'a dusulmez).
        """
        attempt = 0
        while True:
            try:
                return client.models.embed_content(
                    model=self._model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._dimension,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - yalnizca 429 ise yeniden denenir, aksi halde yeniden firlatilir
                if not _is_rate_limit_error(exc) or attempt >= _MAX_RATE_LIMIT_RETRIES:
                    raise
                wait_sec = _RATE_LIMIT_BACKOFF_BASE_SEC * (2**attempt)
                attempt += 1
                logger.warning(
                    "GeminiEmbeddingProvider: 429 RESOURCE_EXHAUSTED (deneme %d/%d), %.0fs bekleniyor: %s",
                    attempt,
                    _MAX_RATE_LIMIT_RETRIES,
                    wait_sec,
                    exc,
                )
                time.sleep(wait_sec)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]


def build_embedding_provider(
    provider: str,
    model_name: str,
    output_dimensionality: Optional[int],
    api_key_env: str = "GEMINI_API_KEY",
) -> EmbeddingProvider:
    """Config'teki `provider` degerine gore uygun `EmbeddingProvider`i uretir.

    Args:
        provider: `configs/config.yaml` -> `memory.embedding.provider` (su an yalnizca "gemini").
        model_name: Embedding model adi.
        output_dimensionality: Istenen vektor boyutu; `None` ise `ConfigurationError`.
        api_key_env: API anahtarinin okunacagi ortam degiskeni adi.

    Returns:
        Kurulmus (ama henuz hicbir AG CAGRISI yapmamis) `EmbeddingProvider`.

    Raises:
        ConfigurationError: `provider` desteklenmiyorsa veya `output_dimensionality` eksikse.
    """
    if provider != "gemini":
        raise ConfigurationError(
            f"Desteklenmeyen embedding saglayicisi: '{provider}'. Su an yalnizca 'gemini' destekleniyor."
        )
    if not output_dimensionality:
        raise ConfigurationError(
            "memory.embedding.output_dimensionality config'te tanimli olmalidir (hard-code edilmez)."
        )
    return GeminiEmbeddingProvider(
        model_name=model_name, output_dimensionality=output_dimensionality, api_key_env=api_key_env
    )
