"""04 - Embedding & RAG Katmani: Cohere Rerank tabanli ikinci-asama siralama.

`EmbeddingRAGService.query()`, FAISS'ten gelen `candidate_k` adayi bu
katmana gonderir; `CohereReranker`, sorgu ile her aday arasindaki gercek
alaka skorunu (relevance_score) hesaplayip adaylari YENIDEN siralar.
`embedding_score` (FAISS/cosine) ile `rerank_score` (Cohere) KARISTIRILMAZ -
cagiran taraf (`EmbeddingRAGService`) ikisini ayri alanlarda tutar.

GUVENLIK KURALI (bkz. gorev tanimi 7. bolum): reranker API cagrisi
basarisiz olursa, bu modul SESSIZCE embedding-siralamasini "final sonuc"
gibi DONDURMEZ - `RerankerUnavailableError` firlatir; cagiran taraf bunu
yakalayip retrieval'i "unavailable" (bos sonuc) olarak ele almalidir.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

logger = logging.getLogger(__name__)


class RerankerUnavailableError(Exception):
    """Reranker API cagrisi basarisiz oldu veya gerekli konfigurasyon (API anahtari) eksik.

    Bu, "alakasiz sonuc" ile KARISTIRILMAMALIDIR - bu, rerank'in HIC
    YAPILAMADIGI anlamina gelir; cagiran taraf bu durumda embedding
    top-k'yi asla "final, dogrulanmis" sonuc gibi sunmamalidir.
    """


class Reranker(ABC):
    """Tum reranker saglayicilari icin ortak sozlesme."""

    @abstractmethod
    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[int, float]]:
        """Adaylari sorguya gore yeniden siralar.

        Args:
            query: Kullanici/sistem sorgusu.
            candidates: Aday dokuman metinleri (orijinal sira - FAISS/embedding sirasi).

        Returns:
            `(orijinal_indeks, relevance_score)` ikililerinin, `relevance_score`e
            gore AZALAN sirali listesi.

        Raises:
            RerankerUnavailableError: API cagrisi basarisiz olursa veya
                gerekli konfigurasyon eksikse.
        """


class CohereReranker(Reranker):
    """Cohere Rerank API (`cohere` SDK, `ClientV2.rerank`) uzerinden calisan `Reranker`."""

    def __init__(self, model_name: str, api_key_env: str = "COHERE_API_KEY") -> None:
        """CohereReranker'i model adiyla kurar (AG CAGRISI YAPMAZ - istemci lazy olusturulur).

        Args:
            model_name: Cohere rerank model adi (orn. "rerank-v4.0-pro").
            api_key_env: API anahtarinin okunacagi ortam degiskeni adi.
        """
        self._model_name = model_name
        self._api_key_env = api_key_env
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self._api_key_env, "").strip()
        if not api_key:
            raise RerankerUnavailableError(
                f"Cohere reranker icin '{self._api_key_env}' ortam degiskeni tanimli degil."
            )

        try:
            import cohere
        except ImportError as exc:  # pragma: no cover - paket kurulu olmali (requirements.txt)
            raise RerankerUnavailableError(f"'cohere' paketi kurulu degil: {exc}") from exc

        self._client = cohere.ClientV2(api_key=api_key)
        return self._client

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[int, float]]:
        if not candidates:
            return []

        client = self._get_client()
        try:
            response = client.rerank(model=self._model_name, query=query, documents=candidates)
        except Exception as exc:  # noqa: BLE001 - HERHANGI bir API hatasi guvenli bir "unavailable"a cevrilir
            logger.error("CohereReranker: rerank cagrisi basarisiz: %s", exc)
            raise RerankerUnavailableError(f"Cohere rerank cagrisi basarisiz: {exc}") from exc

        return [(result.index, float(result.relevance_score)) for result in response.results]
