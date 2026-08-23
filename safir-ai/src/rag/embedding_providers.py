"""04 - Embedding & RAG Katmani: embedding saglayici soyutlamasi (`EmbeddingProvider`).

`EmbeddingRAGService`, hangi embedding backend'inin kullanildigini bilmeden
`embed_documents(texts)`/`embed_query(text)` cagirabilsin diye bu soyutlama
tanimlanmistir.

2026-08-23 guncellemesi (Gemini Embedding API TAMAMEN KALDIRILDI): embedding
artik TAMAMEN LOKAL calisir - `sentence-transformers` ile CPU uzerinde,
harici bir API/kota/API anahtari OLMADAN. Tek gercek implementasyon
`LocalEmbeddingProvider`dir. Gemini Embedding API'ye giden HICBIR kod yolu
KALMADI (ne birincil yol ne de fallback) - bkz. `build_embedding_provider`.

Model (`sentence-transformers`/HuggingFace agirliklari) YALNIZCA ilk gercek
embed cagrisinda (lazy) diskten/HuggingFace Hub onbelleginden yuklenir -
boylece bu modulun import edilmesi veya `LocalEmbeddingProvider` orneklenmesi
ASLA agir bir model yuklemesi TETIKLEMEZ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "
"""`intfloat/multilingual-e5-*` model ailesinin GEREKTIRDIGI, retrieval kalitesi
icin ZORUNLU giris on-ekleri (bkz. modelin resmi kullanim talimatlari) - bu
on-ekler OLMADAN model onemli olcude daha zayif retrieval performansi verir.
Sorgu/dokuman AYRIMI, eski Gemini `task_type=RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT`
ayrimiyla AYNI amaca hizmet eder (asimetrik retrieval semasi)."""


class ConfigurationError(Exception):
    """Embedding saglayicisi icin gerekli konfigurasyon eksik/gecersiz (orn. desteklenmeyen provider, boyut uyusmazligi)."""


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


class LocalEmbeddingProvider(EmbeddingProvider):
    """`sentence-transformers` (HuggingFace) uzerinden CALISAN, TAMAMEN LOKAL `EmbeddingProvider`.

    Harici bir API cagrisi, API anahtari veya kota YOKTUR - model agirliklari
    ilk kullanimda (lazy) HuggingFace Hub onbelleginden/diskten yuklenir, tum
    hesaplama CPU uzerinde yerel olarak yapilir. Varsayilan model
    (`intfloat/multilingual-e5-small`, 384 boyut) kucuk (~118M parametre),
    CPU'da makul hizda calisan, coklu-dilli (Turkce dahil) bir retrieval
    modelidir; dokuman/sorgu embedding'i icin `passage: `/`query: ` on-ekleri
    kullanir (modelin kendi resmi kullanim semasi - eski Gemini
    `task_type=RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY` ayrimiyla ES DEGERDIR).
    """

    def __init__(
        self,
        model_name: str,
        output_dimensionality: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        """LocalEmbeddingProvider'i model adi ve (varsa) beklenen boyutla kurar (HICBIR MODEL YUKLEMEZ).

        Args:
            model_name: HuggingFace/`sentence-transformers` model kimligi
                (orn. "intfloat/multilingual-e5-small").
            output_dimensionality: Config'ten gelen BEKLENEN vektor boyutu;
                verilmisse ilk gercek embed cagrisinda modelin GERCEKTEN
                urettigi boyutla KARSILASTIRILIR (uyusmazlik sessizce KABUL
                EDILMEZ, bkz. `_ensure_dimension_matches`). `None` ise
                modelin kendi boyutu oldugu gibi kabul edilir.
            device: `sentence-transformers` cihaz parametresi ("cpu"/"cuda"); varsayilan "cpu".
        """
        self._model_name = model_name
        self._configured_dimension = output_dimensionality
        self._device = device
        self._model = None  # lazy - ilk gercek cagriya kadar YUKLENMEZ

    @property
    def dimension(self) -> int:
        if self._configured_dimension is not None:
            return self._configured_dimension
        return self._get_model().get_sentence_embedding_dimension()

    def _get_model(self):
        """`sentence-transformers` modelini (lazy) yukler; paket kurulu degilse acik `ConfigurationError` firlatir."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError(
                "'sentence-transformers' paketi kurulu degil. Lokal embedding icin "
                "'pip install sentence-transformers' calistirin."
            ) from exc

        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._ensure_dimension_matches(self._model.get_sentence_embedding_dimension())
        return self._model

    def _ensure_dimension_matches(self, actual_dimension: int) -> None:
        """Modelin GERCEKTEN urettigi boyutu, config'te BEKLENEN boyutla karsilastirir - uyusmazlik SESSIZCE KABUL EDILMEZ."""
        if self._configured_dimension is not None and actual_dimension != self._configured_dimension:
            raise ConfigurationError(
                f"Embedding modeli '{self._model_name}' {actual_dimension} boyutunda vektor uretiyor, "
                f"ancak config'te output_dimensionality={self._configured_dimension} tanimli. "
                "configs/config.yaml -> memory.embedding.output_dimensionality'i duzeltin."
            )

    def _embed(self, texts: List[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        model = self._get_model()
        prefixed = [f"{prefix}{t}" for t in texts]
        vectors = model.encode(
            prefixed,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # kendi _l2_normalize'imizla yapiyoruz (Gemini yolundakiyle AYNI davranis)
        )
        array = np.asarray(vectors, dtype="float32")
        self._ensure_dimension_matches(array.shape[1])
        return _l2_normalize(array)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts, prefix=_PASSAGE_PREFIX)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], prefix=_QUERY_PREFIX)[0]


def build_embedding_provider(
    provider: str,
    model_name: str,
    output_dimensionality: Optional[int],
    device: str = "cpu",
) -> EmbeddingProvider:
    """Config'teki `provider` degerine gore uygun `EmbeddingProvider`i uretir.

    Args:
        provider: `configs/config.yaml` -> `memory.embedding.provider` (su an YALNIZCA "local").
        model_name: Embedding model adi (`sentence-transformers`/HuggingFace kimligi).
        output_dimensionality: Beklenen vektor boyutu; `None` ise `ConfigurationError`.
        device: `sentence-transformers` cihaz parametresi.

    Returns:
        Kurulmus (ama henuz hicbir model agirligi YUKLEMEMIS) `EmbeddingProvider`.

    Raises:
        ConfigurationError: `provider` desteklenmiyorsa veya `output_dimensionality` eksikse.
            "gemini" DAHIL, "local" DISINDAKI HICBIR deger kabul EDILMEZ -
            Gemini Embedding API'ye SESSIZCE fallback YAPILMAZ.
    """
    if provider != "local":
        raise ConfigurationError(
            f"Desteklenmeyen embedding saglayicisi: '{provider}'. Su an YALNIZCA 'local' destekleniyor "
            "(Gemini Embedding API kaldirildi - harici API/kota gerektiren hicbir fallback yoktur)."
        )
    if not output_dimensionality:
        raise ConfigurationError(
            "memory.embedding.output_dimensionality config'te tanimli olmalidir (hard-code edilmez)."
        )
    return LocalEmbeddingProvider(model_name=model_name, output_dimensionality=output_dimensionality, device=device)
