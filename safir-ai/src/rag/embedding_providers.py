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

2026-08-25 guncellemesi (GUVENLI BATCHING): EVREN `bge-m3-embed` istek basina
azami 8192 token toplam giris kabul eder (`_MAX_CONTEXT_TOKENS`). 748
knowledge-base chunk'inin TAMAMINI TEK bir `embeddings.create()` cagrisinda
gondermek bu siniri asip 400 hatasi uretiyordu. `embed_documents()` artik
metinleri, TAHMINI token uzunluguna gore, her istek `_DEFAULT_SAFE_TOKEN_
BUDGET`i (varsayilan 7500 - 8192'nin altinda guvenlik payi) ASMAYACAK
sekilde ARDISIK (sabit-sayida-dokuman DEGIL) batch'lere boler; sonuclar
ORIJINAL sira ile dondurulur. Token sayimi icin EVREN/`bge-m3-embed`in
GERCEK tokenizer'i (coklu-dilli BERT/WordPiece ailesi) YEREL olarak mevcut
DEGIL - `tiktoken` (OpenAI'nin KENDI BPE tokenizer'i) KASITLI KULLANILMADI,
cunku farkli bir model ailesi icin YANLIS bir kesinlik hissi verir. Bunun
yerine, KONSERVATIF (fazla-tahmin eden) bir karakter-sayisi tabanli heuristik
kullanilir (bkz. `_estimate_tokens`) - genis guvenlik payi (8192'nin
~%85'i) bu tahminin kucuk sapmalarini tolere eder. Tek bir dokuman TEK
BASINA guvenli butceyi asarsa (nadir - 748 chunk'in tipik boyutunun COK
uzerinde), metin SESSIZCE KIRPILMAZ - acik bir `DocumentTooLargeError`
firlatilir (bkz. o sinifin dokustringi).
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class ConfigurationError(Exception):
    """Embedding saglayicisi icin gerekli konfigurasyon eksik/gecersiz (orn. desteklenmeyen provider, API anahtari eksik)."""


class DocumentTooLargeError(ConfigurationError):
    """Tek bir dokuman, TEK BASINA, guvenli istek token butcesini asiyor.

    Bu durumda metin SESSIZCE KIRPILMAZ (bkz. modul dokustringi, gorev
    kisiti: "Do NOT truncate document text") - acik bir hata firlatilir;
    cagiran taraf (orn. `scripts/build_kb_chunks.py`) bu dokumani daha kucuk
    chunk'lara bolmek gibi BILINCLI bir karar almalidir.
    """


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


_MAX_CONTEXT_TOKENS = 8192
"""EVREN `bge-m3-embed` icin dokumante edilen, istek basina azami TOPLAM
giris token siniri (tum `input` listesinin toplami, tek bir metnin degil)."""

_DEFAULT_SAFE_TOKEN_BUDGET = 7500
"""Varsayilan, guvenlik-payli hedef (gorev kisiti: "target approximately
<= 7000-7500 estimated input tokens per request") - `_MAX_CONTEXT_TOKENS`in
KENDISI degil, tahmin hatasina karsi bilinçli bir marj birakan deger.
`EmbeddingConfig.max_batch_tokens` ile config'ten override edilebilir."""

_CHARS_PER_TOKEN_ESTIMATE = 3.0
"""Token-sayisi TAHMINI icin kullanilan, KONSERVATIF (token sayisini
OLDUGUNDAN FAZLA tahmin eden) karakter/token orani. `bge-m3-embed`in
gercek (coklu-dilli BERT/WordPiece) tokenizer'i yerel olarak mevcut
olmadigindan TAM bir sayim YAPILAMAZ; dusuk bir oran (English icin tipik
~4 yerine 3) Turkce/coklu-dilli metnin (aksanli karakterler, sondan
eklemeli morfoloji) genelde tokenizer basina daha yogun oldugu bilgisini
YANSITIR - amac ASLA olceginin ALTINDA tahmin ETMEMEK."""


def _estimate_tokens(text: str, chars_per_token: float = _CHARS_PER_TOKEN_ESTIMATE) -> int:
    """Bir metnin token sayisini KONSERVATIF (fazla-tahmin eden) sekilde tahmin eder.

    Args:
        text: Tahmin edilecek metin.
        chars_per_token: Karakter/token orani (dusuk deger = daha fazla token tahmini = daha guvenli).

    Returns:
        Tahmini token sayisi (en az 1, bos metin haric).
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / chars_per_token))


def _build_safe_batches(
    texts: List[str],
    safe_token_budget: int,
    chars_per_token: float = _CHARS_PER_TOKEN_ESTIMATE,
) -> List[List[int]]:
    """`texts`i, HER batch'in TAHMINI toplam token sayisi `safe_token_budget`i ASMAYACAK sekilde, ORIJINAL sirayla ardisik batch'lere boler.

    Sabit-sayida-dokuman batching KULLANILMAZ (gorev kisiti) - her batch,
    icerdigi metinlerin GERCEK uzunluguna gore degisken sayida dokuman
    tasir; kisa dokumanlar bir batch'te birikir, uzun dokumanlar daha az
    dokumanla dolar.

    Args:
        texts: Batch'lenecek dokuman metinleri (orijinal sira KORUNUR).
        safe_token_budget: Bir batch'in asmamasi gereken tahmini token toplami.
        chars_per_token: `_estimate_tokens`e iletilen oran.

    Returns:
        Her biri `texts` icindeki ORIJINAL indekslerin bir listesi olan
        batch'ler (batch'lerin ve batch icindeki indekslerin sirasi,
        `texts`in sirasiyla AYNIDIR).

    Raises:
        DocumentTooLargeError: `texts[i]`nin TEK BASINA tahmini token sayisi
            `safe_token_budget`i asarsa (hangi indeks/tahmini token sayisi
            acikca belirtilir) - metin SESSIZCE KIRPILMAZ/bolunmez.
    """
    batches: List[List[int]] = []
    current: List[int] = []
    current_tokens = 0

    for index, text in enumerate(texts):
        estimated = _estimate_tokens(text, chars_per_token)
        if estimated > safe_token_budget:
            snippet = text.strip().replace("\n", " ")[:120]
            raise DocumentTooLargeError(
                f"Dokuman #{index} tek basina guvenli token butcesini asiyor "
                f"(tahmini {estimated} token > butce {safe_token_budget} token, "
                f"metin uzunlugu={len(text)} karakter). Metin SESSIZCE KIRPILMADI; "
                "bu dokumanin daha kucuk chunk'lara bolunmesi gerekir. "
                f"Metin onizleme: {snippet!r}"
            )
        if current and current_tokens + estimated > safe_token_budget:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(index)
        current_tokens += estimated

    if current:
        batches.append(current)
    return batches


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
        max_batch_tokens: Optional[int] = None,
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
            max_batch_tokens: `embed_documents()`in her istekte hedefleyecegi
                azami TAHMINI token butcesi (bkz. `_build_safe_batches`);
                `None` ise `_DEFAULT_SAFE_TOKEN_BUDGET` kullanilir. EVREN'in
                GERCEK sinirinin (`_MAX_CONTEXT_TOKENS`=8192) UZERINE
                CIKILMAMASI cagiranin sorumlulugundadir - varsayilan zaten
                guvenli bir pay birakir.
        """
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._configured_dimension = output_dimensionality
        self._safe_token_budget = max_batch_tokens or _DEFAULT_SAFE_TOKEN_BUDGET
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

    def _embed_single_request(self, texts: List[str]) -> np.ndarray:
        """TEK bir `embeddings.create()` cagrisi yapar (batch'leme/token-butce kontrolu YAPMAZ - cagiran zaten guvenli boyuta getirmis olmalidir)."""
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        client = self._get_client()
        response = client.embeddings.create(model=self._model_name, input=texts)
        vectors = np.array([item.embedding for item in response.data], dtype="float32")
        return _l2_normalize(vectors)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Dokumanlari, EVREN'in 8192-token istek sinirini asmayacak GUVENLI batch'ler halinde embed eder.

        Metinler, TAHMINI token uzunluguna gore ardisik, degisken-boyutlu
        batch'lere bolunur (bkz. `_build_safe_batches`) - sabit bir dokuman
        sayisina GORE DEGIL. Her batch icin AYRI bir `embeddings.create()`
        cagrisi yapilir; sonuclar ORIJINAL `texts` sirasiyla birlestirilip
        dondurulur - hicbir metin KIRPILMAZ, hicbir sira DEGISMEZ.

        Raises:
            DocumentTooLargeError: `texts` icinde TEK BASINA guvenli token
                butcesini asan bir metin varsa.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        batches = _build_safe_batches(texts, self._safe_token_budget)
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for batch_indices in batches:
            batch_texts = [texts[i] for i in batch_indices]
            batch_vectors = self._embed_single_request(batch_texts)
            for row, original_index in enumerate(batch_indices):
                vectors[original_index] = batch_vectors[row]
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_single_request([text])[0]


def build_embedding_provider(
    provider: str,
    model_name: str,
    output_dimensionality: Optional[int],
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    max_batch_tokens: Optional[int] = None,
    **_ignored: object,
) -> EmbeddingProvider:
    """Config'teki `provider` degerine gore uygun `EmbeddingProvider`i uretir.

    Args:
        provider: `configs/config.yaml` -> `memory.embedding.provider` (su an YALNIZCA "evren").
        model_name: EVREN embedding model takma adi (orn. "bge-m3-embed").
        output_dimensionality: Beklenen vektor boyutu; `None` ise EVREN'in bilinen degeri (1024) kullanilir.
        base_url: EVREN taban adresi.
        api_key_env: API anahtarinin okunacagi ortam degiskeni adi.
        max_batch_tokens: `embed_documents()`in guvenli batch token butcesi
            (bkz. `EvrenEmbeddingProvider.__init__`); `None` ise
            `_DEFAULT_SAFE_TOKEN_BUDGET` kullanilir.

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
        max_batch_tokens=max_batch_tokens,
    )
