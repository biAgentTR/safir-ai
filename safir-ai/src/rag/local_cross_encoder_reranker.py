"""04 - Embedding & RAG Katmani: gelecekteki LOKAL Cross-Encoder icin temiz, opsiyonel extension point.

2026-08-24 (RAG + RISK LAYER FINALIZATION): bu modul, `scripts/rag_benchmark.py`
ile alinacak bir benchmark kararindan SONRA production'a baglanabilecek,
TAMAMEN LOKAL bir ikinci-asama reranker icin arayuzu tanimlar - bu turda
HENUZ hicbir yere BAGLANMAZ (bkz. `EmbeddingRAGService.query()`, bu siniftan
HABERSIZDIR - import zinciri BOS).

NEDEN AYRI BIR MODUL (eski `reranker.py`ye EKLENMEDI): `reranker.py`deki
`GeminiReranker`/`GroqReranker`, bir LLM'e SERBEST "0.0-1.0 arasi puan ver"
sorusu soran, harici-API-bagimli bir arayuzdu (bkz. o modulun KENDI
dokustringi) - production'dan KALDIRILDI ve GERI GETIRILMEYECEK (gorev
tanimi). `LocalCrossEncoderReranker` KAVRAMSAL OLARAK TAMAMEN FARKLIDIR:
- LOKAL calisir (harici API/kota/internet YOK)
- serbest LLM yargisi DEGIL, bir cross-encoder modelinin (query, chunk)
  ciftini TEK BIR forward-pass'te puanlayan, deterministik-tekrarlanabilir
  bir siralama modelidir (ayni girdi -> ayni cikti, LLM sampling'i YOK)
- ciktisi `cross_encoder_score` olarak adlandirilir - "risk_score",
  "confidence" veya "probability" OLARAK ADLANDIRILMAZ (gorev tanimi 2/5. bolum)
Bu yuzden eski `LLMReranker`-ailesi interface'ine (`rerank(candidates) ->
reranked` + API-hata modlari) ZORLA UYDURULMADI; kendi, daha basit
`score(query, texts) -> List[float]` sozlesmesine sahiptir.

EXTENSION POINT (mimari, bkz. gorev tanimi 15. bolum)
------------------------------------------------------
    FAISS top-candidate_k
        -> deterministic relevance/evidence gate (score_candidate)
        -> top-N aday
        -> [BURASI: LocalCrossEncoderReranker.score(query, [c.text for c in top_n])]
        -> top-5 (cross_encoder_score'a gore yeniden siralanmis)
        -> ContextBuilder
        -> Agent

`EmbeddingRAGService`, opsiyonel bir `cross_encoder: Optional[LocalCrossEncoderReranker]`
constructor argumani KABUL EDER (bkz. o modulun dokustringi) - ama hicbir
cagiran kod (bkz. `src/main.py::SafirPipeline.__init__`) su an bunu
GECMIYOR, yani PRODUCTION'DA HER ZAMAN `None`/devre disi kalir. Benchmark
(`scripts/rag_benchmark.py`) C pipeline'inin B'ye karsi ANLAMLI bir
iyilesme gosterdigi GUNE kadar bu boyle KALMALIDIR (gorev tanimi 2/6. bolum:
"benchmark sonucu olmadan production'a zorla ekleme").
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Sequence

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
"""Production varsayilan lokal Cross-Encoder modeli (2026-08-24 RAG+RISK
PRODUCTION KAPANIS turu - bu tur ACIKCA "benchmark yapma, daha once
arastirilmis adaylardan en makul olani sec" talimatiyla secildi, YENI bir
benchmark KOSULMADI):
- mMARCO (100+ dil, Turkce DAHIL) uzerinde egitilmis coklu-dilli bir
  MiniLM cross-encoder - SAFIR'in Turkce ISG mevzuati/sorgulari icin
  makul bir varsayilan.
- ~118M parametre - RTX 4060 Laptop (veya CPU-only) uzerinde, top-N
  (tipik olarak <=10) aday uzerinde calistirildiginda gereksiz agir
  DEGILDIR (buyuk `bge-reranker-v2-m3` (~568M) gibi bir alternatife
  KIYASLA acikca daha hafif).
- `sentence-transformers.CrossEncoder` ile DOGRUDAN uyumludur (ek bir
  adaptor/donusum GEREKMEZ).
Bu, KESIN bir benchmark SONUCU DEGIL - `scripts/rag_benchmark.py --cross-encoder
<model>` ile gercek A/B/C karsilastirmasi kosulabildiginde bu deger
YENIDEN DEGERLENDIRILEBILIR/DEGISTIRILEBILIR (bkz. o script'in dokustringi)."""


class CrossEncoderUnavailableError(RuntimeError):
    """Lokal Cross-Encoder modeli yuklenemedi (paket eksik/model agirligi bulunamadi).

    `reranker.py::RerankerUnavailableError` (eski LLM-reranker'in API-hata
    modu) ile KARISTIRILMAZ - bu SADECE lokal model yukleme basarisizligidir,
    hicbir API/ag hatasi TEMSIL ETMEZ.
    """


class CrossEncoderReranker(ABC):
    """TUM lokal cross-encoder implementasyonlari icin ortak, KUCUK sozlesme."""

    @abstractmethod
    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """Verilen `query` ile her `texts[i]` arasindaki cross-encoder relevance skorunu (`cross_encoder_score`) dondurur.

        Args:
            query: RAG sorgu metni.
            texts: Yeniden siralanacak aday chunk metinleri (deterministic
                relevance/evidence gate'ten GECMIS, zaten daraltilmis bir
                top-N kumesi olmasi beklenir - TUM corpus'u DEGIL).

        Returns:
            `texts` ile AYNI uzunlukta, AYNI sirada skor listesi - "risk_score"/
            "confidence"/"probability" DEGIL, yalnizca bir siralama sinyalidir
            (bkz. modul dokustringi).
        """


class LocalCrossEncoderReranker(CrossEncoderReranker):
    """`sentence-transformers` `CrossEncoder` uzerinden CALISAN, TAMAMEN LOKAL cross-encoder.

    Harici API/kota YOKTUR - model agirliklari ilk gercek `score()` cagrisinda
    (lazy) HuggingFace Hub onbelleginden/diskten yuklenir; bu sinifin
    OLUSTURULMASI (constructor) ASLA bir model yuklemesi TETIKLEMEZ (bkz.
    `LocalEmbeddingProvider` ile AYNI lazy-loading ilkesi).
    """

    def __init__(self, model_name: str, max_length: int = 512, device: str = "cpu") -> None:
        """Model adiyla kurulur - HICBIR AGIRLIK YUKLEMEZ (lazy).

        Args:
            model_name: HuggingFace Cross-Encoder model kimligi (orn.
                "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" - bkz.
                `scripts/rag_benchmark.py` icindeki aday arastirmasi).
            max_length: Model giris token kirpma uzunlugu.
            device: "cpu"/"cuda".
        """
        self._model_name = model_name
        self._max_length = max_length
        self._device = device
        self._model = None  # lazy - ilk gercek score() cagrisina kadar YUKLENMEZ

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise CrossEncoderUnavailableError(
                "'sentence-transformers' paketi kurulu degil. Lokal Cross-Encoder icin "
                "'pip install sentence-transformers' calistirin."
            ) from exc
        try:
            self._model = CrossEncoder(self._model_name, max_length=self._max_length, device=self._device)
        except Exception as exc:  # noqa: BLE001 - model agirligi indirilemedi/bozuk - acik hata, sessiz fallback YOK
            raise CrossEncoderUnavailableError(
                f"Lokal Cross-Encoder modeli '{self._model_name}' yuklenemedi: {exc}"
            ) from exc
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        model = self._get_model()
        pairs = [(query, text) for text in texts]
        try:
            scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001 - `CrossEncoder(...)` CoNSTRUCTORU basarili olsa bile
            # bazi `sentence-transformers` surumlerinde GERCEK model agirligi ilk
            # `predict()` cagrisina kadar (lazy alt-modul yuklemesi) INDIRILMEZ - bu
            # yuzden ag/model hatasi BURADA da olusabilir (yalnizca constructor'da
            # DEGIL). `CrossEncoderUnavailableError`e AYNI sekilde cevrilir - cagiran
            # (`EmbeddingRAGService.query()`) ikisini de TEK bir kontrollu degradasyon
            # yolundan (harici bir API'ye SESSIZCE DUSMEDEN) ele alir.
            raise CrossEncoderUnavailableError(
                f"Lokal Cross-Encoder modeli '{self._model_name}' ile skorlama basarisiz: {exc}"
            ) from exc
        return [float(s) for s in scores]
