"""03 - Gorsel Dil Modeli Katmani: tum VLM implementasyonlari icin soyut taban sinif.

ONEMLI (mimari): Olay kumelemesi ARTIK bu katmanda yapilir (sampler'da
DEGIL). `BaseVLM` alt siniflari (`QwenVLM`, `GemmaVLM`)
`analyze_evidence(evidence_frames, prompt)` implemente eder (`EvrenVLM`,
production'da kullanilan tek video-tabanli implementasyon, bunun yerine
`analyze_video` kullanir - bkz. `src/vlm/evren_vlm.py`); taban sinif
bunun uzerine iki ORTAK yetenek insa eder:

- `analyze_evidence_batched`: TUM evidence karelerini TEK bir dev payload'a
  doldurmak yerine kronolojik, kayipsiz batch'lere boler (batch siniri OLAY
  SINIRI DEGILDIR); her batch icin `analyze_evidence` BAGIMSIZ cagrilir, bir
  batch'in basarisiz olmasi digerlerini ETKILEMEZ.
- `reconcile_events`: birden fazla batch varsa, her batch'in kendi ICINDE
  urettigi (batch-yerel `event_id`li) olaylari, IKINCI (metin-tabanli,
  goruntusuz) bir VLM cagrisiyla TEK bir global olay listesine birlestirir;
  hicbir evidence kaybolmaz.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from src.prompts.vlm_prompts import VLM_RECONCILIATION_SYSTEM_PROMPT
from src.sampler.payload_builder import VLMPayloadBuilder
from src.sampler.schema import EvidenceFrame
from src.utils.config_loader import VLLMEndpointConfig

logger = logging.getLogger(__name__)

# Gecici ag hatalarinda VLM cagrisinin kac kez yeniden deneneceği ve geri-cekilme tabani.
_MAX_INFERENCE_RETRIES = 2
_RETRY_BACKOFF_BASE_SEC = 0.5

# Yeniden denenmesi ANLAMLI olan HTTP durum kodlari. Bunun DISINDAKI 4xx'ler
# (400 bozuk istek, 401/403 gecersiz anahtar, 404, 413 cok buyuk govde, 422)
# tekrar denemekle DUZELMEZ - yeniden denemek yalnizca hata suresini katlar ve
# kota harcar; bu yuzden ANINDA yukseltilir.
#   408 Request Timeout | 409 Conflict | 425 Too Early | 429 Too Many Requests
#   500/502/503/504 sunucu tarafi gecici hatalar
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Saglayici bir bekleme suresi dayatirsa (429/503 + `Retry-After`), ustel
# geri-cekilme yerine ONUN degeri kullanilir - ancak cok uzun bloklanmayi
# onlemek icin bu tavanla sinirlanir.
_MAX_HONORED_RETRY_AFTER_SEC = 30.0


def is_retryable_error(exc: BaseException) -> bool:
    """Bir istisnanin GECICI (yeniden denenebilir) olup olmadigini soyler.

    Args:
        exc: `httpx` cagrisindan yakalanan istisna.

    Returns:
        Ag/baglanti/timeout hatalari ve `_RETRYABLE_STATUS_CODES` icindeki HTTP
        durumlari icin `True`; kalici istemci hatalari (400/401/403/404/413/422)
        icin `False`.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    # `RequestError`: connect/read/write/pool timeout'lari ve baglanti hatalari
    # (DNS, TLS, kopan soket). Hicbiri istegin ICERIGIYLE ilgili degildir.
    return isinstance(exc, httpx.RequestError)


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """`Retry-After` basligini (saniye biciminde verilmisse) saniye olarak dondurur.

    HTTP-date bicimindeki degerler BILEREK yok sayilir (saat farki/kayma riski);
    o durumda cagiran taraf kendi ustel geri-cekilmesine duser.

    Args:
        exc: Yakalanan istisna (yalnizca `HTTPStatusError` anlamlidir).

    Returns:
        Tavanla sinirlanmis bekleme suresi veya `None`.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    raw = exc.response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, _MAX_HONORED_RETRY_AFTER_SEC)


def send_with_retry(
    send: Callable[[], httpx.Response],
    *,
    source: str,
    max_retries: int = _MAX_INFERENCE_RETRIES,
    on_retry: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> httpx.Response:
    """HER VLM/LLM HTTP cagrisi icin ORTAK yeniden-deneme sarmalayicisi.

    NEDEN TEK BIR YERDE: yeniden deneme mantigi daha once YALNIZCA kare-tabanli
    ("hafif") yolda, `_post_chat_completion` icinde SATIR ICI vardi; video-
    dogrudan ("agir") yolda HIC YOKTU - tek bir gecici ag hatasi tum analizi
    dusuruyordu. Ayni mantigi ikinci kez kopyalamak yerine buraya alindi;
    boylece iki yol AYNI politikayi (hangi hata gecicidir, ne kadar beklenir,
    `Retry-After` onurlandirilir mi) paylasir ve politika tek noktadan degisir.

    `send()` yalnizca istegi ATMALIDIR; `raise_for_status()` BURADA cagrilir -
    aksi halde HTTP durum hatalari `send` icinde yakalanip yeniden-deneme
    karari verilemezdi.

    Args:
        send: Istegi atip `httpx.Response` donduren, parametresiz cagrilabilir.
            Her denemede YENIDEN cagrilir (govde tekrar gonderilir).
        source: Teshis/loglama kimligi (model adi veya video yolu).
        max_retries: EK deneme sayisi (toplam deneme = `max_retries + 1`).
        on_retry: Her yeniden denemeden ONCE cagrilan opsiyonel bildirim
            (ör. operator paneline "yeniden deneniyor" bilgisi dusurmek icin).
            Sozluk alanlari: `attempt`, `max_attempts`, `delay_sec`, `error`.

    Returns:
        Basarili (2xx) `httpx.Response`.

    Raises:
        httpx.HTTPError: Hata KALICI ise (yeniden denenmez) oldugu gibi
            yukseltilir - cagiran taraf mevcut hata yollarini (degrade rapor,
            parca basarisizligi) DEGISTIRMEDEN kullanmaya devam eder.
        RuntimeError: Tum denemeler tukendiginde, son hatayi zincirleyerek.
    """
    last_exc: Optional[BaseException] = None
    total_attempts = max_retries + 1

    for attempt in range(total_attempts):
        try:
            response = send()
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            if not is_retryable_error(exc):
                # KALICI hata: hic beklemeden, oldugu gibi yukselt.
                logger.error(
                    "VLM cagrisi KALICI hatayla basarisiz (yeniden denenmeyecek) - kaynak=%s: %s",
                    source,
                    exc,
                )
                raise
            last_exc = exc
            if attempt >= total_attempts - 1:
                break
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = _RETRY_BACKOFF_BASE_SEC * (2**attempt)
            logger.warning(
                "VLM cagrisi basarisiz (deneme %d/%d, kaynak=%s): %s - %.1fs sonra yeniden denenecek",
                attempt + 1,
                total_attempts,
                source,
                exc,
                delay,
            )
            if on_retry is not None:
                on_retry(
                    {
                        "attempt": attempt + 1,
                        "max_attempts": total_attempts,
                        "delay_sec": round(delay, 1),
                        "error": str(exc),
                    }
                )
            time.sleep(delay)

    raise RuntimeError(
        f"VLM cagrisi {total_attempts} denemede basarisiz (kaynak={source}): {last_exc}"
    ) from last_exc


def apply_extra_body(payload: Dict[str, Any], extra_body: Dict[str, Any]) -> None:
    """Saglayici-ozel alanlari (ozellikle `enable_thinking: false`) payload'a ekler.

    Mentor eleştirisi 5 ("Akil Yurutme/Thinking Modu Tuzagindan Kacinma"):
    `enable_thinking` acikken kucuk/orta modeller `max_tokens` butcesinin
    TAMAMINI gizli akil yurutmeye harcayip goruen `content`i BOS birakabilir
    (HTTP 200, ama bos yanit - bkz. `raise_if_empty_content`). Bu fonksiyon,
    `configs/config.yaml`daki `extra_body.chat_template_kwargs.enable_thinking:
    false`i HER `/chat/completions` payload'ina TUTARLI sekilde uygulamak
    icin TEK bir yerden paylasilir (`BaseVLM._build_chat_payload`,
    `BaseVLM._build_reconciliation_payload`, `EvrenVLM`nin video istekleri).
    Cekirdek alanlari (model/messages/...) EZMEZ.

    Args:
        payload: Uzerine yazilacak istek govdesi (yerinde degistirilir).
        extra_body: `VLLMEndpointConfig.extra_body`.
    """
    for key, value in (extra_body or {}).items():
        payload.setdefault(key, value)


def raise_if_empty_content(raw_content: str, source: str, response_data: Dict[str, Any]) -> None:
    """VLM'in ham cevabi bos/yalnizca boslukdan ibaretse, sessizce devam etmek yerine ACIKCA hata firlatir.

    Mentor eleştirisi 5: "Testlerde sistemin sessizce (HTTP 200 donerek ama
    bos icerikle) cokmesi projenin elenmesine neden olabilir." `apply_extra_body`
    bu durumun EN YAYGIN KOK NEDENINI (dusunme modunun acik kalmasi) giderir;
    ancak baska bir nedenle (ag/model hatasi, beklenmedik `finish_reason`)
    yine de bos icerik gelirse, bu SESSIZCE bos bir `VLMResponse.description`/
    cevaba donusup ilerideki asamalarda "hicbir sey gozlemlenmedi" gibi
    YANLIS yorumlanmasin diye burada ACIKCA (RuntimeError ile) yakalanir -
    cagiran taraf (`_post_chat_completion`/`EvrenVLM`) bunu zaten bilinen
    "basarisiz VLM cagrisi" yoluna (degraded rapor/parca basarisizligi/retry)
    yonlendirir; risk UYDURULMAZ, hata GIZLENMEZ.

    Args:
        raw_content: `data["choices"][0]["message"]["content"]` (strip edilmemis olabilir).
        source: Teshis icin loglanacak kimlik (model adi veya video yolu).
        response_data: Ham VLM yaniti (teshis: `finish_reason`/`usage` var mi loglanir).

    Raises:
        RuntimeError: `raw_content` bos/yalnizca boslukdan olusuyorsa.
    """
    if raw_content and raw_content.strip():
        return
    try:
        finish_reason = response_data["choices"][0].get("finish_reason")
    except (KeyError, IndexError, AttributeError):
        finish_reason = None
    usage = response_data.get("usage") if isinstance(response_data, dict) else None
    logger.error(
        "VLM bos icerik dondurdu (HTTP 200, ama content bos) - olasi neden: 'dusunme' "
        "modu tum max_tokens butcesini tuketti. kaynak=%s finish_reason=%s usage=%s",
        source,
        finish_reason,
        usage,
    )
    raise RuntimeError(
        f"VLM bos yanit dondurdu (kaynak={source}, finish_reason={finish_reason}, usage={usage}) - "
        "olasi neden: 'enable_thinking' acik kalip max_tokens butcesi gizli akil yurutmeye tuketildi."
    )


@dataclass
class VLMResponse:
    """Bir VLM cagrisinin (tek bir istek/batch veya reconciliation icin) standardize edilmis ciktisi."""

    description: str
    model_name: str
    frame_count: int
    latency_ms: float
    structured_events: List[Dict[str, Any]] = field(default_factory=list)
    """Modelin dogrudan urettigi tipli olaylar (bkz. `EVENTS_JSON` blogu):
    her biri `{"event_id", "type", "start_time", "end_time", "evidence_ids",
    "description", "risk_score", "confidence"}`. Bos ise `EventEngine`
    anahtar-kelime fallback'ine duser (bkz. `event_engine.detect`)."""
    status: str = "completed"
    """Bu VLM cagrisinin durumu: `"completed"` (basarili), `"failed"`
    (bu batch icin VLM cagrisi basarisiz oldu - description'da
    `[ANALYSIS_FAILED]` notu bulunur) veya (yalnizca batch'leri birlestiren
    agrege yanitlarda) `"partial_failure"` (bazi batch'ler basarili, bazilari
    basarisiz). Asla `risk=0`/basarili gibi yorumlanmamalidir."""
    evidence_ids: List[str] = field(default_factory=list)
    """Bu yanitin kapsadigi (gonderilen) `EvidenceFrame.evidence_id` degerleri
    (bkz. `analyze_evidence_batched`); tek-cagri (eski/agrege) yanitlarda bos olabilir."""
    chunk_analysis_result: Optional[Any] = None
    aggregate_result: Optional[Any] = None
    """Modelin cikti basarimini gosteren tipli sozlesme (bkz. ChunkAnalysisResult)."""
    chunking_summary: Optional[Dict[str, Any]] = None
    """YALNIZCA video-dogrudan (`EvrenVLM.analyze_video`) yolunda doldurulur:
    `{"total_chunks", "encoder", "chunk_duration_sec", "video_duration_sec",
    "per_chunk_elapsed_sec"}` - operator paneline "Video Parçalama" asamasini
    (bkz. `src/main.py::SafirPipeline.run`, `configs/config.yaml ->
    vlm.models.evren.chunk_duration_sec`) GORUNUR/KALICI kilmak icindir.
    Diger tum saglayicilarda/yollarda `None` kalir."""


# VLM ciktisinin sonundaki makine-okunur olay blogunu yakalar:
#   EVENTS_JSON: [ {...}, {...} ]
_EVENTS_JSON_PATTERN = re.compile(r"EVENTS_JSON:\s*(\[.*\])", re.DOTALL | re.IGNORECASE)


def parse_structured_events(content: str) -> Tuple[str, List[Dict[str, Any]]]:
    """VLM metninden `EVENTS_JSON` blogunu ayristirir ve insan-okur metinden ayirir.

    Model, insan-okur gozlem bloklarindan sonra `EVENTS_JSON: [...]` satiri
    ekler. Bu fonksiyon o JSON dizisini ayristirip dondurur ve blogu
    aciklamadan temizler; boylece rapor/panelde yalnizca temiz gozlem metni
    kalir. Blok yoksa veya JSON gecersizse, aciklama oldugu gibi kalir ve bos
    liste doner (EventEngine anahtar-kelime fallback'ine gecer).

    Args:
        content: Modelin ham metin ciktisi.

    Returns:
        `(temiz_aciklama, structured_events)` ikilisi.
    """
    match = _EVENTS_JSON_PATTERN.search(content)
    if not match:
        return content.strip(), []

    events = _loads_events_lenient(match.group(1))
    if not events:
        logger.warning("VLM EVENTS_JSON blogu ayristirilamadi/bos, anahtar-kelime fallback'ine dusulecek.")

    clean_description = content[: match.start()].strip()
    return (clean_description or content.strip()), events


def _loads_events_lenient(raw_block: str) -> List[Dict[str, Any]]:
    """`EVENTS_JSON` dizisini toleransli sekilde ayristirir (kucuk modellerin yaygin hatalarina karsi).

    Once ham metin `json.loads` ile denenir; basarisiz olursa yaygin bir hata
    olan "sondaki virgul" (`,]` / `,}`) temizlenip yeniden denenir. Yalnizca
    sozluk (dict) elemanlar dondurulur; hicbir gecerli JSON elde edilemezse bos
    liste doner (EventEngine anahtar-kelime fallback'ine gecer).

    Args:
        raw_block: `EVENTS_JSON:` isaretcisinden sonra yakalanan `[...]` metni.

    Returns:
        Sozluk elemanlardan olusan liste veya bos liste.
    """
    candidates = [raw_block, re.sub(r",(\s*[\]}])", r"\1", raw_block)]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


class BaseVLM(ABC):
    """Butun Gorsel Dil Modeli (VLM) entegrasyonlari icin soyut taban sinif.

    Somut alt siniflar (`QwenVLM`, `GemmaVLM`), yerel vLLM servisine HTTP
    uzerinden baglanarak `analyze_evidence` metodunu implemente etmelidir. Bu
    soyutlama, ajan/muhakeme katmaninin hangi VLM'in aktif oldugundan bagimsiz
    calismasini saglar.
    """

    #: Bu VLM'in GERCEK girdisi Adaptive Frame Sampler'in urettigi evidence
    #: kareleri mi (`QwenVLM`/`GemmaVLM` - frame-tabanli, `analyze_evidence`
    #: bu karelerin GORUNTULERINI dogrudan model payload'ina gomer), yoksa
    #: videoyu KENDISI dogrudan mi analiz ediyor (`EvrenVLM` - `analyze_video`,
    #: sampler kareleri yalnizca arayuz PARITESI icin alinir, HICBIR ZAMAN
    #: modele gonderilmez). `SafirPipeline.run()` bunu okuyup video-tabanli
    #: saglayicilarda TUM Adaptive Frame Sampler asamasini (CPU'da video
    #: uzunluguyla orantili suren, ~saniyeler-dakikalar mertebesinde bir
    #: on-isleme) ATLAR - calistirilsa bile ciktisi (evidence_frames) bu
    #: saglayicilarda ASLA VLM girdisi olarak KULLANILMIYORDU (bkz.
    #: `EvrenVLM.analyze_video`in `del evidence_frames`i), yani calistirmak
    #: saf zaman kaybiydi. Varsayilan `True` (frame-tabanli, GERIYE-DONUK
    #: UYUMLU - mevcut/gelecek frame-tabanli saglayicilar dokunulmadan calisir).
    requires_frame_sampling: bool = True

    def __init__(self, endpoint: VLLMEndpointConfig) -> None:
        """BaseVLM'i vLLM baglanti bilgileriyle baslatir.

        Args:
            endpoint: Bu modelin vLLM servis adresini ve uretim parametrelerini
                tasiyan konfigurasyon nesnesi.
        """
        self._endpoint = endpoint

    @property
    def base_url(self) -> str:
        """Bu modelin OpenAI-uyumlu servisinin taban URL'sini dondurur (yerel vLLM veya harici saglayici)."""
        return self._endpoint.resolved_base_url()

    @property
    def model_name(self) -> str:
        """Bu VLM icin yapilandirilmis Hugging Face model adini dondurur."""
        return self._endpoint.model_name

    @abstractmethod
    def analyze_evidence(
        self, evidence_frames: List[EvidenceFrame], prompt: str
    ) -> VLMResponse:
        """Kronolojik evidence karelerini analiz edip olay kumeleme + dogal dil aciklama uretir.

        Args:
            evidence_frames: `AdaptiveFrameSampler.process_video` tarafindan
                uretilen, zaman sirali (kronolojik) evidence kareleri (bir
                batch'in tamami veya alt kumesi olabilir).
            prompt: Modelin odaklanmasi istenen olay/soruyu tanimlayan istem.

        Returns:
            Kumelenmis olaylari (`EVENTS_JSON`) ve dogal dile yakin
            aciklamayi iceren `VLMResponse`.

        Raises:
            RuntimeError: vLLM servisine erisilemezse veya yanit gecersizse.
        """
        raise NotImplementedError

    def analyze_evidence_batched(
        self,
        evidence_frames: List[EvidenceFrame],
        prompt: str,
        batch_size: int = 40,
        on_batch: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[VLMResponse]:
        """Evidence karelerini TEK bir dev payload yerine kronolojik, kayipsiz batch'ler halinde analiz eder.

        `evidence_frames`, `batch_size` buyuklugunde ARDISIK (kronolojik)
        gruplara bolunur; batch siniri bir OLAY SINIRI DEGILDIR - tek bir
        gercek olay iki batch'e yayilabilir (bu durum `reconcile_events` ile
        cozulur). Her batch icin `analyze_evidence` BAGIMSIZ olarak cagrilir.
        Bir batch'in VLM cagrisi basarisiz olursa (ag hatasi, gecersiz yanit,
        vb.) o batch icin `status="failed"` ve `[ANALYSIS_FAILED]` ile
        baslayan bir `VLMResponse` uretilir; ISTISNA DIGER batch'lere
        YAYILMAZ (bir batch'in basarisiz olmasi digerlerinin sonucunu
        KAYBETMEZ). Hicbir evidence karesi bu asamada SILINMEZ; basarisiz
        bir batch'in evidence_id'leri `VLMResponse.evidence_ids`de aynen
        korunur (cagiran taraf bunlari `unassigned` olarak isaretleyebilir).

        Args:
            evidence_frames: Analiz edilecek TUM evidence kareleri (video
                geneli, kronolojik, kayipsiz).
            prompt: Kullanici/istem metni (her batch'e aynen iletilir).
            batch_size: Bir VLM istegine dahil edilecek azami evidence karesi
                sayisi.

        Returns:
            Kronolojik sirada, her biri kendi `evidence_ids` alaniyla hangi
            evidence karelerini kapsadigini belirten `VLMResponse` listesi.
            Bos `evidence_frames` icin bos liste doner.
        """
        if not evidence_frames:
            return []

        step = max(1, batch_size)
        total_batches = (len(evidence_frames) + step - 1) // step
        responses: List[VLMResponse] = []
        for start in range(0, len(evidence_frames), step):
            batch_index = start // step
            batch = evidence_frames[start : start + step]
            batch_evidence_ids = [ef.evidence_id for ef in batch]
            try:
                response = self.analyze_evidence(batch, prompt)
                response.evidence_ids = batch_evidence_ids
                response.status = "completed"
            except Exception as exc:  # noqa: BLE001 - izole edilip diger batch'lere yayilmaz
                logger.exception(
                    "VLM batch analizi basarisiz (evidence=%s, model=%s); diger batch'ler etkilenmeyecek.",
                    batch_evidence_ids,
                    self.model_name,
                )
                response = VLMResponse(
                    description=(
                        f"[ANALYSIS_FAILED] Evidence {batch_evidence_ids} icin VLM analizi basarisiz: {exc}"
                    ),
                    model_name=self.model_name,
                    frame_count=len(batch),
                    latency_ms=0.0,
                    structured_events=[],
                    status="failed",
                    evidence_ids=batch_evidence_ids,
                )
            responses.append(response)
            # CANLI YAYIN: batch biter bitmez operatore bildirilir. Bu nokta,
            # sampler ciktisi ile VLM ciktisinin ZATEN ESLESMIS oldugu tek
            # yerdir - `evidence_ids` bu batch'teki kanit karelerini, ayni
            # sozlukteki `description`/`structured_events` ise VLM'in TAM O
            # KARELER icin urettigini tasir. Onceden ikisi de yalnizca TUM
            # analiz bittikten SONRA, tek seferde gorunuyordu.
            if on_batch is not None:
                on_batch(
                    {
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "evidence_ids": batch_evidence_ids,
                        "description": response.description,
                        "structured_events": response.structured_events,
                        "status": response.status,
                        "latency_ms": response.latency_ms,
                    }
                )
        return responses

    def reconcile_events(self, batch_responses: List[VLMResponse], prompt: str) -> VLMResponse:
        """Batch-yerel olaylari IKINCI, metin-tabanli bir VLM cagrisiyla TEK global olay listesine birlestirir.

        Yalnizca `analyze_evidence_batched` BIRDEN FAZLA batch urettiginde
        gereklidir (tek batch zaten globaldir). Basarisiz batch'ler
        reconciliation girdisine DAHIL EDILMEZ (analiz edilememis, dolayisiyla
        birlestirilecek bir olay bilgisi yok) ama evidence_id'leri
        `VLMResponse.evidence_ids`de KORUNUR; cagiran taraf (bkz.
        `src/main.py::_reconcile_unassigned_evidence`) bunlari acikca
        `unassigned` olarak isaretler - evidence hicbir zaman sessizce
        kaybolmaz.

        Args:
            batch_responses: `analyze_evidence_batched` ciktisi (birden fazla
                batch, karisik basarili/basarisiz olabilir).
            prompt: Kullanici/istem metni (baglam icin reconciliation
                promptuna eklenir).

        Returns:
            Tum basarili batch'lerin evidence_id'lerini kapsayan, GLOBAL
            `event_id`li tek bir `VLMResponse`. Basarili batch yoksa
            `status="failed"` doner.

        Raises:
            RuntimeError: Reconciliation VLM cagrisi basarisiz olursa.
        """
        succeeded = [r for r in batch_responses if r.status != "failed"]
        if not succeeded:
            all_ids = [eid for r in batch_responses for eid in r.evidence_ids]
            return VLMResponse(
                description="[HATA] Hicbir batch basariyla analiz edilemedi; birlestirilecek olay yok.",
                model_name=self.model_name,
                frame_count=0,
                latency_ms=0.0,
                structured_events=[],
                status="failed",
                evidence_ids=all_ids,
            )
        if len(succeeded) == 1 and len(batch_responses) == 1:
            return succeeded[0]

        payload = self._build_reconciliation_payload(batch_responses, prompt)
        response = self._post_chat_completion(payload)
        response.evidence_ids = [eid for r in batch_responses for eid in r.evidence_ids]
        response.frame_count = sum(r.frame_count for r in batch_responses)
        return response

    def _build_reconciliation_payload(
        self, batch_responses: List[VLMResponse], prompt: str
    ) -> Dict[str, Any]:
        """Reconciliation icin METIN-TABANLI (goruntusuz) bir chat payload'i kurar.

        Her batch'in `structured_events`ini (zaten ayristirilmis EVENTS_JSON)
        JSON metni olarak reconciliation promptuna gomer; basarisiz
        batch'lerin evidence_id'lerini de ayri bir listede belirtir (boylece
        model bunlari da `unassigned` kaydina dahil edebilir).

        Args:
            batch_responses: `analyze_evidence_batched` ciktisi.
            prompt: Kullanici/istem metni (baglam icin eklenir).

        Returns:
            `/v1/chat/completions` icin JSON-serilestirilebilir, YALNIZCA
            METIN iceren istek govdesi (resim yok).
        """
        batch_summaries = [
            {"batch_index": i, "events": r.structured_events}
            for i, r in enumerate(batch_responses)
            if r.status != "failed"
        ]
        failed_evidence_ids = [eid for r in batch_responses if r.status == "failed" for eid in r.evidence_ids]

        text = (
            f"Kullanici istemi (baglam): {prompt}\n\n"
            f"Batch-yerel olay listeleri (JSON):\n{json.dumps(batch_summaries, ensure_ascii=False, indent=2)}\n\n"
            f"Analiz EDILEMEYEN (basarisiz batch) evidence_id'leri (bunlari da "
            f"'unassigned' kaydina ekle, KAYBETME): {json.dumps(failed_evidence_ids)}"
        )
        content = [
            {"type": "text", "text": VLM_RECONCILIATION_SYSTEM_PROMPT},
            {"type": "text", "text": text},
        ]
        payload: Dict[str, Any] = {
            "model": self._endpoint.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self._endpoint.max_new_tokens,
            "temperature": self._endpoint.temperature,
            "top_p": self._endpoint.top_p,
        }
        apply_extra_body(payload, self._endpoint.extra_body)
        return payload

    def answer_video_question(self, video_source: str, question: str, analysis_summary: str) -> str:
        """Ayni videoya, kalici raporun onceki analizinden SONRA yeni bir soru sorar.

        Varsayilan olarak DESTEKLENMEZ - yalnizca `EvrenVLM` (video-tabanli,
        prefix-cache avantajina sahip tek saglayici, bkz. o modulun
        dokustringi) bunu implemente eder. `QwenVLM`/`GemmaVLM`/mock
        istemciler icin cagiran taraf (`AskService`) bu istisnayi YAKALAYIP
        metin-tabanli soru-cevap akisina SESSIZCE geri doner - bu asla
        kullaniciya gorunen bir hataya donusmez.

        Args:
            video_source: Daha once analiz edilmis videonun yerel dosya yolu.
            question: Kullanicinin bu video hakkindaki yeni (takip) sorusu.
            analysis_summary: Videonun daha once uretilmis kisa metin ozeti
                (baglam icindir, KANIT degildir).

        Returns:
            Modelin videoyu dogrudan izleyerek urettigi Turkce serbest-metin cevap.

        Raises:
            NotImplementedError: Bu saglayici video-QA'yi desteklemiyorsa (varsayilan).
        """
        raise NotImplementedError(
            f"{type(self).__name__} video-tabanli takip sorusu (answer_video_question) desteklemiyor."
        )

    @abstractmethod
    def health_check(self) -> bool:
        """Modelin vLLM servisinin ayakta olup olmadigini kontrol eder.

        Returns:
            Servis erisilebilir ve saglikliysa `True`.
        """
        raise NotImplementedError

    def _build_chat_payload(
        self, evidence_frames: List[EvidenceFrame], prompt: str
    ) -> Dict[str, Any]:
        """vLLM'in OpenAI-uyumlu `/chat/completions` uc noktasi icin istek govdesi kurar.

        Icerik bloklari (evidence karelerinin base64 goruntusu + kimlik/zaman
        metadatasi) `VLMPayloadBuilder` tarafindan uretilir; bu metod yalnizca
        model-ozel alanlarla (model adi, sicaklik, token siniri) sarmalar.

        Args:
            evidence_frames: Modele gonderilecek evidence kareleri.
            prompt: Kullanici/istem metni.

        Returns:
            `/v1/chat/completions` icin JSON-serilestirilebilir istek govdesi.
        """
        content = VLMPayloadBuilder.build_content_blocks(evidence_frames, prompt)

        payload: Dict[str, Any] = {
            "model": self._endpoint.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self._endpoint.max_new_tokens,
            "temperature": self._endpoint.temperature,
            "top_p": self._endpoint.top_p,
        }
        apply_extra_body(payload, self._endpoint.extra_body)
        return payload

    def _post_chat_completion(self, payload: Dict[str, Any]) -> VLMResponse:
        """Hazirlanan istegi vLLM servisine gonderir ve yaniti `VLMResponse`'a cevirir.

        Args:
            payload: `_build_chat_payload` (veya `_build_reconciliation_payload`)
                ile uretilmis istek govdesi.

        Returns:
            Model ciktisini iceren `VLMResponse`.

        Raises:
            RuntimeError: HTTP istegi basarisiz olursa veya yanit beklenmedik
                bicimde gelirse.
        """
        started_at = time.perf_counter()
        # Yeniden deneme politikasi ORTAK `send_with_retry`den gelir (ayni
        # politika video-dogrudan/"agir" yolda da kullanilir - bkz.
        # `src/vlm/gemini_vlm.py::GeminiVLM._generate`). Bozuk yanit
        # (KeyError/IndexError) ve bos icerik yeniden DENENMEZ: bunlar ag
        # kaynakli gecici hatalar degildir.
        response = send_with_retry(
            lambda: httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._endpoint.auth_headers(),
                timeout=60.0,
            ),
            source=self.model_name,
        )
        try:
            data = response.json()
            raw_content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"VLM yaniti beklenmedik bicimde ({self.model_name}): {exc}") from exc
        raise_if_empty_content(raw_content, self.model_name, data)

        # Yeni Typed Parser kullanimi (Legacy Adapter destekli)
        from src.vlm.parser import parse_vlm_response
        from src.vlm.time_normalizer import normalize_observation_time
        from src.vlm.schemas import VLMAnalysisStatus
        
        description, chunk_res = parse_vlm_response(raw_content)
        
        structured_events = []
        has_invalid = False
        all_invalid = True
        
        if chunk_res.report and chunk_res.report.observations:
            for obs in chunk_res.report.observations:
                # EvrenFrames veya diger frame-tabanli cagrilar tum videoyu tek bir 'chunk' olarak 
                # (veya cercevelerin gercek video zamanlarini) kabul ettigi icin offset 0.
                norm = normalize_observation_time(obs, 0.0, None)
                
                if norm.time_status == "invalid":
                    has_invalid = True
                    chunk_res.analysis_status = VLMAnalysisStatus.PARTIAL
                    continue
                    
                all_invalid = False
                
                # `canonical_event_type` + `keywords`: modelin KENDI urettigi
                # taksonomi eslemesi ve serbest terimler. Bu iki alan onceden
                # BURADA dusuruluyordu; sonucta `EventEngine`e giden her olay
                # `keywords=[]` ile ulasiyor ve sistem sabit `_KEYWORD_RULES`
                # (10 kategori) taksonomisine geri dusuyordu. Esleme mantigi
                # artik `src/vlm/parser.py::observations_to_structured_events`
                # ile AYNI sozlesmeyi izler (video-dogrudan yol da onu kullanir).
                _keywords: List[str] = []
                for _term in list(obs.attributes or []) + list(obs.entities or []):
                    _cleaned = str(_term).strip()
                    if _cleaned and _cleaned not in _keywords:
                        _keywords.append(_cleaned)

                structured_events.append({
                    "event_name": obs.observed_label,
                    "canonical_event_type": obs.canonical_type,
                    "taxonomy_status": getattr(obs.taxonomy_status, "value", obs.taxonomy_status),
                    "keywords": _keywords,
                    "description": obs.observed_label,
                    "uncertainties": list(obs.uncertainties or []),
                    "confidence": obs.confidence,
                    "start_time": norm.global_start_sec,
                    "end_time": norm.global_end_sec,
                    "evidence_ids": obs.evidence,
                    "normalized_relative_start_sec": norm.normalized_relative_start_sec,
                    "normalized_relative_end_sec": norm.normalized_relative_end_sec,
                    "was_adjusted": norm.was_adjusted,
                    "adjustment_reasons": norm.adjustment_reasons,
                    "time_status": norm.time_status,
                    "time_base": norm.time_base,
                })
        
        if chunk_res.report and chunk_res.report.observations and all_invalid:
            chunk_res.analysis_status = VLMAnalysisStatus.PARTIAL
            chunk_res.parse_status = "all_times_invalid"

        latency_ms = (time.perf_counter() - started_at) * 1000
        image_count = sum(
            1 for item in payload["messages"][0]["content"] if item["type"] == "image_url"
        )
        return VLMResponse(
            description=description,
            model_name=self.model_name,
            frame_count=image_count,
            latency_ms=latency_ms,
            structured_events=structured_events,
            chunk_analysis_result=chunk_res,
        )

    def health_check_impl(self) -> bool:
        """`/v1/models` uc noktasina istek atarak servisin ayakta olup olmadigini dogrular.

        Returns:
            Servis 200 ile yanit veriyorsa `True`, aksi halde `False`.
        """
        try:
            response = httpx.get(
                f"{self.base_url}/models", headers=self._endpoint.auth_headers(), timeout=5.0
            )
            return response.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("VLM saglik kontrolu basarisiz (%s): %s", self.model_name, exc)
            return False
