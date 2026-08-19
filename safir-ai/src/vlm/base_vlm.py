"""03 - Gorsel Dil Modeli Katmani: tum VLM implementasyonlari icin soyut taban sinif."""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import httpx

from src.sampler.adaptive_sampler import EventCluster
from src.sampler.payload_builder import VLMPayloadBuilder
from src.utils.config_loader import VLLMEndpointConfig

logger = logging.getLogger(__name__)

# Gecici ag hatalarinda VLM cagrisinin kac kez yeniden deneneceği ve geri-cekilme tabani.
_MAX_INFERENCE_RETRIES = 2
_RETRY_BACKOFF_BASE_SEC = 0.5


@dataclass
class VLMResponse:
    """Bir VLM cagrisinin (tek bir istek/batch icin) standardize edilmis ciktisi."""

    description: str
    model_name: str
    frame_count: int
    latency_ms: float
    structured_events: List[Dict[str, Any]] = field(default_factory=list)
    """Modelin dogrudan urettigi tipli olaylar (bkz. `EVENTS_JSON` blogu):
    her biri `{"type", "timestamp", "confidence", "evidence"}`. Bos ise
    `EventEngine` anahtar-kelime fallback'ine duser (bkz. `event_engine.detect`)."""
    status: str = "completed"
    """Bu VLM cagrisinin durumu: `"completed"` (basarili), `"failed"`
    (bu batch/olay icin VLM cagrisi basarisiz oldu - description'da
    `[ANALYSIS_FAILED]` notu bulunur) veya (yalnizca `describe_events_batched`
    ciktilarini birlestiren `main.py::stage_vlm` tarafindan uretilen AGREGE
    yanitlarda) `"partial_failure"` (bazi olaylar basarili, bazilari basarisiz).
    Asla `risk=0`/basarili gibi yorumlanmamalidir - bkz. `describe_events_batched`."""
    cluster_event_ids: List[int] = field(default_factory=list)
    """Bu yanitin kapsadigi `EventCluster.event_id` degerleri (bkz.
    `describe_events_batched`); tek-cagri (eski/agrege) yanitlarda bos olabilir."""


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
    uzerinden baglanarak `describe_events` metodunu implemente etmelidir. Bu
    soyutlama, ajan/muhakeme katmaninin hangi VLM'in aktif oldugundan bagimsiz
    calismasini saglar.
    """

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
    def describe_events(
        self, clusters: List[EventCluster], prompt: str
    ) -> VLMResponse:
        """Olay Gruplarinin zirve karelerini analiz edip dogal dil aciklama uretir.

        Args:
            clusters: `AdaptiveFrameSampler.cluster_events` tarafindan uretilen,
                zaman sirali Olay Gruplari (her biri bir zirve kare tasir).
            prompt: Modelin odaklanmasi istenen olay/soruyu tanimlayan istem.

        Returns:
            Zamansal olaylarin dogal dile yakin aciklamasini iceren `VLMResponse`.

        Raises:
            RuntimeError: vLLM servisine erisilemezse veya yanit gecersizse.
        """
        raise NotImplementedError

    def describe_events_batched(
        self, clusters: List[EventCluster], prompt: str, batch_size: int = 1
    ) -> List[VLMResponse]:
        """Olay Gruplarini TEK bir dev payload yerine kontrollu batch'ler halinde analiz eder.

        `clusters`, `batch_size` buyuklugunde ardisik gruplara bolunur (varsayilan
        `1` = HER olay ayri ayri analiz edilir); her batch icin `describe_events`
        BAGIMSIZ olarak cagrilir. Bir batch'in VLM cagrisi basarisiz olursa
        (ag hatasi, gecersiz yanit, vb.) o batch icin `status="failed"` ve
        `[ANALYSIS_FAILED]` ile baslayan bir `VLMResponse` uretilir; ISTISNA
        DIGER batch'lere YAYILMAZ (bir olayin basarisiz olmasi digerlerinin
        sonucunu KAYBETMEZ). Basarisiz bir batch asla risk=0/basarili olarak
        yorumlanmamalidir - cagiran taraf `status`i kontrol etmelidir.

        Args:
            clusters: Analiz edilecek TUM Olay Gruplari (video geneli).
            prompt: Kullanici/istem metni (her batch'e aynen iletilir).
            batch_size: Bir VLM istegine dahil edilecek azami Olay Grubu
                sayisi. `1` (varsayilan) ile her olay kendi VLM istegini
                alir (kontrollu payload boyutu/timeout); daha buyuk bir
                deger, ilgili olaylari TEK istekte gruplar (daha az istek,
                daha buyuk payload).

        Returns:
            Girdi `clusters` ile AYNI sirada, her biri kendi `cluster_event_ids`
            alaniyla hangi olaylari kapsadigini belirten `VLMResponse` listesi.
            Bos `clusters` icin bos liste doner.
        """
        if not clusters:
            return []

        responses: List[VLMResponse] = []
        for start in range(0, len(clusters), max(1, batch_size)):
            batch = clusters[start : start + max(1, batch_size)]
            batch_event_ids = [c.event_id for c in batch]
            try:
                response = self.describe_events(batch, prompt)
                response.cluster_event_ids = batch_event_ids
                response.status = "completed"
            except Exception as exc:  # noqa: BLE001 - izole edilip diger batch'lere yayilmaz
                logger.exception(
                    "VLM batch analizi basarisiz (olay(lar)=%s, model=%s); diger batch'ler etkilenmeyecek.",
                    batch_event_ids,
                    self.model_name,
                )
                response = VLMResponse(
                    description=(
                        f"[ANALYSIS_FAILED] Olay(lar) {batch_event_ids} icin VLM analizi basarisiz: {exc}"
                    ),
                    model_name=self.model_name,
                    frame_count=sum(len(c.representative_frames) or 1 for c in batch),
                    latency_ms=0.0,
                    structured_events=[],
                    status="failed",
                    cluster_event_ids=batch_event_ids,
                )
            responses.append(response)
        return responses

    @abstractmethod
    def health_check(self) -> bool:
        """Modelin vLLM servisinin ayakta olup olmadigini kontrol eder.

        Returns:
            Servis erisilebilir ve saglikliysa `True`.
        """
        raise NotImplementedError

    def _build_chat_payload(
        self, clusters: List[EventCluster], prompt: str
    ) -> Dict[str, Any]:
        """vLLM'in OpenAI-uyumlu `/chat/completions` uc noktasi icin istek govdesi kurar.

        Icerik bloklari (zirve karelerin base64 goruntusu + olay metadatasi)
        `VLMPayloadBuilder` tarafindan uretilir; bu metod yalnizca model-ozel
        alanlarla (model adi, sicaklik, token siniri) sarmalar.

        Args:
            clusters: Modele gonderilecek Olay Gruplari.
            prompt: Kullanici/istem metni.

        Returns:
            `/v1/chat/completions` icin JSON-serilestirilebilir istek govdesi.
        """
        content = VLMPayloadBuilder.build_content_blocks(clusters, prompt)

        payload: Dict[str, Any] = {
            "model": self._endpoint.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self._endpoint.max_new_tokens,
            "temperature": self._endpoint.temperature,
            "top_p": self._endpoint.top_p,
        }
        # Saglayici-ozel guided decoding alanlari (vLLM); cekirdek alanlar ezilmez.
        for key, value in (self._endpoint.extra_body or {}).items():
            payload.setdefault(key, value)
        return payload

    def _post_chat_completion(self, payload: Dict[str, Any]) -> VLMResponse:
        """Hazirlanan istegi vLLM servisine gonderir ve yaniti `VLMResponse`'a cevirir.

        Args:
            payload: `_build_chat_payload` ile uretilmis istek govdesi.

        Returns:
            Model ciktisini iceren `VLMResponse`.

        Raises:
            RuntimeError: HTTP istegi basarisiz olursa veya yanit beklenmedik
                bicimde gelirse.
        """
        started_at = time.perf_counter()
        # Gecici ag hatalarina (baglanti/timeout/5xx) karsi ustel geri-cekilmeli
        # yeniden deneme; bozuk yanit (KeyError/IndexError) yeniden denenmez.
        last_exc: Exception | None = None
        for attempt in range(_MAX_INFERENCE_RETRIES + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._endpoint.auth_headers(),
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                raw_content = data["choices"][0]["message"]["content"]
                break
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"VLM yaniti beklenmedik bicimde ({self.model_name}): {exc}") from exc
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < _MAX_INFERENCE_RETRIES:
                    backoff = _RETRY_BACKOFF_BASE_SEC * (2**attempt)
                    logger.warning(
                        "VLM cagrisi basarisiz (deneme %d/%d, %s): %s — %.1fs sonra yeniden denenecek",
                        attempt + 1,
                        _MAX_INFERENCE_RETRIES + 1,
                        self.model_name,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
        else:
            raise RuntimeError(
                f"VLM cagrisi {_MAX_INFERENCE_RETRIES + 1} denemede basarisiz ({self.model_name}): {last_exc}"
            ) from last_exc

        # Modelin urettigi makine-okunur EVENTS_JSON blogunu ayristir; insan-okur
        # aciklamadan ayir (bos ise EventEngine anahtar-kelime fallback'ine duser).
        description, structured_events = parse_structured_events(raw_content)

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
