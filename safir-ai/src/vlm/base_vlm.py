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


@dataclass
class VLMResponse:
    """Bir VLM cagrisinin standardize edilmis ciktisi."""

    description: str
    model_name: str
    frame_count: int
    latency_ms: float
    structured_events: List[Dict[str, Any]] = field(default_factory=list)
    """Modelin dogrudan urettigi tipli olaylar (bkz. `EVENTS_JSON` blogu):
    her biri `{"type", "timestamp", "confidence", "evidence"}`. Bos ise
    `EventEngine` anahtar-kelime fallback'ine duser (bkz. `event_engine.detect`)."""


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

    events: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(match.group(1))
        if isinstance(parsed, list):
            events = [item for item in parsed if isinstance(item, dict)]
    except (json.JSONDecodeError, ValueError):
        logger.warning("VLM EVENTS_JSON blogu ayristirilamadi, anahtar-kelime fallback'ine dusulecek.")
        events = []

    clean_description = content[: match.start()].strip()
    return (clean_description or content.strip()), events


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

        return {
            "model": self._endpoint.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self._endpoint.max_new_tokens,
            "temperature": self._endpoint.temperature,
            "top_p": self._endpoint.top_p,
        }

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
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise RuntimeError(f"vLLM cagrisi basarisiz ({self.model_name}): {exc}") from exc

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
