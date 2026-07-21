"""03 - Gorsel Dil Modeli Katmani: tum VLM implementasyonlari icin soyut taban sinif."""

from __future__ import annotations

import base64
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import httpx

from src.sampler.adaptive_sampler import EvidenceFrame
from src.utils.config_loader import VLLMEndpointConfig

logger = logging.getLogger(__name__)


@dataclass
class VLMResponse:
    """Bir VLM cagrisinin standardize edilmis ciktisi."""

    description: str
    model_name: str
    frame_count: int
    latency_ms: float


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
        """Bu modelin vLLM OpenAI-uyumlu servisinin taban URL'sini dondurur."""
        return f"http://{self._endpoint.vllm_host}:{self._endpoint.vllm_port}/v1"

    @property
    def model_name(self) -> str:
        """Bu VLM icin yapilandirilmis Hugging Face model adini dondurur."""
        return self._endpoint.model_name

    @abstractmethod
    def describe_events(
        self, evidence_frames: List[EvidenceFrame], prompt: str
    ) -> VLMResponse:
        """Suzulmus kanit kareleri dizisini analiz edip dogal dil aciklama uretir.

        Args:
            evidence_frames: `AdaptiveSampler` tarafindan uretilen, zamansal
                sirali kanit kareleri.
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

    def _frame_to_data_uri(self, frame) -> str:
        """Bir OpenCV karesini base64 kodlu JPEG data URI'sine cevirir.

        Args:
            frame: BGR formatinda numpy dizisi olarak video karesi.

        Returns:
            `data:image/jpeg;base64,...` formatinda data URI.
        """
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Kare JPEG formatina kodlanamadi.")
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_chat_payload(
        self, evidence_frames: List[EvidenceFrame], prompt: str
    ) -> Dict[str, Any]:
        """vLLM'in OpenAI-uyumlu `/chat/completions` uc noktasi icin istek govdesi kurar.

        Args:
            evidence_frames: Modele gonderilecek kanit kareleri.
            prompt: Kullanici/istem metni.

        Returns:
            `/v1/chat/completions` icin JSON-serilestirilebilir istek govdesi.
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for evidence in evidence_frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._frame_to_data_uri(evidence.frame)},
                }
            )

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
                f"{self.base_url}/chat/completions", json=payload, timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            description = data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise RuntimeError(f"vLLM cagrisi basarisiz ({self.model_name}): {exc}") from exc

        latency_ms = (time.perf_counter() - started_at) * 1000
        image_count = sum(
            1 for item in payload["messages"][0]["content"] if item["type"] == "image_url"
        )
        return VLMResponse(
            description=description,
            model_name=self.model_name,
            frame_count=image_count,
            latency_ms=latency_ms,
        )

    def health_check_impl(self) -> bool:
        """`/v1/models` uc noktasina istek atarak servisin ayakta olup olmadigini dogrular.

        Returns:
            Servis 200 ile yanit veriyorsa `True`, aksi halde `False`.
        """
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("VLM saglik kontrolu basarisiz (%s): %s", self.model_name, exc)
            return False
