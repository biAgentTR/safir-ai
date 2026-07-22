"""03 - Gorsel Dil Modeli Katmani: Gemma 3 Vision implementasyonu (vLLM uzerinden)."""

from __future__ import annotations

import logging
from typing import List

from src.sampler.adaptive_sampler import EventCluster
from src.vlm.base_vlm import BaseVLM, VLMResponse

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "Bir saha analiz asistanisin. Verilen Olay Gruplarinin zirve karelerini "
    "zaman sirasiyla incele; sahnedeki insan/ekipman hareketlerini, olasi "
    "tehlikeleri ve onemli degisimleri Turkce, aciklayici bir sekilde ozetle."
)


class GemmaVLM(BaseVLM):
    """Gemma 3 Vision modelini yerel vLLM servisi uzerinden kullanan VLM implementasyonu."""

    def describe_events(
        self, clusters: List[EventCluster], prompt: str
    ) -> VLMResponse:
        """Olay Gruplarinin zirve karelerini Gemma 3 Vision'a gonderip aciklama uretir.

        Args:
            clusters: `AdaptiveFrameSampler.cluster_events` ciktisi Olay Gruplari.
            prompt: Analiz odagini belirten ek istem (bos olabilir).

        Returns:
            Gemma 3 Vision tarafindan uretilen dogal dil aciklamasini iceren
            `VLMResponse`.

        Raises:
            RuntimeError: vLLM servisine erisilemezse veya Olay Grubu bulunamazsa.
        """
        if not clusters:
            raise RuntimeError("Gemma 3 Vision'a gonderilecek Olay Grubu bulunamadi.")

        full_prompt = f"{_DEFAULT_SYSTEM_PROMPT}\n\nEk istem: {prompt}".strip()
        payload = self._build_chat_payload(clusters, full_prompt)
        logger.info(
            "Gemma 3 Vision cagrisi yapiliyor: %d olay grubu, model=%s",
            len(clusters),
            self.model_name,
        )
        return self._post_chat_completion(payload)

    def health_check(self) -> bool:
        """Gemma 3 Vision vLLM servisinin erisilebilir olup olmadigini kontrol eder."""
        return self.health_check_impl()
