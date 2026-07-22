"""03 - Gorsel Dil Modeli Katmani: Qwen2.5-VL implementasyonu (vLLM uzerinden)."""

from __future__ import annotations

import logging
from typing import List

from src.sampler.adaptive_sampler import EventCluster
from src.vlm.base_vlm import BaseVLM, VLMResponse

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "Sen bir saha guvenlik analistisin. Sana verilen Olay Gruplarinin zirve "
    "karelerini zaman sirasina gore analiz ederek sahnede olan olaylari, "
    "riskli davranislari ve dikkat cekici degisimleri Turkce, ayrintili ve "
    "nesnel bir sekilde tarif et."
)


class QwenVLM(BaseVLM):
    """Qwen2.5-VL modelini yerel vLLM servisi uzerinden kullanan VLM implementasyonu."""

    def describe_events(
        self, clusters: List[EventCluster], prompt: str
    ) -> VLMResponse:
        """Olay Gruplarinin zirve karelerini Qwen2.5-VL'e gonderip aciklama uretir.

        Args:
            clusters: `AdaptiveFrameSampler.cluster_events` ciktisi Olay Gruplari.
            prompt: Analiz odagini belirten ek istem (bos olabilir).

        Returns:
            Qwen2.5-VL tarafindan uretilen dogal dil aciklamasini iceren
            `VLMResponse`.

        Raises:
            RuntimeError: vLLM servisine erisilemezse veya Olay Grubu bulunamazsa.
        """
        if not clusters:
            raise RuntimeError("Qwen2.5-VL'e gonderilecek Olay Grubu bulunamadi.")

        full_prompt = f"{_DEFAULT_SYSTEM_PROMPT}\n\nEk istem: {prompt}".strip()
        payload = self._build_chat_payload(clusters, full_prompt)
        logger.info(
            "Qwen2.5-VL cagrisi yapiliyor: %d olay grubu, model=%s",
            len(clusters),
            self.model_name,
        )
        return self._post_chat_completion(payload)

    def health_check(self) -> bool:
        """Qwen2.5-VL vLLM servisinin erisilebilir olup olmadigini kontrol eder."""
        return self.health_check_impl()
