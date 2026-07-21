"""03 - Gorsel Dil Modeli Katmani: Gemma 3 Vision implementasyonu (vLLM uzerinden)."""

from __future__ import annotations

import logging
from typing import List

from src.sampler.adaptive_sampler import EvidenceFrame
from src.vlm.base_vlm import BaseVLM, VLMResponse

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "Bir saha analiz asistanisin. Verilen kanit karelerini zaman sirasiyla "
    "incele; sahnedeki insan/ekipman hareketlerini, olasi tehlikeleri ve "
    "onemli degisimleri Turkce, aciklayici bir sekilde ozetle."
)


class GemmaVLM(BaseVLM):
    """Gemma 3 Vision modelini yerel vLLM servisi uzerinden kullanan VLM implementasyonu."""

    def describe_events(
        self, evidence_frames: List[EvidenceFrame], prompt: str
    ) -> VLMResponse:
        """Kanit karelerini Gemma 3 Vision'a gonderip zamansal olay aciklamasi uretir.

        Args:
            evidence_frames: `AdaptiveSampler` ciktisi kanit kareleri.
            prompt: Analiz odagini belirten ek istem (bos olabilir).

        Returns:
            Gemma 3 Vision tarafindan uretilen dogal dil aciklamasini iceren
            `VLMResponse`.

        Raises:
            RuntimeError: vLLM servisine erisilemezse.
        """
        if not evidence_frames:
            raise RuntimeError("Gemma 3 Vision'a gonderilecek kanit karesi bulunamadi.")

        full_prompt = f"{_DEFAULT_SYSTEM_PROMPT}\n\nEk istem: {prompt}".strip()
        payload = self._build_chat_payload(evidence_frames, full_prompt)
        logger.info(
            "Gemma 3 Vision cagrisi yapiliyor: %d kare, model=%s",
            len(evidence_frames),
            self.model_name,
        )
        return self._post_chat_completion(payload)

    def health_check(self) -> bool:
        """Gemma 3 Vision vLLM servisinin erisilebilir olup olmadigini kontrol eder."""
        return self.health_check_impl()
