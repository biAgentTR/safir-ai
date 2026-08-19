"""03 - Gorsel Dil Modeli Katmani: Gemma 3 Vision implementasyonu (vLLM uzerinden)."""

from __future__ import annotations

import logging
from typing import List

from src.prompts import VLM_OBSERVER_SYSTEM_PROMPT
from src.sampler.schema import EvidenceFrame
from src.vlm.base_vlm import BaseVLM, VLMResponse

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = VLM_OBSERVER_SYSTEM_PROMPT


class GemmaVLM(BaseVLM):
    """Gemma 3 Vision modelini yerel vLLM servisi uzerinden kullanan VLM implementasyonu."""

    def analyze_evidence(self, evidence_frames: List[EvidenceFrame], prompt: str) -> VLMResponse:
        """Evidence karelerini Gemma 3 Vision'a gonderip olay kumeleme + aciklama uretir.

        Args:
            evidence_frames: `AdaptiveFrameSampler.process_video` ciktisi,
                kronolojik evidence kareleri (bir batch).
            prompt: Analiz odagini belirten ek istem (bos olabilir).

        Returns:
            Gemma 3 Vision tarafindan uretilen, kumelenmis olaylari ve dogal
            dil aciklamasini iceren `VLMResponse`.

        Raises:
            RuntimeError: vLLM servisine erisilemezse veya evidence bulunamazsa.
        """
        if not evidence_frames:
            raise RuntimeError("Gemma 3 Vision'a gonderilecek evidence karesi bulunamadi.")

        full_prompt = f"{_DEFAULT_SYSTEM_PROMPT}\n\nEk istem: {prompt}".strip()
        payload = self._build_chat_payload(evidence_frames, full_prompt)
        logger.info(
            "Gemma 3 Vision cagrisi yapiliyor: %d evidence karesi, model=%s",
            len(evidence_frames),
            self.model_name,
        )
        return self._post_chat_completion(payload)

    def health_check(self) -> bool:
        """Gemma 3 Vision vLLM servisinin erisilebilir olup olmadigini kontrol eder."""
        return self.health_check_impl()
