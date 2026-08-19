"""03 - Gorsel Dil Modeli Katmani: Gemini (harici API) implementasyonu.

GEcici GELISTIRME/TEST backend'i: yerel VRAM (8 GB) iki 3B modeli makul hizda
kosmaya yetmediginde, ayni pipeline'i Google Gemini'nin OpenAI-uyumlu uc
noktasi uzerinden test etmeyi saglar. `BaseVLM._post_chat_completion` zaten
OpenAI-uyumlu `chat/completions` + base64 `image_url` bloklariyla calistigi
icin, bu sinif yalnizca Gemini'ye ozel sistem istemini ekler; taban adres ve
`Authorization` header'i `VLLMEndpointConfig` (`base_url`/`api_key_env`)
uzerinden cozulur.

ONEMLI: Harici API kullanimi sartnamenin "tamamen yerel/offline" gereksinimini
ihlal eder ve yalnizca gelistirme/test icindir. Yarisma teslimi icin VLM
`active_model` degeri her zaman yerel bir modele (qwen/gemma) ayarlanmalidir.
"""

from __future__ import annotations

import logging
from typing import List

from src.prompts import VLM_OBSERVER_SYSTEM_PROMPT
from src.sampler.schema import EvidenceFrame
from src.vlm.base_vlm import BaseVLM, VLMResponse

logger = logging.getLogger(__name__)


class GeminiVLM(BaseVLM):
    """Gemini modelini OpenAI-uyumlu uc nokta uzerinden kullanan (test amacli) VLM implementasyonu."""

    def analyze_evidence(self, evidence_frames: List[EvidenceFrame], prompt: str) -> VLMResponse:
        """Evidence karelerini Gemini'ye gonderip olay kumeleme + Turkce gozlem uretir.

        Args:
            evidence_frames: `AdaptiveFrameSampler.process_video` ciktisi,
                kronolojik evidence kareleri (bir batch).
            prompt: Analiz odagini belirten ek istem (bos olabilir).

        Returns:
            Gemini tarafindan uretilen, kumelenmis olaylari ve dogal dil
            gozlemini iceren `VLMResponse`.

        Raises:
            RuntimeError: Gemini uc noktasina erisilemezse veya evidence bulunamazsa.
        """
        if not evidence_frames:
            raise RuntimeError("Gemini'ye gonderilecek evidence karesi bulunamadi.")

        full_prompt = f"{VLM_OBSERVER_SYSTEM_PROMPT}\n\nEk istem: {prompt}".strip()
        payload = self._build_chat_payload(evidence_frames, full_prompt)
        logger.info(
            "Gemini VLM cagrisi yapiliyor: %d evidence karesi, model=%s",
            len(evidence_frames),
            self.model_name,
        )
        return self._post_chat_completion(payload)

    def health_check(self) -> bool:
        """Gemini uc noktasinin (models listesi) erisilebilir olup olmadigini kontrol eder."""
        return self.health_check_impl()
