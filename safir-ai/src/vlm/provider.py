"""03 - Provider-Agnostic VLM (Vision-Language Model) Katmanı.

Bu modül, görsel dil modelleri (VLM) çağrılarını herhangi bir harici veya yerel
sağlayıcıya sıkı sıkıya bağımlı kalmayacak şekilde soyutlar (Provider-Agnostic Architecture).
Üretim ortamında yerel vLLM (Qwen2-VL) kullanılırken, geliştirme/test ortamında
geçici Gemini API veya birim testlerde MockVLMProvider dinamik olarak seçilebilir.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class VLMProvider(ABC):
    """Soyut VLM Sağlayıcı Arayüzü (Provider-Agnostic VLM Interface).

    Tüm somut VLM sağlayıcıları (GeminiProvider, VLLMQwenProvider, MockVLMProvider)
    bu arayüzü uygulamak zorundadır.
    """

    @abstractmethod
    def analyze_frame(self, image_base64: str, prompt: str) -> str:
        """Tek bir kareyi (base64 formatında) verilen istem doğrultusunda analiz eder.

        Args:
            image_base64: Base64 formatında kodlanmış görsel verisi (veya data URI).
            prompt: Analiz talimatı / istemi.

        Returns:
            VLM tarafından üretilen yanıt metni.
        """
        pass


class GeminiProvider(VLMProvider):
    """
    =============================================================================
    GEÇİCİ GELİŞTİRME VE TEST IMPLEMENTASYONU (TEMPORARY DEV/TEST FALLBACK)
    =============================================================================
    DİKKAT: Bu sınıf yalnızca yerel GPU / VRAM kapasitesi kısıtlı geliştirme ve
    test ortamlarında harici Google Gemini OpenAI-uyumlu API'sini kullanmak için
    geçici olarak tasarlanmıştır.

    ÜRETİM/TESLİM ORTAMINDA:
    Şartname gereği yerel vLLM (Qwen2-VL) sunucusu (`VLLMQwenProvider`)
    kullanılmalıdır. Harici API çağrıları üretim ortamında kullanılmaz.
    =============================================================================
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("VLM_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL_NAME")
            or "gemini-3.5-flash-lite"
        )
        if not self.base_url.endswith("/"):
            self.base_url += "/"

    def analyze_frame(self, image_base64: str, prompt: str) -> str:
        """Gemini OpenAI-compatible endpoint'ine kare analizi isteği atar."""
        if not self.api_key:
            raise ValueError(
                "GeminiProvider için API anahtarı bulunamadı. "
                "Lütfen 'GEMINI_API_KEY' ortam değişkenini tanımlayın."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        img_url = image_base64 if image_base64.startswith("data:") else f"data:image/jpeg;base64,{image_base64}"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
        }

        url = f"{self.base_url}chat/completions"
        logger.info(
            "[GeminiProvider - DEV/TEST] Gemini VLM çağrısı yapılıyor (model=%s)",
            self.model_name,
        )

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("GeminiProvider çağrısı başarısız: %s", exc)
            raise RuntimeError(f"Gemini VLM analizi sırasında hata oluştu: {exc}") from exc


class VLLMQwenProvider(VLMProvider):
    """
    =============================================================================
    ÜRETİM VE TESLİM ORTAMI YEREL VLM İSKELETİ (PRODUCTION LOCAL VLM PROVIDER)
    =============================================================================
    Yerel vLLM sunucusu üzerinde koşan Qwen2-VL modelini OpenAI-compatible API
    üzerinden çağırır. Şartnameye %100 uyumlu, internet bağlantısı gerektirmeyen
    offline yerel çıkarım (local inference) sağlayıcısıdır.
    =============================================================================
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("VLLM_BASE_URL")
            or "http://localhost:8001/v1/"
        )
        self.model_name = (
            model_name
            or os.getenv("VLLM_MODEL_NAME")
            or "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        if not self.base_url.endswith("/"):
            self.base_url += "/"

    def analyze_frame(self, image_base64: str, prompt: str) -> str:
        """Yerel vLLM (Qwen2-VL) sunucusuna OpenAI-compatible chat completion isteği atar."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        img_url = image_base64 if image_base64.startswith("data:") else f"data:image/jpeg;base64,{image_base64}"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }

        url = f"{self.base_url}chat/completions"
        logger.info(
            "[VLLMQwenProvider - LOCAL] Yerel vLLM Qwen2-VL çağrısı yapılıyor: url=%s model=%s",
            url,
            self.model_name,
        )

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("VLLMQwenProvider çağrısı başarısız: %s", exc)
            raise RuntimeError(
                f"Yerel vLLM Qwen2-VL sunucusuna erişilemedi ({url}). "
                "Lütfen vLLM servisinin çalıştığından emin olun."
            ) from exc


class MockVLMProvider(VLMProvider):
    """
    =============================================================================
    BİRİM TEST VE ÇEVRİMDIŞI GELİŞTİRME SAĞLAYICISI (MOCK VLM PROVIDER)
    =============================================================================
    Harici API veya yerel GPU çalıştırmadan, birim testlerde ve offline arayüz
    geliştirmelerinde kullanılmak üzere sabit/temsili test yanıtları döndürür.
    =============================================================================
    """

    def __init__(self, mock_response: Optional[str] = None) -> None:
        self.mock_response = mock_response or (
            "[MOCK VLM PROVIDER] Sahada rutin çalışma devam etmektedir. "
            "Personelin KKD (baret, yansıtıcı yelek) kullanımı uygun görülmektedir. "
            "Herhangi bir duman, yangın veya yetkisiz erişim riski saptanmamıştır."
        )

    def analyze_frame(self, image_base64: str, prompt: str) -> str:
        """Birim testler için sabit/temsili analiz çıktısı döndürür."""
        logger.info("[MockVLMProvider] Sahte VLM çıktısı döndürülüyor.")
        return self.mock_response


def get_vlm_provider(provider_name: Optional[str] = None) -> VLMProvider:
    """`VLM_PROVIDER` ortam değişkeni veya verilen isme göre VLM sağlayıcısı üretir.

    Seçenekler:
    - 'gemini': GeminiProvider (Geliştirme/Test)
    - 'vllm' veya 'qwen': VLLMQwenProvider (Üretim/Yerel vLLM)
    - 'mock': MockVLMProvider (Birim Testler / Offline)

    Args:
        provider_name: 'gemini', 'vllm' veya 'mock'. Verilmezse `VLM_PROVIDER` env
            değişkenine (varsayılan 'gemini') bakılır.

    Returns:
        Somut `VLMProvider` sınıf örneği.
    """
    selected = (
        provider_name
        or os.getenv("VLM_PROVIDER")
        or "gemini"
    ).strip().lower()

    if selected in ("gemini", "google"):
        logger.info("VLMProvider Fabrikası: 'GeminiProvider' seçildi (Geliştirme/Test).")
        return GeminiProvider()
    elif selected in ("vllm", "qwen", "local"):
        logger.info("VLMProvider Fabrikası: 'VLLMQwenProvider' seçildi (Yerel/Üretim).")
        return VLLMQwenProvider()
    elif selected in ("mock", "test"):
        logger.info("VLMProvider Fabrikası: 'MockVLMProvider' seçildi (Birim Test).")
        return MockVLMProvider()
    else:
        logger.warning(
            "Bilinmeyen VLM_PROVIDER '%s', varsayılan 'GeminiProvider'a düşülüyor.",
            selected,
        )
        return GeminiProvider()
