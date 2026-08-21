"""Provider-Agnostic VLM Mimarisi Birim Testleri."""

from __future__ import annotations

import os
import unittest

from src.vlm.provider import (
    GeminiProvider,
    MockVLMProvider,
    VLLMQwenProvider,
    VLMProvider,
    get_vlm_provider,
)


class TestVLMProviderArchitecture(unittest.TestCase):
    """VLMProvider soyutlamasının ve somut sınıflarının doğru yapısını test eder."""

    def test_mock_vlm_provider(self):
        """MockVLMProvider'ın VLMProvider arayüzünü uyguladığını ve mock yanıt döndüğünü doğrular."""
        provider = MockVLMProvider()
        self.assertIsInstance(provider, VLMProvider)

        dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        result = provider.analyze_frame(dummy_b64, "Sahnede tehlike var mı?")

        self.assertIsInstance(result, str)
        self.assertIn("MOCK VLM PROVIDER", result)

    def test_vllm_qwen_provider_structure(self):
        """VLLMQwenProvider iskeletinin varsayılan ayarları doğru okuduğunu doğrular."""
        provider = VLLMQwenProvider()
        self.assertIsInstance(provider, VLMProvider)
        self.assertIn("8001", provider.base_url)
        self.assertEqual(provider.model_name, "Qwen/Qwen2.5-VL-7B-Instruct")

    def test_gemini_provider_structure(self):
        """GeminiProvider'ın geçici test yapılandırmasını doğru kurduğunu doğrular."""
        provider = GeminiProvider(api_key="test_key")
        self.assertIsInstance(provider, VLMProvider)
        self.assertEqual(provider.api_key, "test_key")
        self.assertEqual(provider.model_name, "gemini-3.5-flash-lite")

    def test_factory_get_vlm_provider(self):
        """get_vlm_provider fabrika fonksiyonunun doğru sağlayıcıyı ürettiğini doğrular."""
        # 1. Mock
        mock_p = get_vlm_provider("mock")
        self.assertIsInstance(mock_p, MockVLMProvider)

        # 2. Local / VLLM / Qwen
        vllm_p = get_vlm_provider("vllm")
        self.assertIsInstance(vllm_p, VLLMQwenProvider)

        # 3. Gemini / Dev / Test
        gemini_p = get_vlm_provider("gemini")
        self.assertIsInstance(gemini_p, GeminiProvider)

        # 4. Env variable override
        os.environ["VLM_PROVIDER"] = "mock"
        env_p = get_vlm_provider()
        self.assertIsInstance(env_p, MockVLMProvider)
        del os.environ["VLM_PROVIDER"]


if __name__ == "__main__":
    unittest.main()
