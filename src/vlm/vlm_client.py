from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import List, Optional

import urllib.error
import urllib.request


# =============================================================================
# Cikti veri modeli
# =============================================================================
@dataclass
class VLMDescription:
    """Bir kanit karesi icin VLM'in urettigi yapisal ISG gorsel aciklamasi.

    Attributes:
        description: Serbest metin Turkce sahne aciklamasi.
        detected_hazards: Tespit edilen ISG tehlike/ihlal etiketleri.
        model_name: Aciklamayi ureten model adi (veya "mock-vlm").
        is_mock: Aciklamanin mock olup olmadigi.
        source_frame: Aciklanan karenin dosya yolu (varsa).
    """

    description: str
    detected_hazards: List[str]
    model_name: str
    is_mock: bool
    source_frame: Optional[str] = None


# ISG denetim istemi (gercek Ollama cagrisi icin sistem+kullanici baglamı).
ISG_VLM_PROMPT = (
    "Sen bir Isci Saglığı ve Guvenligi (ISG) denetim uzmanisin. Sana verilen "
    "saha/insaat kamera goruntusunu Turkce olarak degerlendirir; kisisel koruyucu "
    "donanim (KKD) eksikliklerini (baret, is guvenligi yelegi, eldiven), tehlikeli "
    "yakinlik durumlarini (arac-yaya, yuksekte calisma), duzensizlik ve diger ISG "
    "risklerini kisa ve net maddeler halinde raporla. Yalnizca goruntude gordugun "
    "kanitlara dayan."
)


# =============================================================================
# VLM Istemcisi
# =============================================================================
class VLMClient:
    """Ollama tabanli VLM istemcisi; mock modda dis baglanti kurmadan calisir."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        use_mock: bool,
        request_timeout_sec: int = 120,
    ) -> None:
        """VLM istemcisini kurar.

        Args:
            base_url: Ollama sunucu adresi (orn. "http://localhost:11434").
            model_name: Kullanilacak VLM model adi (orn. "qwen2.5-vl").
            use_mock: True ise gercek servise baglanmadan sabit cikti doner.
            request_timeout_sec: Gercek cagri icin zaman asimi (sn).
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.use_mock = use_mock
        self.request_timeout_sec = request_timeout_sec

    # ------------------------------------------------------------------ #
    # Genel API
    # ------------------------------------------------------------------ #
    def describe_frame(self, image_path: str, prompt: Optional[str] = None) -> VLMDescription:
        """Verilen kanit karesi icin ISG odakli Turkce bir gorsel aciklama uretir.

        Args:
            image_path: Aciklanacak kareyi iceren `.jpg` yolu.
            prompt: Isteğe bagli ek kullanici istemi (mock modda goz ardi edilir).

        Returns:
            `VLMDescription` nesnesi.
        """
        if self.use_mock:
            return self._mock_description(image_path)
        return self._ollama_description(image_path, prompt or "")

    # ------------------------------------------------------------------ #
    # MOCK yol
    # ------------------------------------------------------------------ #
    def _mock_description(self, image_path: str) -> VLMDescription:
        """Deterministik, sabit bir Turkce ISG aciklamasi dondurur (mock)."""
        description = (
            "Sahada bir isci, hareket halindeki bir is makinesine tehlikeli olcude "
            "yakin konumda calismaktadir. Iscinin basinda koruyucu baret "
            "bulunmamakta ve is guvenligi yelegi eksik gorunmektedir. Zeminde "
            "duzensiz sekilde birakilmis malzemeler dusme/takilma riski "
            "olusturmaktadir. Genel gorunum, acil operator mudahalesi gerektiren "
            "yuksek riskli bir ISG ihlaline isaret etmektedir."
        )
        hazards = [
            "KKD ihlali: Baret takili degil",
            "KKD ihlali: Is guvenligi yelegi yok",
            "Arac-yaya tehlikeli yakinligi",
            "Zeminde duzensiz malzeme (dusme/takilma riski)",
        ]
        return VLMDescription(
            description=description,
            detected_hazards=hazards,
            model_name="mock-vlm",
            is_mock=True,
            source_frame=image_path,
        )

    # ------------------------------------------------------------------ #
    # GERCEK yol (Ollama /api/chat)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        """Bir resmi Base64 string'e cevirir (Ollama `images` alani icin)."""
        with open(image_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    def _ollama_description(self, image_path: str, extra_prompt: str) -> VLMDescription:
        """Ollama `/api/chat` uc noktasina Base64 resmi + ISG istemini gonderir.

        Args:
            image_path: Kanit karesinin dosya yolu.
            extra_prompt: Kullanicidan gelen ek odak istemi.

        Returns:
            Modelden donen aciklamayi iceren `VLMDescription`.

        Raises:
            RuntimeError: Ollama cagrisi basarisiz olursa.
        """
        image_b64 = self._encode_image_base64(image_path)
        user_content = ISG_VLM_PROMPT
        if extra_prompt.strip():
            user_content = f"{ISG_VLM_PROMPT}\n\nEk odak: {extra_prompt.strip()}"

        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": user_content,
                    "images": [image_b64],
                }
            ],
        }

        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/api/chat"
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Ollama VLM cagrisi basarisiz ({url}, model={self.model_name}): {exc}"
            ) from exc

        content = (body.get("message", {}) or {}).get("content", "").strip()
        if not content:
            raise RuntimeError("Ollama VLM bos yanit dondurdu.")

        return VLMDescription(
            description=content,
            detected_hazards=self._extract_hazards(content),
            model_name=self.model_name,
            is_mock=False,
            source_frame=image_path,
        )

    @staticmethod
    def _extract_hazards(text: str) -> List[str]:
        """Serbest metinden madde/satir bazli kaba tehlike etiketleri cikarir."""
        hazards: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip().lstrip("-*•0123456789. ").strip()
            if line:
                hazards.append(line)
        return hazards[:8]


def build_vlm_client(config) -> VLMClient:
    """`config.yaml`den bir `VLMClient` kurar (mock/gercek secimi dahil).

    Args:
        config: `load_config()` ciktisi (DotDict).

    Returns:
        Yapilandirilmis `VLMClient`.
    """
    return VLMClient(
        base_url=config.ollama.base_url,
        model_name=config.ollama.vlm_model,
        use_mock=bool(config.app.use_mock_vlm),
        request_timeout_sec=int(config.ollama.request_timeout_sec),
    )