"""03 - Gorsel Dil Modeli Katmani: EVREN (TEKNOFEST yarisma cikarim servisi) implementasyonu.

EVREN'in video-analiz ucu (`model="vlm"`), diger VLM implementasyonlarindan
(`QwenVLM`/`GemmaVLM`) FARKLI olarak ayri evidence kareleri (resim) DEGIL,
DOGRUDAN tek bir video dosyasini (base64, `video_url` icerik blogu) kabul
eder (bkz. katilimci dokumantasyonu SS 7.1). Bu nedenle `EvrenVLM`,
`BaseVLM._build_chat_payload`in frame-tabanli akisini KULLANMAZ;
`analyze_video(video_source, evidence_frames, prompt)` ile videoyu TEK bir
istekte gonderir - sampler'in urettigi kareler yalnizca (mock istemciyle
PARITE icin) imzada tasinir, gercek saglayicida KULLANILMAZ.

`analyze_evidence` (frame-tabanli, eski/coklu-goruntu akisi) yalnizca
`BaseVLM` soyut sozlesmesini karsilamak icin tanimlanmistir; production
akisinda (`src/main.py::SafirPipeline.stage_vlm`) CAGRILMAZ - EVREN'in
istek basina en fazla 2 goruntu kabul etmesi nedeniyle (bkz. dokumantasyon
SS 7.5) kare-bolme/coklu-istek deseni kasitli olarak yeniden KURULMAZ.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import List

import httpx

from src.prompts import VLM_OBSERVER_SYSTEM_PROMPT
from src.sampler.schema import EvidenceFrame
from src.vlm.base_vlm import BaseVLM, VLMResponse, parse_structured_events

logger = logging.getLogger(__name__)

# EVREN dokumantasyonu (SS 7.1 "Uyari"): sistem video isteklerinde 1800s'ye
# kadar calisabiliyor; istemci zaman asimi da BUNUNLA UYUMLU ayarlanmalidir
# (aksi halde baglanti modelden ONCE kesilir, sonuc GORUNTULENEMEZ).
_EVREN_VIDEO_TIMEOUT_SEC = 1800.0

_VIDEO_MODE_NOTE = (
    "\n\nNOT: Bu istekte ayri evidence kareleri YOKTUR; tek bir video "
    "dogrudan gonderilmektedir. `evidence_ids` alanini bu nedenle HER ZAMAN "
    "BOS LISTE ([]) birak; `start_time`/`end_time` videonun basindan "
    "itibaren saniye cinsinden GERCEK zaman damgalari olmalidir."
)


class EvrenVLM(BaseVLM):
    """TEKNOFEST EVREN yarisma servisinin video-tabanli VLM ucunu kullanan implementasyon."""

    def analyze_video(
        self, video_source: str, evidence_frames: List[EvidenceFrame], prompt: str
    ) -> VLMResponse:
        """Video dosyasini DOGRUDAN (kare cikarma/bolme olmadan) EVREN'e gonderip olay kumeleme + Turkce gozlem uretir.

        Args:
            video_source: Yerel bir video dosyasinin yolu. RTSP/canli akis
                adresleri DESTEKLENMEZ - EVREN yalnizca base64 kodlanmis bir
                video govdesi kabul eder (bkz. dokumantasyon SS 7.1).
            evidence_frames: Yalnizca `MockVLMClient` ile arayuz PARITESI
                icin alinir; gercek EVREN cagrisinda KULLANILMAZ (EVREN
                videoyu kendisi analiz eder, sampler kareleri gerekmez).
            prompt: Analiz odagini belirten kullanici istemi.

        Returns:
            EVREN tarafindan uretilen, kumelenmis olaylari ve dogal dil
            gozlemini iceren `VLMResponse`.

        Raises:
            RuntimeError: `video_source` bir RTSP/canli akis adresiyse, video
                dosyasi okunamazsa veya EVREN cagrisi basarisiz olursa.
        """
        del evidence_frames  # yalnizca arayuz parametresi; bkz. docstring

        lowered = video_source.strip().lower()
        if lowered.startswith(("rtsp://", "http://", "https://")):
            raise RuntimeError(
                f"EVREN VLM yalnizca yerel video dosyalarini destekler (RTSP/canli akis DESTEKLENMEZ): {video_source}"
            )

        try:
            with open(video_source, "rb") as fh:
                video_b64 = base64.b64encode(fh.read()).decode("ascii")
        except OSError as exc:
            raise RuntimeError(f"EVREN icin video dosyasi okunamadi: {video_source} ({exc})") from exc

        full_prompt = f"{VLM_OBSERVER_SYSTEM_PROMPT}\n\nEk istem: {prompt}{_VIDEO_MODE_NOTE}".strip()
        payload = {
            "model": self._endpoint.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                    ],
                }
            ],
            "max_tokens": self._endpoint.max_new_tokens,
            "temperature": self._endpoint.temperature,
        }
        logger.info(
            "EVREN VLM video cagrisi yapiliyor: video=%s model=%s", video_source, self.model_name
        )

        started_at = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._endpoint.auth_headers(),
            timeout=_EVREN_VIDEO_TIMEOUT_SEC,
        )
        response.raise_for_status()
        data = response.json()
        try:
            raw_content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"EVREN yaniti beklenmedik bicimde: {exc}") from exc

        description, structured_events = parse_structured_events(raw_content)
        latency_ms = (time.perf_counter() - started_at) * 1000
        return VLMResponse(
            description=description,
            model_name=self.model_name,
            frame_count=0,
            latency_ms=latency_ms,
            structured_events=structured_events,
        )

    def analyze_evidence(self, evidence_frames: List[EvidenceFrame], prompt: str) -> VLMResponse:
        """Bu saglayicida DESTEKLENMEZ - production akisi `analyze_video` kullanir.

        Raises:
            NotImplementedError: her zaman - EVREN'de frame-tabanli/coklu-
                istek analiz kasitli olarak yeniden kurulmamistir (gorev
                kapsami: sampler'in VLM'e frame beslemesi kaldirildi).
        """
        raise NotImplementedError(
            "EvrenVLM frame-tabanli analyze_evidence'i desteklemez; production akisi "
            "SafirPipeline.stage_vlm uzerinden analyze_video(video_source, evidence_frames, prompt) kullanir."
        )

    def health_check(self) -> bool:
        """EVREN uc noktasinin (models listesi) erisilebilir olup olmadigini kontrol eder."""
        return self.health_check_impl()
