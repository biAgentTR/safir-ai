"""Modul 2 - VLM Client & Model Interface: VLM icin gercek/mock istemciler.

`VLMClient`, config'te secilen aktif VLM'i (Qwen2.5-VL / Gemma 3 Vision)
`VLMFactory` uzerinden kurup gercek vLLM servisine HTTP istegi atar; boylece
mevcut test edilmis `BaseVLM`/`QwenVLM`/`GemmaVLM` mantigi hic degistirilmeden
tek bir noktadan yeniden kullanilir. `MockVLMClient` ise GPU'su olmayan
gelistiriciler icin ayni `BaseVLM` sozlesmesini (`analyze_evidence`,
`health_check`) saniyenin onda biri kadar surede, sabit Turkce ISG tasvir
metniyle + gonderilen evidence_id'lere dayali bir EVENTS_JSON ile karsilar;
boylece `get_vlm_client(..., use_mock=True)` ile secildiginde main.py'deki
pipeline (kumeleme dahil) hicbir kod degisikligi olmadan uctan uca calisir.
"""

from __future__ import annotations

import logging
import time
from typing import List

from src.sampler.schema import EvidenceFrame
from src.utils.config_loader import VLLMEndpointConfig, VLMConfig
from src.vlm.base_vlm import BaseVLM, VLMResponse
from src.vlm.vlm_factory import VLMFactory

logger = logging.getLogger(__name__)

_MOCK_ENDPOINT = VLLMEndpointConfig(
    model_name="mock-vlm",
    vllm_host="mock",
    vllm_port=0,
    max_new_tokens=0,
    temperature=0.0,
)

_MOCK_DESCRIPTION = (
    "[MOCK] Sahada bir personel korumasiz bir alanda hareket ediyor; baret ve "
    "yansitici yelek eksik gorunuyor. Yakin cevrede forklift trafiginin oldugu "
    "gozlemleniyor. Herhangi bir duman veya yangin belirtisi tespit edilmedi."
)


class VLMClient(BaseVLM):
    """Config'teki aktif VLM'i gercek vLLM servisi uzerinden cagiran genel istemci.

    HTTP/payload mantigini tekrarlamamak icin somut isi `VLMFactory`
    tarafindan secilen `QwenVLM`/`GemmaVLM` orneğine devreder.
    """

    def __init__(self, vlm_config: VLMConfig) -> None:
        """VLMClient'i config'teki aktif VLM secimiyle baslatir.

        Args:
            vlm_config: `configs/config.yaml` icindeki `vlm` blogu.
        """
        self._delegate = VLMFactory.create(vlm_config)
        super().__init__(vlm_config.active_endpoint())

    def analyze_evidence(self, evidence_frames: List[EvidenceFrame], prompt: str) -> VLMResponse:
        """Cagriyi, config'te secilen gercek VLM implementasyonuna devreder.

        Args:
            evidence_frames: Analiz edilecek evidence kareleri.
            prompt: Kullanici/istem metni.

        Returns:
            Gercek vLLM servisinden donen `VLMResponse`.
        """
        return self._delegate.analyze_evidence(evidence_frames, prompt)

    def health_check(self) -> bool:
        """Devredilen gercek VLM implementasyonunun saglik durumunu dondurur."""
        return self._delegate.health_check()


class MockVLMClient(BaseVLM):
    """GPU'su olmayan gelistiriciler icin sahte VLM istemcisi (`use_mock_vlm: true`).

    Gercek vLLM servisine hic baglanmadan, `BaseVLM` sozlesmesine uygun sabit
    bir Turkce ISG sahne tasviri + gonderilen evidence karelerinin TAMAMINI
    tek bir sahte olayda kumeleyen bir `EVENTS_JSON` dondurur; boylece
    event-analysis katmani (evidence_ids dahil) mock modda da gercekci
    sekilde egzersiz edilir.
    """

    def __init__(self) -> None:
        """MockVLMClient'i (gercek endpoint gerekmeden) baslatir."""
        super().__init__(_MOCK_ENDPOINT)

    def analyze_evidence(self, evidence_frames: List[EvidenceFrame], prompt: str) -> VLMResponse:
        """Sahte gecikme sonrasi sabit bir Turkce sahne tasviri + sahte EVENTS_JSON dondurur.

        Args:
            evidence_frames: Analiz edilecek evidence kareleri.
            prompt: Kullanici/istem metni (yalnizca loglanir).

        Returns:
            Sabit metinli, model_name="mock-vlm" olan, gonderilen TUM
            evidence_id'leri TEK bir sahte olayda kumeleyen `VLMResponse`.
        """
        started_at = time.perf_counter()
        time.sleep(0.1)
        logger.info(
            "MockVLMClient: %d evidence karesi icin sahte aciklama uretildi (prompt=%r)",
            len(evidence_frames),
            prompt,
        )
        structured_events = []
        if evidence_frames:
            structured_events = [
                {
                    "event_id": "mock_e1",
                    "type": "arac_yaya_yakinligi",
                    "start_time": evidence_frames[0].timestamp_sec,
                    "end_time": evidence_frames[-1].timestamp_sec,
                    "evidence_ids": [ef.evidence_id for ef in evidence_frames],
                    "description": _MOCK_DESCRIPTION,
                    "keywords": ["forklift", "yaya gecidi", "arac yaklasti"],
                    "risk_score": 35,
                    "confidence": 0.6,
                }
            ]
        return VLMResponse(
            description=_MOCK_DESCRIPTION,
            model_name=self.model_name,
            frame_count=len(evidence_frames),
            latency_ms=(time.perf_counter() - started_at) * 1000,
            structured_events=structured_events,
        )

    def health_check(self) -> bool:
        """Mock istemci her zaman saglikli kabul edilir."""
        return True


if __name__ == "__main__":
    # Modul 2'nin bagimsiz calistirilabilirlik testi:
    #   python -m src.vlm.vlm_client            -> mock istemciyi test eder
    #   python -m src.vlm.vlm_client --real      -> config'teki gercek vLLM'e istek atar
    import sys

    from src.sampler.schema import EvidenceFrame as _EvidenceFrame
    from src.utils.config_loader import load_config

    logging.basicConfig(level=logging.INFO)

    demo_evidence = _EvidenceFrame(
        evidence_id="ev0",
        frame_id=0,
        timestamp_sec=1.0,
        timestamp_str="00:01",
        change_score=0.5,
        image_bytes=b"",
        base64_image="data:image/jpeg;base64,AA==",
        image_shape=(1, 1, 3),
    )

    use_real = "--real" in sys.argv
    if use_real:
        demo_client: BaseVLM = VLMClient(load_config().vlm)
    else:
        demo_client = MockVLMClient()

    demo_response = demo_client.analyze_evidence([demo_evidence], prompt="Test istemi: risk var mi?")
    print(f"model_name={demo_response.model_name}")
    print(f"latency_ms={demo_response.latency_ms:.1f}")
    print(f"description={demo_response.description}")
