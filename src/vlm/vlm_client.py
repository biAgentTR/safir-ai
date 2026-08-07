"""Modul 2 - VLM Client & Model Interface: VLM icin gercek/mock istemciler.

`VLMClient`, config'te secilen aktif VLM'i (Qwen2.5-VL / Gemma 3 Vision)
`VLMFactory` uzerinden kurup gercek vLLM servisine HTTP istegi atar; boylece
mevcut test edilmis `BaseVLM`/`QwenVLM`/`GemmaVLM` mantigi hic degistirilmeden
tek bir noktadan yeniden kullanilir. `MockVLMClient` ise GPU'su olmayan
gelistiriciler icin ayni `BaseVLM` sozlesmesini (`describe_events`,
`health_check`) saniyenin onda biri kadar surede, sabit Turkce ISG tasvir
metniyle karsilar; boylece `get_vlm_client(..., use_mock=True)` ile secildiginde
main.py'deki pipeline hicbir kod degisikligi olmadan calisir.
"""

from __future__ import annotations

import logging
import time
from typing import List

from src.sampler.adaptive_sampler import EventCluster
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
    "[MOCK] Sahada bir personel korumasız bir alanda hareket ediyor; baret ve "
    "yansıtıcı yelek eksik görünüyor. Yakın çevrede forklift trafiğinin olduğu "
    "gözlemleniyor. Herhangi bir duman veya yangın belirtisi tespit edilmedi."
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

    def describe_events(self, clusters: List[EventCluster], prompt: str) -> VLMResponse:
        """Cagriyi, config'te secilen gercek VLM implementasyonuna devreder.

        Args:
            clusters: Analiz edilecek Olay Gruplari.
            prompt: Kullanici/istem metni.

        Returns:
            Gercek vLLM servisinden donen `VLMResponse`.
        """
        return self._delegate.describe_events(clusters, prompt)

    def health_check(self) -> bool:
        """Devredilen gercek VLM implementasyonunun saglik durumunu dondurur."""
        return self._delegate.health_check()


class MockVLMClient(BaseVLM):
    """GPU'su olmayan gelistiriciler icin sahte VLM istemcisi (`use_mock_vlm: true`).

    Gercek vLLM servisine hic baglanmadan, `BaseVLM` sozlesmesine uygun sabit
    bir Turkce ISG sahne tasviri dondurur.
    """

    def __init__(self) -> None:
        """MockVLMClient'i (gercek endpoint gerekmeden) baslatir."""
        super().__init__(_MOCK_ENDPOINT)

    def describe_events(self, clusters: List[EventCluster], prompt: str) -> VLMResponse:
        """Sahte gecikme sonrasi sabit bir Turkce sahne tasviri dondurur.

        Args:
            clusters: Analiz edilecek Olay Gruplari (yalnizca sayisi kullanilir).
            prompt: Kullanici/istem metni (yalnizca loglanir).

        Returns:
            Sabit metinli, model_name="mock-vlm" olan `VLMResponse`.
        """
        started_at = time.perf_counter()
        time.sleep(0.1)
        logger.info("MockVLMClient: %d olay grubu icin sahte aciklama uretildi (prompt=%r)", len(clusters), prompt)
        return VLMResponse(
            description=_MOCK_DESCRIPTION,
            model_name=self.model_name,
            frame_count=len(clusters),
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )

    def health_check(self) -> bool:
        """Mock istemci her zaman saglikli kabul edilir."""
        return True


if __name__ == "__main__":
    # Modul 2'nin bagimsiz calistirilabilirlik testi:
    #   python -m src.vlm.vlm_client            -> mock istemciyi test eder
    #   python -m src.vlm.vlm_client --real      -> config'teki gercek vLLM'e istek atar
    import sys

    from src.sampler.schema import EventCluster as _EventCluster
    from src.sampler.schema import EvidenceFrame as _EvidenceFrame
    from src.utils.config_loader import load_config

    logging.basicConfig(level=logging.INFO)

    demo_evidence = _EvidenceFrame(
        frame_id=0,
        timestamp_sec=1.0,
        timestamp_str="00:01",
        change_score=0.5,
        image_bytes=b"",
        base64_image="data:image/jpeg;base64,AA==",
        image_shape=(1, 1, 3),
    )
    demo_cluster = _EventCluster(
        event_id=1, start_time=1.0, end_time=1.0, peak_frame=demo_evidence, total_candidate_frames=1
    )

    use_real = "--real" in sys.argv
    if use_real:
        demo_client: BaseVLM = VLMClient(load_config().vlm)
    else:
        demo_client = MockVLMClient()

    demo_response = demo_client.describe_events([demo_cluster], prompt="Test istemi: risk var mi?")
    print(f"model_name={demo_response.model_name}")
    print(f"latency_ms={demo_response.latency_ms:.1f}")
    print(f"description={demo_response.description}")
