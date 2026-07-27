"""Modul 2 - VLM/LLM Client Factory: config'teki Mock anahtarlarina gore istemci secimi.

`get_vlm_client()`/`get_llm_client()`, `configs/config.yaml` icindeki
`app.use_mock_vlm`/`app.use_mock_llm` degerlerine gore gercek (`VLMClient`/
`LLMClient`) veya sahte (`MockVLMClient`/`MockLLMClient`) istemciyi dondurur.
Cagiran kod (`src/main.py`, `src/agent/langgraph_agent.py`) hangi modun aktif
oldugunu bilmek zorunda degildir; yalnizca bu iki fonksiyonu cagirir.
"""

from __future__ import annotations

import logging
from typing import Union

from src.utils.config_loader import LLMConfig, VLMConfig
from src.vlm.base_vlm import BaseVLM
from src.vlm.llm_client import LLMClient, MockLLMClient
from src.vlm.vlm_client import MockVLMClient, VLMClient

logger = logging.getLogger(__name__)


def get_vlm_client(vlm_config: VLMConfig, use_mock: bool = False) -> BaseVLM:
    """Config'teki `use_mock_vlm` bayragina gore gercek veya sahte VLM istemcisi uretir.

    Args:
        vlm_config: `configs/config.yaml` icindeki `vlm` blogu.
        use_mock: `True` ise GPU/vLLM'e hic baglanmayan `MockVLMClient` doner.

    Returns:
        `BaseVLM` sozlesmesini uygulayan bir istemci (`VLMClient` veya `MockVLMClient`).
    """
    if use_mock:
        logger.info("get_vlm_client: Mock mod aktif, MockVLMClient donduruluyor.")
        return MockVLMClient()

    logger.info("get_vlm_client: Gercek VLMClient olusturuluyor (active_model=%s).", vlm_config.active_model)
    return VLMClient(vlm_config)


def get_llm_client(llm_config: LLMConfig, use_mock: bool = False) -> Union[LLMClient, MockLLMClient]:
    """Config'teki `use_mock_llm` bayragina gore gercek veya sahte LLM istemcisi uretir.

    Args:
        llm_config: `configs/config.yaml` icindeki `llm` blogu.
        use_mock: `True` ise GPU/vLLM'e hic baglanmayan `MockLLMClient` doner.

    Returns:
        `bind_tools`/`invoke` arayuzunu uygulayan bir istemci (`LLMClient` veya `MockLLMClient`).
    """
    if use_mock:
        logger.info("get_llm_client: Mock mod aktif, MockLLMClient donduruluyor.")
        return MockLLMClient()

    logger.info("get_llm_client: Gercek LLMClient olusturuluyor (active_model=%s).", llm_config.active_model)
    return LLMClient(llm_config)


if __name__ == "__main__":
    # Modul 2'nin fabrika fonksiyonlari icin bagimsiz calistirilabilirlik testi:
    #   python -m src.vlm.factory
    from src.utils.config_loader import load_config

    logging.basicConfig(level=logging.INFO)

    demo_config = load_config()
    demo_vlm = get_vlm_client(demo_config.vlm, use_mock=True)
    demo_llm = get_llm_client(demo_config.llm, use_mock=True)
    print(f"get_vlm_client(mock=True) -> {type(demo_vlm).__name__} (model_name={demo_vlm.model_name})")
    print(f"get_llm_client(mock=True) -> {type(demo_llm).__name__} (model_name={demo_llm.model_name})")
