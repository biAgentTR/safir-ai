"""03 - Gorsel Dil Modeli Katmani: config-driven Factory Pattern ile VLM secimi."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Type

from src.utils.config_loader import VLMConfig
from src.vlm.base_vlm import BaseVLM
from src.vlm.gemma_vlm import GemmaVLM
from src.vlm.qwen_vlm import QwenVLM

logger = logging.getLogger(__name__)


class VLMFactory:
    """`configs/config.yaml` icindeki `vlm.active_model` degerine gore VLM ureten fabrika.

    Yeni bir VLM eklemek icin `BaseVLM`'den tureyen bir sinif yazip
    `_REGISTRY` sozlugune kaydetmek yeterlidir; cagiran kod degismez.
    """

    _REGISTRY: Dict[str, Type[BaseVLM]] = {
        "qwen": QwenVLM,
        "gemma": GemmaVLM,
    }

    @classmethod
    def register(cls, name: str, vlm_cls: Type[BaseVLM]) -> None:
        """Fabrika kayit defterine yeni bir VLM implementasyonu ekler.

        Args:
            name: Config icinde kullanilacak model anahtari (orn. "qwen").
            vlm_cls: `BaseVLM`'den tureyen somut sinif.
        """
        cls._REGISTRY[name] = vlm_cls

    @classmethod
    def create(cls, config: VLMConfig) -> BaseVLM:
        """Config'te belirtilen aktif modele karsilik gelen `BaseVLM` orneği uretir.

        Args:
            config: `configs/config.yaml` icindeki `vlm` blogundan uretilen
                dogrulanmis `VLMConfig` nesnesi.

        Returns:
            Aktif model icin ilklendirilmis `BaseVLM` alt sinif orneği.

        Raises:
            ValueError: `active_model` degeri kayit defterinde tanimli degilse.
        """
        vlm_cls = cls._REGISTRY.get(config.active_model)
        if vlm_cls is None:
            available = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(
                f"Bilinmeyen VLM secimi '{config.active_model}'. "
                f"Kullanilabilir secenekler: {available}"
            )

        endpoint = config.active_endpoint()
        logger.info(
            "VLMFactory: '%s' modeli olusturuluyor (model_name=%s, port=%d)",
            config.active_model,
            endpoint.model_name,
            endpoint.vllm_port,
        )
        return vlm_cls(endpoint)
