"""VLM oncesi katman: CPU uzerinde calisan adaptif kare ornekleyici + VLM payload olusturma.

Olay kumelemesi ARTIK bu katmanda YAPILMAZ (bkz. `adaptive_sampler`/`payload_builder`
modul docstring'leri); kumeleme VLM katmanina tasindi.
"""

from src.sampler.adaptive_sampler import (
    AdaptiveFrameSampler,
    SamplerRunStats,
    sampler_from_config,
)
from src.sampler.payload_builder import VLMPayloadBuilder
from src.sampler.schema import EvidenceFrame

__all__ = [
    "AdaptiveFrameSampler",
    "EvidenceFrame",
    "SamplerRunStats",
    "sampler_from_config",
    "VLMPayloadBuilder",
]
