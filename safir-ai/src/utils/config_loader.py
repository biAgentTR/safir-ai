"""Merkezi YAML konfigurasyonunu tipli (pydantic) modellere donusturen yukleyici."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


class SystemConfig(BaseModel):
    """Genel sistem ayarlari (donanim, ortam, loglama)."""

    name: str
    environment: str
    device: str
    cuda_device_index: int
    log_level: str
    random_seed: int


class SamplerConfig(BaseModel):
    """Adaptive Frame Sampler (CPU) esik ve zamanlama ayarlari."""

    idle_interval_sec: float
    active_fps: float
    noise_floor: float
    motion_threshold: float
    scene_change_threshold: float
    resize_width: int
    max_evidence_buffer: int
    warmup_frames: int


class VLLMEndpointConfig(BaseModel):
    """Tek bir vLLM tarafindan sunulan model icin baglanti bilgileri."""

    model_name: str
    vllm_host: str
    vllm_port: int
    max_new_tokens: int
    temperature: float
    top_p: float = 1.0


class VLMConfig(BaseModel):
    """Aktif VLM secimini ve tum VLM tanimlarini tutar (Factory Pattern icin)."""

    active_model: str
    models: Dict[str, VLLMEndpointConfig]

    def active_endpoint(self) -> VLLMEndpointConfig:
        """Config icinde secilen aktif VLM'in baglanti bilgisini dondurur."""
        if self.active_model not in self.models:
            raise KeyError(f"Tanimsiz VLM secimi: '{self.active_model}'")
        return self.models[self.active_model]


class LLMConfig(BaseModel):
    """Ajan/muhakeme katmani icin aktif LLM secimini ve tanimlarini tutar."""

    active_model: str
    models: Dict[str, VLLMEndpointConfig]

    def active_endpoint(self) -> VLLMEndpointConfig:
        """Config icinde secilen aktif LLM'in baglanti bilgisini dondurur."""
        if self.active_model not in self.models:
            raise KeyError(f"Tanimsiz LLM secimi: '{self.active_model}'")
        return self.models[self.active_model]


class SQLiteMemoryConfig(BaseModel):
    db_path: str


class FaissMemoryConfig(BaseModel):
    index_path: str
    embedding_model: str
    top_k: int


class MemoryConfig(BaseModel):
    """Yapilandirilmis olay bellegi (SQLite) ve anlamsal bellek (FAISS) ayarlari."""

    sqlite: SQLiteMemoryConfig
    faiss: FaissMemoryConfig


class RiskThresholds(BaseModel):
    """0-100 risk skorunu risk seviyelerine esleyen ust sinirlar."""

    low: int
    medium: int
    high: int
    critical: int


class AgentToolsConfig(BaseModel):
    sql_tool_enabled: bool
    rag_tool_enabled: bool
    timeline_tool_enabled: bool


class AgentConfig(BaseModel):
    """LangGraph durum makinesi ve arac yonlendirme ayarlari."""

    max_iterations: int
    risk_thresholds: RiskThresholds
    tools: AgentToolsConfig


class ApiConfig(BaseModel):
    host: str
    port: int
    reload: bool
    cors_origins: List[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    language: str
    json_report_dir: str
    timeline_export_dir: str
    pdf_report_dir: str
    streamlit_port: int


class SafirConfig(BaseModel):
    """SAFIR sisteminin butun katmanlarini kapsayan kok konfigurasyon modeli."""

    system: SystemConfig
    sampler: SamplerConfig
    vlm: VLMConfig
    memory: MemoryConfig
    llm: LLMConfig
    agent: AgentConfig
    api: ApiConfig
    output: OutputConfig


@functools.lru_cache(maxsize=8)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> SafirConfig:
    """`config.yaml` dosyasini okuyup dogrulanmis `SafirConfig` nesnesi olarak dondurur.

    Args:
        config_path: YAML konfigurasyon dosyasinin yolu.

    Returns:
        Pydantic ile dogrulanmis, tipli konfigurasyon nesnesi.

    Raises:
        FileNotFoundError: Belirtilen yolda konfigurasyon dosyasi bulunamazsa.
        ValueError: YAML gecersiz ya da beklenen semaya uymuyorsa.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Konfigurasyon dosyasi bulunamadi: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML ayristirma hatasi ({path}): {exc}") from exc

    try:
        return SafirConfig(**raw)
    except Exception as exc:  # pydantic.ValidationError dahil
        raise ValueError(f"Konfigurasyon semasi gecersiz ({path}): {exc}") from exc
