"""Merkezi YAML konfigurasyonunu tipli (pydantic) modellere donusturen yukleyici."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class AppConfig(BaseModel):
    """Uygulama kimligi ve GPU'suz gelistirme icin Mock mod anahtarlari.

    `use_mock_vlm`/`use_mock_llm` `true` oldugunda `src/vlm/factory.py`
    (`get_vlm_client`/`get_llm_client`) gercek vLLM GPU servislerine hic
    baglanmadan sahte (mock) istemcileri dondurur; boylece GPU'su olmayan
    gelistiriciler tum pipeline'i uctan uca calistirabilir.
    """

    name: str = "SAFIR"
    version: str = "2.0.0"
    use_mock_vlm: bool = False
    use_mock_llm: bool = False


class SamplerConfig(BaseModel):
    """Adaptive Frame Sampler (CPU) esik, pencere ve kumeleme ayarlari.

    `min_change_threshold`/`blur_kernel_size`/`history_window`/
    `min_event_interval_sec`/`sample_fps` `AdaptiveFrameSampler` tarafindan
    dogrudan kullanilan aktif parametrelerdir (bkz. `sampler_from_config`).
    Geri kalan alanlar (idle_interval_sec, active_fps, noise_floor,
    motion_threshold, scene_change_threshold, resize_width, warmup_frames),
    sahadaki alternatif/eski sampler konfigurasyonlariyla geriye donuk
    uyumluluk ve ince ayar icin saklanir.

    ONEMLI: `max_evidence_buffer` (Kanit Karesi sayisinda sabit ust sinir) ve
    `pre_peak_offset_sec`/`post_peak_offset_sec` (eski seek-tabanli pre/post
    pencere ofsetleri) KALDIRILMISTIR: kare sayisinda video geneli sabit bir
    limit YOKTUR, VLM/API kapasitesi olay basina `FrameSelector.
    TARGET_FRAME_COUNT` ile korunur (bkz. `src/sampler/context/frame_selector.py`).
    """

    min_change_threshold: float
    blur_kernel_size: List[int]
    history_window: int
    min_event_interval_sec: float
    sample_fps: int

    # --- Olay kumeleme / zamansal oylama ayarlari ---
    # `sampler_from_config` bu alanlari `AdaptiveFrameSampler.__init__`e gecirir.
    # Varsayilanlar constructor imzasiyla birebir ayni tutulur; config.yaml'da
    # tanimlanmalari zorunlu degildir (verilmezse bu varsayilanlar kullanilir).
    cluster_merge_gap_sec: float = 20.0
    bbox_iou_merge_threshold: float = 0.10
    temporal_vote_window: int = 1
    temporal_vote_min_count: int = 1

    idle_interval_sec: float
    active_fps: float
    noise_floor: float
    motion_threshold: float
    scene_change_threshold: float
    resize_width: int
    warmup_frames: int


class VLLMEndpointConfig(BaseModel):
    """Tek bir model uc noktasi icin baglanti bilgileri (yerel vLLM veya harici saglayici).

    Varsayilan `provider="vllm"` ile davranis, `http://<vllm_host>:<vllm_port>/v1`
    adresindeki yerel OpenAI-uyumlu vLLM servisine baglanmaktir (yarismanin
    offline/yerel gereksinimi). `provider="gemini"` (veya baska bir harici
    saglayici) secildiginde `base_url` ile tam OpenAI-uyumlu taban adres ve
    `api_key_env` ile anahtarin okunacagi ortam degiskeni adi verilir; boylece
    VRAM'i yetmeyen gelistiriciler ayni pipeline'i tek config anahtariyla
    Gemini API uzerinden test edebilir.

    ONEMLI: Harici API kullanimi (Gemini) yalnizca GELISTIRME/TEST amaclidir ve
    sartnamenin "tamamen yerel/offline calisma" gereksinimini ihlal eder;
    yarisma teslimi icin `provider` her zaman `vllm` (yerel) olmalidir.
    """

    model_name: str
    vllm_host: str = "localhost"
    vllm_port: int = 0
    max_new_tokens: int
    temperature: float
    top_p: float = 1.0

    provider: str = "vllm"                       # vllm | gemini
    base_url: Optional[str] = None               # verilirse host:port yerine bu tam OpenAI-uyumlu taban kullanilir
    api_key_env: Optional[str] = None            # anahtarin okunacagi ortam degiskeni adi (orn. "GEMINI_API_KEY")
    extra_body: Dict[str, Any] = Field(default_factory=dict)
    """Istek govdesine eklenecek saglayici-ozel alanlar (vLLM guided decoding icin;
    orn. `{"guided_json": {...}}` veya `{"guided_regex": "..."}`). Varsayilan bos
    (davranis degismez). Cekirdek alanlari (model/messages/...) EZMEZ."""

    def resolved_base_url(self) -> str:
        """Bu uc nokta icin kullanilacak OpenAI-uyumlu taban URL'yi dondurur.

        `base_url` verilmisse (harici saglayici) sondaki `/` temizlenerek o
        kullanilir; aksi halde yerel vLLM adresi (`http://host:port/v1`) uretilir.
        """
        if self.base_url:
            return self.base_url.rstrip("/")
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"

    def resolved_api_key(self) -> str:
        """Bu uc nokta icin API anahtarini dondurur.

        `api_key_env` tanimliysa ilgili ortam degiskeninden okunur; yerel vLLM
        icin anahtar gerekmediginden `"EMPTY"` doner (OpenAI istemcileri bos
        olmayan bir deger bekler).

        Raises:
            RuntimeError: `api_key_env` tanimli ama ortam degiskeni bos/yoksa.
        """
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "").strip()
            if not key:
                raise RuntimeError(
                    f"'{self.model_name}' uc noktasi icin API anahtari bulunamadi: "
                    f"'{self.api_key_env}' ortam degiskenini tanimlayin."
                )
            return key
        return "EMPTY"

    def auth_headers(self) -> Dict[str, str]:
        """Harici saglayici icin `Authorization` header'ini dondurur (yerel vLLM icin bos)."""
        if self.api_key_env:
            return {"Authorization": f"Bearer {self.resolved_api_key()}"}
        return {}


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


class EmbeddingConfig(BaseModel):
    """Embedding & RAG Katmani icin sentence-transformers tabanli model ayarlari."""

    provider: str                       # "sentence-transformers"
    model_name: str                     # orn. "BAAI/bge-m3" veya "Qwen/Qwen3-VL-Embedding"
    device: str                         # "cpu" | "cuda"
    normalize_embeddings: bool = True


class FaissMemoryConfig(BaseModel):
    index_path: str
    embedding_model: str                # bkz. memory.embedding.model_name (ayni deger, senkron tutulmali)
    top_k: int


class MemoryConfig(BaseModel):
    """Yapilandirilmis olay bellegi (SQLite) ve anlamsal bellek (Embedding+FAISS) ayarlari."""

    sqlite: SQLiteMemoryConfig
    embedding: EmbeddingConfig
    faiss: FaissMemoryConfig


class RiskThresholds(BaseModel):
    """0-100 risk skorunu risk seviyelerine esleyen ust sinirlar."""

    low: int
    medium: int
    high: int
    critical: int


class AgentToolsConfig(BaseModel):
    sql_tool_enabled: bool
    rag_tool_enabled: bool              # bkz. retriever_tool_enabled (ayni RAG aracina isaret eder)
    retriever_tool_enabled: bool
    timeline_tool_enabled: bool
    verification_tool_enabled: bool


class AgentConfig(BaseModel):
    """LangGraph durum makinesi ve arac yonlendirme ayarlari."""

    max_iterations: int
    risk_thresholds: RiskThresholds
    tools: AgentToolsConfig
    guided_json: bool = True
    """Ajanin nihai karari gecerli JSON degilse, JSON-modu (response_format=
    json_object; vLLM/Gemini destekler) ile TEK bir yeniden-deneme yapilir.
    Kucuk yerel modellerde bozuk JSON'u kurtarir. `false` ile devre disi."""


class EscalationConfig(BaseModel):
    """Otomatik eskalasyon esikleri (Human-on-the-Loop; bloke edici operator kapisi yok).

    `notify_score` ve uzeri risk skoru -> bildirim (NOTIFY); `auto_alarm_score`
    ve uzeri -> saha alarmi OTOMATIK tetiklenir (ALARM). Varsayilanlar
    `agent.risk_thresholds` ile hizalidir (orta>25, yuksek>50).
    """

    notify_score: int = 26
    auto_alarm_score: int = 51


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

    app: AppConfig = Field(default_factory=AppConfig)
    system: SystemConfig
    sampler: SamplerConfig
    vlm: VLMConfig
    memory: MemoryConfig
    llm: LLMConfig
    agent: AgentConfig
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
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
