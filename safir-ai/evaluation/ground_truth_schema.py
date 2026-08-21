"""Ground Truth JSON Veri Yapısı ve Şema Doğrulama."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class GroundTruthEvent(BaseModel):
    """Ground truth olay tanımı."""

    time: str = Field(description="Olayın 'MM:SS' formatında gerçekleştiği zaman damgası (örn. '00:15').")
    timestamp_sec: Optional[float] = Field(default=None, description="Olayın saniye cinsinden değeri.")
    type: str = Field(description="Olay kategorisi (örn. 'forklift_kazasi', 'kkd_eksikligi', 'sizma_yetkisiz_erisim').")
    is_critical: bool = Field(default=False, description="Kritik/acil olay olup olmadığı.")
    description: Optional[str] = Field(default="", description="İnsan tarafından etiketlenmiş detaylı açıklama.")


class GroundTruthVideo(BaseModel):
    """Tek bir video için etiketlenmiş ground truth verisi."""

    video_id: str = Field(description="Video kimliği / dosya adı.")
    duration_sec: float = Field(description="Videonun toplam uzunluğu (saniye).")
    events: List[GroundTruthEvent] = Field(default_factory=list, description="Etiketli olaylar listesi.")


class GroundTruthDataset(BaseModel):
    """Değerlendirme veri seti koleksiyonu."""

    videos: List[GroundTruthVideo] = Field(default_factory=list)
