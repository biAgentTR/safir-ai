"""06 - Cikti ve Karar Destek Katmani: yapilandirilmis JSON rapor semasi."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TimelineEntry(BaseModel):
    """Zamansal olay cizelgesindeki tek bir giris."""

    timestamp: float = Field(description="Olayin saniye cinsinden zaman damgasi.")
    description: str = Field(description="Olayin dogal dil aciklamasi.")


class SafirReport(BaseModel):
    """Sistemler arasi entegrasyona hazir, mock semayla uyumlu nihai rapor.

    Bu model; Turkce dogal dil ozeti, risk skoru/seviyesi, operator aksiyon
    onerisi ve zaman cizelgesini tek bir yapida birlestirir.
    """

    video_source: str = Field(description="Analiz edilen video/kamera akisinin kaynagi.")
    generated_at: str = Field(description="Raporun ISO-8601 formatinda uretim zamani.")
    natural_language_summary: str = Field(description="Turkce, sade karar ozeti.")
    risk_score: int = Field(ge=0, le=100, description="0-100 arasi hesaplanmis risk skoru.")
    risk_level: str = Field(description="dusuk | orta | yuksek | kritik")
    recommended_action: str = Field(description="Saha operatorune yonelik somut aksiyon onerisi.")
    timeline: List[TimelineEntry] = Field(default_factory=list, description="Kronolojik olay cizelgesi.")
    vlm_model: Optional[str] = Field(default=None, description="Aciklamayi ureten aktif VLM adi.")
    llm_model: Optional[str] = Field(default=None, description="Karari ureten aktif LLM adi.")

    def to_json_file(self, path: str) -> None:
        """Raporu belirtilen dosya yoluna UTF-8 JSON olarak yazar.

        Args:
            path: Yazilacak `.json` dosyasinin yolu.
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))
