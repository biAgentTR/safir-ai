"""07 - Olay Analizi Katmani: paylasilan veri modelleri.

`Event Engine` (T008), `Temporal Reasoning` ve `Rule Engine` (T010) modulleri
arasinda tasinan yapilandirilmis nesneleri tanimlar. Bu katman, `04 Context
Builder` ile `05 LangGraph Agentic Loop` arasindaki ara katmandir; `src/vlm/`
ve `src/memory/` modullerine yalnizca salt-okunur tip referansi icin bakar,
onlari degistirmez.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.vlm.base_vlm import VLMResponse


class EventEngineInput(BaseModel):
    """Event Engine'e beslenen, VLM ciktisindan map'lenmis standardize girdi.

    `src/vlm/base_vlm.py` icindeki `VLMResponse` alanlarindan dogrudan
    turetilir (bkz. `from_vlm_response`); VLM katmanina hicbir degisiklik
    gerektirmez.
    """

    vlm_description: str = Field(description="VLM'in urettigi dogal dil olay aciklamasi.")
    timestamp: float = Field(description="Gozlemin saniye cinsinden zaman damgasi.")
    source_model: str = Field(description="Aciklamayi ureten VLM'in model adi.")
    frame_count: int = Field(default=0, ge=0, description="Aciklamaya kaynaklik eden Kanit Karesi sayisi.")

    @classmethod
    def from_vlm_response(cls, vlm_response: "VLMResponse", timestamp: float) -> "EventEngineInput":
        """`VLMResponse` orneginden bir `EventEngineInput` uretir.

        Args:
            vlm_response: `BaseVLM.describe_events(...)` ciktisi.
            timestamp: Gozlemin zaman damgasi (orn. `EventCluster.end_time`).

        Returns:
            Event Engine'e dogrudan verilebilecek girdi nesnesi.
        """
        return cls(
            vlm_description=vlm_response.description,
            timestamp=timestamp,
            source_model=vlm_response.model_name,
            frame_count=vlm_response.frame_count,
        )


class EventType(str, Enum):
    """Event Engine'in tanidigi olay kategorileri."""

    KKD_IHLALI = "kkd_ihlali"
    ARAC_YAYA_YAKINLIGI = "arac_yaya_yakinligi"
    DUSME_RISKI = "dusme_riski"
    YANGIN_DUMAN = "yangin_duman"
    YETKISIZ_ERISIM = "yetkisiz_erisim"
    GENEL_GOZLEM = "genel_gozlem"


class DetectedEvent(BaseModel):
    """Event Engine'in VLM metninden cikardigi tek bir yapilandirilmis olay."""

    event_type: str = Field(description="Olay kategorisi (bkz. `EventType`).")
    description: str = Field(description="Olayin dogal dil aciklamasi (VLM metninden alinir).")
    timestamp: float = Field(description="Olayin saniye cinsinden zaman damgasi.")
    confidence: float = Field(ge=0.0, le=1.0, description="Tespitin guven skoru (0.0-1.0).")
    matched_keywords: List[str] = Field(
        default_factory=list, description="Tespiti tetikleyen anahtar kelimeler (varsa)."
    )
    source_model: Optional[str] = Field(default=None, description="Aciklamayi ureten VLM'in model adi.")


class RuleMatch(BaseModel):
    """Rule Engine'in (T010) bir `DetectedEvent`e karsi eslestirdigi ISG/saha kurali.

    Bu model burada yalnizca sozlesme olarak tanimlanir; eslestirme mantigi
    T010'da `retriever_tool`/`EmbeddingRAGService` uzerinden implemente edilecektir.
    """

    rule_id: str = Field(description="Kural kimligi (orn. 'ISG-M12').")
    rule_description: str = Field(description="Kuralin kisa aciklamasi.")
    event_type: str = Field(description="Bu kuralin uygulandigi olay kategorisi (bkz. `EventType`).")
    severity: str = Field(description="Kural ihlalinin siddet seviyesi (dusuk/orta/yuksek/kritik).")
