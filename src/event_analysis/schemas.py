
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.vlm.base_vlm import VLMResponse


class EventEngineInput(BaseModel):

    vlm_description: str = Field(description="VLM'in urettigi dogal dil olay aciklamasi.")
    timestamp: float = Field(description="Gozlemin saniye cinsinden zaman damgasi.")
    source_model: str = Field(description="Aciklamayi ureten VLM'in model adi.")
    frame_count: int = Field(default=0, ge=0, description="Aciklamaya kaynaklik eden Kanit Karesi sayisi.")

    @classmethod
    def from_vlm_response(cls, vlm_response: "VLMResponse", timestamp: float) -> "EventEngineInput":
        return cls(
            vlm_description=vlm_response.description,
            timestamp=timestamp,
            source_model=vlm_response.model_name,
            frame_count=vlm_response.frame_count,
        )


class EventType(str, Enum):

    DUSME_RISKI = "dusme_riski"
    """ISG Yonetmeligi Madde 12 (yukseklikte calisma / dusme onleyici ekipman)."""

    KKD_IHLALI = "kkd_ihlali"
    """ISG Yonetmeligi Madde 24 (Kisisel Koruyucu Donanim eksikligi)."""

    ARAC_YAYA_YAKINLIGI = "arac_yaya_yakinligi"
    """Operasyonel Kural OK-07 (forklift/is makinesi - yaya gecidi ihlali)."""

    SICAK_CALISMA_IHLALI = "sicak_calisma_ihlali"
    """ISG Yonetmeligi Madde 31 (izinsiz ates/kivilcim/sicak yuzey islemi)."""

    YANGIN_DUMAN = "yangin_duman"
    """Yangin Guvenligi Talimati YG-03 (duman/alev tespiti)."""

    DAR_ALAN_IHLALI = "dar_alan_ihlali"
    """ISG Yonetmeligi Madde 45 (gaz olcumu/gozetmen olmadan kapali-dar alana giris)."""

    ENERJI_KESME_IHLALI = "enerji_kesme_ihlali"
    """Operasyonel Kural OK-15 (LOTO - Lockout/Tagout prosedurune uyulmadan mudahale)."""

    AGIR_YUK_RISKI = "agir_yuk_riski"
    """ISG Yonetmeligi Madde 52 (sinyalman olmadan vinc/kren ile agir yuk kaldirma)."""

    YETKISIZ_ERISIM = "yetkisiz_erisim"
    """Mevzuat disi, operasyonel izleme kategorisi: mevcut `DEFAULT_ISG_REGULATIONS`
    setinde dogrudan karsiligi yoktur, ancak yasakli/yetkisiz alana giris
    saha guvenligi acisindan izlenmesi gereken yaygin bir durum oldugundan
    enum'da tutulur. Mevzuat seti genisledikce (orn. erisim kontrolu maddesi
    eklenirse) `EVENT_TYPE_REGULATION_MAP`'e baglanabilir."""

    GENEL_GOZLEM = "genel_gozlem"
    """Fallback kategori: hicbir kural kategorisi eslesmedigi durumlar icin (mevzuat karsiligi yok)."""


EVENT_TYPE_REGULATION_MAP: Dict[EventType, Optional[str]] = {
    EventType.DUSME_RISKI: "ISG Yonetmeligi Madde 12",
    EventType.KKD_IHLALI: "ISG Yonetmeligi Madde 24",
    EventType.ARAC_YAYA_YAKINLIGI: "Operasyonel Kural OK-07",
    EventType.SICAK_CALISMA_IHLALI: "ISG Yonetmeligi Madde 31",
    EventType.YANGIN_DUMAN: "Yangin Guvenligi Talimati YG-03",
    EventType.DAR_ALAN_IHLALI: "ISG Yonetmeligi Madde 45",
    EventType.ENERJI_KESME_IHLALI: "Operasyonel Kural OK-15",
    EventType.AGIR_YUK_RISKI: "ISG Yonetmeligi Madde 52",
    EventType.YETKISIZ_ERISIM: None,
    EventType.GENEL_GOZLEM: None,
}


class DetectedEvent(BaseModel):

    event_type: str = Field(description="Olay kategorisi (bkz. `EventType`).")
    description: str = Field(description="Olayin dogal dil aciklamasi (VLM metninden alinir).")
    timestamp: float = Field(description="Olayin saniye cinsinden zaman damgasi.")
    confidence: float = Field(ge=0.0, le=1.0, description="Tespitin guven skoru (0.0-1.0).")
    matched_keywords: List[str] = Field(
        default_factory=list, description="Tespiti tetikleyen anahtar kelimeler (varsa)."
    )
    source_model: Optional[str] = Field(default=None, description="Aciklamayi ureten VLM'in model adi.")


class TemporalEvent(BaseModel):

    event_id: str = Field(description="Bu `TemporalEvent`e ozgu kimlik (orn. 'evt_0').")
    event_type: str = Field(description="Olay kategorisi (bkz. `EventType`).")
    description: str = Field(description="Temsili aciklama (gruptaki en son `DetectedEvent`den alinir).")
    start_timestamp: float = Field(description="Gruptaki en erken olayin zaman damgasi (saniye).")
    end_timestamp: float = Field(description="Gruptaki en son olayin zaman damgasi (saniye).")
    duration: float = Field(
        ge=0.0, description="`end_timestamp - start_timestamp`; tek olay icin 0.0."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Gruptaki en yuksek guven skoru, tekrar eden tespitler icin hafifce artirilmis."
    )
    occurrence_count: int = Field(ge=1, description="Bu `TemporalEvent`e birlesen `DetectedEvent` sayisi.")
    matched_keywords: List[str] = Field(
        default_factory=list, description="Gruptaki tum tespitlerden gelen anahtar kelimelerin birlesimi."
    )
    source_model: Optional[str] = Field(default=None, description="Aciklamayi ureten VLM'in model adi.")
    related_events: List[str] = Field(
        default_factory=list,
        description="Zaman ekseninde yakin duran diger `TemporalEvent`lerin `event_id` listesi.",
    )


class RuleMatch(BaseModel):
    rule_id: str = Field(description="Kural kimligi (orn. 'ISG-M12' veya 'COMBO-01').")
    rule_description: str = Field(description="Kuralin kisa aciklamasi.")
    event_type: str = Field(
        description="Kuralin uygulandigi olay kategorisi; kombinasyon kurallarinda '+' ile birlesik tipler."
    )
    severity: str = Field(description="Kural ihlalinin siddet seviyesi (dusuk/orta/yuksek/kritik).")
    source_event_id: str = Field(description="Bu eslesmeyi tetikleyen `TemporalEvent.event_id`.")
    related_event_ids: List[str] = Field(
        default_factory=list,
        description="Kombinasyon kurallarinda katilan diger `TemporalEvent.event_id`leri; tekli kurallarda bos.",
    )


class StructuredEvent(BaseModel):

    timestamp: float = Field(description="Olayin saniye cinsinden zaman damgasi (`TemporalEvent.end_timestamp`).")
    description: str = Field(
        description="Insan-okunur, VLM aciklamasi + tetiklenen kural(lar)in ozetiyle birlesik metin."
    )
    risk_score: Optional[int] = Field(
        default=None, ge=0, le=100, description="Henuz bilinmiyor; 05 LangGraph Ajaninin karari sonrasi doldurulur."
    )
    risk_level: Optional[str] = Field(
        default=None, description="Henuz bilinmiyor; 05 LangGraph Ajaninin karari sonrasi doldurulur."
    )
    source_model: Optional[str] = Field(default=None, description="Aciklamayi ureten VLM'in model adi.")

    event_type: str = Field(description="Olay kategorisi (bkz. `EventType`).")
    confidence: float = Field(ge=0.0, le=1.0, description="`TemporalEvent.confidence` (tekrar-duzeltilmis guven skoru).")
    temporal_event_id: str = Field(description="Bu olayin turedigi `TemporalEvent.event_id`.")
    related_rule_matches: List[RuleMatch] = Field(
        default_factory=list, description="Bu olaya (tekli veya kombinasyon olarak) uygulanan `RuleMatch` listesi."
    )
    occurrence_count: int = Field(
        default=1, ge=1, description="`TemporalEvent.occurrence_count` (birlesen tespit sayisi)."
    )
    duration: float = Field(default=0.0, ge=0.0, description="`TemporalEvent.duration` (saniye).")

    def to_event_store_kwargs(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "description": self.description,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "source_model": self.source_model,
        }