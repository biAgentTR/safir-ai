"""07 - Olay Analizi Katmani: paylasilan veri modelleri.

`Event Engine` (T008), `Temporal Reasoning` ve `Rule Engine` (T010) modulleri
arasinda tasinan yapilandirilmis nesneleri tanimlar. Bu katman, `04 Context
Builder` ile `05 LangGraph Agentic Loop` arasindaki ara katmandir; `src/vlm/`
ve `src/memory/` modullerine yalnizca salt-okunur tip referansi icin bakar,
onlari degistirmez.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

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
    structured_events: List[dict] = Field(
        default_factory=list,
        description=(
            "VLM'in dogrudan urettigi tipli olaylar (`EVENTS_JSON`); her biri "
            "`{type, timestamp, confidence, evidence}`. Bos ise EventEngine "
            "anahtar-kelime fallback'ine duser."
        ),
    )

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
            structured_events=list(getattr(vlm_response, "structured_events", []) or []),
        )


class EventType(str, Enum):
    """Event Engine'in tanidigi olay kategorileri.

    Ilk 8 deger, `src/memory/embedding_rag_service.py::DEFAULT_ISG_REGULATIONS`
    icindeki 8 mevzuat maddesiyle birebir hizalanmistir (bkz.
    `EVENT_TYPE_REGULATION_MAP`); T010 Rule Engine bu eslesmeyi dogrudan
    kullanabilir. `YETKISIZ_ERISIM` ve `GENEL_GOZLEM` mevcut mevzuat setinde
    karsiligi olmayan, operasyonel amacli iki ozel kategoridir (asagida
    ayrica aciklanmistir).
    """

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
    """Erisim Kontrol Proseduru EK-01 (yasakli/yetkisiz alana giris)."""

    SIZMA_YETKISIZ_ERISIM = "sizma_yetkisiz_erisim"
    """Savunma Sanayi & Kritik Tesis Cevre Koruma Hattinda (tel orgu, duvar, nizamye) sizma ve yetkisiz erisim ihlali."""

    SUPHELI_PAKET_HAREKET = "supheli_paket_hareket"
    """Tesis icinde veya cevresinde sahipsiz paket/canta veya supheli davranis (kesif, gizlenme)."""

    DRONE_IHA_TEHDIDI = "drone_iha_tehdidi"
    """Kritik tesis hava sahasinda izinsiz IHA, drone veya ucan cisim gorulmesi."""

    YARALANMA_KAZA = "yaralanma_kaza"
    """Fiziki is kazasi, yaralanma, dusme, kanama, ezilme, carpisma veya acil mudahale gerektiren durumlar."""

    SINIFLANDIRILAMADI = "siniflandirilamadi"
    """8 ana ISG kurali disinda supheli risk / anormallik / inceleme bekleyen tehlike durumu."""

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
    EventType.YETKISIZ_ERISIM: "Erisim Kontrol Proseduru EK-01",
    EventType.SIZMA_YETKISIZ_ERISIM: "Erisim Kontrol Proseduru EK-01 (Cevre Guvenligi ve Sizma Tespiti)",
    EventType.SUPHELI_PAKET_HAREKET: "Supheli Hareket ve Paket Talimati SHP-02",
    EventType.DRONE_IHA_TEHDIDI: "Hava Savunma ve IHA/Drone Yonergesi IHA-04",
    EventType.YARALANMA_KAZA: "Genel ISG Kanunu Madde 4 (Is Kazasi ve Acil Mudahale)",
    EventType.SINIFLANDIRILAMADI: "8 Ana ISG Kurali Disinda Supheli Risk / Inceleme Bekliyor",
    EventType.GENEL_GOZLEM: None,
}
"""`EventType` -> ilgili mevzuat maddesinin kisa referans etiketi.

Etiketler `src/memory/embedding_rag_service.py::DEFAULT_ISG_REGULATIONS`
icindeki tam metinlerin basliklarina karsilik gelir (o dosyaya bagimlilik
eklememek icin metin burada tekrarlanmaz, yalnizca referans etiketi
tutulur). `None` degeri, o kategorinin mevcut mevzuat setinde karsiligi
olmadigi anlamina gelir (bkz. `YETKISIZ_ERISIM`/`GENEL_GOZLEM` docstring'leri).
"""


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


class TemporalEvent(BaseModel):
    """T009 Temporal Reasoning ciktisi: zaman ekseninde gruplanmis/iliskilendirilmis olay.

    `TemporalReasoner`, ayni `event_type`e sahip ve birbirine yakin zamanli
    `DetectedEvent`leri tek bir `TemporalEvent`e (bir "suregelen olay") indirger
    (`occurrence_count > 1`, `duration > 0`); tek basina kalan `DetectedEvent`ler
    `occurrence_count == 1`, `duration == 0.0` ile birebir bir `TemporalEvent`e
    doner. `related_events`, zaman ekseninde yakin duran (ayni ya da farkli
    tipte) diger `TemporalEvent`lerin `event_id`lerini tasir.
    """

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
    """Rule Engine'in (T010) bir `TemporalEvent`e karsi eslestirdigi ISG/saha kurali.

    Iki kural turu uretir:
    - Tekli kural (deterministik mevzuat eslestirmesi): `event_type`,
      eslesen `TemporalEvent`in tipiyle birebir aynidir; `related_event_ids`
      bostur.
    - Kombinasyon kurali (bilesik ihlal, `src/event_analysis/rules/isg_rules.yaml`
      icinde tanimli): `event_type`, kuralin gerektirdigi tum tiplerin `+`
      ile birlesimidir (orn. `"kkd_ihlali+arac_yaya_yakinligi"`);
      `related_event_ids`, `source_event_id` disindaki katilimci
      `TemporalEvent.event_id`lerini tasir.
    """

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
    """T011 Event Builder ciktisi: `TemporalEvent` + ilgili `RuleMatch`lerin birlestirilmis, nihai temsili.

    Iki hedefi ayni anda karsilar:
    - `timestamp`/`description`/`risk_score`/`risk_level`/`source_model`
      alanlari, `src/memory/event_store.py::EventStore.add_event(...)`
      imzasiyla birebir isim eslesir (bkz. `to_event_store_kwargs`);
      `risk_score`/`risk_level` bu asamada henuz bilinmez (05 LangGraph
      Ajaninin karari sonrasi doldurulur), varsayilan `None`'dir.
    - `event_type`/`confidence`/`temporal_event_id`/`related_rule_matches`
      alanlari, `05 LangGraph Agentic Loop`a giden ek baglami tasir.
    """

    # --- src/memory/event_store.py::EventStore.add_event(...) ile birebir isim eslesen alanlar ---
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

    # --- 05 LangGraph Agentic Loop'a giden ek baglam alanlari ---
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
        """`EventStore.add_event(...)`e dogrudan `**kwargs` olarak verilebilecek sozluk uretir.

        Returns:
            `timestamp`, `description`, `risk_score`, `risk_level`,
            `source_model` anahtarlarini iceren, `EventStore.add_event`
            imzasiyla birebir eslesen sozluk.
        """
        return {
            "timestamp": self.timestamp,
            "description": self.description,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "source_model": self.source_model,
        }
