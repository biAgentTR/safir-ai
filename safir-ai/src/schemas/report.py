"""06 - Cikti ve Karar Destek Katmani: yapilandirilmis JSON rapor semasi."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TimelineEntry(BaseModel):
    """Zamansal olay cizelgesindeki tek bir giris."""

    timestamp: float = Field(description="Olayin saniye cinsinden zaman damgasi.")
    description: str = Field(description="Olayin dogal dil aciklamasi.")


class TimelineEvent(BaseModel):
    """Modul 4 spesifikasyonundaki ortak sema: siddet (severity) alani eklenmis olay girisi.

    `TimelineEntry` ile ayni bilgiyi tasir ve mevcut pipeline/UI tarafindan
    uretilen JSON alanlarini degistirmez; `severity` ekleyen tuketiciler
    (orn. gelecekteki bir siddet-siniflandirici) icin ayrica sunulur.
    """

    timestamp: float = Field(description="Olayin saniye cinsinden zaman damgasi.")
    description: str = Field(description="Olayin dogal dil aciklamasi.")
    severity: Optional[str] = Field(
        default=None, description="Olayin siddet seviyesi (orn. dusuk/orta/yuksek/kritik), bilinmiyorsa None."
    )


class RagContext(BaseModel):
    """Modul 4 spesifikasyonundaki ortak sema: FAISS RAG'dan gelen tek bir mevzuat sonucu.

    `SafirReport.relevant_regulations` (duz metin listesi) ile ayni veriyi,
    kural basligi/skor gibi yapilandirilmis alanlarla birlikte sunmak isteyen
    tuketiciler icin kullanilir (bkz. `EmbeddingRAGService.search_laws`).
    """

    rule_title: str = Field(description="Mevzuat/kural maddesinin kisa basligi (orn. 'ISG Yonetmeligi Madde 12').")
    content: str = Field(description="Maddenin tam metni.")
    score: float = Field(description="FAISS benzerlik skoru.")


class EvidenceFrameOut(BaseModel):
    """UI'da gorsel kanit karti olarak gosterilecek bir Olay Grubu zirve karesi."""

    event_id: int = Field(description="Bu karenin ait oldugu Olay Grubu kimligi.")
    timestamp_sec: float = Field(description="Karenin saniye cinsinden zaman damgasi.")
    timestamp_str: str = Field(description="`MM:SS` formatinda okunabilir zaman damgasi.")
    change_score: float = Field(description="Gurultu-tabani-dusulmus degisim skoru.")
    base64_image: str = Field(description="`data:image/jpeg;base64,...` formatinda goruntu.")
    saved_path: Optional[str] = Field(default=None, description="Karenin diskte kayitli oldugu yol.")
    is_fallback: bool = Field(default=False, description="Esik gecilemedigi icin frame 0 fallback'i mi.")


class SamplerStats(BaseModel):
    """VLM Oncesi Katman (CPU Adaptive Frame Sampler) icin GPU tasarruf istatistikleri."""

    total_frames_scanned: int = Field(description="Videodan okunan toplam ham kare sayisi.")
    sampled_frames_evaluated: int = Field(description="Ornekleme adimina gore degerlendirilen kare sayisi.")
    evidence_frame_count: int = Field(description="Esigi gecip Kanit Karesi sayilan kare sayisi.")
    eliminated_frame_count: int = Field(description="Elenen (VLM'e gonderilmeyen) kare sayisi.")
    gpu_savings_ratio_pct: float = Field(description="Elenen karelerin yuzdesi (GPU tasarruf orani, 0-100).")
    elapsed_sec: float = Field(description="Sampler'in videoyu taramasi icin gecen sure (saniye).")


class SafirReport(BaseModel):
    """Sistemler arasi entegrasyona hazir, mock semayla uyumlu nihai rapor.

    Bu model; Turkce dogal dil ozeti, risk skoru/seviyesi, operator aksiyon
    onerisi ve zaman cizelgesini tek bir yapida birlestirir.
    """

    event_id: Optional[int] = Field(
        default=None, description="Bu analizin SQLite'a yazildigi olay kaydinin kimligi (Human-in-the-Loop geri bildirimi icin)."
    )
    video_source: str = Field(description="Analiz edilen video/kamera akisinin kaynagi.")
    generated_at: str = Field(description="Raporun ISO-8601 formatinda uretim zamani.")
    natural_language_summary: str = Field(description="VLM'in urettigi ham Turkce sahne gozlemi.")
    summary: str = Field(
        default="", description="Ajanin urettigi, operatore yonelik sade Turkce durum ozeti (sartname 'summary')."
    )
    risk_score: int = Field(ge=0, le=100, description="0-100 arasi hesaplanmis risk skoru.")
    risk_level: str = Field(description="dusuk | orta | yuksek | kritik")
    recommended_action: str = Field(
        description="Saha operatorune yonelik birincil aksiyon onerisi (geriye-uyum: actions[0])."
    )
    actions: List[str] = Field(
        default_factory=list, description="Operatore yonelik somut aksiyon onerileri listesi (sartname 'actions')."
    )
    detected_event_types: List[str] = Field(
        default_factory=list,
        description="Bu analizde tespit edilen olay kategorileri (bkz. EventType); aciklanabilirlik/olcumleme icin.",
    )
    timeline: List[TimelineEntry] = Field(default_factory=list, description="Kronolojik olay cizelgesi.")
    evidence_frames: List[EvidenceFrameOut] = Field(
        default_factory=list, description="Her Olay Grubunun zirve karesi (goruntu + metadata)."
    )
    relevant_regulations: List[str] = Field(
        default_factory=list, description="FAISS RAG'dan getirilen ilgili ISG mevzuat maddeleri."
    )
    escalation_tier: Optional[str] = Field(
        default=None, description="Otomatik eskalasyon kademesi: monitor | notify | alarm."
    )
    auto_dispatched: bool = Field(
        default=False, description="Saha alarminin operator onayi beklemeden otomatik tetiklenip tetiklenmedigi."
    )
    alert_id: Optional[str] = Field(
        default=None, description="Otomatik tetiklenen saha alarminin kimligi (operator onayi/geri alma icin)."
    )
    sampler_stats: Optional[SamplerStats] = Field(
        default=None, description="CPU suzgec katmaninin GPU tasarruf istatistikleri."
    )
    vlm_model: Optional[str] = Field(default=None, description="Aciklamayi ureten aktif VLM adi.")
    llm_model: Optional[str] = Field(default=None, description="Karari ureten aktif LLM adi.")

    @staticmethod
    def _seconds_to_mmss(seconds: float) -> str:
        """Saniye degerini `MM:SS` bicimine cevirir (sartname olay zaman damgasi icin)."""
        total = int(round(seconds))
        return f"{total // 60:02d}:{total % 60:02d}"

    def to_sartname_json(self) -> dict:
        """Raporu sartnamedeki mock ornekle birebir ayni sekle indirger.

        Sartname ornegi: `{"summary", "events":[{"time","event"}], "risk", "actions"}`.
        `events`, `timeline` girislerinden (`MM:SS` zaman damgasiyla) uretilir;
        `risk`, insan-okur risk seviyesidir. Tam/zengin cikti icin
        `model_dump()` (tum alanlar) kullanilabilir; bu yardimci yalnizca
        sartname-uyumlu ozet gorunumu icindir.

        Returns:
            Sartname semasiyla uyumlu, JSON-serilestirilebilir sozluk.
        """
        return {
            "summary": self.summary or self.natural_language_summary,
            "events": [
                {"time": self._seconds_to_mmss(entry.timestamp), "event": entry.description}
                for entry in self.timeline
            ],
            "risk": self.risk_level,
            "risk_score": self.risk_score,
            "actions": self.actions or ([self.recommended_action] if self.recommended_action else []),
        }

    def to_json_file(self, path: str) -> None:
        """Raporu belirtilen dosya yoluna UTF-8 JSON olarak yazar.

        Args:
            path: Yazilacak `.json` dosyasinin yolu.
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))
