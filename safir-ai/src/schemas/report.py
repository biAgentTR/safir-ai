"""06 - Cikti ve Karar Destek Katmani: yapilandirilmis JSON rapor semasi."""

from __future__ import annotations

from typing import Dict, List, Optional

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


class EventSummary(BaseModel):
    """T020: Bir `StructuredEvent`in rapora/API'ye tasinan ozeti - UC AYRI kavrami acikca ayirir.

    1. `event_name`: olayin BIRINCIL kimligi - VLM'in KENDI urettigi, ONCEDEN
       TANIMLI bir taksonomiyle SINIRLI OLMAYAN serbest bicimli isim (orn.
       "yerde_hareketsiz_kisi"). HER ZAMAN doludur.
    2. `event_type`: OPSIYONEL canonical baglanti (bkz. `EventType`); yalnizca
       VLM'in gozlemi ZATEN BILINEN bir kategoriye GERCEKTEN karsilik
       geliyorsa doludur. `None` = "eslestirilemedi" (GECERLI, ZORLANMAZ).
    3. `keywords`: VLM'in evidence karelerinde GERCEKTEN gozlemledigi, HICBIR
       taksonomiyle FILTRELENMEYEN serbest kanit ifadeleri (bkz.
       `StructuredEvent.keywords`, `EventEngine._normalize_free_form_keywords`).

    `risk_level`/`risk_score`, bu olaya ait `RuleEngine` eslesmelerinden
    (varsa) deterministik olarak gelir (bkz. `risk_resolver.py`, degistirilmedi);
    eslesme yoksa (orn. `event_type=None` oldugunda COGUNLUKLA boyledir) HER
    IKISI DE `None` kalir - "Degerlendirilmedi" GECERLI bir sonuctur, risk
    UYDURULMAZ.
    """

    event_name: str = Field(
        description="Olayin BIRINCIL, serbest-bicimli kimligi (VLM-uretimi; taksonomiyle SINIRLI DEGIL)."
    )
    event_type: Optional[str] = Field(
        default=None,
        description="OPSIYONEL canonical baglanti (bkz. `EventType`); `None` = 'eslestirilemedi'.",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="VLM'in urettigi serbest-bicimli kanit ifadeleri (taksonomiyle SINIRLI DEGIL, risk KARARI DEGIL).",
    )
    risk_level: Optional[str] = Field(
        default=None, description="Bu olaya ait deterministik RuleEngine eslesmesi (varsa); yoksa 'Degerlendirilmedi'."
    )
    risk_score: Optional[int] = Field(default=None, ge=0, le=100, description="Bkz. `risk_level`.")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Izlenebilirlik: bu olayin dayandigi kanit karesi kimlikleri (bkz. "
        "`StructuredEvent.evidence_ids`) - Evidence -> Event -> Rapor zincirini rapor "
        "duzeyinde de takip edilebilir kilar.",
    )
    rule_ids: List[str] = Field(
        default_factory=list,
        description="Izlenebilirlik: bu olaya uygulanan `RuleMatch.rule_id` degerleri (bkz. "
        "`StructuredEvent.related_rule_matches`) - `risk_level`in HANGI kural(lar)dan "
        "geldigini gosterir; kurallar INSAN-YAZIMI ve sabittir, LLM tarafindan uretilmez.",
    )


class RagContext(BaseModel):
    """Modul 4 spesifikasyonundaki ortak sema: FAISS RAG'dan gelen tek bir mevzuat sonucu.

    `SafirReport.relevant_regulations` (duz metin listesi, RuleEngine-turevli
    kisa basliklar) ile ayni veriyi, kural basligi/skor gibi yapilandirilmis
    alanlarla birlikte sunmak isteyen tuketiciler icin kullanilir (bkz.
    `EmbeddingRAGService.search_laws`).

    2026-08-24 (RAG entegrasyon dogrulama turu - traceability gap kapatildi):
    Onceden bu sema TANIMLIYDI ama HICBIR YERDE POPULATE EDILMIYORDU - operator,
    canli pipeline SSE trace'i disinda (kalici DEGIL, yalnizca o anki run icin),
    "bu karar hangi mevzuat maddesine dayaniyor?" sorusunu rapor UZERINDEN
    CEVAPLAYAMIYORDU. `SafirReport.semantic_rag_sources` alani artik bunu
    (chunk_id/document_id/article_number/source_url ile birlikte) KALICI
    olarak tasir - bkz. `main.py::build_report`. Bu, risk_score/risk_level
    hesaplamasini ETKILEMEZ (semantik RAG zaten karar mekanizmasina HIC
    girmiyordu, bkz. `context_builder.py` modul dokustringi) - yalnizca
    ZATEN var olan bilginin kalici hale getirilmesidir.
    """

    rule_title: str = Field(description="Mevzuat/kural maddesinin kisa basligi (orn. 'ISG Yonetmeligi Madde 12').")
    content: str = Field(description="Maddenin tam metni.")
    score: float = Field(description="GERIYE-UYUMLULUK: `embedding_score` ile AYNI deger (bkz. asagida) - eski tuketiciler icin korunur.")
    embedding_score: Optional[float] = Field(
        default=None,
        description=(
            "E5/FAISS `IndexFlatIP` cosine benzerlik skoru (semantic_score'un HAM girdisi) - "
            "`relevance_score`/`cross_encoder_score` ile KARISTIRILMAZ (bkz. RAG finalizasyon "
            "turu gorev tanimi 5. bolum: 'embedding_score = E5/FAISS semantic similarity'). "
            "`score` alaniyla AYNI degeri tasir, yalnizca ismi kanonik/acik."
        ),
    )
    chunk_id: Optional[str] = Field(default=None, description="Kaynak chunk'in kimligi (persisted KB index'teki).")
    document_id: Optional[str] = Field(default=None, description="Kaynak mevzuat dokumaninin kimligi.")
    article_number: Optional[str] = Field(default=None, description="Madde/ek numarasi (orn. 'I.3.1').")
    source_url: Optional[str] = Field(default=None, description="Resmi mevzuat kaynak URL'si (varsa).")
    relevance_score: Optional[float] = Field(
        default=None,
        description=(
            "Deterministik, agirlikli-toplam relevance skoru (bkz. "
            "`src/rag/deterministic_reranker.py::score_candidate`) - LLM'e SORULMAZ; "
            "relevance skorlama devre disiysa `None`."
        ),
    )
    semantic_score: Optional[float] = Field(
        default=None,
        description=(
            "`relevance_score`e giden bes bilesenden biri (bkz. "
            "`deterministic_reranker.RelevanceBreakdown.semantic_score`) - E5/FAISS "
            "benzerliginin [0,1] araligina kirpilmis hali. Relevance skorlama devre "
            "disiysa/hesaplanmadiysa `None` (UYDURULMAZ)."
        ),
    )
    lexical_score: Optional[float] = Field(
        default=None, description="`RelevanceBreakdown.lexical_score` - sorgu/chunk token orusumu. `None` = hesaplanmadi."
    )
    keyword_score: Optional[float] = Field(
        default=None, description="`RelevanceBreakdown.keyword_score` - VLM matched_keywords orusumu. `None` = hesaplanmadi."
    )
    metadata_score: Optional[float] = Field(
        default=None, description="`RelevanceBreakdown.metadata_score` - baslik/madde numarasi eslesmesi. `None` = hesaplanmadi."
    )
    phrase_score: Optional[float] = Field(
        default=None, description="`RelevanceBreakdown.phrase_score` - tam ifade eslesmesi. `None` = hesaplanmadi."
    )
    relevance_status: Optional[str] = Field(
        default=None,
        description=(
            "'accepted' | 'rejected' (bkz. `RetrievedDocument.relevance_status`) - bu kaynagin "
            "threshold/top_k SONRASI nihai sonuc kumesine (Agent'in GORDUGU `semantic_rag_sources`) "
            "GIRIP GIRMEDIGINI backend'in KENDI canonical karar alanindan tasir; UI 'Seçildi' "
            "kolonu bunu okur, kendi basina bir esik hesaplamasi YAPMAZ."
        ),
    )
    relevance_reason: Optional[str] = Field(
        default=None,
        description="`relevance_status`un insan-okunur gerekcesi (bkz. `RetrievedDocument.relevance_reason`).",
    )
    cross_encoder_score: Optional[float] = Field(
        default=None,
        description=(
            "LOKAL Cross-Encoder'in (bkz. `src/rag/local_cross_encoder_reranker.py`) (query, chunk) "
            "cift relevance skoru - `embedding_score`/`relevance_score`den AYRI, `risk_score`/"
            "`confidence`/`probability` OLARAK ADLANDIRILMAZ. Cross-Encoder bu cagrida devreye "
            "GIRMEDIYSE (model kullanilamadi/devre disi) `None`."
        ),
    )
    final_rank: Optional[int] = Field(
        default=None,
        description="Nihai (Cross-Encoder SONRASI, devredeyse) 1-index'li sira (bkz. `RetrievedDocument.final_rank`).",
    )
    source_verified: bool = Field(
        default=True,
        description=(
            "Bu kaynagin GERCEKTEN persisted KB index'inden geldigi (bkz. "
            "`RetrievedDocument.source_verified`) - retrieval sonucu OLDUGU icin HER ZAMAN `True`dur. "
            "YUKSEK bir `relevance_score`, source_verified=False bir kaynagi ASLA 'dogrulanmis "
            "mevzuat kaniti' YAPMAZ - bu iki alan BAGIMSIZDIR (bkz. gorev tanimi 15. bolum)."
        ),
    )


class EvidenceFrameOut(BaseModel):
    """UI'da gorsel kanit karti olarak gosterilecek, VLM'in kumeledigi bir olayin temsili karesi."""

    event_id: str = Field(description="Bu karenin ait oldugu VLM olayinin kimligi (bkz. EVENTS_JSON.event_id).")
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
    risk_score: Optional[int] = Field(
        default=None, ge=0, le=100, description="0-100 arasi hesaplanmis risk skoru; guvenilir karar uretilemediyse None."
    )
    risk_level: str = Field(description="dusuk | orta | yuksek | kritik | unknown")
    risk_status: str = Field(
        default="assessed",
        description=(
            "'assessed': risk_score/risk_level guvenilir sekilde hesaplandi (0 dahil gecerli bir deger). "
            "'unclassified': olay/risk tespit edildi veya kumeleme yapildi ancak 8 temel ISG kategorisine oturtulamadi; "
            "bu durumda risk_score null (None) olarak set edilir ve 'risk yok' (0 risk) durumundan kesin olarak ayrilir. "
            "'unknown': VLM/LLM/ajan karar zincirinde hata olustu veya guvenilir karar uretilemedi."
        ),
    )
    risk_source: Optional[str] = Field(
        default=None,
        description=(
            "Nihai `risk_score`/`risk_level`in HANGI mekanizmadan geldigi (izlenebilirlik): "
            "'rule_engine' (deterministik RuleEngine eslesmesi, HER ZAMAN Agent'in kendi tahminini "
            "EZER), 'agent' (hicbir kural eslesmedi, Agent'in KENDI dogrulanmamis tahmini korundu), "
            "'unknown' (analiz basarisiz oldu, risk hic belirlenemedi)."
        ),
    )
    risk_explanation: Optional[str] = Field(
        default=None,
        description=(
            "Nihai riskin deterministik, LLM'e SORULMAMIS Turkce gerekcesi (bkz. "
            "`risk_resolver.RiskProvenance.explanation`). Operatorun 'bu risk neden bu deger?' "
            "sorusunu, structured RuleMatch verisinden turemis bir cumleyle cevaplar."
        ),
    )
    contributing_rule_ids: List[str] = Field(
        default_factory=list,
        description="Nihai risk_level'i belirleyen (en yuksek siddetli) RuleMatch(ler)in rule_id'leri; "
        "risk_source='rule_engine' degilse bos liste.",
    )
    scoring_method: Optional[str] = Field(
        default=None,
        description=(
            "RISK ENGINE V2: nihai skoru ureten matematiksel modelin kimligi (orn. "
            "'safir_evidence_weighted_v2') - bkz. `src/event_analysis/risk_model.py`. "
            "risk_source='rule_engine' degilse `None`."
        ),
    )
    risk_features: Optional[Dict[str, Optional[float]]] = Field(
        default=None,
        description=(
            "Nihai skoru ureten sekiz normalize edilmis (0.0-1.0) feature: severity/likelihood/"
            "exposure/duration/recurrence/protection_gap/rule_support/regulatory_support. "
            "`None` deger = bu cagirida OLCULEMEDI (notr/guvenli varsayilan kullanildi, UYDURULMADI)."
        ),
    )
    risk_feature_contributions: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Skora giden ara carpim adimlari (base_risk + temporal/exposure/protection/evidence_factor + "
            "raw_score) - jurinin/operatorun 'bir feature degisince skor NEDEN degisti?' sorusunu "
            "izleyebilmesi icin (bkz. `risk_model.RiskScoreBreakdown.as_contributions_dict`)."
        ),
    )
    llm_proposed_score: Optional[int] = Field(
        default=None,
        description=(
            "Agent'in (05 LangGraph) KENDI, dogrulanmamis taslak risk_score'u - ASLA final_score/"
            "risk_score'u BELIRLEMEZ, yalnizca karsilastirma/kalibrasyon icin izlenir "
            "(bkz. gorev tanimi 8. bolum)."
        ),
    )
    recommended_action: str = Field(
        description="Saha operatorune yonelik birincil aksiyon onerisi (geriye-uyum: actions[0])."
    )
    actions: List[str] = Field(
        default_factory=list, description="Operatore yonelik somut aksiyon onerileri listesi (sartname 'actions')."
    )
    onset_timestamp_str: Optional[str] = Field(
        default=None, description="Olayin/kazanin ILK BAŞLADIGI kareden alinan zaman damgasi (MM:SS)."
    )
    safe_timestamps: List[str] = Field(
        default_factory=list, description="Hicbir kazanin/riskin olmadigi rutin karesel zaman damgalari (orn. 00:07, 00:10, 00:12, 00:15)."
    )
    incident_timestamps: List[str] = Field(
        default_factory=list, description="Tehlikenin/kazanin aktif oldugu zaman damgalari (orn. 00:18, 00:22, 00:25)."
    )
    detected_event_names: List[str] = Field(
        default_factory=list,
        description="Bu analizde tespit edilen olaylarin BIRINCIL, serbest-bicimli kimlikleri (event_name); "
        "ONCEDEN TANIMLI bir taksonomiyle SINIRLI DEGILDIR (bkz. `EventSummary.event_name`).",
    )
    detected_event_types: List[str] = Field(
        default_factory=list,
        description="Bu analizde tespit edilen olaylardan, YALNIZCA ZATEN BILINEN bir kategoriye (bkz. EventType) "
        "GERCEKTEN karsilik gelenlerin canonical listesi; eslesmeyenler burada GORUNMEZ (bkz. `detected_event_names`).",
    )
    events: List[EventSummary] = Field(
        default_factory=list,
        description="Her tespit edilen olayin (StructuredEvent) ozeti: event_name (birincil, serbest-bicimli), "
        "opsiyonel canonical event_type, VLM-uretimi keywords, ve (varsa) deterministik risk (bkz. `EventSummary`). "
        "Hicbiri taksonomiyle zorla FILTRELENMEZ/DEGISTIRILMEZ.",
    )
    timeline: List[TimelineEntry] = Field(default_factory=list, description="Kronolojik olay cizelgesi.")
    evidence_frames: List[EvidenceFrameOut] = Field(
        default_factory=list, description="Her Olay Grubunun zirve karesi (goruntu + metadata)."
    )
    relevant_regulations: List[str] = Field(
        default_factory=list,
        description="RuleEngine-dogrulanmis (deterministik) mevzuat basliklari - risk kararina baglidir.",
    )
    unverified_references: List[str] = Field(
        default_factory=list,
        description=(
            "Ajanin serbest metninde (summary/actions) gecen, mevzuat-atfi GIBI GORUNEN "
            "ama bu cagrida GERCEKTEN retrieved olan `semantic_rag_sources`den HICBIRIYLE "
            "eslesmeyen ifadeler (bkz. `main.py::_unverified_citations`, gorev tanimi 10. "
            "bolum). Deterministik, regex-tabanli bir kontroldur (LLM'e SORULMAZ); TAM bir "
            "NLP atif-dogrulamasi DEGILDIR - yalnizca acikca UYDURULMUS gorunen, corpus'ta "
            "karsiligi bulunamayan referanslari isaretler. Bos liste = ya hicbir mevzuat-"
            "benzeri ifade gecmedi ya da gecenlerin TAMAMI retrieved evidence'la eslesti."
        ),
    )
    semantic_rag_sources: List[RagContext] = Field(
        default_factory=list,
        description=(
            "Bu analizde semantik RAG sorgusunun (VLM keyword'lerinden kurulan, bkz. "
            "`main.py::_build_semantic_query`) persisted KB index'inden GERCEKTEN sectigi "
            "kaynaklar - chunk_id/document_id/article_number/source_url ile birlikte. "
            "`relevant_regulations`den BAGIMSIZDIR, risk_score/risk_level'i ETKILEMEZ "
            "(bkz. RagContext docstring'i); yalnizca 'bu karar/gozlem hangi mevzuat "
            "maddesine dayaniyor?' sorusunun KALICI, iz-surulebilir cevabidir. Esik-uzeri "
            "sonuc bulunamadiysa VEYA reranker basarisiz olduysa (bkz. `rag_security` trace "
            "stage'i) BOS LISTE - GECERLI bir sonuctur, uydurulmus bir kaynak EKLENMEZ."
        ),
    )
    cross_encoder_status: Optional[str] = Field(
        default=None,
        description=(
            "Bu analizin semantik RAG sorgusunda LOKAL Cross-Encoder'in GERCEKTEN calisip "
            "calismadigi: 'used' (calisti, `semantic_rag_sources[i].cross_encoder_score` dolu) | "
            "'unavailable' (bir Cross-Encoder verildi ama model agirligi yuklenemedi - kontrollu "
            "sekilde deterministic relevance siralamasina dusuldu, HARICI BIR API'YE DUSULMEDI) | "
            "'disabled' (bu cagriya Cross-Encoder hic verilmedi). `None` = bu cagrida semantik RAG "
            "sorgusu hic yapilmadi (matched_keywords yoktu). UI, `cross_encoder_score` alani `None` "
            "GORUNDUGUNDE bunu SESSIZCE '-' olarak degil, bu alana gore ('kullanilamadi'/'devre disi') "
            "ACIKCA gostermelidir - bkz. `src/rag/embedding_rag_service.py::RagQueryTelemetry.cross_encoder_status`."
        ),
    )
    relevance_threshold: Optional[float] = Field(
        default=None,
        description=(
            "Bu analizde deterministic relevance/evidence gate'in KULLANDIGI esik degeri (bkz. "
            "`RagQueryTelemetry.threshold`, `configs/config.yaml -> memory.reranker.score_threshold`den "
            "okunur, HARD-CODE DEGIL). `relevance_score < relevance_threshold` olan adaylar "
            "REJECTED sayilir. Bu cagrida hic semantik RAG sorgusu yapilmadiysa `None`."
        ),
    )
    relevance_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Bu analizde `deterministic_reranker.score_candidate()`e GERCEKTEN gecirilen "
            "agirliklar (bkz. `EmbeddingRAGService.relevance_weights`, `configs/config.yaml -> "
            "memory.reranker.weights`den okunur, HARD-CODE DEGIL): "
            "{'semantic','lexical','keyword','metadata','phrase'} -> agirlik. UI'nin 'Deterministic "
            "Relevance' aciklamasi/formulu bunu KENDI varsayimindan degil BURADAN okumalidir. "
            "Bu cagrida hic semantik RAG sorgusu yapilmadiysa `None`."
        ),
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
        """Raporu sartnamedeki mock ornekle birebir ayni sekle indirger."""
        return {
            "summary": self.summary or self.natural_language_summary,
            "onset_timestamp": self.onset_timestamp_str or (self._seconds_to_mmss(self.timeline[0].timestamp) if self.timeline else "00:00"),
            "safe_timestamps": self.safe_timestamps,
            "incident_timestamps": self.incident_timestamps,
            "events": [
                {"time": self._seconds_to_mmss(entry.timestamp), "event": entry.description}
                for entry in self.timeline
            ],
            "risk": self.risk_level,
            "risk_score": self.risk_score,
            "risk_status": self.risk_status,
            "actions": self.actions or ([self.recommended_action] if self.recommended_action else []),
        }

    def to_json_file(self, path: str) -> None:
        """Raporu belirtilen dosya yoluna UTF-8 JSON olarak yazar.

        Args:
            path: Yazilacak `.json` dosyasinin yolu.
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))
