"""T008 - Event Engine: VLM metin ciktisindan yapilandirilmis olay tespiti.

Kategori kaynagi
-----------------
`_KEYWORD_RULES` anahtarlari `EventType` degerleridir; ilk 8 kategori
`src/memory/embedding_rag_service.py::DEFAULT_ISG_REGULATIONS` icindeki 8
mevzuat maddesiyle birebir hizalanmistir (bkz. `schemas.EVENT_TYPE_REGULATION_MAP`).
`YETKISIZ_ERISIM` ve `GENEL_GOZLEM` mevcut mevzuat setinde karsiligi olmayan,
operasyonel amacli iki ozel kategoridir (bkz. `EventType` docstring'i).

Yaklasim karsilastirmasi
-------------------------
Bu katmanin gorevi, `03 VLM` katmaninin ürettigi serbest metin aciklamayi
(`VLMResponse.description`), `05 LangGraph Agentic Loop`un tuketebilecegi
yapilandirilmis `DetectedEvent` nesnelerine cevirmek. Iki yaklasim
degerlendirildi:

1. Kural tabanli / anahtar kelime eslestirme (secilen yaklasim)
   (+) Deterministik: ayni girdi her zaman ayni ciktiyi uretir, testte
       kolayca dogrulanir.
   (+) GPU/ag bagimliligi yok; milisaniyeler icinde calisir, `02 Adaptive
       Sampler` katmaninin "CPU ONLY" felsefesiyle tutarlidir.
   (+) `05 LangGraph Ajani` zaten ayni metni bir LLM ile degerlendirip risk
       skoru uretiyor; burada ikinci bir LLM cagrisi islevsel tekrar
       olustururdu.
   (-) VLM'in ifade cesitliligine (esanlamli kelimeler, dolayli anlatim,
       olumsuzlama) karsi kirilgan; yeni olay tipleri icin manuel anahtar
       kelime bakimi gerektirir.

2. Kucuk bir LLM'e (Qwen3) siniflandirma sorusu sormak
   (+) Ifade cesitliligine karsi daha dayanikli; yeni kategori eklemek
       kod degil, prompt guncellemesi ister.
   (-) Her VLM ciktisi icin ekstra bir LLM round-trip'i: gecikme ve
       GPU/vLLM bagimliligi ekler.
   (-) Non-deterministik cikti; birim testte mutlaka mock gerektirir.
   (-) `05` katmani zaten ayni metni LLM ile degerlendiriyor; bu adimda
       ikinci bir LLM cagrisi maliyeti karsiliksiz kalir.

Karar: kural tabanli yaklasim secildi (yukaridaki (+) gerekceleriyle).
`EventEngine.__init__`'teki `classifier` parametresi, ileride dusuk-guven
tespitlerini iyilestirmek icin `src.vlm.llm_client.LLMClient` ile ayni
`invoke(messages) -> AIMessage` sozlesmesine sahip opsiyonel bir siniflandirici
enjekte etme imkani birakir; varsayilan `None` ile devre disidir ve `vlm/`
katmanina hicbir bagimlilik eklemez.

T014 - Olumsuzlama tespiti (ve sinirlari)
-------------------------------------------
Yukaridaki (-) maddesinde belirtilen "olumsuzlama" zayifligi, T013'un gercek
pipeline entegrasyonunda somut olarak gozlemlendi: `MockVLMClient`in sabit
aciklamasindaki "Herhangi bir duman veya yangin belirtisi tespit edilmedi."
cumlesi (duman/yangin OLMADIGINI soyluyor), salt substring eslestirmesiyle
`yangin_duman` olarak YANLIS POZITIF tespit ediliyordu. Bunu gidermek icin
`_is_negated()` ile basit bir kelime-penceresi sezgiseli eklendi: bir anahtar
kelime, ayni klanuz (cumle) icinde, kelimenin oncesindeki VE sonrasindaki
`_NEGATION_WINDOW_WORDS` kelime icinde `_NEGATION_CUES`'ten biri geciyorsa
"olumsuzlanmis" sayilir ve o eslesme atlanir. Pencere hem geriye hem ileriye
tarandi, cunku Turkce SOV (Ozne-Nesne-Yuklem) sozdiziminde olumsuzlama eki
genellikle klanuzun SONUNDAKI yuklemde bulunur (yukaridaki "tespit edilmedi"
ornegindeki gibi) - yalnizca geriye bakan bir pencere bu yaygin durumu
kacirirdi.

Bu, TAM bir NLP/bagimlilik ayristirma (dependency parsing) cozumu DEGILDIR;
bilinen sinirlamalari:
- Cok cumlecikli/ic ice gecmis olumsuzlamalari (orn. "X olmadigini soylemek
  yanlis olmaz" gibi cift olumsuzlama) ayirt edemez.
- Pencere disina tasan (uzun ana cumleler, ara cumlecikler) olumsuzlamalari
  kacirabilir.
- `_NEGATION_CUES` listesi kapsamli degildir; yeni olumsuzlama ifadeleri
  (esanlamlilar) elle eklenmelidir.
- Klanuz icinde AYNI anahtar kelimenin birden fazla gectigi durumlarda,
  yalnizca ILK gecislerin klanuzu kontrol edilir (`str.find` tabanli).

Bu sinirlamalar, T008'de zaten belgelenen "kural tabanli yaklasimin ifade
cesitliligine kirilganligi" zayifliginin bir devami/kismi iyilestirmesidir;
tam giderilmesi icin (2) numarali LLM-tabanli alternatif (veya gercek bir
NLP kutuphanesi) gerekir.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.event_analysis.schemas import DetectedEvent, EventEngineInput, EventType

logger = logging.getLogger(__name__)

_KEYWORD_RULES: Dict[EventType, List[str]] = {
    EventType.DUSME_RISKI: [
        "dusme onleyici",
        "yukseklikte calisma",
        "guvenlik kemeri",
        "korkuluk yok",
        "iskele",
    ],
    EventType.KKD_IHLALI: [
        "baretsiz",
        "yeleksiz",
        "koruyucu ekipman eksik",
        "kkd eksik",
        "korumasiz alan",
    ],
    EventType.ARAC_YAYA_YAKINLIGI: [
        "forklift",
        "yaya gecidi",
        "arac yaklas",
        "carpisma riski",
        "yaya trafigi",
    ],
    EventType.SICAK_CALISMA_IHLALI: [
        "sicak calisma izni",
        "kaynak islemi",
        "kivilcim",
        "izinsiz ates",
    ],
    EventType.YANGIN_DUMAN: [
        "duman",
        "yangin",
        "alev",
        "yanik kokusu",
    ],
    EventType.DAR_ALAN_IHLALI: [
        "kapali alan",
        "dar alan",
        "gaz olcumu yapilmadan",
        "gozetmen olmadan",
    ],
    EventType.ENERJI_KESME_IHLALI: [
        "elektrik pano",
        "enerji kesme",
        "loto",
        "kilitleme etiketleme",
    ],
    EventType.AGIR_YUK_RISKI: [
        "vinc",
        "kren",
        "agir yuk kaldirma",
        "sinyalman olmadan",
    ],
    EventType.YETKISIZ_ERISIM: [
        "yetkisiz",
        "izinsiz giris",
        "yasakli alan",
        "guvenlik ihlali",
    ],
    EventType.SINIFLANDIRILAMADI: [
        "siniflandirilamayan",
        "tanimlanamayan risk",
        "belirsiz ihlal",
        "anormallik",
        "bilinmeyen durum",
    ],
}

_BASE_CONFIDENCE = 0.5
_CONFIDENCE_STEP_PER_EXTRA_MATCH = 0.1

# VLM'in dogrudan urettigi tipli olaylari dogrulamak icin gecerli EventType degerleri.
_VALID_EVENT_TYPES = {event_type.value for event_type in EventType}

_NEGATION_CUES: List[str] = [
    "degil",
    "yok",
    "yoktur",
    "olmadi",
    "bulunmuyor",
    "gorulmedi",
    "gozlemlenmedi",
    "gozlenmedi",
    "tespit edilmedi",
    "rastlanmadi",
]
"""Olumsuzlama ifadeleri (TR, kod tabaninin geri kalaniyla tutarli ASCII
harflerle). Kapsamli DEGILDIR; bkz. modul dokustringindeki T014 bolumu."""

_NEGATION_WINDOW_WORDS = 5
"""Bir anahtar kelimenin olumsuzlanmis sayilmasi icin, kelimenin basladigi/
bittigi konumdan ONCE ve SONRA (ayni klanuz icinde) taranacak kelime sayisi."""

_CLAUSE_SPLIT_PATTERN = re.compile(r"[.!?;]+")


def _split_into_clauses(text: str) -> List[str]:
    """Metni cumle/klanuz sinirlarina (`. ! ? ;`) gore ayri parcalara boler.

    Olumsuzlama kontrolu, bir anahtar kelimenin yalnizca KENDI klanuzu
    icindeki baglamini dikkate almalidir (baska bir cumledeki olumsuzlama
    bu kelimeyi etkilememeli).

    Args:
        text: Kucuk harfe cevrilmis VLM aciklama metni.

    Returns:
        Bos olmayan, bas/son bosluklari temizlenmis klanuz listesi. Hicbir
        ayirici bulunmazsa, tum metni tek elemanli liste olarak dondurur.
    """
    return [clause.strip() for clause in _CLAUSE_SPLIT_PATTERN.split(text) if clause.strip()]


def _is_negated(clause: str, keyword: str, window_words: int = _NEGATION_WINDOW_WORDS) -> bool:
    """Bir anahtar kelimenin, bulundugu klanuz icinde olumsuzlanip olumsuzlanmadigini kontrol eder.

    Basit bir kelime-penceresi sezgiseli kullanir: anahtar kelimenin
    basladigi kelime konumundan `window_words` kadar ONCESI ve bittigi
    konumdan `window_words` kadar SONRASI (ayni klanuz icinde) taranir; bu
    pencerede `_NEGATION_CUES`'ten biri geciyorsa anahtar kelime
    olumsuzlanmis sayilir. Bkz. modul dokustringindeki T014 bolumu icin
    bu yaklasimin (SOV sozdizimi nedeniyle iki yonlu tarama) gerekcesi ve
    bilinen sinirlamalari.

    Args:
        clause: Anahtar kelimeyi iceren, kucuk harfli klanuz metni.
        keyword: Kontrol edilecek anahtar kelime (tek veya cok kelimeli).
        window_words: Her iki yonde taranacak kelime sayisi.

    Returns:
        Pencere icinde bir olumsuzlama ifadesi geciyorsa `True`.
    """
    start_char = clause.find(keyword)
    if start_char == -1:
        return False

    words = clause.split()
    prefix_word_count = len(clause[:start_char].split())
    keyword_word_count = len(keyword.split())

    window_start = max(0, prefix_word_count - window_words)
    window_end = min(len(words), prefix_word_count + keyword_word_count + window_words)
    context = " ".join(words[window_start:window_end])

    return any(cue in context for cue in _NEGATION_CUES)


class EventEngine:
    """VLM aciklama metnini tarayip yapilandirilmis `DetectedEvent` listesi ureten katman.

    Bkz. modul dokustringi icin kural-tabanli/LLM-tabanli yaklasim
    karsilastirmasi ve secilen yaklasimin gerekcesi.
    """

    def __init__(self, classifier: Optional[Any] = None, min_confidence: float = 0.0) -> None:
        """EventEngine'i opsiyonel bir siniflandirici ve guven esigiyle baslatir.

        Args:
            classifier: `invoke(messages) -> AIMessage` sozlesmesine sahip
                opsiyonel bir LLM istemcisi (orn. `src.vlm.llm_client.LLMClient`
                veya `MockLLMClient`). Su an tespit mantiginda kullanilmaz;
                ileride dusuk-guven tespitlerini iyilestirmek icin ayrilmis
                bir genisletme noktasidir.
            min_confidence: Bu esigin altindaki tespitler `detect()`
                ciktisindan elenir (varsayilan 0.0: hicbiri elenmez).
        """
        self._classifier = classifier
        self._min_confidence = min_confidence

    def detect(self, engine_input: EventEngineInput) -> List[DetectedEvent]:
        """VLM ciktisindan yapilandirilmis olaylar uretir (once model-tabanli, sonra anahtar-kelime).

        Oncelik sirasi:
        1. VLM dogrudan tipli olaylar urettiyse (`EVENTS_JSON`,
           `engine_input.structured_events`), bunlar dogrulanip kullanilir
           (model-tabanli siniflandirma; sartname "statik kural" cezasindan kacinir).
        2. Yapilandirilmis olay yoksa/gecersizse, mevcut anahtar-kelime kurali
           (olumsuzlama tespitiyle) FALLBACK olarak devreye girer.

        Her iki durumda da hicbir olay uretilemezse, dusuk guvenli tek bir
        `genel_gozlem` olayi dondurulur; boylece her VLM aciklamasi en az bir
        `DetectedEvent`e karsilik gelir.

        Args:
            engine_input: `EventEngineInput.from_vlm_response(...)` ile
                uretilmis, VLM aciklamasi + (varsa) tipli olaylar + zaman damgasi.

        Returns:
            Guven skoruna gore azalan sirali `DetectedEvent` listesi.
        """
        structured = self._detect_from_structured(engine_input)
        if structured:
            structured.sort(key=lambda event: event.confidence, reverse=True)
            logger.debug(
                "EventEngine: t=%.2f -> %d olay (model-tabanli/structured: %s)",
                engine_input.timestamp,
                len(structured),
                ", ".join(d.event_type for d in structured),
            )
            return structured

        text_lower = engine_input.vlm_description.lower()
        clauses = _split_into_clauses(text_lower)
        detections: List[DetectedEvent] = []

        for event_type, keywords in _KEYWORD_RULES.items():
            matched = self._match_keywords_excluding_negated(keywords, text_lower, clauses)
            if not matched:
                continue

            confidence = self._compute_confidence(matched, keywords)
            if confidence < self._min_confidence:
                continue

            detections.append(
                DetectedEvent(
                    event_type=event_type.value,
                    description=engine_input.vlm_description,
                    timestamp=engine_input.timestamp,
                    confidence=confidence,
                    matched_keywords=matched,
                    source_model=engine_input.source_model,
                )
            )

        if not detections:
            detections.append(
                DetectedEvent(
                    event_type=EventType.GENEL_GOZLEM.value,
                    description=engine_input.vlm_description,
                    timestamp=engine_input.timestamp,
                    confidence=0.0,
                    matched_keywords=[],
                    source_model=engine_input.source_model,
                )
            )

        detections.sort(key=lambda event: event.confidence, reverse=True)
        logger.debug(
            "EventEngine: t=%.2f -> %d olay tespit edildi (%s)",
            engine_input.timestamp,
            len(detections),
            ", ".join(d.event_type for d in detections),
        )
        return detections

    def _detect_from_structured(self, engine_input: EventEngineInput) -> List[DetectedEvent]:
        """VLM'in dogrudan urettigi tipli olaylari (`structured_events`) dogrulayip `DetectedEvent`e cevirir.

        Yalnizca `EventType` enum'unda tanimli tipler kabul edilir (gecersiz
        tipler atlanir); guven skoru 0-1 araligina kirpilir ve zaman damgasi
        item'da yoksa cagrinin zaman damgasina duser. `event_id`/`evidence_ids`/
        `end_time`/`risk_score` (VLM'in olay kumeleme ciktisi - bkz.
        `src/prompts/vlm_prompts.py`) varsa aynen tasinir; hicbiri VLM
        katmaninda UYDURULMAZ, yalnizca burada oldugu gibi aktarilir (kimlik
        dogrulamasi/`unassigned` yonetimi `src/main.py` katmaninda yapilir).
        Gecerli hicbir olay yoksa bos liste doner ve cagiran (`detect`)
        anahtar-kelime fallback'ine gecer.

        Args:
            engine_input: (varsa) `structured_events` tasiyan girdi.

        Returns:
            Dogrulanmis `DetectedEvent` listesi (siralama cagirana birakilir).
        """
        detections: List[DetectedEvent] = []
        for item in engine_input.structured_events:
            if not isinstance(item, dict):
                continue

            raw_type = str(item.get("type", "")).strip().lower()
            if raw_type not in _VALID_EVENT_TYPES:
                logger.debug("Structured olay atlandi (gecersiz tip): %r", raw_type)
                continue

            confidence = self._coerce_confidence(item.get("confidence"))
            if confidence < self._min_confidence:
                continue

            evidence = str(item.get("description") or item.get("evidence") or "").strip() or engine_input.vlm_description
            start_time = self._coerce_timestamp(
                item.get("start_time", item.get("timestamp")), engine_input.timestamp
            )
            end_time_raw = item.get("end_time")
            evidence_ids = [str(eid) for eid in (item.get("evidence_ids") or []) if isinstance(eid, (str, int))]
            detections.append(
                DetectedEvent(
                    event_type=raw_type,
                    description=evidence,
                    timestamp=start_time,
                    end_timestamp=self._coerce_timestamp(end_time_raw, start_time) if end_time_raw is not None else None,
                    confidence=confidence,
                    matched_keywords=self._extract_keywords_for_type(raw_type, evidence),
                    source_model=engine_input.source_model,
                    vlm_event_id=str(item.get("event_id")) if item.get("event_id") is not None else None,
                    evidence_ids=evidence_ids,
                    risk_hint=self._coerce_risk_hint(item.get("risk_score")),
                )
            )
        return detections

    @staticmethod
    def _coerce_risk_hint(value: Any) -> Optional[int]:
        """VLM'in kaba `risk_score` ipucunu 0-100 araligina kirpilmis bir int'e cevirir (gecersizse `None`)."""
        try:
            risk = int(round(float(value)))
        except (TypeError, ValueError):
            return None
        return max(0, min(100, risk))

    @staticmethod
    def _coerce_confidence(value: Any, default: float = 0.5) -> float:
        """Guven skorunu 0.0-1.0 araligina kirpilmis bir float'a cevirir (gecersizse `default`)."""
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return default
        return round(max(0.0, min(1.0, confidence)), 2)

    @staticmethod
    def _coerce_timestamp(value: Any, fallback: float) -> float:
        """Zaman damgasini float'a cevirir; gecersizse cagrinin zaman damgasina (`fallback`) duser."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _extract_keywords_for_type(event_type: str, description: str) -> List[str]:
        """VLM'in yapilandirilmis (`EVENTS_JSON`) olayindaki serbest metin aciklamadan canonical anahtar kelime cikarir.

        Model-tabanli (structured) path daha once `matched_keywords=[]`
        birakiyordu (yalnizca `type` alani vardi); bu, `TemporalReasoner`in
        kelime birlestirmesini ve `RuleEngine`in kombinasyon kurallarini
        zayiflatiyordu. Bu metod, ayni `_KEYWORD_RULES` canonical listesini
        (anahtar-kelime fallback'inde zaten kullanilan, gercek VLM
        ciktilarindan derlenmis kelime dagarcigi) olayin kendi `event_type`ina
        karsi calistirarak VLM'in farkli ifadelerini ayni canonical kelimeye
        indirger (normalizasyon); olumsuzlanmis gecisler `_is_negated` ile
        elenir (bkz. T014).

        Args:
            event_type: `DetectedEvent.event_type` (VLM'in verdigi tip).
            description: Bu olaya ait serbest metin aciklama.

        Returns:
            Eslesen canonical anahtar kelime listesi (bulunamazsa bos liste;
            `EventType`e karsilik gelen bir `_KEYWORD_RULES` girdisi yoksa da bos).
        """
        keywords = _KEYWORD_RULES.get(EventType(event_type)) if event_type in _VALID_EVENT_TYPES else None
        if not keywords:
            return []

        text_lower = description.lower()
        clauses = _split_into_clauses(text_lower)
        return EventEngine._match_keywords_excluding_negated(keywords, text_lower, clauses)

    @staticmethod
    def _match_keywords_excluding_negated(
        keywords: List[str], text_lower: str, clauses: List[str]
    ) -> List[str]:
        """Metinde gecen ama olumsuzlanmamis anahtar kelimeleri dondurur (bkz. T014, `_is_negated`).

        Args:
            keywords: Bu kategori icin tanimli anahtar kelime listesi.
            text_lower: Kucuk harfli tam VLM aciklamasi (hizli on-filtre icin).
            clauses: `text_lower`in klanuzlara bolunmus hali.

        Returns:
            Metinde gecen VE olumsuzlanmamis anahtar kelimelerin listesi
            (orijinal `keywords` sirasi korunur).
        """
        matched: List[str] = []
        for keyword in keywords:
            if keyword not in text_lower:
                continue

            containing_clause = next((clause for clause in clauses if keyword in clause), None)
            if containing_clause is not None and _is_negated(containing_clause, keyword):
                continue

            matched.append(keyword)
        return matched

    @staticmethod
    def _compute_confidence(matched: List[str], keywords: List[str]) -> float:
        """Eslesen anahtar kelime sayisina gore basit, deterministik bir guven skoru hesaplar.

        Args:
            matched: Metinde bulunan anahtar kelimeler.
            keywords: Bu kategori icin tanimli tum anahtar kelimeler.

        Returns:
            0.0-1.0 araliginda, tek eslesme icin `_BASE_CONFIDENCE`'tan
            baslayip her ek eslesme icin `_CONFIDENCE_STEP_PER_EXTRA_MATCH`
            artan, 1.0'da tavanlanan bir skor.
        """
        extra_matches = len(matched) - 1
        confidence = _BASE_CONFIDENCE + extra_matches * _CONFIDENCE_STEP_PER_EXTRA_MATCH
        return round(min(1.0, confidence), 2)


if __name__ == "__main__":
    # T008'in bagimsiz calistirilabilirlik testi:
    #   python -m src.event_analysis.event_engine
    logging.basicConfig(level=logging.INFO)

    demo_input = EventEngineInput(
        vlm_description=(
            "Sahada bir personel korumasiz alanda, baretsiz calisiyor. Yakin "
            "cevrede forklift bir yaya gecidine yaklasiyor."
        ),
        timestamp=12.4,
        source_model="demo-vlm",
        frame_count=1,
    )
    demo_events = EventEngine().detect(demo_input)
    for demo_event in demo_events:
        print(
            f"[{demo_event.event_type}] conf={demo_event.confidence} "
            f"keywords={demo_event.matched_keywords}"
        )
