"""T008 - Event Engine: VLM metin ciktisindan yapilandirilmis olay tespiti.

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
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.event_analysis.schemas import DetectedEvent, EventEngineInput, EventType

logger = logging.getLogger(__name__)

_KEYWORD_RULES: Dict[EventType, List[str]] = {
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
    EventType.DUSME_RISKI: [
        "dusme onleyici",
        "yukseklikte calisma",
        "guvenlik kemeri",
        "korkuluk yok",
        "iskele",
    ],
    EventType.YANGIN_DUMAN: [
        "duman",
        "yangin",
        "alev",
        "yanik kokusu",
    ],
    EventType.YETKISIZ_ERISIM: [
        "yetkisiz",
        "izinsiz giris",
        "yasakli alan",
        "guvenlik ihlali",
    ],
}

_BASE_CONFIDENCE = 0.5
_CONFIDENCE_STEP_PER_EXTRA_MATCH = 0.1


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
        """VLM aciklama metnini kural tabanindaki anahtar kelimelere karsi tarar.

        Birden fazla kategori eslesirse, her biri icin ayri bir `DetectedEvent`
        uretilir (guven skoruna gore azalan sirali). Hicbir kategori
        eslesmezse, dusuk guvenli tek bir `genel_gozlem` olayi dondurulur;
        boylece her VLM aciklamasi en az bir `DetectedEvent`e karsilik gelir
        ve `05 LangGraph Ajani`/`Event Gecmisi` katmani bos girdiyle
        karsilasmaz.

        Args:
            engine_input: `EventEngineInput.from_vlm_response(...)` ile
                uretilmis, VLM aciklamasi + zaman damgasini tasiyan girdi.

        Returns:
            Guven skoruna gore azalan sirali `DetectedEvent` listesi.
        """
        text_lower = engine_input.vlm_description.lower()
        detections: List[DetectedEvent] = []

        for event_type, keywords in _KEYWORD_RULES.items():
            matched = [kw for kw in keywords if kw in text_lower]
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
