"""T010 - Rule Engine: `TemporalEvent` listesini ISG/saha kurallarina karsi sorgular.

Iki kural katmani uretir:

1. Tekli kural (deterministik mevzuat eslestirmesi): her `TemporalEvent`in
   `event_type`'i, T008'de tanimlanan `EVENT_TYPE_REGULATION_MAP`
   (`src/event_analysis/schemas.py`) uzerinden ilgili mevzuat maddesiyle
   eslestirilir. Mevzuat karsiligi olmayan kategoriler (`yetkisiz_erisim`,
   `genel_gozlem`) icin eslesme uretilmez.
2. Kombinasyon kurali (bilesik ihlal): `TemporalEvent.related_events`
   uzerinden kurulan "iliski kumeleri" (bir olay + `related_events`'inin
   isaret ettigi diger olaylar), `src/event_analysis/rules/isg_rules.yaml`
   icinde tanimli `required_event_types` kumelerini kapsiyorsa, daha yuksek
   onem/risk sinyali tasiyan bir kombinasyon `RuleMatch`'i uretilir. Kural
   seti koda gomulmez, YAML'dan okunur; boylece yeni bir bilesik ihlal
   kurali eklemek kod degisikligi gerektirmez.

`agent/` bagimliligi
--------------------
Bu modul, `src.agent.tools.RetrieverTool`'u dogrudan import ETMEZ (yasakli
`src/agent/` klasorune bagimlilik eklememek icin). Bunun yerine
`RegulationRetriever` adinda bir `Protocol` tanimlar; `query(question, top_k)
-> List[...]` imzasina uyan herhangi bir nesne (gercek kullanimda dogrudan
`src.rag.embedding_rag_service.EmbeddingRAGService`, testte basit bir mock)
`RuleEngine`e opsiyonel olarak enjekte edilebilir. Retriever verilmezse,
tekli kural eslesmeleri yalnizca `EVENT_TYPE_REGULATION_MAP`'teki kisa
etiketle doner (zenginlestirme atlanir).

2026-08-25 duzeltmesi (getirilen mevzuat metni GERCEK KB'den gelmiyordu):
`_describe_regulation` ONCEDEN `RetrieverTool.run(...) -> str` (LLM-agent icin
bicimlendirilmis DUZ METIN) kullanip, donen metnin sorgulanan KISA ETIKETI
(orn. "Yangin Guvenligi Talimati YG-03" - `EVENT_TYPE_REGULATION_MAP`, bu
etiketler ESKI `DEFAULT_ISG_REGULATIONS` PLACEHOLDER corpus'unun basliklarina
gore yazilmisti, bkz. o sabitin dokustringi) BIREBIR ICERIP ICERMEDIGINI
kontrol ediyordu. Sisteme GERCEK, ~748 chunk'lik Qdrant/EVREN tabanli mevzuat
KB'si baglandiktan SONRA bu kontrol ARTIK HICBIR ZAMAN gecemiyordu (gercek
mevzuat metni, uydurma/placeholder etiketi HARFIYEN icermez) - bu yuzden
zenginlestirme HER ZAMAN reddedilip UI'da hep placeholder etiket gosteriliyordu.
Artik `RegulationRetriever.query(...)` DOGRUDAN `EmbeddingRAGService.query()`e
baglanir; bu metod ZATEN yalnizca deterministik relevance esiginden GECMIS
("accepted") adaylari dondurur (bkz. `embedding_rag_service.py::query`) - o
yuzden `_describe_regulation` ayrica bir metin-icerme kontrolu YAPMAZ, dondurulen
(varsa) ilk sonucu GUVENILIR sayar. Alakasiz sonuclara karsi korunma, artik TEK
bir yerde (deterministik reranker'in `score_threshold` gate'i) yasar - burada
tekrarlanan ve GERCEK KB ile hicbir zaman calismayan kirik bir kontrol degil.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

import yaml

from src.event_analysis.schemas import (
    EVENT_TYPE_REGULATION_MAP,
    EventType,
    RuleMatch,
    TemporalEvent,
)

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "rules" / "isg_rules.yaml"

_RULE_ID_BY_EVENT_TYPE: Dict[EventType, str] = {
    EventType.DUSME_RISKI: "ISG-M12",
    EventType.KKD_IHLALI: "ISG-M24",
    EventType.ARAC_YAYA_YAKINLIGI: "OK-07",
    EventType.SICAK_CALISMA_IHLALI: "ISG-M31",
    EventType.YANGIN_DUMAN: "YG-03",
    EventType.DAR_ALAN_IHLALI: "ISG-M45",
    EventType.ENERJI_KESME_IHLALI: "OK-15",
    EventType.AGIR_YUK_RISKI: "ISG-M52",
}
"""`EventType` -> kisa kural kimligi. `EVENT_TYPE_REGULATION_MAP` (schemas.py)
ile ayni 8 mevzuat maddesini kapsar; burada yalnizca kisa, okunabilir bir
`rule_id` uretmek icin ayrilmistir, mevzuat metnini tekrarlamaz."""

_DEFAULT_SEVERITY_BY_EVENT_TYPE: Dict[EventType, str] = {
    EventType.DUSME_RISKI: "yuksek",
    EventType.KKD_IHLALI: "orta",
    EventType.ARAC_YAYA_YAKINLIGI: "yuksek",
    EventType.SICAK_CALISMA_IHLALI: "yuksek",
    EventType.YANGIN_DUMAN: "kritik",
    EventType.DAR_ALAN_IHLALI: "yuksek",
    EventType.ENERJI_KESME_IHLALI: "kritik",
    EventType.AGIR_YUK_RISKI: "yuksek",
}
"""`EventType` -> Rule Engine'in tekli kurallara atadigi varsayilan siddet.
Bu, `05 LangGraph Ajani`nin urettigi risk skorundan bagimsiz, salt kural
tabanli bir on-degerlendirmedir."""


@runtime_checkable
class RegulationRetriever(Protocol):
    """`src.rag.embedding_rag_service.EmbeddingRAGService.query` ile ayni imzaya sahip herhangi bir nesne icin duck-typing sozlesmesi.

    `RuleEngine`, `agent/` klasorune bagimlilik eklememek icin
    `src.agent.tools.RetrieverTool`'u import etmez; bunun yerine bu Protocol'e
    uyan herhangi bir nesne (gercek kullanimda dogrudan `EmbeddingRAGService`,
    testte basit bir mock) enjekte edilebilir. `src/rag/` bir alt katmandir
    (agent'a bagimli DEGILDIR), bu yuzden ona dogrudan baglanmak yasakli
    `agent/` bagimliligini IHLAL ETMEZ.
    """

    def query(self, question: str, top_k: int = 1) -> List[object]:
        """Verilen soruya en ilgili (ZATEN deterministik relevance esiginden GECMIS) dokuman listesini dondurur."""
        ...


def _safe_event_type(value: Optional[str]) -> Optional[EventType]:
    """Bir string'i `EventType`e cevirir; bilinmeyen/gecersiz/`None` degerlerde `None` dondurur.

    T020: `TemporalEvent.event_type` artik OPSIYONEL (canonical baglanti) -
    `value=None` (VLM'in gozlemi bilinen hicbir kategoriye karsilik gelmiyor,
    yani serbest/yeni bir `event_name`) GECERLI bir girdidir; bu durumda bu
    fonksiyon (kasitli olarak) `None` dondurur ve cagiran taraf HICBIR
    RuleMatch URETMEZ - bir kategoriye ZORLAMA YAPILMAZ.

    Args:
        value: `TemporalEvent.event_type` (opsiyonel canonical baglanti) gibi
            serbest bir string veya `None`.

    Returns:
        Eslesen `EventType` uyesi, yoksa (bilinmeyen deger VEYA `None`) `None`.
    """
    try:
        return EventType(value)
    except ValueError:
        return None


class RuleEngine:
    """`TemporalEvent` listesini tekli mevzuat eslestirmesi ve kombinasyon kurallarina karsi sorgular.

    Bkz. modul dokustringi icin tekli/kombinasyon kural katmanlarinin tam
    aciklamasi ve `agent/`'a bagimlilik eklenmemesini saglayan Protocol
    yaklasimi.
    """

    def __init__(
        self,
        retriever: Optional[RegulationRetriever] = None,
        rules_path: str | Path = DEFAULT_RULES_PATH,
    ) -> None:
        """RuleEngine'i opsiyonel bir retriever ve kombinasyon kurallari dosyasiyla baslatir.

        Args:
            retriever: `RegulationRetriever` sozlesmesine (`.query(question,
                top_k) -> List[...]`) uyan opsiyonel mevzuat arama nesnesi
                (orn. dogrudan `EmbeddingRAGService`). `None` ise tekli kural
                eslesmeleri yalnizca kisa mevzuat etiketiyle doner.
            rules_path: `combination_rules` listesini iceren YAML dosyasinin
                yolu (varsayilan: `src/event_analysis/rules/isg_rules.yaml`).

        Raises:
            FileNotFoundError: `rules_path` mevcut degilse.
        """
        self._retriever = retriever
        self._combination_rules = self._load_combination_rules(Path(rules_path))

    @staticmethod
    def _load_combination_rules(rules_path: Path) -> List[dict]:
        """Kombinasyon kurallari YAML dosyasini okuyup `combination_rules` listesini dondurur.

        Args:
            rules_path: `isg_rules.yaml` formatindaki dosyanin yolu.

        Returns:
            Her biri `rule_id`/`description`/`required_event_types`/`severity`
            anahtarlarini iceren sozluklerden olusan liste.

        Raises:
            FileNotFoundError: Dosya mevcut degilse.
        """
        if not rules_path.exists():
            raise FileNotFoundError(f"Kombinasyon kurallari dosyasi bulunamadi: {rules_path}")

        with rules_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        return data.get("combination_rules", [])

    def evaluate(self, temporal_events: List[TemporalEvent]) -> List[RuleMatch]:
        """`TemporalEvent` listesini tekli ve kombinasyon kurallarina karsi degerlendirir.

        Args:
            temporal_events: `TemporalReasoner.reason(...)` ciktisi.

        Returns:
            Tekli mevzuat eslesmeleri ve kombinasyon (bilesik ihlal)
            eslesmelerinin birlestirilmis listesi. Girdi bossa bos liste
            doner.
        """
        if not temporal_events:
            return []

        sorted_events = sorted(temporal_events, key=lambda event: event.start_timestamp)

        matches: List[RuleMatch] = []
        for event in sorted_events:
            matches.extend(self._match_single_event_rule(event))
        matches.extend(self._match_combination_rules(sorted_events))

        logger.debug(
            "RuleEngine: %d TemporalEvent -> %d RuleMatch", len(temporal_events), len(matches)
        )
        return matches

    def _match_single_event_rule(self, event: TemporalEvent) -> List[RuleMatch]:
        """Tek bir `TemporalEvent`i `EVENT_TYPE_REGULATION_MAP` uzerinden deterministik olarak eslestirir.

        Args:
            event: Degerlendirilecek `TemporalEvent`.

        Returns:
            Mevzuat karsiligi varsa tek elemanli, yoksa bos liste.
        """
        event_type = _safe_event_type(event.event_type)
        if event_type is None:
            return []

        regulation_label = EVENT_TYPE_REGULATION_MAP.get(event_type)
        if regulation_label is None:
            return []

        return [
            RuleMatch(
                rule_id=_RULE_ID_BY_EVENT_TYPE.get(event_type, event_type.value),
                rule_description=self._describe_regulation(regulation_label),
                event_type=event.event_type,
                severity=_DEFAULT_SEVERITY_BY_EVENT_TYPE.get(event_type, "orta"),
                source_event_id=event.event_id,
                related_event_ids=[],
            )
        ]

    def _describe_regulation(self, regulation_label: str) -> str:
        """Mevzuat etiketini, varsa retriever uzerinden GERCEK KB metniyle zenginlestirir.

        T017 dogrulamasi: bu, event_type -> mevzuat uygulanabilirligi (HANGI
        mevzuat) kararini VERMEZ - o zaten `EVENT_TYPE_REGULATION_MAP`
        uzerinden deterministik olarak alinmistir. Burada retriever yalnizca
        ZATEN BILINEN bu etiketin TAM METNINI getirir (bir dokuman-lookup
        adimidir).

        2026-08-25 duzeltmesi: `self._retriever.query(...)`,
        `EmbeddingRAGService.query()`nin KENDISIDIR - bu metod ZATEN yalnizca
        deterministik relevance esiginden (`RerankerConfig.score_threshold`)
        GECMIS ("accepted") adaylari dondurur; esigin ALTINDA kalan/alakasiz
        adaylar `query()` icinde SESSIZCE ELENIR (bos liste doner). Bu yuzden
        burada AYRICA bir metin-icerme kontrolu YAPILMAZ (eski surumdeki
        kontrol, ESKI placeholder corpus icin yazilmisti ve gercek KB'nin
        metniyle hicbir zaman eslesmiyordu - bkz. modul dokustringi); donen
        ilk sonuc dogrudan GUVENILIR sayilir. Retriever `None`, hata verir
        veya BOS liste donduruyse (deterministik gate onu zaten eledi) guvenli
        kisa etikete geri donulur - uydurma bir mevzuat metni ASLA rapora sizmaz.

        Args:
            regulation_label: `EVENT_TYPE_REGULATION_MAP`'ten gelen kisa etiket
                (orn. "ISG Yonetmeligi Madde 12"); ayni zamanda retriever'a
                gonderilen sorgu metnidir.

        Returns:
            Retriever mevcutsa VE en az bir (zaten deterministik gate'ten
            gecmis) sonuc donduysa `"<kisa etiket>: <gercek chunk metni>"`;
            retriever yoksa, cagri basarisiz olursa VEYA hicbir sonuc yoksa
            orijinal (guvenli) kisa etiket.
        """
        if self._retriever is None:
            return regulation_label

        try:
            results = self._retriever.query(regulation_label, top_k=1)
        except Exception:
            logger.exception(
                "RuleEngine: retriever cagrisi basarisiz, kisa mevzuat etiketiyle devam ediliyor."
            )
            return regulation_label

        if not results:
            return regulation_label

        text = getattr(results[0], "text", None)
        if not text:
            return regulation_label

        return f"{regulation_label}: {text}"

    def _match_combination_rules(self, sorted_events: List[TemporalEvent]) -> List[RuleMatch]:
        """`related_events` iliski kumelerini YAML'daki kombinasyon kurallarina karsi degerlendirir.

        Her `TemporalEvent` icin iliski kumesi, kendisi + `related_events`
        alaninin isaret ettigi diger olaylardan olusur. Bir kural, gerektirdigi
        tum tipler bu kumede bulunuyorsa tetiklenir. Ayni katilimci kumesi
        (`rule_id` + katilimci `event_id` kumesi) birden fazla ceviri
        noktasindan (farkli anchor'lardan) tekrar tetiklenirse, yalnizca ilk
        (zaman sirali en erken) eslesme tutulur.

        Args:
            sorted_events: `start_timestamp`'a gore artan sirali `TemporalEvent` listesi.

        Returns:
            Kombinasyon kurali eslesmelerinin listesi (tekrarsiz).
        """
        events_by_id = {event.event_id: event for event in sorted_events}
        seen_matches: set = set()
        matches: List[RuleMatch] = []

        for anchor in sorted_events:
            cluster = [anchor] + [
                events_by_id[related_id]
                for related_id in anchor.related_events
                if related_id in events_by_id
            ]

            for rule in self._combination_rules:
                required_types = {
                    event_type
                    for raw in rule.get("required_event_types", [])
                    if (event_type := _safe_event_type(raw)) is not None
                }
                if not required_types:
                    continue

                participants_by_id = {
                    member.event_id: member
                    for member in cluster
                    if _safe_event_type(member.event_type) in required_types
                }
                participant_types = {
                    _safe_event_type(member.event_type) for member in participants_by_id.values()
                }
                if not required_types.issubset(participant_types):
                    continue

                participant_ids = tuple(sorted(participants_by_id.keys()))
                dedup_key = (rule["rule_id"], participant_ids)
                if dedup_key in seen_matches:
                    continue
                seen_matches.add(dedup_key)

                source_event_id = anchor.event_id if anchor.event_id in participant_ids else participant_ids[0]
                related_event_ids = [pid for pid in participant_ids if pid != source_event_id]
                combined_event_type = "+".join(
                    sorted(t.value for t in required_types)
                )

                matches.append(
                    RuleMatch(
                        rule_id=rule["rule_id"],
                        rule_description=rule["description"].strip(),
                        event_type=combined_event_type,
                        severity=rule["severity"],
                        source_event_id=source_event_id,
                        related_event_ids=related_event_ids,
                    )
                )

        return matches


if __name__ == "__main__":
    # T010'un bagimsiz calistirilabilirlik testi:
    #   python -m src.event_analysis.rule_engine
    logging.basicConfig(level=logging.INFO)

    demo_events = [
        TemporalEvent(
            event_id="evt_0",
            event_name="kkd_ihlali",
            event_type="kkd_ihlali",
            description="Personel baretsiz calisiyor.",
            start_timestamp=0.0,
            end_timestamp=0.0,
            duration=0.0,
            confidence=0.6,
            occurrence_count=1,
            matched_keywords=["baretsiz"],
            source_model="demo-vlm",
            related_events=["evt_1"],
        ),
        TemporalEvent(
            event_id="evt_1",
            event_name="arac_yaya_yakinligi",
            event_type="arac_yaya_yakinligi",
            description="Forklift yaya gecidine yaklasiyor.",
            start_timestamp=8.0,
            end_timestamp=8.0,
            duration=0.0,
            confidence=0.6,
            occurrence_count=1,
            matched_keywords=["forklift"],
            source_model="demo-vlm",
            related_events=["evt_0"],
        ),
    ]

    for demo_match in RuleEngine().evaluate(demo_events):
        print(
            f"[{demo_match.rule_id}] ({demo_match.severity}) {demo_match.event_type} "
            f"source={demo_match.source_event_id} related={demo_match.related_event_ids}"
        )
