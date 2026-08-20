"""T009 - Temporal Reasoning: `DetectedEvent`ler arasinda kronolojik bag kurma.

Bu modul, `Event Engine`in (T008) birden fazla VLM cikisi/zaman penceresi
uzerinden urettigi `List[DetectedEvent]`'i alip iki farkli zamansal iliskiyi
kurar:

1. Gruplama/birlestirme ("suregelen olay"): ayni `event_type`e sahip
   `DetectedEvent`ler, aralarindaki zaman farki `merge_window_sec` esigini
   asmadigi surece tek bir `TemporalEvent`e indirgenir. Ornegin bir dusme
   riski 3 VLM cikisinda da (5'er saniye arayla) tespit edildiyse, bunlar 3
   ayri olay degil, `occurrence_count=3`, `duration=10.0` olan TEK bir
   suregelen `TemporalEvent` olarak modellenir. Gerekce: ayni riskin her
   VLM karesinde yeniden "yeni bir olay" sayilmasi, asagi akıştaki `05
   LangGraph Ajani`nin ve `Event Gecmisi`nin (T012) ayni durumu tekrar
   tekrar islemesine/raporlamasina yol acardi.

   Birlestirme karari YALNIZCA "ayni tip + zaman farki `merge_window_sec`
   icinde" kriterine gore verilir; iki ayni-tip olay arasina baska tipte
   bir olay girmesi birlestirmeyi ENGELLEMEZ (T015 - bkz.
   `_group_by_type_and_proximity`). Ornegin `kkd_ihlali(t=0)` ->
   `arac_yaya_yakinligi(t=3)` -> `kkd_ihlali(t=5)` sirasinda, iki
   `kkd_ihlali` (araya farkli tip girmis olsa da) birlesir; `arac_yaya_yakinligi`
   ayri, tek olaylik bir grup olarak kalir. (T015 oncesi: yalnizca listede
   dogrudan BITISIK ayni-tip olaylar birlesiyordu; bu, T013'un gercek
   pipeline entegrasyonunda tespit edilen bir kisitlamaydi.)
2. Iliskilendirme: birlestirme sonrasi elde edilen `TemporalEvent`ler
   arasinda, zaman ekseninde birbirine yakin duranlar (`relation_window_sec`
   icinde, tip fark etmeksizin) `related_events` alaninda birbirine
   baglanir. Ornegin bir `arac_yaya_yakinligi` olayindan 8 saniye sonra
   tespit edilen bir `kkd_ihlali`, birbirinden bagimsiz iki olay olsa da,
   Rule Engine'in (T010) "ayni saha aninda birden fazla ihlal" gibi bilesik
   bir degerlendirme yapabilmesi icin birbirine referans verir.

Esikler (`merge_window_sec`, `relation_window_sec`) su an `configs/config.yaml`
uzerinden degil, `TemporalReasoner` constructor parametreleri olarak
tutulur (varsayilan degerler modul sabitleridir); `configs/` dosyasina
dokunulmadan, `SafirConfig`'e sonradan tasinabilecek sekilde izole edilmistir.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from src.event_analysis.schemas import DetectedEvent, TemporalEvent

logger = logging.getLogger(__name__)

DEFAULT_MERGE_WINDOW_SEC = 10.0
"""Ayni `event_type`e sahip ardisik `DetectedEvent`lerin tek bir suregelen
`TemporalEvent`e birlesmesi icin izin verilen maksimum zaman farki (saniye)."""

DEFAULT_RELATION_WINDOW_SEC = 30.0
"""Farkli (veya birlesmemis ayni) `TemporalEvent`lerin birbirine
`related_events` olarak baglanmasi icin izin verilen maksimum zaman
bosluğu (saniye)."""

_CONFIDENCE_BOOST_PER_EXTRA_OCCURRENCE = 0.05


class TemporalReasoner:
    """`DetectedEvent` listesini zaman ekseninde gruplayip iliskilendiren katman.

    Bkz. modul dokustringi icin gruplama/iliskilendirme mantiginin tam
    aciklamasi ve gerekcesi.
    """

    def __init__(
        self,
        merge_window_sec: float = DEFAULT_MERGE_WINDOW_SEC,
        relation_window_sec: float = DEFAULT_RELATION_WINDOW_SEC,
    ) -> None:
        """TemporalReasoner'i gruplama ve iliskilendirme esikleriyle baslatir.

        Args:
            merge_window_sec: Ayni tipteki ardisik olaylarin birlesmesi icin
                izin verilen maksimum zaman farki (saniye).
            relation_window_sec: Birlesmis olaylarin birbirine `related_events`
                olarak baglanmasi icin izin verilen maksimum zaman bosluğu
                (saniye). `merge_window_sec`'ten kucuk olmamalidir.
        """
        self._merge_window_sec = merge_window_sec
        self._relation_window_sec = relation_window_sec

    def reason(self, detected_events: List[DetectedEvent]) -> List[TemporalEvent]:
        """`DetectedEvent` listesini gruplayip iliskilendirerek `TemporalEvent` listesine cevirir.

        Args:
            detected_events: `EventEngine.detect(...)` cagrilarindan biriken,
                zaman damgasina gore sirali olmasi beklenen (savunmaci olarak
                yeniden siralanir) `DetectedEvent` listesi.

        Returns:
            Baslangic zaman damgasina gore sirali `TemporalEvent` listesi.
            Girdi bossa bos liste doner.
        """
        if not detected_events:
            return []

        sorted_events = sorted(detected_events, key=lambda event: event.timestamp)
        groups = self._group_by_type_and_proximity(sorted_events)

        temporal_events = [
            self._build_temporal_event(group, index) for index, group in enumerate(groups)
        ]
        self._link_related_events(temporal_events)

        logger.debug(
            "TemporalReasoner: %d DetectedEvent -> %d TemporalEvent",
            len(detected_events),
            len(temporal_events),
        )
        return temporal_events

    def _group_by_type_and_proximity(
        self, sorted_events: List[DetectedEvent]
    ) -> List[List[DetectedEvent]]:
        """Zaman sirali olaylari, `event_name` bazinda "en son acik grup" mantigiyla gruplar.

        T020: gruplama artik `event_type`e (opsiyonel/canonical) DEGIL,
        olayin BIRINCIL kimligi olan `event_name`e (VLM-uretimi, serbest
        bicimli) gore yapilir - boylece canonical taksonomiye hic
        oturmayan (`event_type=None`) yeni/serbest olaylar da kendi
        `event_name`leri altinda dogru sekilde kumelenir.

        Her `event_name` icin, o isimdeki EN SON grubun indeksi ayri ayri
        izlenir (`last_group_index_by_name`). Yeni bir olay geldiginde:
        - Ayni isimdeki en son grubun son olayina zaman farki
          `merge_window_sec` icindeyse, o gruba eklenir (araya baska isimde
          kac olay girmis olursa olsun - listede bitisik olma sarti YOKTUR).
        - Degilse (ya bu isim ilk kez goruluyor ya da esik asilmis), bu isim
          icin YENI bir grup acilir; o ismin "en son grup" isaretcisi bu
          yeni gruba guncellenir (eski grup bir daha asla yeniden acilmaz -
          zaten zaman ilerledikce ona olan uzaklik sadece artabilir).

        Args:
            sorted_events: Zaman damgasina gore artan sirali `DetectedEvent` listesi.

        Returns:
            Her biri tek bir suregelen olaya karsilik gelecek, olusturulma
            sirasina gore (bu da baslangic zaman damgasina gore artan
            siraya esittir) dizili `DetectedEvent` gruplarinin listesi.
        """
        groups: List[List[DetectedEvent]] = []
        last_group_index_by_name: Dict[str, int] = {}

        for event in sorted_events:
            last_index = last_group_index_by_name.get(event.event_name)
            if last_index is not None:
                last_event = groups[last_index][-1]
                if (event.timestamp - last_event.timestamp) <= self._merge_window_sec:
                    groups[last_index].append(event)
                    continue

            groups.append([event])
            last_group_index_by_name[event.event_name] = len(groups) - 1

        return groups

    def _build_temporal_event(self, group: List[DetectedEvent], index: int) -> TemporalEvent:
        """Ayni `event_name`e sahip, birlesmis bir `DetectedEvent` grubundan tek bir `TemporalEvent` uretir.

        Args:
            group: `_group_by_type_and_proximity` tarafindan uretilmis, zaman
                sirali, ayni `event_name`e sahip `DetectedEvent` grubu.
            index: Bu grubun `event_id` uretiminde kullanilacak sira numarasi.

        Returns:
            Grubu temsil eden tek bir `TemporalEvent`.
        """
        start_timestamp = group[0].timestamp
        end_timestamp = group[-1].timestamp
        latest = group[-1]

        max_confidence = max(event.confidence for event in group)
        occurrence_count = len(group)
        boosted_confidence = round(
            min(1.0, max_confidence + _CONFIDENCE_BOOST_PER_EXTRA_OCCURRENCE * (occurrence_count - 1)),
            2,
        )

        risk_hints = [event.risk_hint for event in group if event.risk_hint is not None]
        # Canonical `event_type` (opsiyonel): gruptaki EN SON bilinen (None
        # olmayan) deger tercih edilir - VLM tutarsiz davranip bir gozlemde
        # canonical baglanti verip digerinde vermeyebilir; bilgi varsa
        # KAYBEDILMEZ. Hicbir gozlemde canonical yoksa `None` kalir
        # ("eslestirilemedi", zorlanmaz).
        canonical_event_type = next(
            (event.event_type for event in reversed(group) if event.event_type is not None), None
        )

        return TemporalEvent(
            event_id=f"evt_{index}",
            event_name=latest.event_name,
            event_type=canonical_event_type,
            description=latest.description,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            duration=round(end_timestamp - start_timestamp, 2),
            confidence=boosted_confidence,
            occurrence_count=occurrence_count,
            matched_keywords=self._union_keywords(group),
            source_model=latest.source_model,
            related_events=[],
            evidence_ids=self._union_evidence_ids(group),
            risk_hint=max(risk_hints) if risk_hints else None,
        )

    @staticmethod
    def _union_keywords(group: List[DetectedEvent]) -> List[str]:
        """Gruptaki tum tespitlerin anahtar kelimelerini, ilk gorulme sirasini koruyarak birlestirir.

        Args:
            group: Anahtar kelimeleri birlestirilecek `DetectedEvent` grubu.

        Returns:
            Tekrarsiz, ilk gorulme sirali anahtar kelime listesi.
        """
        seen: List[str] = []
        for event in group:
            for keyword in event.matched_keywords:
                if keyword not in seen:
                    seen.append(keyword)
        return seen

    @staticmethod
    def _union_evidence_ids(group: List[DetectedEvent]) -> List[str]:
        """Gruptaki tum tespitlerin `evidence_ids`lerini, ilk gorulme sirasini koruyarak birlestirir.

        Args:
            group: Evidence kimlikleri birlestirilecek `DetectedEvent` grubu.

        Returns:
            Tekrarsiz, ilk gorulme sirali evidence_id listesi.
        """
        seen: List[str] = []
        for event in group:
            for evidence_id in event.evidence_ids:
                if evidence_id not in seen:
                    seen.append(evidence_id)
        return seen

    def _link_related_events(self, temporal_events: List[TemporalEvent]) -> None:
        """Zaman ekseninde `relation_window_sec` icinde duran `TemporalEvent`leri birbirine baglar.

        Iliskilendirme simetriktir ve tipten bagimsizdir: iki olayin zaman
        araliklari (`start_timestamp`-`end_timestamp`) `relation_window_sec`
        icinde ortusuyor veya birbirine yakinsa, ikisi de digerinin
        `related_events` listesine eklenir. `temporal_events` yerinde
        (in-place) guncellenir.

        Args:
            temporal_events: `_build_temporal_event` ile uretilmis, henuz
                iliskilendirilmemis `TemporalEvent` listesi.
        """
        for i, current in enumerate(temporal_events):
            related_ids = [
                other.event_id
                for j, other in enumerate(temporal_events)
                if i != j and self._time_gap(current, other) <= self._relation_window_sec
            ]
            current.related_events = related_ids

    @staticmethod
    def _time_gap(a: TemporalEvent, b: TemporalEvent) -> float:
        """Iki `TemporalEvent`in zaman araliklari arasindaki bosluğu hesaplar (ortusuyorsa 0.0).

        Args:
            a: Birinci olay.
            b: Ikinci olay.

        Returns:
            Saniye cinsinden bosluk; araliklar ortusuyor veya iceriyorsa 0.0.
        """
        if a.end_timestamp < b.start_timestamp:
            return b.start_timestamp - a.end_timestamp
        if b.end_timestamp < a.start_timestamp:
            return a.start_timestamp - b.end_timestamp
        return 0.0


if __name__ == "__main__":
    # T009'un bagimsiz calistirilabilirlik testi:
    #   python -m src.event_analysis.temporal_reasoner
    logging.basicConfig(level=logging.INFO)

    demo_events = [
        DetectedEvent(
            event_name="dusme_riski",
            event_type="dusme_riski",
            description="Personel korkuluksuz iskelede calisiyor.",
            timestamp=0.0,
            confidence=0.5,
            matched_keywords=["iskele"],
            source_model="demo-vlm",
        ),
        DetectedEvent(
            event_name="dusme_riski",
            event_type="dusme_riski",
            description="Personel hala korkuluksuz iskelede.",
            timestamp=5.0,
            confidence=0.5,
            matched_keywords=["iskele"],
            source_model="demo-vlm",
        ),
        DetectedEvent(
            event_name="kkd_ihlali",
            event_type="kkd_ihlali",
            description="Ayni bolgede baretsiz bir personel daha goruldu.",
            timestamp=12.0,
            confidence=0.5,
            matched_keywords=["baretsiz"],
            source_model="demo-vlm",
        ),
    ]

    for demo_temporal_event in TemporalReasoner().reason(demo_events):
        print(
            f"[{demo_temporal_event.event_id}] {demo_temporal_event.event_name} "
            f"(canonical={demo_temporal_event.event_type}) "
            f"occurrence_count={demo_temporal_event.occurrence_count} "
            f"duration={demo_temporal_event.duration} "
            f"related={demo_temporal_event.related_events}"
        )
