"""Sampler icin TEK, ortak kare secim mekanizmasi.

Onceden iki bagimsiz mekanizma vardi (`RepresentativeFrameExtractor` VLM icin,
`PeakFrameExporter` disk arsivi icin); ikisi de kaynak videoyu YENIDEN acip
seek ile kare okuyor, aralarinda koordinasyon olmadigindan VLM'in gordugu
kareler ile diske yazilan/kullaniciya gosterilen kareler farklilasabiliyordu.

`FrameSelector`, bir Olay Grubunun (`EventCluster`) zaten `AdaptiveFrameSampler.
process_video` tarafindan bellekte JPEG/base64 olarak uretilmis, evidence
esigini GECMIS Kanit Karelerinden (`EvidenceFrame`) en fazla `TARGET_FRAME_COUNT`
(varsayilan 5) benzersiz evidence karesi secer. Video dosyasina HICBIR
sekilde erismez, hicbir kareyi yeniden JPEG'e KODLAMAZ; yalnizca mevcut
baytlari yeniden kullanir. Bu secimin ciktisi hem VLM payload'ina hem
(istege bagli) diske ayni kaynaktan aktarilir.

ONEMLI (konumsal ROL YOK): Cikti modelinde ('RepresentativeFrame') hicbir
kare kalici bir 'pre'/'peak'/'post' etiketiyle ISARETLENMEZ. En yuksek
evidence skoruna sahip kare secime dahil edilir (`selection_reason=
"highest_evidence_score"`), ama listede baska bir kareden farkli bir alan/
tip tasimaz - hepsi ayni `RepresentativeFrame` modelinde, yalnizca
`selection_reason` ve `timestamp_sec` ile ayirt edilen birer evidence
karesidir. Kalan kareler, olayin farkli zaman kesitlerini temsil edecek
sekilde ZAMAN EKSENINDE dengeli dagitilir (`selection_reason=
"temporal_coverage"`); bu dagitimin dahili uygulamasi (en yuksek skorlu
karenin ONCESINDEN/SONRASINDAN butce ayirma) yalnizca bir secim
STRATEJISIDIR, cikti modelinde herhangi bir konumsal alan/etiket olarak
YANSITILMAZ.

ONEMLI: Girdi havuzu (`candidate_frames`) YALNIZCA cagiranin ait cluster'a
dahil ettigi, evidence esigini gecmis kareler olmalidir (bkz.
`AdaptiveFrameSampler.cluster_events`/`_close_group`, ki zaten yalnizca
esigi gecmis `EvidenceFrame`lerden olusan gruplari besler). `FrameSelector`
kendi basina bir evidence esigi uygulamaz; cagiranin sozlesmesine guvenir.
"""

from __future__ import annotations

import logging
from typing import List

from src.sampler.schema import EvidenceFrame, RepresentativeFrame

logger = logging.getLogger(__name__)

# Her Olay Grubu icin hedeflenen evidence karesi sayisi (en yuksek skorlu
# kare dahil). Aday havuzunda bu sayidan az benzersiz kare varsa, kare
# COGALTILMAZ; mevcut tum benzersiz kareler donulur (bkz. `RepresentativeFrame`
# schema docstring'i).
TARGET_FRAME_COUNT = 5

# En yuksek evidence skoruna sahip karenin ONCESINDEN/SONRASINDAN secilecek
# azami dengeleme (zamansal dagitim) kare sayisi. Bu, YALNIZCA dahili bir
# secim butcesidir; cikti modelinde konumsal bir alan/etiket olarak
# YANSITILMAZ (bkz. modul docstring'i).
_BUDGET_BEFORE_HIGHEST_SCORE = 2
_BUDGET_AFTER_HIGHEST_SCORE = 2

_REASON_HIGHEST_SCORE = "highest_evidence_score"
_REASON_TEMPORAL_COVERAGE = "temporal_coverage"


class FrameSelector:
    """Bir Olay Grubunun evidence esigini gecmis Kanit Karelerinden VLM+arsiv icin ortak kare kumesini secer."""

    @staticmethod
    def select(
        peak_frame: EvidenceFrame, candidate_frames: List[EvidenceFrame], event_id: int
    ) -> List[RepresentativeFrame]:
        """En yuksek skorlu kare + zamansal dengeleme ile en fazla `TARGET_FRAME_COUNT` (5) evidence karesi secer.

        Adimlar:
        1. Aday havuzu `timestamp_sec`e gore kronolojik siralanir, `frame_id`
           bazinda benzersizlestirilir (ayni kare iki kez sayilmaz).
        2. En yuksek evidence skoruna sahip kare HER ZAMAN secilir
           (`selection_reason="highest_evidence_score"`); ancak cikti
           modelinde kalici bir konumsal etiket TASIMAZ.
        3. Kalan kareler, bu karenin ZAMAN ONCESI ve ZAMAN SONRASI olarak iki
           dahili havuza ayrilir (yalnizca secim stratejisi icin; cikti
           modeline yansimaz).
        4. Her havuzdan en fazla `_BUDGET_BEFORE_HIGHEST_SCORE`/
           `_BUDGET_AFTER_HIGHEST_SCORE` (2'ser) kare, zaman ekseninde esit
           araliklarla (havuz ici en erken ve en gec kareyi de kapsayacak
           sekilde) secilir; boylece skor yogunlasmasindan bagimsiz olarak
           olayin farkli zaman kesitleri temsil edilir
           (`selection_reason="temporal_coverage"`).
        5. Bir taraf (ör. en yuksek skorlu kare olayin en basinda/sonunda
           ise) bos veya yetersizse, o tarafin kullanilmayan butcesi DIGER
           tarafa aktarilir.
        6. Sonuc kronolojik siraya sokulur (`timestamp_sec` artan).

        Aday havuzunda toplam `TARGET_FRAME_COUNT`den az benzersiz kare varsa,
        kare COGALTILMAZ; yalnizca mevcut benzersiz kareler donulur.

        Args:
            peak_frame: Bu Olay Grubunun en yuksek evidence/degisim skoruna
                sahip karesi (`candidate_frames` icinde olmasa bile otomatik
                dahil edilir).
            candidate_frames: Bu Olay Grubuna (cluster'a) dahil edilen,
                evidence esigini ZATEN gecmis Kanit Kareleri (zaten
                `process_video` tarafindan bellekte JPEG/base64 encode
                edilmis). Esigi gecmemis hicbir kare bu listede OLMAMALIDIR
                (cagiranin sorumlulugu; bkz. modul docstring'i).
            event_id: Bu kareleri secen `EventCluster.event_id`; her
                `RepresentativeFrame.event_id` alanina damgalanir.

        Returns:
            `timestamp_sec` artan sirada, benzersiz (`frame_index`)
            `RepresentativeFrame` listesi; en yuksek skorlu kareyi HER ZAMAN
            icerir, ama hicbirini konumsal olarak ETIKETLEMEZ.
        """
        pool = [f for f in candidate_frames if f.base64_image] if candidate_frames else []
        if not any(f.frame_id == peak_frame.frame_id for f in pool):
            pool.append(peak_frame)

        unique_by_id = {f.frame_id: f for f in pool}
        unique = sorted(unique_by_id.values(), key=lambda f: f.timestamp_sec)

        before_pool = [f for f in unique if f.timestamp_sec < peak_frame.timestamp_sec]
        after_pool = [f for f in unique if f.timestamp_sec > peak_frame.timestamp_sec]

        # Once her tarafa kendi butcesi kadar (en fazla) pay ayrilir; sonra
        # kullanilmayan (havuz kucuklugu nedeniyle harcanamayan) butce,
        # DIGER tarafin kapasitesi elveriyorsa ona aktarilir. Boylece en
        # yuksek skorlu kare olayin basinda/sonunda olsa bile toplam 4
        # dengeleme karesi butcesi (mumkun oldugunca) kullanilir, ekstra kare
        # eklenmez/cogaltilmaz.
        before_take = min(_BUDGET_BEFORE_HIGHEST_SCORE, len(before_pool))
        after_take = min(_BUDGET_AFTER_HIGHEST_SCORE, len(after_pool))
        leftover = (_BUDGET_BEFORE_HIGHEST_SCORE + _BUDGET_AFTER_HIGHEST_SCORE) - before_take - after_take
        if leftover > 0:
            extra_before_capacity = len(before_pool) - before_take
            give_before = min(leftover, extra_before_capacity)
            before_take += give_before
            leftover -= give_before
        if leftover > 0:
            extra_after_capacity = len(after_pool) - after_take
            give_after = min(leftover, extra_after_capacity)
            after_take += give_after
            leftover -= give_after

        before_selected = FrameSelector._evenly_spaced_subset(before_pool, before_take)
        after_selected = FrameSelector._evenly_spaced_subset(after_pool, after_take)

        representative_frames: List[RepresentativeFrame] = [
            FrameSelector._to_representative(peak_frame, event_id, reason=_REASON_HIGHEST_SCORE)
        ]
        for f in before_selected:
            representative_frames.append(
                FrameSelector._to_representative(f, event_id, reason=_REASON_TEMPORAL_COVERAGE)
            )
        for f in after_selected:
            representative_frames.append(
                FrameSelector._to_representative(f, event_id, reason=_REASON_TEMPORAL_COVERAGE)
            )

        representative_frames.sort(key=lambda rf: rf.timestamp_sec)

        if len(representative_frames) < TARGET_FRAME_COUNT:
            logger.debug(
                "FrameSelector: Olay #%d icin yalnizca %d benzersiz evidence karesi mevcut "
                "(hedef=%d); kare cogaltilmadan mevcut kareler kullanildi.",
                event_id,
                len(representative_frames),
                TARGET_FRAME_COUNT,
            )

        return representative_frames

    @staticmethod
    def _to_representative(frame: EvidenceFrame, event_id: int, reason: str) -> RepresentativeFrame:
        return RepresentativeFrame(
            frame_index=frame.frame_id,
            event_id=event_id,
            timestamp_sec=frame.timestamp_sec,
            timestamp_str=frame.timestamp_str,
            evidence_score=frame.change_score,
            selection_reason=reason,
            base64_image=frame.base64_image,
        )

    @staticmethod
    def _evenly_spaced_subset(pool: List[EvidenceFrame], count: int) -> List[EvidenceFrame]:
        """Kronolojik `pool`dan `count` kadar kareyi zaman ekseninde esit araliklarla secer.

        `count == 0` -> bos liste. `count >= len(pool)` -> havuzun tamami
        (cogaltma yok). Aksi halde, havuzun ilk ve son karesini de kapsayacak
        sekilde esit araliklarla `count` indeks secilir (yuvarlama
        cakismalari benzersizlestirilir; bu durumda donen liste `count`den
        az olabilir — yine de cogaltma YAPILMAZ).
        """
        if count <= 0 or not pool:
            return []
        if count >= len(pool):
            return list(pool)
        if count == 1:
            # Tek kare istenirse havuzun ortasindaki (zamansal olarak en
            # temsili) kareyi sec.
            return [pool[len(pool) // 2]]

        step = (len(pool) - 1) / (count - 1)
        indices = sorted({round(i * step) for i in range(count)})
        return [pool[i] for i in indices]
