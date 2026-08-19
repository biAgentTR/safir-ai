"""Sampler icin TEK, ortak kare secim mekanizmasi.

Onceden iki bagimsiz mekanizma vardi (`RepresentativeFrameExtractor` VLM icin,
`PeakFrameExporter` disk arsivi icin); ikisi de kaynak videoyu YENIDEN acip
seek ile kare okuyor, aralarinda koordinasyon olmadigindan VLM'in gordugu
kareler ile diske yazilan/kullaniciya gosterilen kareler farklilasabiliyordu.

`FrameSelector`, bir Olay Grubunun (`EventCluster`) zaten `AdaptiveFrameSampler.
process_video` tarafindan bellekte JPEG/base64 olarak uretilmis, evidence
esigini GECMIS Kanit Karelerinden (`EvidenceFrame`) en fazla `TARGET_FRAME_COUNT`
(varsayilan 5) benzersiz kare secer: 1 zirve (en yuksek evidence/degisim
skoru) + zirve oncesinden en fazla `PRE_CONTEXT_BUDGET` (2) + zirve
sonrasindan en fazla `POST_CONTEXT_BUDGET` (2) baglam karesi. Video dosyasina
HICBIR sekilde erismez, hicbir kareyi yeniden JPEG'e KODLAMAZ; yalnizca
mevcut baytlari yeniden kullanir. Bu secimin ciktisi hem VLM payload'ina hem
(istege bagli) diske ayni kaynaktan aktarilir.

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

# Her Olay Grubu icin hedeflenen temsili kare sayisi (zirve dahil). Aday havuzunda
# bu sayidan az benzersiz kare varsa, kare COGALTILMAZ; mevcut tum benzersiz
# kareler donulur (bkz. `RepresentativeFrame` schema docstring'i).
TARGET_FRAME_COUNT = 5

# Zirve oncesinden/sonrasindan secilecek azami baglam (context) kare sayisi.
# PRE_CONTEXT_BUDGET + 1 (zirve) + POST_CONTEXT_BUDGET == TARGET_FRAME_COUNT.
PRE_CONTEXT_BUDGET = 2
POST_CONTEXT_BUDGET = 2


class FrameSelector:
    """Bir Olay Grubunun evidence esigini gecmis Kanit Karelerinden VLM+arsiv icin ortak kare kumesini secer."""

    @staticmethod
    def select(
        peak_frame: EvidenceFrame, candidate_frames: List[EvidenceFrame], event_id: int
    ) -> List[RepresentativeFrame]:
        """Zirve + zirve oncesi/sonrasi baglamdan en fazla `TARGET_FRAME_COUNT` (5) kare secer.

        Adimlar:
        1. Aday havuzu `timestamp_sec`e gore kronolojik siralanir, `frame_id`
           bazinda benzersizlestirilir (ayni kare iki kez sayilmaz).
        2. Zirve kare (en yuksek `change_score`) HER ZAMAN secilir.
        3. Kalan kareler zirve ONCESI ve zirve SONRASI iki havuza ayrilir.
        4. Her havuzdan en fazla `PRE_CONTEXT_BUDGET`/`POST_CONTEXT_BUDGET`
           (2'ser) kare, zaman ekseninde esit araliklarla (havuz ici en erken
           ve en gec kareyi de kapsayacak sekilde) secilir; boylece skor
           yogunlasmasindan bagimsiz olarak olayin farkli zaman kesitleri
           temsil edilir.
        5. Bir taraf (ör. zirve olayin en basinda/sonunda ise) bos veya
           yetersizse, o tarafin kullanilmayan butcesi DIGER tarafa aktarilir
           (ör. pre-pool bossa post-pool'dan 4 kareye kadar secilebilir).
        6. Sonuc tekrar kronolojik siraya sokulur.

        Aday havuzunda toplam `TARGET_FRAME_COUNT`den az benzersiz kare varsa,
        kare COGALTILMAZ; yalnizca mevcut benzersiz kareler donulur.

        Args:
            peak_frame: Bu Olay Grubunun en yuksek evidence/degisim skoruna
                sahip zirve karesi (`candidate_frames` icinde olmasa bile
                otomatik dahil edilir).
            candidate_frames: Bu Olay Grubuna (cluster'a) dahil edilen,
                evidence esigini ZATEN gecmis Kanit Kareleri (zaten
                `process_video` tarafindan bellekte JPEG/base64 encode
                edilmis). Esigi gecmemis hicbir kare bu listede OLMAMALIDIR
                (cagiranin sorumlulugu; bkz. modul docstring'i).
            event_id: Bu kareleri secen `EventCluster.event_id`; her
                `RepresentativeFrame.event_id` alanina damgalanir.

        Returns:
            `timestamp_sec` artan sirada, benzersiz (`frame_id`)
            `RepresentativeFrame` listesi; zirve karesini HER ZAMAN icerir.
        """
        pool = [f for f in candidate_frames if f.base64_image] if candidate_frames else []
        if not any(f.frame_id == peak_frame.frame_id for f in pool):
            pool.append(peak_frame)

        unique_by_id = {f.frame_id: f for f in pool}
        unique = sorted(unique_by_id.values(), key=lambda f: f.timestamp_sec)

        pre_pool = [f for f in unique if f.timestamp_sec < peak_frame.timestamp_sec]
        post_pool = [f for f in unique if f.timestamp_sec > peak_frame.timestamp_sec]

        # Once her tarafa kendi butcesi kadar (en fazla) pay ayrilir; sonra
        # kullanilmayan (havuz kucuklugu nedeniyle harcanamayan) butce,
        # DIGER tarafin kapasitesi elveriyorsa ona aktarilir. Boylece zirve
        # olayin basinda/sonunda olsa bile toplam 4 baglam karesi butcesi
        # (mumkun oldugunca) kullanilir, ekstra kare eklenmez/cogaltilmaz.
        pre_take = min(PRE_CONTEXT_BUDGET, len(pre_pool))
        post_take = min(POST_CONTEXT_BUDGET, len(post_pool))
        leftover = (PRE_CONTEXT_BUDGET + POST_CONTEXT_BUDGET) - pre_take - post_take
        if leftover > 0:
            extra_pre_capacity = len(pre_pool) - pre_take
            give_pre = min(leftover, extra_pre_capacity)
            pre_take += give_pre
            leftover -= give_pre
        if leftover > 0:
            extra_post_capacity = len(post_pool) - post_take
            give_post = min(leftover, extra_post_capacity)
            post_take += give_post
            leftover -= give_post

        pre_selected = FrameSelector._evenly_spaced_subset(pre_pool, pre_take)
        post_selected = FrameSelector._evenly_spaced_subset(post_pool, post_take)

        representative_frames: List[RepresentativeFrame] = []
        representative_frames.append(
            FrameSelector._to_representative(
                peak_frame, event_id, label="peak", reason="en yuksek evidence skoru (zirve)"
            )
        )
        for f in pre_selected:
            representative_frames.append(
                FrameSelector._to_representative(
                    f, event_id, label="pre_context", reason="zirve oncesi baglam (zaman ekseninde esit dagitilmis)"
                )
            )
        for f in post_selected:
            representative_frames.append(
                FrameSelector._to_representative(
                    f, event_id, label="post_context", reason="zirve sonrasi baglam (zaman ekseninde esit dagitilmis)"
                )
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
    def _to_representative(
        frame: EvidenceFrame, event_id: int, label: str, reason: str
    ) -> RepresentativeFrame:
        return RepresentativeFrame(
            label=label,
            frame_id=frame.frame_id,
            event_id=event_id,
            timestamp_sec=frame.timestamp_sec,
            timestamp_str=frame.timestamp_str,
            change_score=frame.change_score,
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
