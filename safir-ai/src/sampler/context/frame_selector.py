"""Sampler icin TEK, ortak kare secim mekanizmasi.

Onceden iki bagimsiz mekanizma vardi (`RepresentativeFrameExtractor` VLM icin,
`PeakFrameExporter` disk arsivi icin); ikisi de kaynak videoyu YENIDEN acip
seek ile kare okuyor, aralarinda koordinasyon olmadigindan VLM'in gordugu
kareler ile diske yazilan/kullaniciya gosterilen kareler farklilasabiliyordu.

`FrameSelector`, bir Olay Grubunun (`EventCluster`) zaten `AdaptiveFrameSampler.
process_video` tarafindan bellekte JPEG/base64 olarak uretilmis Kanit
Karelerinden (`EvidenceFrame`) en fazla `TARGET_FRAME_COUNT` benzersiz,
kronolojik kare secer. Video dosyasina HICBIR sekilde erismez, hicbir kareyi
yeniden JPEG'e KODLAMAZ; yalnizca mevcut baytlari yeniden kullanir. Bu secimin
ciktisi hem VLM payload'ina hem (istege bagli) diske ayni kaynaktan aktarilir.
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


class FrameSelector:
    """Bir Olay Grubunun Kanit Karelerinden VLM+arsiv icin ortak temsili kare kumesini secer."""

    @staticmethod
    def select(peak_frame: EvidenceFrame, candidate_frames: List[EvidenceFrame]) -> List[RepresentativeFrame]:
        """En fazla `TARGET_FRAME_COUNT` benzersiz, kronolojik kare secer; zirve HER ZAMAN dahildir.

        `candidate_frames` havuzu `TARGET_FRAME_COUNT`den fazla benzersiz kare
        icerirse, kareler zaman ekseninde olabildigince esit araliklarla
        (`timestamp_sec` sirasina gore) secilir; zirve karenin kendi slotu
        korunur (en yakin sirali slotun YERINE gecer, ekstra kare eklemez).
        Havuzda `TARGET_FRAME_COUNT`den az benzersiz kare varsa (kisa olay,
        video sinirlari, tekrarli zaman damgalari vb.), kare COGALTILMADAN
        mevcut tum benzersiz kareler donulur.

        Args:
            peak_frame: Bu Olay Grubunun en yuksek degisim skoruna sahip zirve karesi.
            candidate_frames: Bu Olay Grubuna dahil edilen Kanit Kareleri (zaten
                `process_video` tarafindan bellekte JPEG/base64 encode edilmis).
                Zirve karesi bu listede olmasa bile otomatik eklenir.

        Returns:
            `timestamp_sec` artan sirada, benzersiz (`frame_id`) `RepresentativeFrame`
            listesi; zirve karesini HER ZAMAN icerir. Girdi tamamen bossa bile
            (teorik olarak imkansiz, `peak_frame` her zaman verilir) en az
            zirve karesini icerir.
        """
        pool = list(candidate_frames) if candidate_frames else []
        if not any(f.frame_id == peak_frame.frame_id for f in pool):
            pool.append(peak_frame)

        unique_by_id = {f.frame_id: f for f in pool}
        unique = sorted(unique_by_id.values(), key=lambda f: f.timestamp_sec)

        if len(unique) <= TARGET_FRAME_COUNT:
            selected = unique
        else:
            selected = FrameSelector._evenly_spaced_including_peak(unique, peak_frame.frame_id)

        representative_frames = [
            RepresentativeFrame(
                label="peak" if f.frame_id == peak_frame.frame_id else "context",
                frame_id=f.frame_id,
                timestamp_sec=f.timestamp_sec,
                timestamp_str=f.timestamp_str,
                base64_image=f.base64_image,
            )
            for f in selected
        ]

        if len(representative_frames) < TARGET_FRAME_COUNT:
            logger.debug(
                "FrameSelector: bu Olay Grubunda yalnizca %d benzersiz kare mevcut "
                "(hedef=%d); kare cogaltilmadan mevcut kareler kullanildi.",
                len(representative_frames),
                TARGET_FRAME_COUNT,
            )

        return representative_frames

    @staticmethod
    def _evenly_spaced_including_peak(
        unique: List[EvidenceFrame], peak_frame_id: int
    ) -> List[EvidenceFrame]:
        """`unique` (kronolojik, benzersiz) havuzdan `TARGET_FRAME_COUNT` kareyi esit araliklarla secer.

        Zirve karenin index'i, esit-aralikli secimin en yakin noktasinin
        YERINE gecer; boylece zirve ekstra kare eklemeden ve ust siniri
        asmadan HER ZAMAN secilenler arasinda kalir.
        """
        n = len(unique)
        step = (n - 1) / (TARGET_FRAME_COUNT - 1)
        indices = [round(i * step) for i in range(TARGET_FRAME_COUNT)]

        # Yuvarlama cakismalarini temizle (benzersizligi koru, sirayi bozma).
        seen: set[int] = set()
        dedup_indices: List[int] = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                dedup_indices.append(idx)

        peak_idx = next(i for i, f in enumerate(unique) if f.frame_id == peak_frame_id)
        if peak_idx not in seen:
            nearest_pos = min(range(len(dedup_indices)), key=lambda p: abs(dedup_indices[p] - peak_idx))
            dedup_indices[nearest_pos] = peak_idx
            dedup_indices = sorted(set(dedup_indices))

        return [unique[i] for i in dedup_indices]
