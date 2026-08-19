"""Modul 1 - VLM Oncesi Katman icin ortak Pydantic veri sozlesmeleri.

`AdaptiveFrameSampler` (`adaptive_sampler.py`) tarafindan uretilen `EvidenceFrame`
modeli burada tanimlanir; boylece sampler modulu, VLM/UI gibi tuketici
modullerle net bir arayuz sozlesmesi (schema) uzerinden konusur ve modul
kendi basina (`python -m src.sampler.adaptive_sampler`) test edilebilir.

ONEMLI (mimari): Sampler artik hicbir OLAY KUMELEMESI (event clustering)
YAPMAZ - yalnizca evidence esigini gecen kareleri uretir. Kumeleme (hangi
evidence karelerinin ayni gercek olaya ait oldugu) VLM katmaninda yapilir
(bkz. `src/vlm/base_vlm.py`); bu yuzden burada `EventCluster`/
`RepresentativeFrame` gibi kumeleme-ozel modeller ARTIK BULUNMAZ.
"""

from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, Field


class EvidenceFrame(BaseModel):
    """Esik-gecmis, VLM'e gonderilecek Kanit Karesi veri modeli.

    Video genelinde SIRALI, kayipsiz uretilir: hicbir global buffer/kare
    limiti, temporal voting/clustering/deduplication veya liste kesme
    nedeniyle evidence esigini gecen bir kare burada elenmez (bkz.
    `AdaptiveFrameSampler.process_video`).
    """

    evidence_id: str = Field(
        description="Bu evidence karesinin video genelinde benzersiz, kararli kimligi "
        "(`f'ev{frame_id}'`); VLM'in `evidence_ids` referanslarinda kullandigi kimliktir."
    )
    frame_id: int = Field(description="Karenin video icindeki sirasi (frame index).")
    timestamp_sec: float = Field(description="Karenin saniye cinsinden zaman damgasi.")
    timestamp_str: str = Field(description="`MM:SS` formatinda okunabilir zaman damgasi.")
    change_score: float = Field(description="Gurultu-tabani-dusulmus evidence/degisim skoru.")
    image_bytes: bytes = Field(description="JPEG-kodlu ham kare baytlari.", repr=False)
    base64_image: str = Field(description="`data:image/jpeg;base64,...` formatinda goruntu.")
    image_shape: Tuple[int, int, int] = Field(description="(yukseklik, genislik, kanal) kare boyutu.")
    saved_path: Optional[str] = Field(default=None, description="Karenin diskte kayitli oldugu yol.")
    is_fallback: bool = Field(default=False, description="Esik gecilemedigi icin frame 0 fallback'i mi.")
    motion_bbox: Optional[Tuple[int, int, int, int]] = Field(
        default=None,
        description=(
            "Bu karenin ureten hareket maskesinin (x_min, y_min, x_max, y_max) sinirlayici "
            "kutusu (bilgi amaclidir; artik kumeleme icin kullanilmaz - bkz. modul docstring'i). "
            "Hareket maskesi bossa (ör. fallback kare) `None`."
        ),
    )
