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

from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field

SelectionReason = Literal[
    "threshold_exceeded",
    "temporal_coverage",
    "early_change",
    "significant_change",
    "single_frame_change",
    "fallback",
]


class EvidenceFrame(BaseModel):
    """Esik-gecmis, VLM'e gonderilecek Kanit Karesi veri modeli.

    Video genelinde SIRALI, kayipsiz uretilir: hicbir global buffer/kare
    limiti, temporal voting/clustering/deduplication veya liste kesme
    nedeniyle evidence esigini gecen bir kare burada elenmez (bkz.
    `AdaptiveFrameSampler.process_video`).

    ONEMLI: `selection_reason` bir pre/peak/post KONUMSAL ROLU DEGILDIR -
    sadece bu karenin NEDEN evidence listesine girdigini (esik mi gecti,
    yoksa uzun bir sessiz araligi mi kapatti) aciklayan, olay
    kumelemesinden bagimsiz bir MEDATADIR. Olay kumelemesi hala tamamen
    VLM katmaninin sorumlulugundadir.
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
    selection_reason: SelectionReason = Field(
        default="threshold_exceeded",
        description=(
            "Bu karenin evidence listesine GIRME NEDENI (pozisyonel bir rol "
            "DEGILDIR, olay sinifi DEGILDIR - yalnizca sampler'in kendi secim "
            "gerekcesidir): 'threshold_exceeded' = kare ana esik testini "
            "gecti; 'temporal_coverage' = sakin bolgede son evidence "
            "karesinden bu yana `max_temporal_gap_sec` asildigi icin secildi; "
            "'early_change' = ana esigin ALTINDA ama surdurulen (tek karelik "
            "sicrama degil) bir onset sinyali tespit edildiginde, ana esik "
            "beklenmeden secildi - olayin baslangic karesini geriye donuk "
            "buffer gerektirmeden korur; 'significant_change' = ana esik "
            "gecildikten sonraki kisa bir hysteresis/cooldown penceresinde, "
            "skor tekrar esigin altina dustugunde bile hala yuksek yogunlukta "
            "secilen bir kare; 'single_frame_change' = ana esigin ALTINDA "
            "ama TEK bir karede zaten GUCLU, LOKAL ve gurultu tabanindan "
            "acikca ayrisan bir sinyal - 'early_change'in cok-kareli "
            "(early_change_min_count) onayi BEKLENMEDEN, ayri ve bagimsiz "
            "bir yoldan secilir; 'fallback' = videoda hicbir kare esigi "
            "gecemedi, sistem cokmesin diye ilk kare temsilen kullanildi "
            "(bkz. `is_fallback`)."
        ),
    )
    motion_bbox: Optional[Tuple[int, int, int, int]] = Field(
        default=None,
        description=(
            "Bu karenin ureten hareket maskesinin (x_min, y_min, x_max, y_max) sinirlayici "
            "kutusu (bilgi amaclidir; artik kumeleme icin kullanilmaz - bkz. modul docstring'i). "
            "Hareket maskesi bossa (ör. fallback kare) `None`."
        ),
    )
