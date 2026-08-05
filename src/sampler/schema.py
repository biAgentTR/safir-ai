from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, Field


class EvidenceFrame(BaseModel):
    """VLM'e gonderilecek Kanit Karesi veri modeli."""

    frame_id: int = Field(description="Karenin video icindeki sirasi.")
    timestamp_sec: float = Field(description="Karenin saniye cinsinden zaman damgasi.")
    timestamp_str: str = Field(description="`MM:SS` formatinda okunabilir zaman damgasi.")
    change_score: float = Field(description="Gurultu-tabani-dusulmus degisim skoru.")
    image_bytes: bytes = Field(description="JPEG-kodlu ham kare baytlari.", repr=False)
    base64_image: str = Field(description="`data:image/jpeg;base64,...` formatinda goruntu.")
    image_shape: Tuple[int, int, int] = Field(description="(yukseklik, genislik, kanal) kare boyutu.")
    saved_path: Optional[str] = Field(default=None, description="Karenin diskte kayitli oldugu yol.")
    is_fallback: bool = Field(default=False, description="Esik gecilemedigi icin frame 0 fallback'i mi.")
    motion_bbox: Optional[Tuple[int, int, int, int]] = Field(
        default=None,
        description=(
            "Bu karenin ureten hareket maskesinin (x_min, y_min, x_max, y_max) sinirlayici "
            "kutusu; `cluster_events`in ayni fiziksel olayin devami mi yoksa farkli bir olay mi "
            "oldugunu ayirt etmek icin kullandigi konumsal sinyal. Hareket maskesi bossa "
            "(ör. fallback kare) `None`."
        ),
    )


class RepresentativeFrame(BaseModel):
    """Bir Olay Grubu icin VLM'e gonderilen tekil temsili kare (pre/peak/post)."""

    label: str = Field(description="Karenin olay icindeki rolu: 'pre-event', 'peak' veya 'post-event'.")
    timestamp_sec: float = Field(description="Karenin saniye cinsinden zaman damgasi.")
    timestamp_str: str = Field(description="`MM:SS` formatinda okunabilir zaman damgasi.")
    base64_image: str = Field(description="`data:image/jpeg;base64,...` formatinda goruntu.")


class EventCluster(BaseModel):
    """Arka arkaya gerceklesen degisim karelerinin kumelenmis olay grubu."""

    event_id: int = Field(description="Bu Olay Grubunun kimligi.")
    start_time: float = Field(description="Grubun baslangic zaman damgasi (sn).")
    end_time: float = Field(description="Grubun bitis zaman damgasi (sn).")
    peak_frame: EvidenceFrame = Field(description="Grubun en yuksek degisim skoruna sahip zirve karesi.")
    total_candidate_frames: int = Field(description="Bu gruba dahil edilen Kanit Karesi sayisi.")
    duration_sec: float = Field(
        default=0.0, description="`end_time - start_time` (saniye); birlestirilmis olayin toplam suresi."
    )
    representative_frames: list["RepresentativeFrame"] = Field(
        default_factory=list,
        description=(
            "Zaman sirali pre-event/peak/post-event temsili kareler (0-3 adet arasi); "
            "VLM'e context.mp4 yerine dogrudan gonderilir. Bos ise VLMPayloadBuilder yalnizca "
            "peak_frame ile eski (tek-kare) davranisa duser."
        ),
    )