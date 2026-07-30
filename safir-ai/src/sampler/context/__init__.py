"""Kayitli video (VOD) icin pre/peak/post temsili kare cikarma katmani.

`AdaptiveFrameSampler`'in evidence/clustering/peak-secim mantigindan
tamamen ayri, opsiyonel bir katmandir; yalnizca zirve kare etrafindaki
kisa zaman penceresinden pre-event/post-event JPEG kareleri seek ile
cikarir (diske `.mp4` yazmaz).

- `RepresentativeFrameExtractor`: VLM'e gonderilecek base64 JPEG'leri
  bellekte uretir (diske yazmaz).
- `PeakFrameExporter`: her Olay Grubu icin `pre_peak.jpg`/`peak.jpg`/
  `post_peak.jpg` + `metadata.json`'i diske yazar.
"""

from src.sampler.context.peak_frame_exporter import PeakFrameExporter
from src.sampler.context.representative_frame_extractor import RepresentativeFrameExtractor

__all__ = ["PeakFrameExporter", "RepresentativeFrameExtractor"]
