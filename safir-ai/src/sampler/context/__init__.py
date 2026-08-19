"""Olay Gruplari icin ORTAK kare secim + (istege bagli) diske arsivleme katmani.

Onceden iki bagimsiz mekanizma vardi (VLM icin `RepresentativeFrameExtractor`,
disk arsivi icin `PeakFrameExporter`); ikisi de kaynak videoyu ayri ayri
seek ediyor, VLM'in gordugu kareler ile diske yazilan kareler farklilasabiliyordu.
Artik TEK bir secim kaynagi var:

- `FrameSelector`: `AdaptiveFrameSampler.cluster_events` icinde her Olay Grubu
  icin, zaten bellekte JPEG/base64 encode edilmis Kanit Karelerinden
  (video YENIDEN acilmadan) en fazla 5 benzersiz, kronolojik, zirve dahil
  temsili kare secer.
- `FrameArchiver`: `FrameSelector`'in sectigi kareleri (bagimsiz bir secim
  YAPMADAN) `event_XXXX/frame_NN_<label>.jpg` + `metadata.json` olarak
  diske yazan pasif persistence bileseni.
"""

from src.sampler.context.frame_archiver import FrameArchiver
from src.sampler.context.frame_selector import FrameSelector

__all__ = ["FrameArchiver", "FrameSelector"]
