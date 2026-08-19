"""Ortak secilmis (`FrameSelector`) temsili kareleri diske arsivleyen PASIF persistence katmani.

`FrameArchiver` kendi basina hicbir kare SECIMI yapmaz ve kaynak videoya
HICBIR sekilde erismez: yalnizca `AdaptiveFrameSampler.cluster_events` icinde
`FrameSelector` tarafindan ZATEN secilmis `EventCluster.representative_frames`
listesini alip JPEG + `metadata.json` olarak diske yazar. Boylece diske
yazilan/kullaniciya arsiv olarak gosterilen kareler ile VLM'e giden kareler
HER ZAMAN ayni kaynaktan (ayni secimden) gelir; ikinci bir seek/JPEG-encode
dongusu YOKTUR.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.sampler.schema import EventCluster

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = "outputs/sampler"


class FrameArchiver:
    """Her Olay Grubu icin `FrameSelector` tarafindan ZATEN secilmis kareleri diske yazar."""

    @staticmethod
    def export(clusters: List[EventCluster], output_dir: str = _DEFAULT_OUTPUT_DIR) -> List[str]:
        """Her Olay Grubu icin `event_XXXX/frame_NN_<label>.jpg` + `metadata.json` yazar.

        Secilmis kare yoksa (`representative_frames` bos; teorik olarak
        `FrameSelector` her zaman en az zirveyi doner, ama guvenlik icin
        kontrol edilir) o olay icin yalnizca bos bir `frames` listesiyle
        `metadata.json` yazilir; hata firlatilmaz.

        Args:
            clusters: `cluster_events` ciktisi, `representative_frames` ZATEN
                doldurulmus Olay Gruplari.
            output_dir: `event_XXXX/` alt klasorlerinin olusturulacagi kok dizin.

        Returns:
            Yazilan her `event_XXXX` klasorunun yolu (string listesi).
        """
        if not clusters:
            return []

        event_dirs: List[str] = []
        for cluster in clusters:
            event_dirs.append(FrameArchiver._export_one_event(cluster, output_dir))
        return event_dirs

    @staticmethod
    def _export_one_event(cluster: EventCluster, output_dir: str) -> str:
        event_dir = Path(output_dir) / f"event_{cluster.event_id:04d}"
        event_dir.mkdir(parents=True, exist_ok=True)

        frame_entries: List[Dict[str, Any]] = []
        for idx, rf in enumerate(cluster.representative_frames):
            try:
                _, b64_payload = rf.base64_image.split(",", 1)
                raw = base64.b64decode(b64_payload)
            except (ValueError, IndexError):
                logger.warning(
                    "Kare (event=%d, frame_id=%d) base64 cozulemedi, atlaniyor.",
                    cluster.event_id,
                    rf.frame_id,
                )
                continue

            filename = f"frame_{idx:02d}_{rf.label}.jpg"
            (event_dir / filename).write_bytes(raw)
            frame_entries.append(
                {
                    "filename": filename,
                    "label": rf.label,
                    "frame_id": rf.frame_id,
                    "timestamp_sec": rf.timestamp_sec,
                    "timestamp_str": rf.timestamp_str,
                }
            )

        metadata = {
            "event_id": cluster.event_id,
            "start_time": round(cluster.start_time, 2),
            "end_time": round(cluster.end_time, 2),
            "peak_time": round(cluster.peak_frame.timestamp_sec, 2),
            "peak_change_score": cluster.peak_frame.change_score,
            "total_candidate_frames": cluster.total_candidate_frames,
            # FrameSelector'in bu olay icin fiilen sectigi benzersiz kare sayisi;
            # hedef (TARGET_FRAME_COUNT=5) her zaman ulasilamayabilir (kisa olay/
            # az benzersiz kare) - bkz. RepresentativeFrame schema docstring'i.
            "selected_frame_count": len(frame_entries),
            "frames": frame_entries,
        }
        (event_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(event_dir)
