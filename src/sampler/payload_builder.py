from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.sampler.adaptive_sampler import EventCluster

logger = logging.getLogger(__name__)


class VLMPayloadBuilder:
    """Olay Gruplarinin zirve karelerini VLM icin OpenAI-uyumlu icerik bloklarina cevirir."""

    @staticmethod
    def build_content_blocks(clusters: List[EventCluster], prompt: str) -> List[Dict[str, Any]]:
        """Zirve karelerini, zaman damgasi/metadata metniyle birlikte icerik bloklarina cevirir.

        Her Olay Grubu icin once o grubu tanimlayan bir metin blogu (zaman
        araligi, aday kare sayisi, degisim skoru), ardindan zirve karenin
        base64 goruntusu eklenir. Bu, VLM'in her karenin hangi olaya ait
        oldugunu ayirt edebilmesini saglar.

        Args:
            clusters: `cluster_events` tarafindan uretilen Olay Gruplari.
            prompt: Analiz odagini belirten kullanici/istem metni.

        Returns:
            `/v1/chat/completions` mesaj icerigi olarak kullanilabilecek,
            `{"type": "text" | "image_url", ...}` bloklarindan olusan liste.

        Raises:
            ValueError: `clusters` bos verilirse.
        """
        if not clusters:
            raise ValueError("VLMPayloadBuilder: bos Olay Grubu listesiyle payload uretilemez.")

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for cluster in clusters:
            peak = cluster.peak_frame
            metadata_text = (
                f"[Olay #{cluster.event_id}] zaman araligi={peak.timestamp_str} "
                f"({cluster.start_time:.2f}s - {cluster.end_time:.2f}s), "
                f"aday_kare_sayisi={cluster.total_candidate_frames}, "
                f"degisim_skoru={peak.change_score:.4f}"
            )
            content.append({"type": "text", "text": metadata_text})

            if cluster.representative_frames:
                for rf in cluster.representative_frames:
                    content.append(
                        {"type": "text", "text": f"[{rf.label}, {rf.timestamp_str}]"}
                    )
                    content.append({"type": "image_url", "image_url": {"url": rf.base64_image}})
            else:
                content.append({"type": "image_url", "image_url": {"url": peak.base64_image}})

        logger.debug(
            "VLMPayloadBuilder: %d olay grubu icin %d icerik blogu uretildi.",
            len(clusters),
            len(content),
        )
        return content

    @staticmethod
    def build_metadata_summary(clusters: List[EventCluster]) -> List[Dict[str, Any]]:
        """Her Olay Grubu icin serilestirilebilir bir metadata sozlugu uretir.

        Rapor/log ciktilarinda kullanilmak uzere; goruntu verisi icermez.

        Args:
            clusters: `cluster_events` tarafindan uretilen Olay Gruplari.

        Returns:
            Her biri bir Olay Grubunu tanimlayan sozlukler listesi.
        """
        return [
            {
                "event_id": cluster.event_id,
                "start_time": cluster.start_time,
                "end_time": cluster.end_time,
                "peak_timestamp": cluster.peak_frame.timestamp_sec,
                "peak_change_score": cluster.peak_frame.change_score,
                "total_candidate_frames": cluster.total_candidate_frames,
            }
            for cluster in clusters
        ]