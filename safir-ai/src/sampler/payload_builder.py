"""04 - VLM Payload Builder: zirve karelerini vLLM'e gonderilecek JSON'a donusturur.

Bu modul, `AdaptiveFrameSampler.cluster_events` tarafindan uretilen Olay
Gruplarinin zirve karelerini (peak frames) alip her biri icin zaman damgasi
ve olay metadatasi iceren bir metin blogu ile birlikte, OpenAI/vLLM
`chat/completions` API'sinin bekledigi `image_url` (base64) icerik bloklarina
donusturur. VLM katmani (`BaseVLM`), bu modulun uretttigi icerik bloklarini
model adi/sicaklik gibi model-ozel alanlarla sarmalayarak nihai istegi olusturur.
"""

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

        all_timestamps = []
        for c in clusters:
            if c.representative_frames:
                for rf in c.representative_frames:
                    all_timestamps.append(rf.timestamp_str)
            elif c.peak_frame:
                all_timestamps.append(c.peak_frame.timestamp_str)

        # 1. Payload Debug Log (Kesin Zaman Damgasi Teyidi)
        print(f"[DEBUG VLM PAYLOAD] Modele giden karelerin zaman damgaları (toplam {len(all_timestamps)} kare): {all_timestamps}")
        logger.info(
            "[DEBUG VLM PAYLOAD] Modele giden karelerin zaman damgalari (toplam %d kare): %s",
            len(all_timestamps),
            all_timestamps,
        )

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for idx, cluster in enumerate(clusters, start=1):
            peak = cluster.peak_frame
            start_mins, start_secs = divmod(int(cluster.start_time), 60)
            start_str = f"{start_mins:02d}:{start_secs:02d}"
            end_mins, end_secs = divmod(int(cluster.end_time), 60)
            end_str = f"{end_mins:02d}:{end_secs:02d}"

            block_header = (
                f"\n--- [BLOK #{idx} | Zaman Aralığı: {start_str} - {end_str}] ---\n"
                f"Aşağıdaki görseller bu 20 saniyelik zaman penceresine aittir. "
                f"Olayın İLK KEZ başladığı görselin zaman damgasını bulunuz:"
            )
            content.append({"type": "text", "text": block_header})

            if cluster.representative_frames:
                for rf_idx, rf in enumerate(cluster.representative_frames):
                    if "REFERANS" in rf.label:
                        content.append(
                            {"type": "text", "text": f"[Görsel #0: REFERANS TEMİZ KARE | Zaman: {rf.timestamp_str}] (Sahnenin temiz referans zeminidir)"}
                        )
                    else:
                        content.append(
                            {"type": "text", "text": f"[Görsel #{rf_idx}: Zaman: {rf.timestamp_str} | PTS: {rf.timestamp_sec:.2f}s]"}
                        )
                    content.append({"type": "image_url", "image_url": {"url": rf.base64_image}})
            else:
                content.append(
                    {"type": "text", "text": f"[Görsel: Zaman: {peak.timestamp_str} | PTS: {peak.timestamp_sec:.2f}s]"}
                )
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
