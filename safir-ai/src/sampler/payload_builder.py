"""04 - VLM Payload Builder: evidence karelerini vLLM'e gonderilecek JSON'a donusturur.

Bu modul, `AdaptiveFrameSampler.process_video` tarafindan uretilen, esik
gecmis TUM `EvidenceFrame`leri (video geneli, kronolojik, kayipsiz) alip her
biri icin kimlik/zaman/evidence-skoru bilgisi iceren bir metin blogu ile
birlikte, OpenAI/vLLM `chat/completions` API'sinin bekledigi `image_url`
(base64) icerik bloklarina donusturur. Kareler arasinda hicbir onceden
belirlenmis olay/kumeleme bilgisi VERILMEZ - olay kumelemesi VLM'in kendi
gorevidir (bkz. `src/prompts/vlm_prompts.py`, `src/vlm/base_vlm.py`).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.sampler.schema import EvidenceFrame

logger = logging.getLogger(__name__)


class VLMPayloadBuilder:
    """Evidence karelerini VLM icin OpenAI-uyumlu, kronolojik icerik bloklarina cevirir."""

    @staticmethod
    def build_content_blocks(evidence_frames: List[EvidenceFrame], prompt: str) -> List[Dict[str, Any]]:
        """Evidence karelerini, kimlik/zaman/skor metniyle birlikte kronolojik icerik bloklarina cevirir.

        Her evidence karesi icin once o kareyi tanimlayan kisa bir metin
        blogu (`evidence_id`/`frame_index`/zaman/evidence skoru), ardindan
        karenin base64 goruntusu eklenir; boylece VLM her karenin kimligini
        ve zamanini goruntuyle birlikte gorur ve kendi olay kumelemesini
        yapabilir (`evidence_ids` referanslariyla).

        Args:
            evidence_frames: Gonderilecek `EvidenceFrame` listesi (cagiranin
                sorumlulugunda kronolojik sirada olmalidir; bu bir batch'in
                TAMAMI veya alt kumesi olabilir - bkz.
                `BaseVLM.analyze_evidence_batched`).
            prompt: Analiz odagini belirten kullanici/istem metni.

        Returns:
            `/v1/chat/completions` mesaj icerigi olarak kullanilabilecek,
            `{"type": "text" | "image_url", ...}` bloklarindan olusan liste.

        Raises:
            ValueError: `evidence_frames` bos verilirse.
        """
        if not evidence_frames:
            raise ValueError("VLMPayloadBuilder: bos evidence listesiyle payload uretilemez.")

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for ef in evidence_frames:
            metadata_text = (
                f"[evidence_id={ef.evidence_id} | frame_index={ef.frame_id} | "
                f"zaman={ef.timestamp_str} ({ef.timestamp_sec:.2f}s) | "
                f"evidence_skoru={ef.change_score:.4f} | "
                f"secim_nedeni={ef.selection_reason}]"
            )
            content.append({"type": "text", "text": metadata_text})
            content.append({"type": "image_url", "image_url": {"url": ef.base64_image}})

        logger.debug(
            "VLMPayloadBuilder: %d evidence karesi icin %d icerik blogu uretildi.",
            len(evidence_frames),
            len(content),
        )
        return content

    @staticmethod
    def build_metadata_summary(evidence_frames: List[EvidenceFrame]) -> List[Dict[str, Any]]:
        """Her evidence karesi icin serilestirilebilir bir metadata sozlugu uretir.

        Rapor/log ciktilarinda kullanilmak uzere; goruntu verisi icermez.

        Args:
            evidence_frames: `process_video` tarafindan uretilen evidence kareleri.

        Returns:
            Her biri bir evidence karesini tanimlayan sozlukler listesi.
        """
        return [
            {
                "evidence_id": ef.evidence_id,
                "frame_index": ef.frame_id,
                "timestamp_sec": ef.timestamp_sec,
                "evidence_score": ef.change_score,
                "selection_reason": ef.selection_reason,
            }
            for ef in evidence_frames
        ]
