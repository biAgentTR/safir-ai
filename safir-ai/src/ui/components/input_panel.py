"""Operator Paneli - girdi bileseni (video kaynagi + inceleme istemi)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from src.ui.theme import ALLOWED_VIDEO_EXTENSIONS, DATA_DIR


@dataclass
class AnalysisInput:
    """Operatorun sagladigi analiz girdisi."""

    video_source: Optional[str]
    user_prompt: str
    start_requested: bool


class InputPanel:
    """Video/istem girdisini toplar ve 'Baslat' istegi olup olmadigini bildirir."""

    def render(self, input_mode: str) -> AnalysisInput:
        """Girdi panelini cizer ve secilen kaynagi/istemi/baslat-istegini dondurur.

        Args:
            input_mode: Yan menuden gelen kaynak modu (dosya/canli yayin).
        """
        st.header("📥 Girdi Paneli")

        video_source: Optional[str] = None
        if input_mode == "📹 Video Dosyası Sürükle":
            uploaded_file = st.file_uploader(
                "Video dosyasini surukle veya sec", type=ALLOWED_VIDEO_EXTENSIONS
            )
            if uploaded_file is not None:
                video_source = self._save_uploaded_file(uploaded_file)
                st.success(f"Dosya kaydedildi: `data/{video_source}`")
        else:
            video_source = st.text_input(
                "RTSP/HTTP canli yayin adresi",
                placeholder="rtsp://192.168.1.10:554/stream veya http://kamera-ip/video",
            )

        user_prompt = st.text_area(
            "Sahaya ozel ISG talimati / inceleme istemi",
            value="Sahnede baret veya yelek takmayan personel ya da duman tespiti yap.",
            height=80,
        )

        start_requested = st.button(
            "🚀 Pipeline & Ajan Düşünme Sürecini Başlat", type="primary", disabled=not video_source
        )

        return AnalysisInput(
            video_source=video_source, user_prompt=user_prompt, start_requested=start_requested
        )

    @staticmethod
    def _save_uploaded_file(uploaded_file) -> str:
        """Yuklenen video dosyasini `data/` altina kaydeder ve dosya adini dondurur."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        destination = DATA_DIR / uploaded_file.name
        destination.write_bytes(uploaded_file.getvalue())
        return uploaded_file.name
