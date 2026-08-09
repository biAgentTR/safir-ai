"""Operator Paneli - yan menu bileseni (veri kaynagi, sampler ayarlari, saglik)."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.ui.api_client import SafirApiClient


@dataclass
class SidebarSettings:
    """Yan menuden toplanan operator ayarlari."""

    input_mode: str
    sample_fps: int
    min_change_threshold: float


class SidebarView:
    """Yan menudeki veri kaynagi secimi, sampler slider'lari ve backend saglik durumunu render eder."""

    def __init__(self, api_client: SafirApiClient) -> None:
        """Bileseni, saglik kontrolu icin kullanacagi API istemcisiyle baslatir."""
        self._api = api_client

    def render(self) -> SidebarSettings:
        """Yan menuyu cizer ve secilen ayarlari dondurur."""
        with st.sidebar:
            st.header("⚙️ Ayarlar")

            st.subheader("Saha Veri Kaynagi")
            input_mode = st.radio(
                "Kaynak",
                ["📹 Video Dosyası Sürükle", "🔴 RTSP / Canlı Kamera Akışı"],
                label_visibility="collapsed",
            )

            st.subheader("CPU Sampler Esik Ayarlari")
            sample_fps = st.slider("Ornekleme FPS (sample_fps)", min_value=1, max_value=10, value=5)
            min_change_threshold = st.slider(
                "Hassasiyet Esigi (min_change_threshold)",
                min_value=0.001,
                max_value=0.050,
                value=0.011,
                step=0.001,
                format="%.3f",
            )

            st.subheader("Backend Servis Sagligi")
            if self._api.check_health():
                st.success(f"API ayakta: `{self._api.base_url}`")
            else:
                st.error(f"Backend'e ulasilamiyor: `{self._api.base_url}`")

            return SidebarSettings(
                input_mode=input_mode,
                sample_fps=sample_fps,
                min_change_threshold=min_change_threshold,
            )
