"""Operator Paneli - uygulama orkestratoru (`SafirDashboardApp`).

Yan menu, girdi paneli, canli ilerleme ve rapor gorunumu bilesenlerini tek bir
akista birlestirir. Streamlit'e ozgu tum durum (session_state) ve sayfa kurulumu
burada yonetilir; bilesenler yalnizca kendi render sorumluluklarini tasir.
"""

from __future__ import annotations

import streamlit as st

from src.ui.api_client import SafirApiClient
from src.ui.components import InputPanel, LiveProgressView, ReportView, SidebarView
from src.ui.theme import PAGE_STYLES


class SafirDashboardApp:
    """SAFIR operator panelinin bilesenlerini kuran ve calistiran ana uygulama."""

    def __init__(self) -> None:
        """Uygulamayi API istemcisi ve tum UI bilesenleriyle baslatir."""
        self._api = SafirApiClient()
        self._sidebar = SidebarView(self._api)
        self._input_panel = InputPanel()
        self._live_progress = LiveProgressView(self._api)
        self._report_view = ReportView(self._api)

    def run(self) -> None:
        """Sayfayi kurar ve tam operator paneli akisini render eder."""
        st.set_page_config(page_title="SAFIR Operator Paneli", page_icon="🛡️", layout="wide")
        st.markdown(PAGE_STYLES, unsafe_allow_html=True)
        st.title("🛡️ SAFIR — Saha Analiz ve Farkındalık Operatör Paneli")

        if "last_report" not in st.session_state:
            st.session_state.last_report = None

        settings = self._sidebar.render()
        analysis_input = self._input_panel.render(settings.input_mode)

        if analysis_input.start_requested and analysis_input.video_source:
            report = self._live_progress.run(
                analysis_input.video_source,
                analysis_input.user_prompt,
                settings.sample_fps,
                settings.min_change_threshold,
            )
            if report is not None:
                st.session_state.last_report = report

        if st.session_state.last_report:
            self._report_view.render(st.session_state.last_report)
