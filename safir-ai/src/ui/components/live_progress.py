"""Operator Paneli - canli ilerleme bileseni (analiz isini baslatir ve akil yurutmeyi anlatir)."""

from __future__ import annotations

import time
from typing import Optional

import httpx
import streamlit as st

from src.ui.api_client import SafirApiClient
from src.ui.theme import POLL_INTERVAL_SEC, STAGE_NARRATION


class LiveProgressView:
    """Analiz isini baslatir ve `st.status` ile 5 adimli akil yurutme surecini canli gosterir.

    Anlatim, backend'in gercek 3 asamali ilerleme sinyaline (`JobStatusResponse.step`)
    baglidir; sahte bir zamanlayici degildir. Is tamamlaninca nihai raporu dondurur.
    """

    def __init__(self, api_client: SafirApiClient) -> None:
        """Bileseni, is olusturma/sorgulama icin kullanacagi API istemcisiyle baslatir."""
        self._api = api_client

    def run(
        self, video_source: str, user_prompt: str, sample_fps: int, min_change_threshold: float
    ) -> Optional[dict]:
        """Isi baslatir, durumu canli sorgular ve tamamlaninca `SafirReport` sozlugunu dondurur.

        Returns:
            Basarili tamamlanmada rapor sozlugu; hata/iptalde `None`.
        """
        with st.status("🧠 Ajan Dusunme Sureci Baslatiliyor...", expanded=True) as status:
            try:
                job_id = self._api.create_analyze_job(
                    video_source, user_prompt, sample_fps, min_change_threshold
                )
            except httpx.HTTPError as exc:
                status.update(label="❌ Pipeline baslatilamadi", state="error")
                st.error(f"Analiz baslatilamadi: {exc}")
                return None

            narrated_steps: set = set()
            while True:
                try:
                    data = self._api.get_job_status(job_id)
                except httpx.HTTPError as exc:
                    status.update(label="❌ Durum sorgulanamadi", state="error")
                    st.error(f"Is durumu alinamadi: {exc}")
                    return None

                backend_step = data["step"]
                if backend_step not in narrated_steps and backend_step in STAGE_NARRATION:
                    for line in STAGE_NARRATION[backend_step]:
                        st.write(line)
                    narrated_steps.add(backend_step)

                status.update(
                    label=f"{data['stage_name'] or 'Kuyrukta bekleniyor...'} "
                    f"({backend_step}/{data['total_steps']})"
                )

                if data["status"] == "done":
                    status.update(label="✅ Analiz tamamlandi", state="complete", expanded=False)
                    return data["result"]
                if data["status"] == "error":
                    status.update(label="❌ Analiz basarisiz oldu", state="error")
                    st.error(f"Analiz basarisiz oldu: {data['error']}")
                    return None

                time.sleep(POLL_INTERVAL_SEC)
