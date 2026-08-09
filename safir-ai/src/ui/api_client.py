"""Operator Paneli - SAFIR API istemcisi (tum HTTP cagrilari tek sinifta).

`SafirApiClient`, panelin backend ile tum iletisimini (saglik, analiz isi
olusturma/sorgulama, geri bildirim, alarm onayi/override) kapsuller; boylece UI
bilesenleri `httpx` detaylarindan yalitilir ve tek noktadan test edilebilir.
"""

from __future__ import annotations

from typing import Any, Dict

import httpx

from src.ui.theme import API_BASE_URL, HTTP_TIMEOUT_SEC


class SafirApiClient:
    """SAFIR FastAPI servisiyle konusan ince HTTP istemcisi."""

    def __init__(self, base_url: str = API_BASE_URL, timeout: float = HTTP_TIMEOUT_SEC) -> None:
        """Istemciyi backend adresi ve zaman asimiyla baslatir.

        Args:
            base_url: SAFIR API taban adresi (varsayilan `SAFIR_API_URL`).
            timeout: HTTP istekleri icin zaman asimi (saniye).
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        """Backend taban adresini dondurur (UI mesajlarinda gostermek icin)."""
        return self._base_url

    def check_health(self) -> bool:
        """`/health` uzerinden backend'in ayakta olup olmadigini kontrol eder (kisa timeout)."""
        try:
            response = httpx.get(f"{self._base_url}/health", timeout=3.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def create_analyze_job(
        self, video_source: str, user_prompt: str, sample_fps: int, min_change_threshold: float
    ) -> str:
        """`/analyze/jobs`'a POST atarak arka planda analiz isi baslatir ve `job_id` dondurur.

        Raises:
            httpx.HTTPError: Istek basarisiz olursa.
        """
        response = httpx.post(
            f"{self._base_url}/analyze/jobs",
            json={
                "video_source": video_source,
                "user_prompt": user_prompt,
                "sample_fps": sample_fps,
                "min_change_threshold": min_change_threshold,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["job_id"]

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """`/analyze/jobs/{job_id}` uzerinden bir isin guncel durumunu dondurur.

        Raises:
            httpx.HTTPError: Istek basarisiz olursa.
        """
        response = httpx.get(f"{self._base_url}/analyze/jobs/{job_id}", timeout=self._timeout)
        response.raise_for_status()
        return response.json()

    def send_feedback(self, event_id: int, feedback: str) -> Dict[str, Any]:
        """`/events/{event_id}/feedback`'e operator dogrulamasini gonderir.

        Raises:
            httpx.HTTPError: Istek basarisiz olursa.
        """
        response = httpx.post(
            f"{self._base_url}/events/{event_id}/feedback",
            json={"feedback": feedback},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def acknowledge_alert(self, alert_id: str, operator_note: str = "") -> Dict[str, Any]:
        """Otomatik tetiklenmis bir saha alarmini operator adina onaylar (Human-on-the-Loop).

        Raises:
            httpx.HTTPError: Istek basarisiz olursa.
        """
        response = httpx.post(
            f"{self._base_url}/alerts/{alert_id}/acknowledge",
            json={"operator_note": operator_note},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def trigger_manual_alert(
        self, risk_score: int, risk_level: str, recommended_action: str, operator_note: str = ""
    ) -> Dict[str, Any]:
        """Operatorun otomatik akis disinda manuel/override saha alarmi tetiklemesini saglar.

        Raises:
            httpx.HTTPError: Istek basarisiz olursa.
        """
        response = httpx.post(
            f"{self._base_url}/alerts/trigger",
            json={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "recommended_action": recommended_action,
                "operator_note": operator_note,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()
