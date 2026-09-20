"""Ortak VLM yeniden-deneme politikasi (`base_vlm.send_with_retry`) davranis testleri.

NEDEN BU DOSYA VAR
------------------
Yeniden deneme mantigi eskiden YALNIZCA kare-tabanli ("hafif") yolda,
`_post_chat_completion` icinde SATIR ICI vardi; video-dogrudan ("agir") yolda
HIC YOKTU - tek bir gecici ag hatasi tum analizi dusuruyordu. Ayrica mevcut
satir-ici dongu `raise_for_status()`u try BLOGUNUN ICINDE cagirdigi ve
`httpx.HTTPError` yakaladigi icin KALICI hatalari (400 bozuk istek, 401
gecersiz anahtar, 413 cok buyuk govde) de bosuna yeniden deniyordu - gecersiz
bir API anahtarinda hata suresi 3 katina cikiyordu.

Bu testler iki sozlesmeyi kilitler:
  1. GECICI hatalar (ag hatasi, 429, 5xx) yeniden denenir ve sonunda basarili
     olabilir.
  2. KALICI hatalar (429 disindaki 4xx) ASLA yeniden denenmez - ilk denemede
     yukseltilir.
Ucuncu olarak, `Retry-After` basliginin onurlandirildigi ve her iki yolun
(hafif + agir) AYNI politikayi kullandigi dogrulanir.

Testler AG'A CIKMAZ: `httpx.post` sahte bir fonksiyonla degistirilir ve
`time.sleep` yamalanir (geri-cekilme beklemesi testi yavaslatmaz).
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

from src.vlm import base_vlm
from src.vlm.base_vlm import is_retryable_error, send_with_retry


def _response(status_code: int, *, headers: Dict[str, str] | None = None) -> httpx.Response:
    """Belirtilen durum kodunu tasiyan, istege bagli bir `httpx.Response` uretir."""
    return httpx.Response(
        status_code,
        headers=headers or {},
        json={"ok": status_code < 400},
        request=httpx.Request("POST", "https://example.invalid/v1/test"),
    )


@pytest.fixture(autouse=True)
def slept(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    """Gercek beklemeyi kaldirir ve beklenen surelerin kaydini dondurur."""
    recorded: List[float] = []
    monkeypatch.setattr(base_vlm.time, "sleep", lambda s: recorded.append(s))
    return recorded


# --------------------------------------------------------------- siniflandirma


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504])
def test_gecici_http_durumlari_yeniden_denenebilir(status: int) -> None:
    """Sunucu tarafi/gecici durumlar yeniden denenebilir sayilir."""
    resp = _response(status)
    exc = httpx.HTTPStatusError("x", request=resp.request, response=resp)
    assert is_retryable_error(exc) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_kalici_istemci_hatalari_yeniden_denenmez(status: int) -> None:
    """Istegin ICERIGIYLE ilgili hatalar tekrar denemekle DUZELMEZ."""
    resp = _response(status)
    exc = httpx.HTTPStatusError("x", request=resp.request, response=resp)
    assert is_retryable_error(exc) is False


def test_ag_hatalari_yeniden_denenebilir() -> None:
    """Baglanti/timeout hatalari istegin icerigiyle ilgili degildir - yeniden denenir."""
    request = httpx.Request("POST", "https://example.invalid/v1/test")
    assert is_retryable_error(httpx.ConnectError("dns", request=request)) is True
    assert is_retryable_error(httpx.ReadTimeout("slow", request=request)) is True


# ------------------------------------------------------------------- davranis


def test_gecici_hatadan_sonra_basarili_olur(slept: List[float]) -> None:
    """Ilk iki deneme gecici hatayla duser, ucuncu basarili olur."""
    calls: List[int] = []

    def send() -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return _response(503)
        return _response(200)

    response = send_with_retry(send, source="test")

    assert response.status_code == 200
    assert len(calls) == 3, "iki yeniden deneme yapilmaliydi"
    # Ustel geri-cekilme: 0.5s, 1.0s
    assert slept == [pytest.approx(0.5), pytest.approx(1.0)]


def test_kalici_hatada_HIC_beklenmez_ve_aninda_yukselir(slept: List[float]) -> None:
    """401 gibi kalici bir hatada tek deneme yapilir, hic uyunmaz.

    Bu, gecersiz bir API anahtarinda hata suresinin 3 katina cikmasini onleyen
    davranistir (eski satir-ici dongunun somut kusuru).
    """
    calls: List[int] = []

    def send() -> httpx.Response:
        calls.append(1)
        return _response(401)

    with pytest.raises(httpx.HTTPStatusError):
        send_with_retry(send, source="test")

    assert len(calls) == 1, "kalici hata yeniden DENENMEMELIYDI"
    assert slept == [], "kalici hatada beklenmemeliydi"


def test_denemeler_tukenirse_RuntimeError_ve_son_hata_zincirlenir(slept: List[float]) -> None:
    """Butun denemeler gecici hatayla duserse acik bir RuntimeError uretilir."""

    def send() -> httpx.Response:
        return _response(500)

    with pytest.raises(RuntimeError, match="3 denemede basarisiz"):
        send_with_retry(send, source="video.mp4")

    assert len(slept) == 2, "toplam 3 deneme -> 2 bekleme"


def test_retry_after_basligi_onurlandirilir(slept: List[float]) -> None:
    """Saglayici `Retry-After` verirse ustel geri-cekilme yerine O kullanilir."""
    calls: List[int] = []

    def send() -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return _response(429, headers={"Retry-After": "7"})
        return _response(200)

    send_with_retry(send, source="test")
    assert slept == [pytest.approx(7.0)]


def test_retry_after_tavanla_sinirlanir(slept: List[float]) -> None:
    """Asiri uzun bir `Retry-After` degeri sistemi kilitlemez."""
    calls: List[int] = []

    def send() -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return _response(429, headers={"Retry-After": "99999"})
        return _response(200)

    send_with_retry(send, source="test")
    assert slept == [pytest.approx(base_vlm._MAX_HONORED_RETRY_AFTER_SEC)]


def test_on_retry_bildirimi_cagrilir(slept: List[float]) -> None:
    """Yeniden deneme operatore bildirilebilsin diye `on_retry` tetiklenir."""
    events: List[Dict[str, Any]] = []
    calls: List[int] = []

    def send() -> httpx.Response:
        calls.append(1)
        return _response(200) if len(calls) > 1 else _response(502)

    send_with_retry(send, source="video.mp4", on_retry=events.append)

    assert len(events) == 1
    assert events[0]["attempt"] == 1
    assert events[0]["max_attempts"] == 3
    assert events[0]["delay_sec"] == pytest.approx(0.5)


# ------------------------------------------ iki yolun AYNI politikayi kullanmasi


def test_agir_yol_gemini_generate_yeniden_dener(monkeypatch: pytest.MonkeyPatch) -> None:
    """Video-dogrudan ("agir") yol da gecici hatada yeniden dener.

    Regresyon kilidi: bu yolda ONCEDEN hic retry YOKTU - tek bir gecici ag
    hatasi tum video analizini dusuruyordu.
    """
    from src.utils.config_loader import VLLMEndpointConfig
    from src.vlm.gemini_vlm import GeminiVLM

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    endpoint = VLLMEndpointConfig(
        model_name="gemini-2.5-flash",
        max_new_tokens=128,
        temperature=0.0,
        provider="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
    )
    vlm = GeminiVLM(endpoint)

    calls: List[int] = []

    def fake_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return _response(503)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "gozlem metni"}]}}]},
            request=httpx.Request("POST", "https://example.invalid"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    retries: List[Dict[str, Any]] = []
    text = vlm._generate([{"text": "merhaba"}], "video.mp4", on_retry=retries.append)

    assert text == "gozlem metni"
    assert len(calls) == 2, "agir yol gecici hatada yeniden denemeliydi"
    assert len(retries) == 1, "yeniden deneme bildirilmeliydi"
