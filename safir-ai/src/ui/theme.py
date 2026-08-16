"""Operator Paneli - tema, sabitler ve sunum yardimcilari (Streamlit'ten bagimsiz saf mantik).

UI bilesenlerinin paylastigi sabitler (API adresi, esikler), CSS temasi, risk
rozeti cozumleme mantigi ve sesli uyari (beep) uretimi burada toplanir; boylece
bilesenler yalnizca kendi render sorumluluklarina odaklanir.
"""

from __future__ import annotations

import base64
import io
import math
import os
import struct
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Servis / davranis sabitleri ---
API_BASE_URL = os.environ.get("SAFIR_API_URL", "http://localhost:8000")
DATA_DIR = Path(os.environ.get("SAFIR_DATA_DIR", "data"))
ALLOWED_VIDEO_EXTENSIONS = ["mp4", "avi", "mkv"]
POLL_INTERVAL_SEC = 1.0
HTTP_TIMEOUT_SEC = 30.0
CRITICAL_ALARM_THRESHOLD = 70

# --- Risk seviyesi -> rozet (etiket, renk) ---
RISK_BADGE_MAP: Dict[str, Tuple[str, str]] = {
    "kritik": ("CRITICAL", "#dc2626"),
    "yuksek": ("HIGH", "#ea580c"),
    "orta": ("MEDIUM", "#d97706"),
    "dusuk": ("LOW", "#2563eb"),
}
NORMAL_BADGE: Tuple[str, str] = ("NORMAL", "#16a34a")
UNKNOWN_BADGE: Tuple[str, str] = ("BELIRSIZ", "#64748b")

# Backend'in 3 asamali gercek ilerlemesini, panelde 5 adimli anlatima esler.
STAGE_NARRATION: Dict[int, List[str]] = {
    1: [
        "**Adim 1:** CPU Adaptive Sampler Calistiriliyor...",
        "**Adim 2:** Hareket & Degisim Kontrolu (Noise Floor Filter)...",
    ],
    2: ["**Adim 3:** Multimodal VLM Gorsel Anlama..."],
    3: [
        "**Adim 4:** FAISS ISG Mevzuat RAG Sorgulamasi...",
        "**Adim 5:** Nihai Risk Skoru & Otomatik Eskalasyon...",
    ],
}

PAGE_STYLES = """
<style>
.filter-banner {
    background: linear-gradient(90deg, #0f172a 0%, #1e3a5f 100%);
    border: 1px solid #2563eb;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: #e2e8f0;
    margin-bottom: 1rem;
}
.filter-banner b { color: #93c5fd; }
.risk-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-weight: 700;
    color: white;
    letter-spacing: 0.03em;
}
@keyframes safir-flash-critical {
    0%, 100% { background-color: #dc2626; }
    50% { background-color: #7f1d1d; }
}
.critical-alarm-banner {
    animation: safir-flash-critical 1s infinite;
    color: white;
    font-weight: 800;
    font-size: 1.15rem;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 1rem;
    border: 2px solid #fecaca;
}
</style>
"""


def resolve_risk_badge(risk_level: str, risk_score: Optional[int]) -> Tuple[str, str]:
    """Risk seviyesi/skoruna gore `(etiket, renk)` rozetini cozer.

    Args:
        risk_level: `dusuk`/`orta`/`yuksek`/`kritik`/`unknown`.
        risk_score: 0-100 arasi skor (NORMAL esigini belirlemek icin); guvenilir
            bir karar uretilemediyse `None` (bkz. `risk_status="unknown"`).

    Returns:
        `(etiket, renk_kodu)` ikilisi. `risk_score` `None` ise ASLA NORMAL/dusuk
        risk olarak yorumlanmaz — ayri bir `UNKNOWN_BADGE` dondurulur.
    """
    if risk_score is None:
        return UNKNOWN_BADGE
    if risk_score <= 5:
        return NORMAL_BADGE
    return RISK_BADGE_MAP.get(risk_level, NORMAL_BADGE)


def risk_badge_html(risk_level: str, risk_score: Optional[int]) -> str:
    """Risk seviyesine gore renkli bir HTML rozet dondurur."""
    label, color = resolve_risk_badge(risk_level, risk_score)
    return f'<span class="risk-badge" style="background-color:{color};">{label}</span>'


def generate_beep_data_uri(frequency: float = 880.0, duration_sec: float = 0.35, volume: float = 0.5) -> str:
    """Harici bagimlilik olmadan stdlib ile kisa bir bip sesi `data:` URI'si uretir.

    Args:
        frequency: Bip frekansi (Hz).
        duration_sec: Sure (saniye).
        volume: 0-1 arasi ses seviyesi.

    Returns:
        `<audio>` etiketine dogrudan verilebilecek `data:audio/wav;base64,...`.
    """
    sample_rate = 22050
    n_samples = int(sample_rate * duration_sec)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(volume * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wav_file.writeframes(struct.pack("<h", value))
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{encoded}"


# Bir kez uretilip yeniden kullanilir (her render'da yeniden hesaplanmaz).
CRITICAL_ALARM_AUDIO_DATA_URI = generate_beep_data_uri()
