"""SAFIR Operator Paneli: Streamlit tabanli gorsel destekli gercek zamanli arayuz.

Bu panel, `src/main.py` FastAPI servisinin `/health`, `/analyze/jobs`
(asenkron is kuyrugu), `/analyze/jobs/{job_id}` (canli asama sorgulama) ve
`/alerts/trigger` (Human-in-the-Loop saha alarmi) uc noktalarini tuketir.

Yan menude operator; saha veri kaynagini (dosya/canli yayin) secer, CPU
Sampler'in ornekleme FPS ve hassasiyet esigini slider'larla ayarlar ve
backend saglik durumunu canli izler. Ana panelde video/istem girdisini verip
analizi baslatir; `st.status` ile ajanin 5 adimli akil yurutme surecini canli
izler (backend'in gercek 3 asamali ilerleme sinyaline baglidir). Sonuclar;
CPU Filtre/GPU Tasarruf banner'i + renkli risk rozetiyle ozetlenir ve 4
sekmeli bir panelde (Kanit Galerisi, Ajan Dusunme & RAG, Olay Cizelgesi &
Alarm, Sartname JSON Raporu) detaylandirilir.

Calistirma:
    streamlit run src/ui/dashboard.py
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import struct
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import streamlit as st
from history import render_history_section
from pdf_fonts import register_tr_fonts, TR_FONT, TR_FONT_BOLD
register_tr_fonts()


API_BASE_URL = os.environ.get("SAFIR_API_URL", "http://localhost:8000")
DATA_DIR = Path(os.environ.get("SAFIR_DATA_DIR", "data"))
ALLOWED_VIDEO_EXTENSIONS = ["mp4", "avi", "mkv"]
POLL_INTERVAL_SEC = 1.0
HTTP_TIMEOUT_SEC = 30.0
CRITICAL_ALARM_THRESHOLD = 70

# Risk seviyesi (backend: dusuk/orta/yuksek/kritik) -> rozet etiketi ve rengi.
_RISK_BADGE_MAP = {
    "kritik": ("CRITICAL", "#dc2626"),
    "yuksek": ("HIGH", "#ea580c"),
    "orta": ("MEDIUM", "#d97706"),
    "dusuk": ("LOW", "#2563eb"),
}
_NORMAL_BADGE = ("NORMAL", "#16a34a")

# Backend'in gercek 3 asamali ilerleme adimlarini, panelde istenen 5 adimli
# akil yurutme anlatimina esler. Anahtar = JobStatusResponse.step (1..3).
_STAGE_NARRATION: Dict[int, List[str]] = {
    1: [
        "**Adim 1:** CPU Adaptive Sampler Calistiriliyor...",
        "**Adim 2:** Hareket & Degisim Kontrolu (Noise Floor Filter)...",
    ],
    2: [
        "**Adim 3:** Multimodal vLLM (Qwen2.5-VL) Gorsel Anlama...",
    ],
    3: [
        "**Adim 4:** FAISS ISG Mevzuat RAG Sorgulamasi...",
        "**Adim 5:** Nihai Risk Skoru & Sartname JSON Ciktisi...",
    ],
}

st.set_page_config(page_title="SAFIR Operator Paneli", page_icon="🛡️", layout="wide")

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)


def _generate_beep_data_uri(frequency: float = 880.0, duration_sec: float = 0.35, volume: float = 0.5) -> str:
    """Harici bir ses dosyasina/aga bagimli olmadan, stdlib ile kisa bir bip sesi uretir.

    Args:
        frequency: Bip sesinin frekansi (Hz).
        duration_sec: Bip sesinin suresi (saniye).
        volume: 0-1 arasi ses seviyesi.

    Returns:
        `data:audio/wav;base64,...` formatinda, `<audio>` etiketine dogrudan verilebilecek data URI.
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


_CRITICAL_ALARM_AUDIO_DATA_URI = _generate_beep_data_uri()


def _check_backend_health() -> bool:
    """`/health` uc noktasina kisa sureli bir istek atarak backend'in ayakta olup olmadigini kontrol eder.

    Returns:
        Backend 200 ile yanit veriyorsa `True`, aksi halde `False`.
    """
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _save_uploaded_file(uploaded_file) -> str:
    """Streamlit'e yuklenen video dosyasini `data/` klasorune kaydeder.

    Args:
        uploaded_file: `st.file_uploader` tarafindan dondurulen yuklenen dosya nesnesi.

    Returns:
        Backend'e gonderilecek dosya adi (basename).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getvalue())
    return uploaded_file.name


def _create_analyze_job(
    video_source: str, user_prompt: str, sample_fps: int, min_change_threshold: float
) -> str:
    """`/analyze/jobs` uc noktasina POST atarak arka planda bir analiz isi baslatir.

    Args:
        video_source: Video dosya adi/yolu veya canli yayin URI'si.
        user_prompt: Operatorun ozel inceleme istemi.
        sample_fps: Yan menudeki slider'dan gelen ornekleme FPS degeri.
        min_change_threshold: Yan menudeki slider'dan gelen hassasiyet esigi.

    Returns:
        Olusturulan isin kimligi.

    Raises:
        httpx.HTTPError: Istek basarisiz olursa.
    """
    response = httpx.post(
        f"{API_BASE_URL}/analyze/jobs",
        json={
            "video_source": video_source,
            "user_prompt": user_prompt,
            "sample_fps": sample_fps,
            "min_change_threshold": min_change_threshold,
        },
        timeout=HTTP_TIMEOUT_SEC,
    )
    response.raise_for_status()
    return response.json()["job_id"]


def _get_job_status(job_id: str) -> Dict[str, Any]:
    """`/analyze/jobs/{job_id}` uzerinden bir isin guncel durumunu getirir.

    Args:
        job_id: `_create_analyze_job` tarafindan donen is kimligi.

    Returns:
        `JobStatusResponse` sozlugu.
    """
    response = httpx.get(f"{API_BASE_URL}/analyze/jobs/{job_id}", timeout=HTTP_TIMEOUT_SEC)
    response.raise_for_status()
    return response.json()


def _run_pipeline_with_live_status(
    video_source: str, user_prompt: str, sample_fps: int, min_change_threshold: float
) -> None:
    """Analiz isini baslatir ve `st.status` ile 5 adimli canli akil yurutme surecini gosterir.

    Anlatim adimlari, backend'in gercek 3 asamali ilerleme sinyaline
    (`JobStatusResponse.step`) baglidir; ilerleme sahte bir zamanlayici
    degil, `/analyze/jobs/{job_id}` sorgulamasindan gelen gercek durumdur.

    Args:
        video_source: Video dosya adi/yolu veya canli yayin URI'si.
        user_prompt: Operatorun ozel inceleme istemi.
        sample_fps: Yan menudeki slider'dan gelen ornekleme FPS degeri.
        min_change_threshold: Yan menudeki slider'dan gelen hassasiyet esigi.
    """
    with st.status("🧠 Ajan Dusunme Sureci Baslatiliyor...", expanded=True) as status:
        try:
            job_id = _create_analyze_job(video_source, user_prompt, sample_fps, min_change_threshold)
        except httpx.HTTPError as exc:
            status.update(label="❌ Pipeline baslatilamadi", state="error")
            st.error(f"Analiz baslatilamadi: {exc}")
            return

        narrated_steps: set = set()
        while True:
            try:
                data = _get_job_status(job_id)
            except httpx.HTTPError as exc:
                status.update(label="❌ Durum sorgulanamadi", state="error")
                st.error(f"Is durumu alinamadi: {exc}")
                return

            backend_step = data["step"]
            if backend_step not in narrated_steps and backend_step in _STAGE_NARRATION:
                for line in _STAGE_NARRATION[backend_step]:
                    st.write(line)
                narrated_steps.add(backend_step)

            status.update(
                label=f"{data['stage_name'] or 'Kuyrukta bekleniyor...'} "
                f"({backend_step}/{data['total_steps']})"
            )

            if data["status"] == "done":
                status.update(label="✅ Analiz tamamlandi", state="complete", expanded=False)
                st.session_state.last_report = data["result"]
                return
            if data["status"] == "error":
                status.update(label="❌ Analiz basarisiz oldu", state="error")
                st.error(f"Analiz basarisiz oldu: {data['error']}")
                return

            time.sleep(POLL_INTERVAL_SEC)


def _resolve_risk_badge(risk_level: str, risk_score: int) -> Tuple[str, str]:
    """Risk seviyesi/skoruna gore rozet etiketini ve rengini cozer.

    HTML rozet, HTML/PDF ozet raporlari gibi birden fazla tuketici tarafindan
    kullanilabilmesi icin renk/etiket cozumleme mantigi burada merkezilestirilir.

    Args:
        risk_level: Backend risk seviyesi (`dusuk`/`orta`/`yuksek`/`kritik`).
        risk_score: 0-100 arasi risk skoru (NORMAL esigini belirlemek icin).

    Returns:
        `(etiket, renk_kodu)` ikilisi (orn. `("CRITICAL", "#dc2626")`).
    """
    if risk_score <= 5:
        return _NORMAL_BADGE
    return _RISK_BADGE_MAP.get(risk_level, _NORMAL_BADGE)


def _risk_badge_html(risk_level: str, risk_score: int) -> str:
    """Risk seviyesine gore renkli bir HTML rozet dondurur.

    Args:
        risk_level: Backend risk seviyesi (`dusuk`/`orta`/`yuksek`/`kritik`).
        risk_score: 0-100 arasi risk skoru (NORMAL esigini belirlemek icin).

    Returns:
        `st.markdown(..., unsafe_allow_html=True)` ile basilabilecek HTML.
    """
    label, color = _resolve_risk_badge(risk_level, risk_score)
    return f'<span class="risk-badge" style="background-color:{color};">{label}</span>'


def _render_critical_alarm_banner(report: Dict[str, Any]) -> None:
    """Risk skoru >= 70 oldugunda yanip sonen kirmizi banner ve sesli bip uyarisi gosterir.

    Sesli uyari, ayni rapor icin yalnizca bir kez calinir (`st.session_state`
    ile takip edilir); boylece raporu degistirmeyen her Streamlit yeniden
    calistirmasinda (orn. baska bir sekmedeki butona tiklama) tekrar
    tetiklenmez.

    Args:
        report: Guncel `SafirReport` sozlugu.
    """
    if report["risk_score"] < CRITICAL_ALARM_THRESHOLD:
        return

    st.markdown(
        f'<div class="critical-alarm-banner">🚨 KRITIK RISK TESPIT EDILDI — '
        f'Risk Skoru: {report["risk_score"]}/100 ({report["risk_level"].upper()}) — '
        f"Acil operator mudahalesi gerekiyor!</div>",
        unsafe_allow_html=True,
    )

    report_signature = f"{report.get('event_id')}::{report.get('generated_at')}"
    if st.session_state.get("alarm_played_for") != report_signature:
        st.markdown(
            f'<audio autoplay><source src="{_CRITICAL_ALARM_AUDIO_DATA_URI}" type="audio/wav"></audio>',
            unsafe_allow_html=True,
        )
        st.session_state.alarm_played_for = report_signature


def _render_filter_banner(stats: Optional[Dict[str, Any]]) -> None:
    """CPU suzgec (Adaptive Sampler) GPU tasarruf istatistiklerini banner olarak gosterir.

    Args:
        stats: `SafirReport.sampler_stats` sozlugu (yoksa banner atlanir).
    """
    if not stats:
        st.info("Bu analiz icin suzgec istatistigi bulunamadi.")
        return

    st.markdown(
        f"""
        <div class="filter-banner">
            <b>⚡ CPU Suzgec & GPU Tasarruf Panosu</b><br/>
            Taranan ham kare: <b>{stats['total_frames_scanned']}</b> &nbsp;|&nbsp;
            Degerlendirilen ornek kare: <b>{stats['sampled_frames_evaluated']}</b> &nbsp;|&nbsp;
            Suzulen Kanit Karesi: <b>{stats['evidence_frame_count']}</b> &nbsp;|&nbsp;
            Elenen kare: <b>{stats['eliminated_frame_count']}</b><br/>
            <b>GPU Tasarruf Orani: %{stats['gpu_savings_ratio_pct']:.1f}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_metrics(report: Dict[str, Any]) -> None:
    """Saha KPI metrik kartlarini (ms/kare, CPU filtre orani, ihlal sayisi, aktif modeller) gosterir.

    Tum degerler backend'in gercekten urettigi `sampler_stats`/`evidence_frames`/
    `vlm_model`/`llm_model` alanlarindan hesaplanir; sabit/uydurma deger
    kullanilmaz.

    Args:
        report: Guncel `SafirReport` sozlugu.
    """
    st.subheader("📊 Saha KPI Metrikleri")
    stats = report.get("sampler_stats") or {}
    total_frames = stats.get("total_frames_scanned", 0)
    elapsed_sec = stats.get("elapsed_sec", 0.0)
    ms_per_frame = (elapsed_sec / total_frames * 1000.0) if total_frames else 0.0
    detected_event_count = len(report.get("evidence_frames", []))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kare Isleme Hizi", f"{ms_per_frame:.1f} ms/kare")
    col2.metric("CPU Filtre Orani", f"%{stats.get('gpu_savings_ratio_pct', 0.0):.1f}")
    col3.metric("Tespit Edilen Olay", f"{detected_event_count}")
    with col4:
        st.markdown("**Aktif Modeller**")
        st.markdown(f"🟢 VLM: `{report.get('vlm_model') or '-'}`")
        st.markdown(f"🟢 LLM: `{report.get('llm_model') or '-'}`")


def _send_feedback(event_id: int, feedback: str) -> None:
    """Operatorun Human-in-the-Loop dogrulamasini `/events/{event_id}/feedback`'e gonderir.

    Args:
        event_id: Bu analizin SQLite kaydi (`SafirReport.event_id`).
        feedback: `"true_positive"` veya `"false_positive"`.
    """
    try:
        response = httpx.post(
            f"{API_BASE_URL}/events/{event_id}/feedback",
            json={"feedback": feedback},
            timeout=HTTP_TIMEOUT_SEC,
        )
        response.raise_for_status()
        st.toast(response.json()["message"], icon="✅")
    except httpx.HTTPError as exc:
        st.error(f"Geri bildirim kaydedilemedi: {exc}")


def _render_evidence_gallery(report: Dict[str, Any]) -> None:
    """Zirve Kanıt Karelerini, her karti icin Dogru Ihbar/Yanlis Alarm butonlariyla birlikte gosterir.

    Not: Backend her analiz icin SQLite'a tek bir toplu olay kaydi yazar
    (`SafirReport.event_id`); dolayisiyla bu sekmedeki her kart ayni
    `event_id`'ye geri bildirim gonderir — kartlar arasindaki fark yalnizca
    hangi zirve karenin operatore gosterildigidir.

    Args:
        report: Guncel `SafirReport` sozlugu.
    """
    evidence_frames = report.get("evidence_frames", [])
    db_event_id = report.get("event_id")

    if not evidence_frames:
        st.info("Bu analiz icin kanit karesi uretilmedi.")
        return

    columns = st.columns(3)
    for i, evidence in enumerate(evidence_frames):
        with columns[i % 3]:
            try:
                _, b64_data = evidence["base64_image"].split(",", 1)
                image_bytes = base64.b64decode(b64_data)
                st.image(
                    image_bytes,
                    caption=(
                        f"Olay #{evidence['event_id']} | {evidence['timestamp_str']} | "
                        f"change_score={evidence['change_score']:.4f}"
                    ),
                    use_container_width=True,
                )
                if evidence.get("is_fallback"):
                    st.caption("⚠️ Fallback kare (esik gecen kare bulunamadi, frame 0 kullanildi)")
            except (KeyError, ValueError, base64.binascii.Error) as exc:
                st.warning(f"Kare goruntulenemedi: {exc}")
                continue

            if db_event_id is not None:
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    if st.button("✅ Dogru Ihbar", key=f"fb_true_{db_event_id}_{i}"):
                        _send_feedback(db_event_id, "true_positive")
                with fb_col2:
                    if st.button("❌ Yanlis Alarm", key=f"fb_false_{db_event_id}_{i}"):
                        _send_feedback(db_event_id, "false_positive")


def _render_agent_rag_tab(report: Dict[str, Any]) -> None:
    """VLM'in Turkce sahne tasvirini ve FAISS RAG mevzuat maddelerini gosterir.

    Args:
        report: Guncel `SafirReport` sozlugu.
    """
    st.subheader("🧠 vLLM Görsel Anlama Çıktısı (Turkce)")
    st.write(report["natural_language_summary"])

    st.subheader("📚 RAG & Mevzuat Karti (FAISS)")
    with st.container(border=True):
        regulations = report.get("relevant_regulations", [])
        if not regulations:
            st.caption("Bu analiz icin ilgili mevzuat maddesi bulunamadi.")
        else:
            st.caption("LangGraph Ajaninin `retriever_tool` uzerinden FAISS'ten getirdigi maddeler:")
            for regulation in regulations:
                st.markdown(f"- {regulation}")


def _render_timeline_alert_tab(report: Dict[str, Any]) -> None:
    """Olay Zaman Çizelgesini ve Human-in-the-Loop alarm tetikleme bolumunu gosterir.

    Args:
        report: Guncel `SafirReport` sozlugu.
    """
    st.subheader("🕒 Olay Zaman Çizelgesi")
    timeline = report.get("timeline", [])
    if not timeline:
        st.caption("Zaman Çizelgesinde Kayıt yok.")
    else:
        for entry in timeline:
            with st.container(border=True):
                st.write(f"`[{entry['timestamp']:.1f}s]` {entry['description']}")

    st.divider()
    st.subheader("🚨 Human-in-the-Loop: Saha Alarmi")
    risk_level = report["risk_level"]
    risk_score = report["risk_score"]

    if risk_level in ("yuksek", "kritik"):
        st.warning(f"Risk seviyesi **{risk_level.upper()}** (skor: {risk_score}/100) — operator onayi bekleniyor.")
    else:
        st.success(f"Risk seviyesi **{risk_level}** (skor: {risk_score}/100).")

    operator_note = st.text_input("Operator notu (opsiyonel)", key="operator_note")

    if st.button("✅ Riski Onayla ve Saha Alarmini Tetikle", type="primary"):
        try:
            response = httpx.post(
                f"{API_BASE_URL}/alerts/trigger",
                json={
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "recommended_action": report["recommended_action"],
                    "operator_note": operator_note,
                },
                timeout=HTTP_TIMEOUT_SEC,
            )
            response.raise_for_status()
            st.success(response.json()["message"])
        except httpx.HTTPError as exc:
            st.error(f"Alarm tetiklenemedi: {exc}")


def _report_file_stub(report: Dict[str, Any]) -> str:
    """Indirme dosya adlari icin Video Kaynağından guvenli bir dosya adi kokü uretir.

    Args:
        report: Guncel `SafirReport` sozlugu.

    Returns:
        Dosya sisteminde sorunsuz kullanilabilecek, `/` icermeyen bir ad.
    """
    safe_name = report["video_source"].replace("/", "_").replace("\\", "_")
    return f"safir_report_{safe_name}"


def _build_html_report(report: Dict[str, Any]) -> str:
    """Kanit goruntulerini gomulu olarak iceren, bagimsiz (self-contained) bir HTML ozet raporu uretir.

    Args:
        report: Guncel `SafirReport` sozlugu.

    Returns:
        Tarayicida dogrudan acilabilen veya yazdirilip PDF'e cevrilebilen HTML metni.
    """
    _, risk_color = _resolve_risk_badge(report["risk_level"], report["risk_score"])

    evidence_html = "".join(
        f'<div class="evidence-card">'
        f'<img src="{ef["base64_image"]}" alt="Olay #{ef["event_id"]}"/>'
        f'<p>Olay #{ef["event_id"]} | {ef["timestamp_str"]} | skor={ef["change_score"]:.4f}</p>'
        f"</div>"
        for ef in report.get("evidence_frames", [])
    ) or "<p>Kanit karesi yok.</p>"

    regulations_html = "".join(
        f"<li>{regulation}</li>" for regulation in report.get("relevant_regulations", [])
    ) or "<li>	İlgili mevzuat bulunamadı.</li>"

    timeline_html = "".join(
        f"<li>[{entry['timestamp']:.1f}s] {entry['description']}</li>"
        for entry in report.get("timeline", [])
    ) or "<li>Kayıt yok.</li>"

    return f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8"/>
<title>SAFIR Rapor - {report['video_source']}</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; margin: 2rem; color: #0f172a; background: #f8fafc; }}
  h1 {{ color: #1e3a5f; }}
  h2 {{ color: #1e3a5f; border-bottom: 2px solid #e2e8f0; padding-bottom: .3rem; }}
  .risk-badge {{ display: inline-block; padding: .4rem 1rem; border-radius: 999px; color: #fff; font-weight: 700; }}
  .section {{ margin-top: 1.5rem; background: #fff; padding: 1rem 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  .evidence-card {{ display: inline-block; margin: .5rem; text-align: center; vertical-align: top; }}
  .evidence-card img {{ max-width: 280px; border-radius: 8px; }}
</style>
</head>
<body>
<h1>🛡️ SAFIR Saha Analiz Raporu</h1>
<p><b>Video Kaynağı:</b> {report['video_source']}<br/>
<b>Üretim Zamanı:</b> {report['generated_at']}<br/>
<b>Aktif Modeller:</b> VLM={report.get('vlm_model') or '-'} / LLM={report.get('llm_model') or '-'}</p>

<div class="section">
<h2>Risk Degerlendirmesi</h2>
<p><b>Risk Skoru:</b> {report['risk_score']}/100 &nbsp;
<span class="risk-badge" style="background-color:{risk_color};">{report['risk_level'].upper()}</span></p>
<p><b>Önerilen Aksiyon:</b> {report['recommended_action']}</p>
</div>

<div class="section">
<h2>vLLM Görsel Anlama Çıktısı</h2>
<p>{report['natural_language_summary']}</p>
</div>

<div class="section">
<h2>Kanıt Kareleri</h2>
{evidence_html}
</div>

<div class="section">
<h2>	İlgili İSG Mevzuatı (FAISS RAG)</h2>
<ul>{regulations_html}</ul>
</div>

<div class="section">
<h2>Zaman Çizelgesi</h2>
<ul>{timeline_html}</ul>
</div>
</body>
</html>"""


def _build_pdf_report(report: Dict[str, Any]) -> bytes:
    """`reportlab` ile kanit goruntulerini iceren bicimli bir PDF ozet raporu uretir.

    Args:
        report: Guncel `SafirReport` sozlugu.

    Returns:
        PDF dosyasinin ham baytlari.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image as RLImage,
    )
    from reportlab.platypus import (
        ListFlowable,
        ListItem,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    _, risk_color = _resolve_risk_badge(report["risk_level"], report["risk_score"])

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    for _stil in styles.byName.values():
        _stil.fontName = TR_FONT_BOLD if ("Heading" in _stil.name or _stil.name == "Title") else TR_FONT
    risk_style = ParagraphStyle(
        "SafirRisk", parent=styles["BodyText"], textColor=colors.HexColor(risk_color), fontSize=14, spaceAfter=6
    )

    story = [
        Paragraph("SAFIR Saha Analiz Raporu", styles["Title"]),
        Spacer(1, 0.3 * cm),
        Paragraph(f"Video Kaynağı: {report['video_source']}", styles["BodyText"]),
        Paragraph(f"Üretim Zamanı: {report['generated_at']}", styles["BodyText"]),
        Spacer(1, 0.3 * cm),
        Paragraph(f"Risk Skoru: {report['risk_score']}/100 — {report['risk_level'].upper()}", risk_style),
        Paragraph(f"Önerilen Aksiyon: {report['recommended_action']}", styles["BodyText"]),
        Spacer(1, 0.4 * cm),
        Paragraph("vLLM Görsel Anlama Çıktısı", styles["Heading2"]),
        Paragraph(report["natural_language_summary"], styles["BodyText"]),
        Spacer(1, 0.4 * cm),
        Paragraph("Kanıt Kareleri", styles["Heading2"]),
    ]

    for evidence in report.get("evidence_frames", []):
        try:
            _, b64_data = evidence["base64_image"].split(",", 1)
            image_bytes = base64.b64decode(b64_data)
            story.append(RLImage(io.BytesIO(image_bytes), width=8 * cm, height=6 * cm))
            story.append(
                Paragraph(
                    f"Olay #{evidence['event_id']} | {evidence['timestamp_str']} | "
                    f"skor={evidence['change_score']:.4f}",
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 0.3 * cm))
        except (KeyError, ValueError, base64.binascii.Error):
            continue

    story.append(Paragraph("	İlgili İSG Mevzuatı (FAISS RAG)", styles["Heading2"]))
    regulations = report.get("relevant_regulations", [])
    if regulations:
        story.append(
            ListFlowable(
                [ListItem(Paragraph(r, styles["BodyText"])) for r in regulations], bulletType="bullet"
            )
        )
    else:
        story.append(Paragraph("	İlgili mevzuat bulunamadı.", styles["BodyText"]))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Zaman Çizelgesi", styles["Heading2"]))
    timeline = report.get("timeline", [])
    if timeline:
        story.append(
            ListFlowable(
                [
                    ListItem(Paragraph(f"[{e['timestamp']:.1f}s] {e['description']}", styles["BodyText"]))
                    for e in timeline
                ],
                bulletType="bullet",
            )
        )
    else:
        story.append(Paragraph("Kayıt yok.", styles["BodyText"]))

    doc.build(story)
    return buffer.getvalue()


def _render_json_report_tab(report: Dict[str, Any]) -> None:
    """Sartname uyumlu JSON raporunun onizlemesini ve JSON/HTML/PDF indirme secilerini gosterir.

    Args:
        report: Guncel `SafirReport` sozlugu.
    """
    st.subheader("📄 Sartname Uyumlu JSON Raporu")
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    st.code(report_json, language="json")

    file_stub = _report_file_stub(report)
    st.download_button(
        "⬇️ JSON Raporu Indir",
        data=report_json,
        file_name=f"{file_stub}.json",
        mime="application/json",
    )

    st.divider()
    st.subheader("📥 HTML/PDF Ozet Rapor Indir")
    col1, col2 = st.columns(2)
    with col1:
        html_report = _build_html_report(report)
        st.download_button(
            "📥 HTML Ozet Rapor Indir",
            data=html_report,
            file_name=f"{file_stub}.html",
            mime="text/html",
        )
    with col2:
        try:
            pdf_bytes = _build_pdf_report(report)
            st.download_button(
                "📥 PDF Ozet Rapor Indir",
                data=pdf_bytes,
                file_name=f"{file_stub}.pdf",
                mime="application/pdf",
            )
        except Exception as exc:  # noqa: BLE001 - reportlab eksikse/hata olursa panel cokmesin
            st.warning(f"PDF raporu olusturulamadi: {exc}")


def _render_report(report: Dict[str, Any]) -> None:
    """Tamamlanmis bir `SafirReport`'u banner + rozet + 4 sekmeli panel olarak render eder.

    Args:
        report: `/analyze/jobs/{job_id}` uzerinden gelen nihai rapor sozlugu.
    """
    st.divider()
    st.header("📋 Analiz Sonucu")

    _render_critical_alarm_banner(report)
    _render_filter_banner(report.get("sampler_stats"))
    _render_kpi_metrics(report)

    col1, col2, col3 = st.columns([1, 1, 2])
    col1.metric("Risk Skoru", f"{report['risk_score']}/100")
    with col2:
        st.markdown("**Risk Seviyesi**")
        st.markdown(_risk_badge_html(report["risk_level"], report["risk_score"]), unsafe_allow_html=True)
    col3.metric("VLM / LLM", f"{report.get('vlm_model', '-')} / {report.get('llm_model', '-')}")

    st.markdown("**Önerilen Aksiyon**")
    st.info(report["recommended_action"])

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📸 Süzülen Kanıt Kareleri",
            "🧠 Ajan Düşünme & İSG RAG Bağlamı",
            "📊 Olay Çizelgesi & Saha Alarmı",
            "📄 Şartname JSON Raporu",
        ]
    )
    with tab1:
        _render_evidence_gallery(report)
    with tab2:
        _render_agent_rag_tab(report)
    with tab3:
        _render_timeline_alert_tab(report)
    with tab4:
        _render_json_report_tab(report)


def _render_sidebar() -> Dict[str, Any]:
    """Yan menudeki veri kaynagi secimi, sampler slider'lari ve saglik durumunu render eder.

    Returns:
        `{"input_mode": str, "sample_fps": int, "min_change_threshold": float}` sozlugu.
    """
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
        if _check_backend_health():
            st.success(f"API & vLLM ayakta: `{API_BASE_URL}`")
        else:
            st.error(f"Backend'e ulasilamiyor: `{API_BASE_URL}`")

        return {
            "input_mode": input_mode,
            "sample_fps": sample_fps,
            "min_change_threshold": min_change_threshold,
        }


def main() -> None:
    """Streamlit operator panelinin ana giris noktasi."""
    st.title("🛡️ SAFIR — Saha Analiz ve Farkındalık Operatör Paneli")

    if "last_report" not in st.session_state:
        st.session_state.last_report = None

    sidebar = _render_sidebar()

    st.header("📥 Girdi Paneli")
    video_source: Optional[str] = None
    if sidebar["input_mode"] == "📹 Video Dosyası Sürükle":
        uploaded_file = st.file_uploader(
            "Video dosyasini surukle veya sec", type=ALLOWED_VIDEO_EXTENSIONS
        )
        if uploaded_file is not None:
            video_source = _save_uploaded_file(uploaded_file)
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

    start_disabled = not video_source
    if st.button("🚀 Pipeline & Ajan Düşünme Sürecini Başlat", type="primary", disabled=start_disabled):
        _run_pipeline_with_live_status(
            video_source,
            user_prompt,
            sidebar["sample_fps"],
            sidebar["min_change_threshold"],
        )

    if st.session_state.last_report:
        _render_report(st.session_state.last_report)


if __name__ == "__main__":
    main()
render_history_section()
