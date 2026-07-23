"""SAFIR Operator Paneli: Streamlit tabanli gorsel destekli gercek zamanli arayuz.

Bu panel, `src/main.py` FastAPI servisinin `/analyze/jobs` (asenkron is
kuyrugu), `/analyze/jobs/{job_id}` (canli asama sorgulama) ve `/alerts/trigger`
(Human-in-the-Loop saha alarmi) uc noktalarini tuketir. Operator; bir video
dosyasi yukleyebilir veya bir RTSP/HTTP canli yayin adresi girebilir,
ozellestirilmis bir inceleme istemi yazabilir, VLM Oncesi Katman -> VLM ->
LangGraph Ajan asamalarinin canli ilerlemesini izleyebilir, zirve kanit
karelerini ve FAISS RAG'dan gelen ilgili ISG mevzuatini gorebilir, yuksek
riskli durumlari onaylayip saha alarmini tetikleyebilir ve nihai raporu JSON
olarak indirebilir.

Calistirma:
    streamlit run src/ui/dashboard.py
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("SAFIR_API_URL", "http://localhost:8000")
DATA_DIR = Path(os.environ.get("SAFIR_DATA_DIR", "data"))
ALLOWED_VIDEO_EXTENSIONS = ["mp4", "avi", "mkv"]
POLL_INTERVAL_SEC = 1.0
HTTP_TIMEOUT_SEC = 30.0

st.set_page_config(page_title="SAFIR Operator Paneli", page_icon="🛡️", layout="wide")


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


def _create_analyze_job(video_source: str, user_prompt: str) -> str:
    """`/analyze/jobs` uc noktasina POST atarak arka planda bir analiz isi baslatir.

    Args:
        video_source: Video dosya adi/yolu veya canli yayin URI'si.
        user_prompt: Operatorun ozel inceleme istemi.

    Returns:
        Olusturulan isin kimligi.

    Raises:
        httpx.HTTPError: Istek basarisiz olursa.
    """
    response = httpx.post(
        f"{API_BASE_URL}/analyze/jobs",
        json={"video_source": video_source, "user_prompt": user_prompt},
        timeout=HTTP_TIMEOUT_SEC,
    )
    response.raise_for_status()
    return response.json()["job_id"]


def _poll_analyze_job(job_id: str, progress_bar, stage_placeholder) -> Optional[Dict[str, Any]]:
    """Bir analiz isini tamamlanana kadar sorgulayip ilerleme cubugunu gunceller.

    Args:
        job_id: `_create_analyze_job` tarafindan donen is kimligi.
        progress_bar: Guncellenecek `st.progress` bileseni.
        stage_placeholder: Asama metninin yazilacagi `st.empty` bilesigi.

    Returns:
        Basariliysa nihai `SafirReport` sozlugu, hata olursa `None`.
    """
    while True:
        response = httpx.get(f"{API_BASE_URL}/analyze/jobs/{job_id}", timeout=HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        data = response.json()

        step, total = data["step"], data["total_steps"] or 3
        pct = int(100 * step / total) if total else 0
        label = f"{step}/{total} - {data['stage_name'] or 'Kuyrukta bekleniyor...'}"
        progress_bar.progress(min(max(pct, 0), 100), text=label)
        stage_placeholder.caption(f"Durum: `{data['status']}`")

        if data["status"] == "done":
            progress_bar.progress(100, text=f"{total}/{total} - Tamamlandi")
            return data["result"]
        if data["status"] == "error":
            st.error(f"Analiz basarisiz oldu: {data['error']}")
            return None

        time.sleep(POLL_INTERVAL_SEC)


def _render_evidence_frames(evidence_frames: list) -> None:
    """Zirve kanit karelerini resim kartlari olarak (zaman damgasi + skorla) gosterir.

    Args:
        evidence_frames: `SafirReport.evidence_frames` listesi (dict olarak).
    """
    st.subheader("🖼️ Gorsel Kanit Kareleri (Zirve Kareler)")
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


def _render_rag_card(relevant_regulations: list) -> None:
    """FAISS RAG'dan getirilen ilgili ISG mevzuat maddelerini seffaf bir kartta gosterir.

    Args:
        relevant_regulations: `SafirReport.relevant_regulations` listesi.
    """
    st.subheader("📚 RAG & Mevzuat Karti (FAISS)")
    with st.container(border=True):
        if not relevant_regulations:
            st.caption("Bu analiz icin ilgili mevzuat maddesi bulunamadi.")
        else:
            st.caption("LangGraph Ajaninin `retriever_tool` uzerinden FAISS'ten getirdigi maddeler:")
            for regulation in relevant_regulations:
                st.markdown(f"- {regulation}")


def _render_human_in_the_loop(report: Dict[str, Any]) -> None:
    """Operatorun riski onaylayip saha alarmini tetikleyebilecegi bolumu render eder.

    Args:
        report: Guncel `SafirReport` sozlugu.
    """
    st.subheader("🚨 Human-in-the-Loop: Karar Onayi")
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


def _render_report(report: Dict[str, Any]) -> None:
    """Tamamlanmis bir `SafirReport`'u tum bolumleriyle (ozet, kanitlar, RAG, HITL, indirme) render eder.

    Args:
        report: `/analyze/jobs/{job_id}` uzerinden gelen nihai rapor sozlugu.
    """
    st.divider()
    st.header("📋 Analiz Sonucu")

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Skoru", f"{report['risk_score']}/100")
    col2.metric("Risk Seviyesi", report["risk_level"].upper())
    col3.metric("VLM / LLM", f"{report.get('vlm_model', '-')} / {report.get('llm_model', '-')}")

    st.markdown("**Turkce Dogal Dil Ozeti**")
    st.write(report["natural_language_summary"])

    st.markdown("**Onerilen Aksiyon**")
    st.info(report["recommended_action"])

    _render_evidence_frames(report.get("evidence_frames", []))
    _render_rag_card(report.get("relevant_regulations", []))

    st.subheader("🕒 Zaman Cizelgesi")
    timeline = report.get("timeline", [])
    if not timeline:
        st.caption("Zaman cizelgesinde kayit yok.")
    else:
        for entry in timeline:
            st.write(f"`[{entry['timestamp']:.1f}s]` {entry['description']}")

    _render_human_in_the_loop(report)

    st.subheader("⬇️ Rapor Indirme")
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    st.download_button(
        "JSON Raporu Indir",
        data=report_json,
        file_name=f"safir_report_{report['video_source'].replace('/', '_')}.json",
        mime="application/json",
    )


def main() -> None:
    """Streamlit operator panelinin ana giris noktasi."""
    st.title("🛡️ SAFIR — Saha Analiz ve Farkindalik Operator Paneli")

    if _check_backend_health():
        st.caption("🟢 Backend baglantisi: `%s` (saglikli)" % API_BASE_URL)
    else:
        st.caption("🔴 Backend'e ulasilamiyor: `%s`" % API_BASE_URL)

    if "last_report" not in st.session_state:
        st.session_state.last_report = None

    st.header("1️⃣ Video Girdisi")
    input_mode = st.radio(
        "Girdi Modu",
        ["📁 Dosya Yukle", "📡 Canli Yayin (RTSP/HTTP)"],
        horizontal=True,
    )

    video_source: Optional[str] = None
    if input_mode == "📁 Dosya Yukle":
        uploaded_file = st.file_uploader(
            "Video dosyasi sec", type=ALLOWED_VIDEO_EXTENSIONS
        )
        if uploaded_file is not None:
            video_source = _save_uploaded_file(uploaded_file)
            st.success(f"Dosya kaydedildi: `data/{video_source}`")
    else:
        video_source = st.text_input(
            "RTSP/HTTP canli yayin adresi",
            placeholder="rtsp://192.168.1.10:554/stream veya http://kamera-ip/video",
        )

    st.header("2️⃣ Operator Istemi")
    user_prompt = st.text_area(
        "Ozellestirilmis inceleme istemi",
        value="Sahnede baret veya yelek takmayan personel ya da duman tespiti yap.",
        height=80,
    )

    st.header("3️⃣ Analiz")
    start_disabled = not video_source
    if st.button("🔍 Analizi Baslat", type="primary", disabled=start_disabled):
        progress_bar = st.progress(0, text="Kuyruga aliniyor...")
        stage_placeholder = st.empty()
        try:
            job_id = _create_analyze_job(video_source, user_prompt)
            result = _poll_analyze_job(job_id, progress_bar, stage_placeholder)
            st.session_state.last_report = result
        except httpx.HTTPError as exc:
            st.error(f"Analiz baslatilamadi: {exc}")

    if st.session_state.last_report:
        _render_report(st.session_state.last_report)


if __name__ == "__main__":
    main()
