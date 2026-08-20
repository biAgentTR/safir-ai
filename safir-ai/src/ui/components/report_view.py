"""Operator Paneli - rapor gorunumu bilesenleri (banner, KPI, kanit, RAG, eskalasyon, JSON).

Her panel tek bir sorumluluk tasir; `ReportView` bunlari 4 sekmeli bir duzende
orkestreye eder. Human-in-the-Loop onay KAPISI kaldirilmistir: eskalasyon paneli,
yuksek/kritik risk icin alarmin OTOMATIK tetiklendigini gosterir ve operatore
yalnizca sonradan ONAYLAMA (Human-on-the-Loop) imkani sunar.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List

import httpx
import streamlit as st

from src.ui.api_client import SafirApiClient
from src.ui.report_export import ReportExporter
from src.ui.theme import (
    CRITICAL_ALARM_AUDIO_DATA_URI,
    CRITICAL_ALARM_THRESHOLD,
    resolve_risk_badge,
    risk_badge_html,
)


def _actions_of(report: Dict[str, Any]) -> List[str]:
    """Raporun aksiyon listesini dondurur (yoksa `recommended_action`'a duser)."""
    actions = report.get("actions") or []
    if not actions and report.get("recommended_action"):
        actions = [report["recommended_action"]]
    return actions


class CriticalAlarmBanner:
    """Risk skoru esigi asildiginda yanip sonen kirmizi banner + tek seferlik sesli uyari."""

    def render(self, report: Dict[str, Any]) -> None:
        if report["risk_score"] < CRITICAL_ALARM_THRESHOLD:
            return

        st.markdown(
            f'<div class="critical-alarm-banner">🚨 KRITIK RISK TESPIT EDILDI — '
            f'Risk Skoru: {report["risk_score"]}/100 ({report["risk_level"].upper()}) — '
            f"Otomatik saha alarmi tetiklendi!</div>",
            unsafe_allow_html=True,
        )

        report_signature = f"{report.get('event_id')}::{report.get('generated_at')}"
        if st.session_state.get("alarm_played_for") != report_signature:
            st.markdown(
                f'<audio autoplay><source src="{CRITICAL_ALARM_AUDIO_DATA_URI}" type="audio/wav"></audio>',
                unsafe_allow_html=True,
            )
            st.session_state.alarm_played_for = report_signature


class FilterBanner:
    """CPU suzgec (Adaptive Sampler) GPU tasarruf istatistiklerini banner olarak gosterir."""

    def render(self, stats: Dict[str, Any]) -> None:
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


class KpiPanel:
    """Saha KPI metrik kartlarini (ms/kare, CPU filtre orani, olay sayisi, aktif modeller) gosterir."""

    def render(self, report: Dict[str, Any]) -> None:
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


class EvidenceGallery:
    """Zirve kanit karelerini, her kart icin Dogru Ihbar/Yanlis Alarm butonlariyla gosterir."""

    def __init__(self, api_client: SafirApiClient) -> None:
        self._api = api_client

    def render(self, report: Dict[str, Any]) -> None:
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
                            self._feedback(db_event_id, "true_positive")
                    with fb_col2:
                        if st.button("❌ Yanlis Alarm", key=f"fb_false_{db_event_id}_{i}"):
                            self._feedback(db_event_id, "false_positive")

    def _feedback(self, event_id: int, feedback: str) -> None:
        try:
            result = self._api.send_feedback(event_id, feedback)
            st.toast(result["message"], icon="✅")
        except httpx.HTTPError as exc:
            st.error(f"Geri bildirim kaydedilemedi: {exc}")


class AgentRagPanel:
    """Ajan ozetini, VLM ham gozlemini ve FAISS RAG mevzuat maddelerini gosterir."""

    def render(self, report: Dict[str, Any]) -> None:
        summary = report.get("summary")
        if summary:
            st.subheader("📝 Operator Ozeti (Ajan)")
            st.write(summary)

        st.subheader("🧠 VLM Gorsel Anlama Ciktisi (Turkce)")
        st.write(report["natural_language_summary"])

        st.subheader("🏷️ Olay Kanit Ifadeleri (VLM-uretimi, serbest bicimli)")
        with st.container(border=True):
            event_keywords = report.get("event_keywords") or []
            if not event_keywords:
                st.caption("Bu analiz icin VLM kanit ifadesi (keyword) uretilmedi.")
            else:
                st.caption(
                    "Bu ifadeler VLM'in evidence karelerinde GERCEKTEN gozlemledigi, "
                    "onceden tanimli bir taksonomiyle SINIRLI OLMAYAN serbest kanitlardir "
                    "- risk KARARI DEGILDIR (risk RuleEngine'den gelir)."
                )
                for entry in event_keywords:
                    keywords = entry.get("keywords") or []
                    if not keywords:
                        continue
                    badges = " ".join(f"`{kw}`" for kw in keywords)
                    st.markdown(f"**{entry.get('event_type', '?')}**: {badges}")

        st.subheader("📚 RAG & Mevzuat Karti (FAISS)")
        with st.container(border=True):
            regulations = report.get("relevant_regulations", [])
            if not regulations:
                st.caption(
                    "Mevzuat eşleştirilemedi. Gerekçe: bu analiz için RuleEngine tarafından "
                    "doğrulanmış, güvenilir bir İSG mevzuat eşleşmesi bulunamadı."
                )
            else:
                st.caption("RuleEngine tarafindan deterministik olarak dogrulanmis mevzuat maddeleri:")
                for regulation in regulations:
                    st.markdown(f"- {regulation}")


class TimelineEscalationPanel:
    """Olay zaman cizelgesini + OTOMATIK eskalasyon durumunu ve operator onayini gosterir.

    Human-in-the-Loop onay KAPISI yoktur: yuksek/kritik risk icin alarm zaten
    otomatik tetiklenmistir; operator burada onu yalnizca ONAYLAR/DENETLER.
    """

    def __init__(self, api_client: SafirApiClient) -> None:
        self._api = api_client

    def render(self, report: Dict[str, Any]) -> None:
        st.subheader("🕒 Olay Zaman Cizelgesi")
        timeline = report.get("timeline", [])
        if not timeline:
            st.caption("Zaman cizelgesinde kayit yok.")
        else:
            for entry in timeline:
                with st.container(border=True):
                    st.write(f"`[{entry['timestamp']:.1f}s]` {entry['description']}")

        st.divider()
        self._render_escalation(report)

    def _render_escalation(self, report: Dict[str, Any]) -> None:
        st.subheader("🚨 Otomatik Eskalasyon (Human-on-the-Loop)")
        tier = report.get("escalation_tier")
        auto_dispatched = report.get("auto_dispatched", False)
        alert_id = report.get("alert_id")
        risk_level = report["risk_level"]
        risk_score = report["risk_score"]

        if auto_dispatched and alert_id:
            st.error(
                f"🚨 Saha alarmi OTOMATIK tetiklendi — risk **{risk_level.upper()}** "
                f"(skor: {risk_score}/100). Operator onayi BEKLENMEDI.\n\n`alert_id={alert_id}`"
            )
            operator_note = st.text_input("Operator denetim notu (opsiyonel)", key="ack_note")
            if st.button("👁️ Alarmi Denetle / Onayla", type="primary"):
                try:
                    result = self._api.acknowledge_alert(alert_id, operator_note)
                    st.success(result["message"])
                except httpx.HTTPError as exc:
                    st.error(f"Alarm onaylanamadi: {exc}")
        elif tier == "notify":
            st.warning(
                f"Bildirim: risk **{risk_level.upper()}** (skor: {risk_score}/100). "
                "Otomatik alarm esigi asilmadi; izleme surdurulur."
            )
        else:
            st.success(
                f"Izlemeye devam: risk **{risk_level}** (skor: {risk_score}/100). Eskalasyon gerekmiyor."
            )

        with st.expander("Manuel alarm tetikle (override)"):
            st.caption("Otomatik akis disinda, operatorun manuel saha alarmi baslatmasi icin.")
            note = st.text_input("Manuel alarm notu", key="manual_alert_note")
            if st.button("Manuel Saha Alarmi Tetikle"):
                actions = _actions_of(report)
                try:
                    result = self._api.trigger_manual_alert(
                        risk_score=risk_score,
                        risk_level=risk_level,
                        recommended_action=actions[0] if actions else "",
                        operator_note=note,
                    )
                    st.success(result["message"])
                except httpx.HTTPError as exc:
                    st.error(f"Manuel alarm tetiklenemedi: {exc}")


class JsonReportPanel:
    """Sartname-uyumlu ozet JSON'u, tam SafirReport JSON'unu ve HTML/PDF indirmeyi gosterir."""

    @staticmethod
    def _sartname_view(report: Dict[str, Any]) -> Dict[str, Any]:
        """Raporu sartnamedeki `{summary, events, risk, actions}` sekline indirger."""
        def mmss(seconds: float) -> str:
            total = int(round(seconds))
            return f"{total // 60:02d}:{total % 60:02d}"

        return {
            "summary": report.get("summary") or report.get("natural_language_summary", ""),
            "events": [
                {"time": mmss(e["timestamp"]), "event": e["description"]}
                for e in report.get("timeline", [])
            ],
            "risk": report["risk_level"],
            "risk_score": report["risk_score"],
            "actions": _actions_of(report),
        }

    def render(self, report: Dict[str, Any]) -> None:
        exporter = ReportExporter(report)

        st.subheader("📄 Sartname Uyumlu Ozet JSON")
        st.code(json.dumps(self._sartname_view(report), ensure_ascii=False, indent=2), language="json")

        with st.expander("Tam SafirReport JSON (superset)"):
            st.code(json.dumps(report, ensure_ascii=False, indent=2), language="json")

        st.download_button(
            "⬇️ Tam JSON Raporu Indir",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name=f"{exporter.file_stub}.json",
            mime="application/json",
        )

        st.divider()
        st.subheader("📥 HTML/PDF Ozet Rapor Indir")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 HTML Ozet Rapor Indir",
                data=exporter.to_html(),
                file_name=f"{exporter.file_stub}.html",
                mime="text/html",
            )
        with col2:
            try:
                st.download_button(
                    "📥 PDF Ozet Rapor Indir",
                    data=exporter.to_pdf(),
                    file_name=f"{exporter.file_stub}.pdf",
                    mime="application/pdf",
                )
            except Exception as exc:  # noqa: BLE001 - reportlab eksikse panel cokmesin
                st.warning(f"PDF raporu olusturulamadi: {exc}")


class ReportView:
    """Tamamlanmis bir `SafirReport`'u banner + rozet + 4 sekmeli panel olarak orkestreye eder."""

    def __init__(self, api_client: SafirApiClient) -> None:
        self._critical_banner = CriticalAlarmBanner()
        self._filter_banner = FilterBanner()
        self._kpi = KpiPanel()
        self._evidence = EvidenceGallery(api_client)
        self._agent_rag = AgentRagPanel()
        self._timeline = TimelineEscalationPanel(api_client)
        self._json = JsonReportPanel()

    def render(self, report: Dict[str, Any]) -> None:
        st.divider()
        st.header("📋 Analiz Sonucu")

        self._critical_banner.render(report)
        self._filter_banner.render(report.get("sampler_stats"))
        self._kpi.render(report)

        col1, col2, col3 = st.columns([1, 1, 2])
        col1.metric("Risk Skoru", f"{report['risk_score']}/100")
        with col2:
            st.markdown("**Risk Seviyesi**")
            st.markdown(risk_badge_html(report["risk_level"], report["risk_score"]), unsafe_allow_html=True)
        col3.metric("VLM / LLM", f"{report.get('vlm_model', '-')} / {report.get('llm_model', '-')}")

        st.markdown("**Onerilen Aksiyonlar**")
        actions = _actions_of(report)
        if actions:
            for action in actions:
                st.markdown(f"- {action}")
        else:
            st.info("Aksiyon onerisi uretilmedi.")

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📸 Süzülen Kanıt Kareleri",
                "🧠 Ajan Düşünme & İSG RAG Bağlamı",
                "📊 Olay Çizelgesi & Otomatik Eskalasyon",
                "📄 Şartname JSON Raporu",
            ]
        )
        with tab1:
            self._evidence.render(report)
        with tab2:
            self._agent_rag.render(report)
        with tab3:
            self._timeline.render(report)
        with tab4:
            self._json.render(report)
