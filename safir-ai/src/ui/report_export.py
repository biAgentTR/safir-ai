"""Operator Paneli - rapor disa aktarimi (HTML & PDF) tek bir sinifta.

`ReportExporter`, bir `SafirReport` sozlugunu bagimsiz (self-contained) HTML veya
PDF ozet raporuna cevirir. Kanit goruntuleri gomulu (base64) tasinir; ozet ve
aksiyon listesi (sartname ile hizali) rapora dahildir.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

from src.ui.theme import resolve_risk_badge


class ReportExporter:
    """Bir `SafirReport` sozlugunu HTML/PDF ozet raporuna ceviren yardimci."""

    def __init__(self, report: Dict[str, Any]) -> None:
        """Disa aktariciyi bir rapor sozlugu ile baslatir.

        Args:
            report: `/analyze/jobs/{job_id}` uzerinden gelen `SafirReport` sozlugu.
        """
        self._report = report

    @property
    def file_stub(self) -> str:
        """Indirme dosya adlari icin, video kaynagindan guvenli bir kok uretir."""
        safe_name = self._report["video_source"].replace("/", "_").replace("\\", "_")
        return f"safir_report_{safe_name}"

    def _actions(self) -> List[str]:
        """Aksiyon listesini dondurur (yoksa `recommended_action`'a duser)."""
        actions = self._report.get("actions") or []
        if not actions and self._report.get("recommended_action"):
            actions = [self._report["recommended_action"]]
        return actions

    def to_html(self) -> str:
        """Kanit goruntulerini gomulu iceren, bagimsiz bir HTML ozet raporu uretir."""
        report = self._report
        _, risk_color = resolve_risk_badge(report["risk_level"], report["risk_score"])

        evidence_html = "".join(
            f'<div class="evidence-card">'
            f'<img src="{ef["base64_image"]}" alt="Olay #{ef["event_id"]}"/>'
            f'<p>Olay #{ef["event_id"]} | {ef["timestamp_str"]} | skor={ef["change_score"]:.4f}</p>'
            f"</div>"
            for ef in report.get("evidence_frames", [])
        ) or "<p>Kanit karesi yok.</p>"

        actions_html = "".join(f"<li>{a}</li>" for a in self._actions()) or "<li>Aksiyon onerisi yok.</li>"

        regulations_html = "".join(
            f"<li>{regulation}</li>" for regulation in report.get("relevant_regulations", [])
        ) or "<li>Ilgili mevzuat bulunamadi.</li>"

        timeline_html = "".join(
            f"<li>[{entry['timestamp']:.1f}s] {entry['description']}</li>"
            for entry in report.get("timeline", [])
        ) or "<li>Kayit yok.</li>"

        summary = report.get("summary") or report.get("natural_language_summary", "")

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
<p><b>Video Kaynagi:</b> {report['video_source']}<br/>
<b>Uretim Zamani:</b> {report['generated_at']}<br/>
<b>Aktif Modeller:</b> VLM={report.get('vlm_model') or '-'} / LLM={report.get('llm_model') or '-'}</p>

<div class="section">
<h2>Ozet</h2>
<p>{summary}</p>
</div>

<div class="section">
<h2>Risk Degerlendirmesi</h2>
<p><b>Risk Skoru:</b> {report['risk_score']}/100 &nbsp;
<span class="risk-badge" style="background-color:{risk_color};">{report['risk_level'].upper()}</span>
&nbsp; <b>Eskalasyon:</b> {report.get('escalation_tier') or '-'}
{'(otomatik alarm tetiklendi)' if report.get('auto_dispatched') else ''}</p>
<p><b>Onerilen Aksiyonlar:</b></p>
<ul>{actions_html}</ul>
</div>

<div class="section">
<h2>VLM Gorsel Anlama Ciktisi</h2>
<p>{report['natural_language_summary']}</p>
</div>

<div class="section">
<h2>Kanit Kareleri</h2>
{evidence_html}
</div>

<div class="section">
<h2>Ilgili ISG Mevzuati (FAISS RAG)</h2>
<ul>{regulations_html}</ul>
</div>

<div class="section">
<h2>Zaman Cizelgesi</h2>
<ul>{timeline_html}</ul>
</div>
</body>
</html>"""

    def to_pdf(self) -> bytes:
        """`reportlab` ile kanit goruntulerini iceren bicimli bir PDF ozet raporu uretir."""
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

        report = self._report
        _, risk_color = resolve_risk_badge(report["risk_level"], report["risk_score"])

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
        styles = getSampleStyleSheet()
        risk_style = ParagraphStyle(
            "SafirRisk", parent=styles["BodyText"], textColor=colors.HexColor(risk_color), fontSize=14, spaceAfter=6
        )
        summary = report.get("summary") or report.get("natural_language_summary", "")

        story = [
            Paragraph("SAFIR Saha Analiz Raporu", styles["Title"]),
            Spacer(1, 0.3 * cm),
            Paragraph(f"Video Kaynagi: {report['video_source']}", styles["BodyText"]),
            Paragraph(f"Uretim Zamani: {report['generated_at']}", styles["BodyText"]),
            Spacer(1, 0.3 * cm),
            Paragraph("Ozet", styles["Heading2"]),
            Paragraph(summary, styles["BodyText"]),
            Spacer(1, 0.3 * cm),
            Paragraph(f"Risk Skoru: {report['risk_score']}/100 — {report['risk_level'].upper()}", risk_style),
            Paragraph(
                f"Eskalasyon: {report.get('escalation_tier') or '-'}"
                + (" (otomatik alarm tetiklendi)" if report.get("auto_dispatched") else ""),
                styles["BodyText"],
            ),
        ]

        actions = self._actions()
        story.append(Paragraph("Onerilen Aksiyonlar", styles["Heading2"]))
        if actions:
            story.append(
                ListFlowable([ListItem(Paragraph(a, styles["BodyText"])) for a in actions], bulletType="bullet")
            )
        else:
            story.append(Paragraph("Aksiyon onerisi yok.", styles["BodyText"]))

        story += [
            Spacer(1, 0.4 * cm),
            Paragraph("VLM Gorsel Anlama Ciktisi", styles["Heading2"]),
            Paragraph(report["natural_language_summary"], styles["BodyText"]),
            Spacer(1, 0.4 * cm),
            Paragraph("Kanit Kareleri", styles["Heading2"]),
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

        story.append(Paragraph("Ilgili ISG Mevzuati (FAISS RAG)", styles["Heading2"]))
        regulations = report.get("relevant_regulations", [])
        if regulations:
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(r, styles["BodyText"])) for r in regulations], bulletType="bullet"
                )
            )
        else:
            story.append(Paragraph("Ilgili mevzuat bulunamadi.", styles["BodyText"]))

        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("Zaman Cizelgesi", styles["Heading2"]))
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
            story.append(Paragraph("Kayit yok.", styles["BodyText"]))

        doc.build(story)
        return buffer.getvalue()
