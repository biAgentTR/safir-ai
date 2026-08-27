"""Operator Paneli - rapor disa aktarimi (HTML & PDF) tek bir sinifta.

`ReportExporter`, bir `SafirReport` sozlugunu bagimsiz (self-contained) HTML veya
PDF ozet raporuna cevirir. Kanit goruntuleri gomulu (base64) tasinir; ozet ve
aksiyon listesi (sartname ile hizali) rapora dahildir.

Turkce karakter notu: `to_pdf()`, reportlab'in varsayilan 14 standart fontunu
(Helvetica/Times - WinAnsiEncoding/cp1252) KULLANMAZ. cp1252 kodlamasinda
"ş", "ğ", "ı" ve "İ" harfleri BULUNMAZ; bu yuzden varsayilan fontlarla uretilen
bir PDF'te bu dort harf ya bosluk, ya farkli bir glif, ya da hic gorunmez
olurdu. Bunun yerine (bkz. `_register_fonts`) serbestce dagitilabilir, Latin
Extended-A (Turkce dahil) destekleyen bir Unicode TTF (DejaVu Sans, `assets/
fonts/`) kayit edilir ve TUM stiller bu fonta yonlendirilir.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List

from src.ui.theme import resolve_risk_badge

# SAFIR masaustu uygulamasiyla ayni marka rengi (bkz. desktop/app/assets/css/
# main.css --c-accent) - rapor disa aktarimlarinin uygulamanin geri kalaniyla
# GORSEL OLARAK TUTARLI olmasi icin.
_ACCENT = "#0d9488"
_ACCENT_DARK = "#0f172a"
_MUTED = "#64748b"
_BORDER = "#e2e8f0"

_FONTS_DIR = Path(__file__).parent / "assets" / "fonts"
_FONT_REGULAR = "SafirSans"
_FONT_BOLD = "SafirSans-Bold"
_fonts_registered = False


def _register_pdf_fonts() -> str:
    """Turkce-uyumlu bir Unicode TTF fontu reportlab'e kayit eder (tek seferlik).

    Bundle edilmis DejaVu Sans bulunamazsa (beklenmeyen bir dagitim hatasi),
    PDF uretimi COKMEZ - varsayilan Helvetica'ya geri duser (ş/ğ/ı/İ o
    durumda hatali gorunebilir, ama rapor yine de uretilir).

    Returns:
        Kullanilacak normal-agirlik font adi (`SafirSans` veya `Helvetica`).
    """
    global _fonts_registered
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if _fonts_registered:
        return _FONT_REGULAR
    regular_path = _FONTS_DIR / "DejaVuSans.ttf"
    bold_path = _FONTS_DIR / "DejaVuSans-Bold.ttf"
    if not (regular_path.exists() and bold_path.exists()):
        return "Helvetica"
    pdfmetrics.registerFont(TTFont(_FONT_REGULAR, str(regular_path)))
    pdfmetrics.registerFont(TTFont(_FONT_BOLD, str(bold_path)))
    pdfmetrics.registerFontFamily(_FONT_REGULAR, normal=_FONT_REGULAR, bold=_FONT_BOLD)
    _fonts_registered = True
    return _FONT_REGULAR


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

    def _risk_score_text(self) -> str:
        """Risk skoru metnini uretir; `risk_score=None` (risk_status='unknown') icin ASLA '.../100' gostermez."""
        report = self._report
        score = report.get("risk_score")
        if score is None or report.get("risk_status") == "unknown":
            return "Belirsiz (analiz güvenilir şekilde tamamlanamadı — manuel inceleme gerekli)"
        return f"{score}/100"

    def to_html(self) -> str:
        """Kanit goruntulerini gomulu iceren, bagimsiz, modern bir HTML ozet raporu uretir."""
        report = self._report
        _, risk_color = resolve_risk_badge(report["risk_level"], report["risk_score"])

        evidence_html = "".join(
            f'<figure class="evidence-card">'
            f'<img src="{ef["base64_image"]}" alt="Olay #{ef["event_id"]}"/>'
            f'<figcaption>Olay #{ef["event_id"]} · {ef["timestamp_str"]} · skor {ef["change_score"]:.4f}</figcaption>'
            f"</figure>"
            for ef in report.get("evidence_frames", [])
        ) or '<p class="empty">Kanıt karesi yok.</p>'

        actions_html = "".join(f"<li>{a}</li>" for a in self._actions()) or '<li class="empty">Aksiyon önerisi yok.</li>'

        regulations_html = "".join(
            f"<li>{regulation}</li>" for regulation in report.get("relevant_regulations", [])
        ) or '<li class="empty">Mevzuat eşleştirilemedi (güvenilir/doğrulanmış bir eşleşme bulunamadı).</li>'

        events_html = "".join(
            f"<li><b>{entry.get('event_name', '?')}</b> "
            f"<span class=\"tag\">{entry.get('event_type') or 'Eşleştirilemedi'}</span> "
            f"<span class=\"tag tag-risk\">{entry.get('risk_level') or 'Değerlendirilmedi'}</span><br/>"
            f'<span class="muted">{", ".join(entry.get("keywords") or [])}</span></li>'
            for entry in report.get("events", [])
        ) or '<li class="empty">VLM olay üretmedi.</li>'

        def _mmss(seconds: float) -> str:
            total = int(round(seconds))
            return f"{total // 60:02d}:{total % 60:02d}"

        timeline_html = "".join(
            f'<li><span class="ts">{_mmss(entry["timestamp"])}</span> {entry["description"]}</li>'
            for entry in report.get("timeline", [])
        ) or '<li class="empty">Kayıt yok.</li>'

        summary = report.get("summary") or report.get("natural_language_summary", "")

        return f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SAFİR Raporu — {report['video_source']}</title>
<style>
  :root {{
    color-scheme: light;
    --accent: {_ACCENT};
    --ink: {_ACCENT_DARK};
    --muted: {_MUTED};
    --border: {_BORDER};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Segoe UI", "Inter", "DejaVu Sans", Arial, sans-serif;
    margin: 0; padding: 2.5rem 3rem 4rem;
    color: var(--ink); background: #f6f8fa; line-height: 1.55;
  }}
  header.report-header {{
    background: linear-gradient(120deg, var(--ink) 0%, #14343f 100%);
    color: #f1f5f9; border-radius: 14px; padding: 1.75rem 2rem; margin-bottom: 1.75rem;
    box-shadow: 0 8px 24px rgba(15,23,42,.18);
  }}
  header.report-header h1 {{ margin: 0 0 .35rem; font-size: 1.5rem; letter-spacing: .04em; }}
  header.report-header p {{ margin: 0; color: #94a3b8; font-size: .85rem; }}
  header.report-header .meta-row {{ margin-top: .9rem; display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: .82rem; color: #cbd5e1; }}
  header.report-header .meta-row b {{ color: #e2e8f0; }}
  h2 {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; color: var(--accent); margin: 0 0 .9rem; }}
  .section {{
    margin-top: 1.1rem; background: #fff; padding: 1.35rem 1.6rem;
    border-radius: 12px; border: 1px solid var(--border); box-shadow: 0 1px 2px rgba(15,23,42,.04);
  }}
  .risk-row {{ display: flex; align-items: center; gap: .9rem; flex-wrap: wrap; margin-bottom: .6rem; }}
  .risk-score {{ font-size: 1.3rem; font-weight: 700; }}
  .risk-badge {{ display: inline-block; padding: .3rem .95rem; border-radius: 999px; color: #fff; font-weight: 700; font-size: .78rem; letter-spacing: .05em; }}
  ul {{ margin: 0; padding-left: 1.15rem; }}
  li {{ margin-bottom: .45rem; }}
  li.empty, p.empty {{ color: var(--muted); font-style: italic; list-style: none; margin-left: -1.15rem; }}
  .muted {{ color: var(--muted); font-size: .85rem; }}
  .tag {{ display: inline-block; font-size: .72rem; padding: .1rem .5rem; border-radius: 999px; background: #eef2f7; color: #334155; margin-left: .35rem; }}
  .tag-risk {{ background: var(--accent); color: #fff; }}
  .ts {{ display: inline-block; min-width: 3.4rem; font-variant-numeric: tabular-nums; font-weight: 600; color: var(--accent); }}
  .evidence-grid {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
  .evidence-card {{ margin: 0; text-align: center; width: 220px; }}
  .evidence-card img {{ width: 100%; border-radius: 10px; border: 1px solid var(--border); }}
  .evidence-card figcaption {{ margin-top: .4rem; font-size: .78rem; color: var(--muted); }}
</style>
</head>
<body>
<header class="report-header">
  <h1>SAFİR — Saha Analiz Raporu</h1>
  <p>Yapay zekâ destekli video risk analizi</p>
  <div class="meta-row">
    <span><b>Video:</b> {report['video_source']}</span>
    <span><b>Üretim zamanı:</b> {report['generated_at']}</span>
    <span><b>Modeller:</b> VLM={report.get('vlm_model') or '—'} · LLM={report.get('llm_model') or '—'}</span>
  </div>
</header>

<div class="section">
<h2>Özet</h2>
<p>{summary}</p>
</div>

<div class="section">
<h2>Risk Değerlendirmesi</h2>
<div class="risk-row">
  <span class="risk-score">{self._risk_score_text()}</span>
  <span class="risk-badge" style="background-color:{risk_color};">{report['risk_level'].upper()}</span>
  <span class="muted">Eskalasyon: {report.get('escalation_tier') or '—'}{' · otomatik alarm tetiklendi' if report.get('auto_dispatched') else ''}</span>
</div>
<p class="muted" style="margin-bottom:.4rem;">Önerilen aksiyonlar</p>
<ul>{actions_html}</ul>
</div>

<div class="section">
<h2>VLM Görsel Anlama Çıktısı</h2>
<p>{report['natural_language_summary']}</p>
</div>

<div class="section">
<h2>Kanıt Kareleri</h2>
<div class="evidence-grid">{evidence_html}</div>
</div>

<div class="section">
<h2>Tespit Edilen Olaylar</h2>
<ul>{events_html}</ul>
</div>

<div class="section">
<h2>İlgili İSG Mevzuatı (RAG)</h2>
<ul>{regulations_html}</ul>
</div>

<div class="section">
<h2>Zaman Çizelgesi</h2>
<ul>{timeline_html}</ul>
</div>
</body>
</html>"""

    def to_pdf(self) -> bytes:
        """`reportlab` ile kanit goruntulerini iceren bicimli, Turkce-uyumlu bir PDF ozet raporu uretir."""
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image as RLImage,
        )
        from reportlab.platypus import (
            HRFlowable,
            ListFlowable,
            ListItem,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )

        font_regular = _register_pdf_fonts()
        font_bold = _FONT_BOLD if font_regular == _FONT_REGULAR else "Helvetica-Bold"

        report = self._report
        _, risk_color = resolve_risk_badge(report["risk_level"], report["risk_score"])
        accent = colors.HexColor(_ACCENT)
        ink = colors.HexColor(_ACCENT_DARK)
        muted = colors.HexColor(_MUTED)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4, topMargin=1.8 * cm, bottomMargin=1.8 * cm, leftMargin=2 * cm, rightMargin=2 * cm
        )
        base = getSampleStyleSheet()["BodyText"]
        body_style = ParagraphStyle("SafirBody", parent=base, fontName=font_regular, fontSize=10, leading=14.5)
        title_style = ParagraphStyle(
            "SafirTitle", parent=base, fontName=font_bold, fontSize=19, leading=23, textColor=ink, alignment=TA_LEFT
        )
        subtitle_style = ParagraphStyle("SafirSubtitle", parent=body_style, fontSize=9, textColor=muted)
        heading_style = ParagraphStyle(
            "SafirHeading",
            parent=body_style,
            fontName=font_bold,
            fontSize=10.5,
            textColor=accent,
            spaceBefore=12,
            spaceAfter=6,
        )
        risk_style = ParagraphStyle(
            "SafirRisk", parent=body_style, fontName=font_bold, textColor=colors.HexColor(risk_color), fontSize=14
        )
        list_style = ParagraphStyle("SafirListItem", parent=body_style)
        bullet = dict(bulletFontName=font_regular, bulletColor=accent)

        summary = report.get("summary") or report.get("natural_language_summary", "")

        story: list = [
            Paragraph("SAFİR — Saha Analiz Raporu", title_style),
            Spacer(1, 0.15 * cm),
            Paragraph(f"Video kaynağı: {report['video_source']}", subtitle_style),
            Paragraph(f"Üretim zamanı: {report['generated_at']}", subtitle_style),
            Spacer(1, 0.25 * cm),
            HRFlowable(width="100%", thickness=1.2, color=accent, spaceAfter=8),
            Paragraph("ÖZET", heading_style),
            Paragraph(summary, body_style),
            Paragraph(f"{self._risk_score_text()} — {report['risk_level'].upper()}", risk_style),
            Paragraph(
                f"Eskalasyon: {report.get('escalation_tier') or '—'}"
                + (" (otomatik alarm tetiklendi)" if report.get("auto_dispatched") else ""),
                body_style,
            ),
        ]

        actions = self._actions()
        story.append(Paragraph("ÖNERİLEN AKSİYONLAR", heading_style))
        if actions:
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(a, list_style)) for a in actions], bulletType="bullet", **bullet
                )
            )
        else:
            story.append(Paragraph("Aksiyon önerisi yok.", body_style))

        story += [
            Paragraph("VLM GÖRSEL ANLAMA ÇIKTISI", heading_style),
            Paragraph(report["natural_language_summary"], body_style),
            Paragraph("KANIT KARELERİ", heading_style),
        ]

        for evidence in report.get("evidence_frames", []):
            try:
                _, b64_data = evidence["base64_image"].split(",", 1)
                image_bytes = base64.b64decode(b64_data)
                story.append(RLImage(io.BytesIO(image_bytes), width=8 * cm, height=6 * cm))
                story.append(
                    Paragraph(
                        f"Olay #{evidence['event_id']} · {evidence['timestamp_str']} · "
                        f"skor {evidence['change_score']:.4f}",
                        subtitle_style,
                    )
                )
                story.append(Spacer(1, 0.3 * cm))
            except (KeyError, ValueError, base64.binascii.Error):
                continue
        if not report.get("evidence_frames"):
            story.append(Paragraph("Kanıt karesi yok.", body_style))

        story.append(Paragraph("TESPİT EDİLEN OLAYLAR", heading_style))
        events = report.get("events", [])
        if events:
            story.append(
                ListFlowable(
                    [
                        ListItem(
                            Paragraph(
                                f"<b>{entry.get('event_name', '?')}</b> "
                                f"(Kategori: {entry.get('event_type') or 'Eşleştirilemedi'}, "
                                f"Risk: {entry.get('risk_level') or 'Değerlendirilmedi'}): "
                                f"{', '.join(entry.get('keywords') or [])}",
                                list_style,
                            )
                        )
                        for entry in events
                    ],
                    bulletType="bullet",
                    **bullet,
                )
            )
        else:
            story.append(Paragraph("VLM olay üretmedi.", body_style))

        story.append(Paragraph("İLGİLİ İSG MEVZUATI (RAG)", heading_style))
        regulations = report.get("relevant_regulations", [])
        if regulations:
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(r, list_style)) for r in regulations], bulletType="bullet", **bullet
                )
            )
        else:
            story.append(
                Paragraph("Mevzuat eşleştirilemedi (güvenilir/doğrulanmış bir eşleşme bulunamadı).", body_style)
            )

        story.append(Paragraph("ZAMAN ÇİZELGESİ", heading_style))
        timeline = report.get("timeline", [])
        if timeline:
            story.append(
                ListFlowable(
                    [
                        ListItem(Paragraph(f"[{e['timestamp']:.1f}s] {e['description']}", list_style))
                        for e in timeline
                    ],
                    bulletType="bullet",
                    **bullet,
                )
            )
        else:
            story.append(Paragraph("Kayıt yok.", body_style))

        doc.build(story)
        return buffer.getvalue()
