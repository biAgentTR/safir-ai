/**
 * Report export — all three formats produce a REAL file, no placeholders:
 *   - JSON : full SafirReport, pretty-printed, built client-side from the
 *            already-loaded report (no extra request).
 *   - HTML : a clean standalone document built client-side from the same data.
 *   - PDF  : GET /history/{job_id}/report.pdf — a REAL backend-generated PDF
 *            (reportlab, `ReportExporter.to_pdf()`, the exact class already
 *            used by the Operator Panel/Streamlit — reused, not reimplemented).
 *            This REPLACES the earlier window.print() workaround: printing to
 *            PDF depends on the OS print dialog and could not be verified
 *            headlessly, while this downloads real, testable PDF bytes.
 *
 * All three go through the same `download()` helper (Blob + `<a download>`),
 * which is the standard web-platform download mechanism — supported by the
 * Tauri v2 webview (WebView2/WebKitGTK) without any extra Tauri plugin.
 */
import type { SafirReport } from '~/types/api'

function fileStub(r: SafirReport): string {
  const ts = (r.generated_at || new Date().toISOString()).replace(/[:.]/g, '-')
  return `safir_report_${ts}`
}

function download(filename: string, content: string | Blob, mime?: string) {
  const blob = content instanceof Blob ? content : new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function esc(s: unknown): string {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

export function buildReportHtml(r: SafirReport): string {
  const mmssLocal = (sec: number) => {
    const t = Math.round(sec)
    return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`
  }
  const rows = (r.timeline ?? []).map((e) => `<tr><td>${mmssLocal(e.timestamp)}</td><td>${esc(e.description)}</td></tr>`).join('')
  const actions = (r.actions ?? []).map((a) => `<li>${esc(a)}</li>`).join('')
  const regs = (r.relevant_regulations ?? []).map((a) => `<li>${esc(a)}</li>`).join('')
  const st = r.sampler_stats

  const isUnclassified = r.risk_status === 'unclassified' || (r.detected_event_types ?? []).includes('siniflandirilamadi')
  const isUnknownRisk = !isUnclassified && (r.risk_status === 'unknown' || r.risk_score === null || r.risk_score === undefined)

  let riskText = `${esc(r.risk_score)} / 100 — ${esc(r.risk_level)}`
  if (isUnclassified) {
    riskText = 'Sınıflandırılamayan Risk (null) — 8 Ana İSG kuralı dışında şüpheli anormallik'
  } else if (isUnknownRisk) {
    riskText = 'Belirsiz — analiz güvenilir şekilde tamamlanamadı, manuel inceleme gerekli'
  }

  const onsetHtml = r.onset_timestamp_str ? `<p><b>📍 Olay Başlangıç Anı (Onset):</b> <span style="color:#d97706;font-weight:bold">${esc(r.onset_timestamp_str)}</span></p>` : ''
  const safeHtml = (r.safe_timestamps ?? []).length ? `<p><b>🟢 Rutin Kareler (Safe):</b> ${r.safe_timestamps.map(t => `<code style="background:#e6f4ea;color:#137333;padding:2px 6px;border-radius:4px">${esc(t)}</code>`).join(' ')}</p>` : ''
  const incidentHtml = (r.incident_timestamps ?? []).length ? `<p><b>🔴 Olay Kareleri (Incident):</b> ${r.incident_timestamps.map(t => `<code style="background:#fce8e6;color:#c5221f;padding:2px 6px;border-radius:4px">${esc(t)}</code>`).join(' ')}</p>` : ''

  const unclassifiedBox = isUnclassified
    ? `<div style="background:#fffbeb;border:1px solid #fef3c7;padding:12px;border-radius:6px;margin:1rem 0;color:#92400e">
        <b>🏷️ 8 Ana İSG Kuralı Dışında Şüpheli Risk / İnceleme Bekliyor</b>
        <p style="margin:4px 0 0 0;font-size:13px">Sahada anormallik/şüpheli olay tespit edilmiştir ancak tanımlı 8 temel İSG mevzuat kategorisinden birine eşleştirilememiştir. Risk skoru null olarak işaretlenmiştir.</p>
       </div>`
    : ''

  return `<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>SAFİR — Saha Analiz Raporu</title>
<style>
  body{font-family:system-ui,Arial,sans-serif;color:#111;margin:2rem;line-height:1.5}
  h1{font-size:22px;letter-spacing:.1em;color:#0f172a;border-bottom:2px solid #0f172a;padding-bottom:6px}
  h2{font-size:14px;margin-top:1.5rem;text-transform:uppercase;color:#475569;border-bottom:1px solid #cbd5e1;padding-bottom:4px}
  .risk{font-size:24px;font-weight:700;color:#0f172a} table{border-collapse:collapse;width:100%} td,th{border:1px solid #cbd5e1;padding:6px 8px;text-align:left;font-size:13px}
  .muted{color:#64748b;font-size:12px} ul{margin:.3rem 0}
</style></head><body>
<h1>SAFİR — Saha Analiz ve İSG Denetim Raporu</h1>
<p class="muted">Video Kaynağı: ${esc(r.video_source)} · Tarih: ${esc(r.generated_at)}</p>
<h2>Executive Summary</h2><p>${esc(r.summary || r.natural_language_summary)}</p>
${unclassifiedBox}
<h2>Risk Değerlendirmesi</h2><p class="risk">${riskText}</p>
${onsetHtml}
${safeHtml}
${incidentHtml}
<p style="margin-top:1rem"><b>Önerilen Aksiyon:</b> ${esc(r.recommended_action)}</p>
<h2>Önerilen Somut Aksiyonlar</h2><ul>${actions || '<li class="muted">—</li>'}</ul>
<h2>Zaman Çizelgesi</h2><table><tr><th>Zaman</th><th>Olay Açıklaması</th></tr>${rows || '<tr><td colspan=2 class="muted">—</td></tr>'}</table>
<h2>İlgili Mevzuat (RAG / FAISS)</h2><ul>${regs || '<li class="muted">—</li>'}</ul>
<h2>Eskalasyon & Bildirim</h2><p>Kademe: <b>${esc(r.escalation_tier ?? '-')}</b> · Otomatik Alarm: ${r.auto_dispatched ? 'Evet' : 'Hayır'}${r.alert_id ? ` · Alert ID: ${esc(r.alert_id)}` : ''}</p>
<h2>Teknik Metrikler & Performans</h2>
<p class="muted">VLM Modeli: ${esc(r.vlm_model ?? '-')} · LLM Modeli: ${esc(r.llm_model ?? '-')}${st ? ` · Taranan Kare: ${esc(st.total_frames_scanned)} · GPU Tasarrufu: %${esc(st.gpu_savings_ratio_pct)} · Analiz Süresi: ${esc(st.elapsed_sec)}s` : ''}</p>
</body></html>`
}

export function useReportExport() {
  const api = useSafirApi()

  function exportJson(r: SafirReport) {
    download(`${fileStub(r)}.json`, JSON.stringify(r, null, 2), 'application/json')
  }
  function exportHtml(r: SafirReport) {
    download(`${fileStub(r)}.html`, buildReportHtml(r), 'text/html')
  }
  /**
   * Real backend PDF. Needs `jobId` (live job still in memory OR already
   * persisted to History — the endpoint tries both). Throws a human-readable
   * error on failure; callers surface it (no silent no-op).
   */
  async function exportPdf(jobId: string | null, r: SafirReport) {
    if (!jobId) throw new Error('PDF dışa aktarmak için analiz kimliği bulunamadı.')
    let blob: Blob
    try {
      blob = await api.getReportPdf(jobId)
    } catch (e: unknown) {
      const detail = (e as { data?: { detail?: string } })?.data?.detail
      throw new Error(detail ?? 'PDF oluşturulamadı. Backend\'e ulaşılamıyor olabilir.')
    }
    download(`${fileStub(r)}.pdf`, blob)
  }
  return { exportJson, exportHtml, exportPdf }
}
