/**
 * Report export. The backend does NOT expose an export endpoint — `ReportExporter`
 * (HTML/PDF) is Python-only, used inside Streamlit. So we export the REAL report
 * data client-side (this is not a mock — every value is the backend's report):
 *   - JSON : full SafirReport, pretty-printed
 *   - HTML : a clean standalone document built from the report
 *   - PDF  : the same document opened in a print window -> the webview's native
 *            "Save as PDF" (works in the Tauri/Chromium webview)
 */
import type { SafirReport } from '~/types/api'

function fileStub(r: SafirReport): string {
  const ts = (r.generated_at || new Date().toISOString()).replace(/[:.]/g, '-')
  return `safir_report_${ts}`
}

function download(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime })
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
  return `<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>SAFİR Raporu</title>
<style>
  body{font-family:system-ui,Arial,sans-serif;color:#111;margin:2rem;line-height:1.5}
  h1{font-size:20px;letter-spacing:.15em} h2{font-size:14px;margin-top:1.5rem;text-transform:uppercase;color:#555;border-bottom:1px solid #ddd;padding-bottom:4px}
  .risk{font-size:28px;font-weight:700} table{border-collapse:collapse;width:100%} td,th{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:13px}
  .muted{color:#666;font-size:12px} ul{margin:.3rem 0}
</style></head><body>
<h1>SAFİR — Saha Analiz Raporu</h1>
<p class="muted">${esc(r.video_source)} · ${esc(r.generated_at)}</p>
<h2>Executive Summary</h2><p>${esc(r.summary || r.natural_language_summary)}</p>
<h2>Risk</h2><p class="risk">${esc(r.risk_score)} / 100 — ${esc(r.risk_level)}</p>
<p><b>Önerilen aksiyon:</b> ${esc(r.recommended_action)}</p>
<h2>Aksiyonlar</h2><ul>${actions || '<li class="muted">—</li>'}</ul>
<h2>Zaman Çizelgesi</h2><table><tr><th>Zaman</th><th>Olay</th></tr>${rows || '<tr><td colspan=2 class="muted">—</td></tr>'}</table>
<h2>İlgili Mevzuat (RAG)</h2><ul>${regs || '<li class="muted">—</li>'}</ul>
<h2>Eskalasyon</h2><p>Kademe: <b>${esc(r.escalation_tier ?? '-')}</b> · otomatik: ${r.auto_dispatched ? 'evet' : 'hayır'}${r.alert_id ? ` · alert_id: ${esc(r.alert_id)}` : ''}</p>
<h2>Teknik Metrikler</h2>
<p class="muted">VLM: ${esc(r.vlm_model ?? '-')} · LLM: ${esc(r.llm_model ?? '-')}${st ? ` · taranan kare: ${esc(st.total_frames_scanned)} · GPU tasarrufu: %${esc(st.gpu_savings_ratio_pct)} · süre: ${esc(st.elapsed_sec)}s` : ''}</p>
</body></html>`
}

export function useReportExport() {
  function exportJson(r: SafirReport) {
    download(`${fileStub(r)}.json`, JSON.stringify(r, null, 2), 'application/json')
  }
  function exportHtml(r: SafirReport) {
    download(`${fileStub(r)}.html`, buildReportHtml(r), 'text/html')
  }
  function exportPdf(r: SafirReport) {
    // Open the printable HTML and invoke the webview's native print -> Save as PDF.
    const w = window.open('', '_blank')
    if (!w) return
    w.document.write(buildReportHtml(r))
    w.document.close()
    w.focus()
    setTimeout(() => w.print(), 300)
  }
  return { exportJson, exportHtml, exportPdf }
}
