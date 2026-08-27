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
 * `download()` also best-effort opens the same blob in a new tab/window
 * right away, so the operator can immediately see the file loaded instead
 * of only trusting a download notification.
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
  // Indirmeye EK olarak, dosyayi dogrudan bir sekmede/pencerede acmayi da
  // DENER (operator "indirildi mi?" diye Indirilenler klasorunu aramak
  // zorunda kalmasin diye) — best-effort: Tauri webview'de engellenirse veya
  // tarayici pop-up'i durdurursa sessizce yutulur, indirme YINE DE gecerlidir.
  try {
    window.open(url, '_blank')
  } catch {
    // yoksay — indirme zaten tamamlandi
  }
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
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
  const isUnknownRisk = r.risk_status === 'unknown' || r.risk_score === null || r.risk_score === undefined
  const riskText = isUnknownRisk
    ? 'Belirsiz — analiz güvenilir şekilde tamamlanamadı, manuel inceleme gerekli'
    : `${esc(r.risk_score)} / 100 — ${esc(r.risk_level)}`
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
<h2>Risk</h2><p class="risk">${riskText}</p>
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
  const api = useSafirApi()

  function exportJson(r: SafirReport): string {
    const filename = `${fileStub(r)}.json`
    download(filename, JSON.stringify(r, null, 2), 'application/json')
    return filename
  }
  function exportHtml(r: SafirReport): string {
    const filename = `${fileStub(r)}.html`
    download(filename, buildReportHtml(r), 'text/html')
    return filename
  }
  /**
   * Real backend PDF. Needs `jobId` (live job still in memory OR already
   * persisted to History — the endpoint tries both). Throws a human-readable
   * error on failure; callers surface it (no silent no-op).
   */
  async function exportPdf(jobId: string | null, r: SafirReport): Promise<string> {
    if (!jobId) throw new Error('PDF dışa aktarmak için analiz kimliği bulunamadı.')
    let blob: Blob
    try {
      blob = await api.getReportPdf(jobId)
    } catch (e: unknown) {
      const detail = (e as { data?: { detail?: string } })?.data?.detail
      throw new Error(detail ?? 'PDF oluşturulamadı. Backend\'e ulaşılamıyor olabilir.')
    }
    const filename = `${fileStub(r)}.pdf`
    download(filename, blob)
    return filename
  }
  return { exportJson, exportHtml, exportPdf }
}
