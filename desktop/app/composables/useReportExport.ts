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
  try {
    window.open(url, '_blank')
  } catch {
    // yoksay
  }
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

function mmss(seconds: number): string {
  const total = Math.round(seconds)
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`
}

export function buildSartnameJson(r: SafirReport): Record<string, unknown> {
  const onsetTimestamp = r.onset_timestamp_str || (r.timeline?.length ? mmss(r.timeline[0].timestamp) : '00:00')
  return {
    summary: r.summary || r.natural_language_summary,
    onset_timestamp: onsetTimestamp,
    safe_timestamps: r.safe_timestamps ?? [],
    incident_timestamps: r.incident_timestamps ?? [],
    events: (r.timeline ?? []).map((entry) => ({ time: mmss(entry.timestamp), event: entry.description })),
    risk: r.risk_level,
    risk_score: r.risk_score,
    risk_status: r.risk_status,
    risk_accuracy: {
      deterministic_score: r.deterministic_score ?? null,
      deterministic_level: r.deterministic_level ?? null,
      llm_proposed_score: r.llm_proposed_score ?? null,
      final_score: r.risk_score,
      final_level: r.risk_level,
      method: r.llm_proposed_score != null ? 'ortalama(deterministic_score, llm_proposed_score)' : 'deterministic_score',
    },
    actions: r.actions?.length ? r.actions : r.recommended_action ? [r.recommended_action] : [],
  }
}

function esc(s: unknown): string {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

export function buildReportHtml(r: SafirReport): string {
  const rows = (r.timeline ?? []).map((e) => `<tr><td>${mmss(e.timestamp)}</td><td>${esc(e.description)}</td></tr>`).join('')
  const actions = (r.actions ?? []).map((a) => `<li>${esc(a)}</li>`).join('')
  const regs = (r.relevant_regulations ?? []).map((a) => `<li>${esc(a)}</li>`).join('')
  const st = r.sampler_stats
  const isUnknownRisk = r.risk_status === 'unknown' || r.risk_score === null || r.risk_score === undefined
  const riskText = isUnknownRisk
    ? 'Belirsiz — analiz güvenilir şekilde tamamlanamadı, manuel inceleme gerekli'
    : `${esc(r.risk_score)} / 100 — ${esc(r.risk_level)}`
  const hasBreakdown = r.deterministic_score != null || r.llm_proposed_score != null
  const breakdownHtml = hasBreakdown
    ? `<div class="risk-breakdown">
        <span><b>Deterministik (RuleEngine):</b> ${r.deterministic_score ?? '—'}/100 (${esc(r.deterministic_level ?? '—')})</span>
        <span><b>Ajan (LLM) taslağı:</b> ${r.llm_proposed_score ?? '—'}/100</span>
        <span class="muted">${r.llm_proposed_score != null ? 'Nihai skor = ortalama(deterministik, ajan taslağı)' : 'Nihai skor = deterministik skor (ajan taslağı yok)'}</span>
      </div>`
    : ''
  return `<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAFİR Raporu — ${esc(r.video_source)}</title>
<style>
  :root{color-scheme:light;--accent:#0d9488;--ink:#0f172a;--muted:#64748b;--border:#e2e8f0}
  *{box-sizing:border-box}
  body{font-family:"Segoe UI","Inter","DejaVu Sans",Arial,sans-serif;color:var(--ink);background:#f6f8fa;margin:0;padding:2.5rem 3rem 4rem;line-height:1.6}
  header{background:linear-gradient(120deg,var(--ink) 0%,#14343f 100%);color:#f1f5f9;border-radius:14px;padding:1.75rem 2rem;margin-bottom:1.5rem;box-shadow:0 8px 24px rgba(15,23,42,.18)}
  header h1{margin:0 0 .35rem;font-size:1.4rem;letter-spacing:.05em}
  header p{margin:0;color:#94a3b8;font-size:.85rem}
  h2{font-size:.75rem;text-transform:uppercase;letter-spacing:.09em;color:var(--accent);margin:0 0 .8rem}
  .section{margin-top:1.1rem;background:#fff;padding:1.3rem 1.6rem;border-radius:12px;border:1px solid var(--border);box-shadow:0 1px 2px rgba(15,23,42,.04)}
  .risk{font-size:1.6rem;font-weight:700}
  table{border-collapse:collapse;width:100%}
  td,th{border-bottom:1px solid var(--border);padding:.5rem .3rem;text-align:left;font-size:.85rem}
  th{color:var(--muted);text-transform:uppercase;font-size:.7rem;letter-spacing:.05em}
  .muted{color:var(--muted);font-size:.82rem}
  ul{margin:.3rem 0;padding-left:1.15rem}
  li{margin-bottom:.35rem}
  .risk-breakdown{display:flex;gap:1.5rem;flex-wrap:wrap;font-size:.82rem;margin:.6rem 0 0;padding:.6rem .8rem;background:#f6f8fa;border-radius:8px;border:1px solid var(--border)}
</style></head><body>
<header>
  <h1>SAFİR — Saha Analiz Raporu</h1>
  <p>${esc(r.video_source)} · ${esc(r.generated_at)}</p>
</header>
<div class="section"><h2>Özet</h2><p>${esc(r.summary || r.natural_language_summary)}</p></div>
<div class="section"><h2>Risk</h2><p class="risk">${riskText}</p>
${breakdownHtml}
<p><b>Önerilen aksiyon:</b> ${esc(r.recommended_action)}</p></div>
<div class="section"><h2>Aksiyonlar</h2><ul>${actions || '<li class="muted">—</li>'}</ul></div>
<div class="section"><h2>Zaman Çizelgesi</h2><table><tr><th>Zaman</th><th>Olay</th></tr>${rows || '<tr><td colspan=2 class="muted">—</td></tr>'}</table></div>
<div class="section"><h2>İlgili Mevzuat (RAG)</h2><ul>${regs || '<li class="muted">—</li>'}</ul></div>
<div class="section"><h2>Eskalasyon</h2><p>Kademe: <b>${esc(r.escalation_tier ?? '-')}</b> · otomatik: ${r.auto_dispatched ? 'evet' : 'hayır'}${r.alert_id ? ` · alert_id: ${esc(r.alert_id)}` : ''}</p></div>
<div class="section"><h2>Teknik Metrikler</h2>
<p class="muted">VLM: ${esc(r.vlm_model ?? '-')} · LLM: ${esc(r.llm_model ?? '-')}${st ? ` · taranan kare: ${esc(st.total_frames_scanned)} · GPU tasarrufu: %${esc(st.gpu_savings_ratio_pct)} · süre: ${esc(st.elapsed_sec)}s` : ''}</p></div>
</body></html>`
}

export function useReportExport() {
  const api = useSafirApi()

  function exportJson(r: SafirReport): string {
    const filename = `${fileStub(r)}.json`
    download(filename, JSON.stringify(buildSartnameJson(r), null, 2), 'application/json')
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
