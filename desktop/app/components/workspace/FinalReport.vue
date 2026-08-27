<script setup lang="ts">
// User-friendly final report: executive summary, risk recap, recommended
// actions, RAG regulations, escalation (+ manual override trigger), technical
// metrics, and export (JSON / HTML client-side, PDF from the real backend
// endpoint — see useReportExport.ts). Timeline and Evidence live in their own tabs.
const store = useAnalysisStore()
const { exportJson, exportHtml, exportPdf } = useReportExport()
const manualNote = ref('')

const r = computed(() => store.report)
const isUnknownRisk = computed(() => r.value?.risk_status === 'unknown' || r.value?.risk_score == null)
const tone = computed(() => (isUnknownRisk.value ? 'unknown' : riskTone(r.value?.risk_level)))

type ExportKind = 'json' | 'html' | 'pdf'
type ExportPhase = 'idle' | 'loading' | 'ok' | 'error'
const exportPhase = reactive<Record<ExportKind, ExportPhase>>({ json: 'idle', html: 'idle', pdf: 'idle' })
const exportError = reactive<Record<ExportKind, string>>({ json: '', html: '', pdf: '' })

async function runExport(kind: ExportKind, fn: () => void | Promise<void>) {
  exportPhase[kind] = 'loading'
  exportError[kind] = ''
  try {
    await fn()
    exportPhase[kind] = 'ok'
    setTimeout(() => {
      if (exportPhase[kind] === 'ok') exportPhase[kind] = 'idle'
    }, 2000)
  } catch (e) {
    exportPhase[kind] = 'error'
    exportError[kind] = e instanceof Error ? e.message : 'Dışa aktarma başarısız oldu.'
  }
}

function doExportJson() {
  if (r.value) runExport('json', () => exportJson(r.value as NonNullable<typeof r.value>))
}
function doExportHtml() {
  if (r.value) runExport('html', () => exportHtml(r.value as NonNullable<typeof r.value>))
}
function doExportPdf() {
  if (r.value) runExport('pdf', () => exportPdf(store.jobId, r.value as NonNullable<typeof r.value>))
}

// notify_health_team_tool/dispatch_security_tool/trigger_area_lockdown_tool
// (src/agent/tools.py) — sabit, insan-okunur Türkçe etiketler; bilinmeyen bir
// arac adi gelirse adin kendisi fallback olarak kalır.
const MOCK_ACTION_LABELS: Record<string, string> = {
  notify_health_team_tool: 'Sağlık Ekibi Bilgilendirildi',
  dispatch_security_tool: 'Güvenlik Ekibi Yönlendirildi',
  trigger_area_lockdown_tool: 'Alan Tahliye/Kilitleme Tetiklendi',
}
function mockActionLabel(tool: string): string {
  return MOCK_ACTION_LABELS[tool] ?? tool
}
</script>

<template>
  <div v-if="r" class="space-y-6">
    <!-- executive summary -->
    <section>
      <div class="field-label">Yönetici Özeti</div>
      <p class="text-sm text-slate-200 leading-relaxed">{{ r.summary || r.natural_language_summary || '—' }}</p>
    </section>

    <div class="grid md:grid-cols-2 gap-6">
      <!-- risk + actions -->
      <section class="space-y-4">
        <div>
          <div class="field-label">Risk</div>
          <div v-if="isUnknownRisk" class="text-2xl font-bold" :class="RISK_TEXT[tone]">
            Risk Belirsiz <span class="uppercase text-base font-normal text-slate-400">— MANUEL İNCELEME GEREKLİ</span>
          </div>
          <div v-else class="text-2xl font-bold" :class="RISK_TEXT[tone]">{{ r.risk_score }} / 100 · <span class="uppercase text-base">{{ trUpper(r.risk_level) }}</span></div>
        </div>
        <div>
          <div class="field-label">Önerilen aksiyonlar</div>
          <ol v-if="r.actions?.length" class="list-decimal list-inside space-y-1 text-sm text-slate-200">
            <li v-for="(a, i) in r.actions" :key="i">{{ a }}</li>
          </ol>
          <p v-else class="text-sm text-slate-400">{{ r.recommended_action || '—' }}</p>
        </div>
        <div v-if="r.triggered_mock_actions?.length">
          <div class="field-label">Ajanın Çağırdığı Mock Aksiyon Araçları</div>
          <ul class="space-y-1.5">
            <li
              v-for="(t, i) in r.triggered_mock_actions"
              :key="i"
              class="rounded-md border border-accent/30 bg-accent/10 px-3 py-2 text-xs text-slate-200"
            >
              <span class="font-mono text-accent">{{ mockActionLabel(t.tool) }}</span>
              <span class="text-slate-400"> — {{ t.result }}</span>
            </li>
          </ul>
        </div>
      </section>

      <!-- RAG regulations -->
      <section>
        <div class="field-label">İlgili İSG mevzuatı (RAG / FAISS)</div>
        <ul v-if="r.relevant_regulations?.length" class="space-y-1 text-sm text-slate-200">
          <li v-for="(reg, i) in r.relevant_regulations" :key="i" class="bg-surface-2 border border-edge rounded-md px-3 py-2">{{ reg }}</li>
        </ul>
        <p v-else class="text-sm text-slate-400">Bu analiz için ilgili mevzuat maddesi bulunamadı.</p>
      </section>
    </div>

    <!-- escalation + manual override -->
    <section class="card p-4 bg-surface-2/40">
      <div class="field-label">Eskalasyon (Human-on-the-Loop)</div>
      <p class="text-sm text-slate-200">
        Kademe: <b>{{ r.escalation_tier ?? '—' }}</b>
        · otomatik tetik: {{ r.auto_dispatched ? 'evet' : 'hayır' }}
        <span v-if="r.alert_id" class="font-mono text-slate-400"> · {{ r.alert_id }}</span>
      </p>
      <details class="mt-3">
        <summary class="cursor-pointer text-xs text-slate-400">Manuel saha alarmı tetikle (override)</summary>
        <div class="mt-2 flex flex-col sm:flex-row gap-2">
          <input v-model="manualNote" class="field-input" placeholder="Manuel alarm notu (opsiyonel)" />
          <button
            class="btn-ghost shrink-0"
            :disabled="store.manualAlert.state === 'pending'"
            @click="store.triggerManualAlert(manualNote)"
          >Manuel Alarm Tetikle</button>
        </div>
        <p v-if="store.manualAlert.message" class="mt-2 text-xs" :class="store.manualAlert.state === 'error' ? 'text-risk-crit' : 'text-risk-low'">
          {{ store.manualAlert.message }}
        </p>
      </details>
    </section>

    <!-- technical metrics -->
    <section>
      <div class="field-label">Teknik metrikler</div>
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCell label="VLM" :value="r.vlm_model ?? '—'" mono />
        <MetricCell label="LLM" :value="r.llm_model ?? '—'" mono />
        <MetricCell label="Olay Kimliği" :value="r.event_id ?? '—'" mono />
        <MetricCell label="Otomatik Yönlendirildi" :value="r.auto_dispatched ? 'evet' : 'hayır'" />
      </div>
    </section>

    <!-- technical JSON (real SafirReport; no raw_response/secret/reasoning) -->
    <section>
      <details>
        <summary class="cursor-pointer text-xs text-slate-400">Technical JSON (tam SafirReport)</summary>
        <pre class="mt-2 text-[11px] font-mono text-slate-400 bg-surface-2 border border-edge rounded-md p-3 max-h-96 overflow-auto">{{ JSON.stringify(r, null, 2) }}</pre>
      </details>
    </section>

    <!-- export -->
    <section class="pt-2 border-t border-edge">
      <div class="flex items-center gap-3">
        <span class="text-xs text-slate-500">Dışa aktar:</span>
        <button class="btn-ghost" :disabled="exportPhase.json === 'loading'" @click="doExportJson">
          <span v-if="exportPhase.json === 'loading'">…</span>
          <span v-else-if="exportPhase.json === 'ok'">✓ JSON</span>
          <span v-else>JSON</span>
        </button>
        <button class="btn-ghost" :disabled="exportPhase.html === 'loading'" @click="doExportHtml">
          <span v-if="exportPhase.html === 'loading'">…</span>
          <span v-else-if="exportPhase.html === 'ok'">✓ HTML</span>
          <span v-else>HTML</span>
        </button>
        <button class="btn-ghost" :disabled="exportPhase.pdf === 'loading'" @click="doExportPdf">
          <span v-if="exportPhase.pdf === 'loading'">PDF oluşturuluyor…</span>
          <span v-else-if="exportPhase.pdf === 'ok'">✓ PDF</span>
          <span v-else>PDF</span>
        </button>
        <span class="text-[11px] text-slate-600 ml-2">(JSON/HTML gerçek rapor verisinden; PDF backend'de reportlab ile üretilir)</span>
      </div>
      <p v-if="exportPhase.json === 'error'" class="mt-2 text-xs text-risk-crit">JSON: {{ exportError.json }}</p>
      <p v-if="exportPhase.html === 'error'" class="mt-2 text-xs text-risk-crit">HTML: {{ exportError.html }}</p>
      <p v-if="exportPhase.pdf === 'error'" class="mt-2 text-xs text-risk-crit">PDF: {{ exportError.pdf }}</p>
    </section>
  </div>

  <div v-else class="text-sm text-slate-500 py-8 text-center">
    Nihai rapor henüz hazır değil.
  </div>
</template>
