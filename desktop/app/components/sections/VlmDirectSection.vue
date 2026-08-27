<script setup lang="ts">
// VLM Direct Analysis dashboard. Independent PRESENTATION from the
// low-budget mode's Workspace, but the SAME backend underneath: both modes
// go through POST /analyze/jobs, and src/main.py::SafirPipeline.stage_vlm
// already sends the video directly to EVREN (config-driven — configs/
// config.yaml `vlm.active_model`/`llm.active_model`, not hardcoded to any
// provider/model in either mode). This page just renders that result as a
// video timeline + event table instead of the pipeline-stage workspace.
//
// Reuses the SAME Pinia analysis store / SSE stream as workspace/[jobId].vue
// and new-analysis.vue — there is one active job at a time app-wide, which
// is the existing assumption those pages already make.
import { summarize, riskLevelCounts, eventTypeCounts } from '~/composables/useVlmMockData'
import { mapVlmDirectEvents } from '~/composables/useVlmDirectEvents'
import type { VlmStageEventData, TraceEvent } from '~/types/api'

const store = useAnalysisStore()
const stream = useAnalysisStream()
const { state: backendHealth } = useBackendHealth()
const { trigger: newAnalysisTrigger } = useVlmDirectReset()

onBeforeUnmount(() => stream.stop())

// ---- video source: a REAL local filesystem path (Tauri dialog), same
// pattern as pages/new-analysis.vue — the backend reads the file from disk,
// it is never uploaded as bytes. Preview uses Tauri's convertFileSrc to turn
// that path into a webview-loadable asset:// URL; outside Tauri (plain
// browser dev) there is no preview, only the path-driven analysis itself. ----
const videoPath = ref('')
const videoUrl = ref<string | null>(null)
const fileName = ref<string | null>(null)
const userPrompt = ref('Sahnede riskli bir durum var mi degerlendir.')

async function pickVideo() {
  try {
    const dialog = await import('@tauri-apps/plugin-dialog')
    const selected = await dialog.open({
      multiple: false,
      filters: [{ name: 'Video', extensions: ['mp4', 'avi', 'mov', 'mkv', 'webm'] }],
    })
    if (typeof selected !== 'string') return
    videoPath.value = selected
    fileName.value = selected.split(/[\\/]/).pop() || selected
    try {
      const core = await import('@tauri-apps/api/core')
      videoUrl.value = core.convertFileSrc(selected)
    } catch {
      videoUrl.value = null // non-Tauri browser dev — no preview, analysis still works
    }
  } catch {
    // Not running inside Tauri (or plugin unavailable) — no picker available;
    // operator has no way to enter a path here (unlike new-analysis.vue's
    // manual text field), since this dashboard is built around a real preview.
  }
  duration.value = 0
  currentTime.value = 0
  activeEventId.value = null
}

// ---- playback / timeline state ----
const duration = ref(0)
const currentTime = ref(0)
const activeEventId = ref<string | null>(null)

function onSelectEvent(id: string) {
  activeEventId.value = id
}

// ---- analysis lifecycle ----
const submitError = ref<string | null>(null)
const canSubmit = computed(() => !!videoPath.value && !store.submitting && !store.isRunning)

// Full reset — clears everything, including the video/prompt/results left
// from a previous run, not just the store's job. Runs on first mount and
// every time Ana Sayfa's "VLM Direct Analiz" card is clicked again (see
// composables/useVlmDirectReset.ts — this component stays mounted across
// hash navigations, it never remounts on its own).
function resetForNewAnalysis() {
  stream.stop()
  store.resetJob()
  videoPath.value = ''
  videoUrl.value = null
  fileName.value = null
  userPrompt.value = 'Sahnede riskli bir durum var mi degerlendir.'
  duration.value = 0
  currentTime.value = 0
  activeEventId.value = null
  submitError.value = null
  showJsonReport.value = false
  exportPhase.json = 'idle'
  exportPhase.html = 'idle'
  exportPhase.pdf = 'idle'
}
onMounted(resetForNewAnalysis)
watch(newAnalysisTrigger, (v) => {
  if (v > 0) resetForNewAnalysis()
})

async function startAnalysis() {
  if (!canSubmit.value) return
  submitError.value = null
  duration.value = 0
  currentTime.value = 0
  activeEventId.value = null
  try {
    const jobId = await store.createAnalysis({
      video_source: videoPath.value,
      user_prompt: userPrompt.value.trim() || undefined,
    })
    stream.start(jobId)
    store.pollUntilDone(jobId).catch(() => {})
  } catch (e: unknown) {
    submitError.value =
      (e as { data?: { detail?: string } })?.data?.detail ?? (e as Error)?.message ?? 'Analiz başlatılamadı.'
  }
}

const statusLabel = computed(() => {
  if (store.submitting || store.status === 'queued') return 'Kuyrukta'
  if (store.status === 'running') return 'Analiz Çalışıyor'
  if (store.status === 'done') return 'Tamamlandı'
  if (store.status === 'error') return 'Hata'
  return null
})

// ---- real events, from the VLM stage's structured per-event data (SSE
// trace) or the report timeline as a fallback — see useVlmDirectEvents.ts.
// No mock/placeholder data: before an analysis has run (or when it found
// nothing) this is empty and the components below render their real empty
// states instead of a fabricated preview. ----
const vlmStage = computed(() => store.eventForStage('vlm') as TraceEvent<VlmStageEventData> | undefined)
// Step-by-step VLM progress (video chunking/sending) — same trace data the
// low-budget Workspace's StageCard shows, surfaced here too since this
// dashboard has no pipeline-stage rail of its own to show it in otherwise.
const vlmProgress = computed(() => {
  const d = vlmStage.value?.data
  return d && 'progress' in d ? d.progress : null
})
const events = computed(() => mapVlmDirectEvents(vlmStage.value, store.report))
const hasRunAnalysis = computed(() => store.status === 'done' || store.status === 'error')

// Marker/timeline positioning needs SOME notion of video length even when
// there's no real <video> preview (non-Tauri dev) — fall back to the
// furthest event timestamp so the timeline strip still renders sensibly.
const effectiveDuration = computed(() => {
  if (duration.value > 0) return duration.value
  if (!events.value.length) return 0
  return Math.max(...events.value.map((e) => e.timestamp)) + 5
})

const summary = computed(() => summarize(events.value))
const riskCounts = computed(() => riskLevelCounts(events.value))
const typeCounts = computed(() => eventTypeCounts(events.value))

// ---- report export (Olay Türü Dağılımı panel) — JSON/HTML/PDF from the
// same SafirReport backing Workspace's FinalReport.vue (useReportExport.ts).
// Clicking JSON additionally opens the raw report below the dashboard; all
// three also open in a new tab right away (useReportExport's download())
// and show an explicit "indirildi, kontrol edin" confirmation here — bir
// indirmenin sessizce Indirilenler klasorune duşup fark edilmemesi yerine. ----
const { exportJson, exportHtml, exportPdf } = useReportExport()
const reportReady = computed(() => !!store.report)
const showJsonReport = ref(false)

type ExportKind = 'json' | 'html' | 'pdf'
const exportPhase = reactive<Record<ExportKind, 'idle' | 'loading' | 'ok' | 'error'>>({
  json: 'idle',
  html: 'idle',
  pdf: 'idle',
})
const exportMessage = reactive<Record<ExportKind, string>>({ json: '', html: '', pdf: '' })
const exportOkTimers: Record<ExportKind, ReturnType<typeof setTimeout> | null> = { json: null, html: null, pdf: null }

function markExported(kind: ExportKind, filename: string) {
  exportPhase[kind] = 'ok'
  exportMessage[kind] = `${filename} — yeni sekmede açıldı, lütfen dosyayı kontrol edin.`
  if (exportOkTimers[kind]) clearTimeout(exportOkTimers[kind]!)
  exportOkTimers[kind] = setTimeout(() => {
    if (exportPhase[kind] === 'ok') exportPhase[kind] = 'idle'
  }, 6000)
}
function markExportFailed(kind: ExportKind, e: unknown) {
  exportPhase[kind] = 'error'
  exportMessage[kind] = e instanceof Error ? e.message : 'Dışa aktarma başarısız oldu.'
}

function doExportJson() {
  if (!store.report) return
  exportPhase.json = 'loading'
  try {
    const filename = exportJson(store.report)
    showJsonReport.value = true
    markExported('json', filename)
  } catch (e: unknown) {
    markExportFailed('json', e)
  }
}
function doExportHtml() {
  if (!store.report) return
  exportPhase.html = 'loading'
  try {
    markExported('html', exportHtml(store.report))
  } catch (e: unknown) {
    markExportFailed('html', e)
  }
}
async function doExportPdf() {
  if (!store.report) return
  exportPhase.pdf = 'loading'
  try {
    markExported('pdf', await exportPdf(store.jobId, store.report))
  } catch (e: unknown) {
    markExportFailed('pdf', e)
  }
}
</script>

<template>
  <div id="vlm-direct" class="scroll-mt-16 max-w-7xl mx-auto px-6 py-6">
    <div class="mb-6 text-center max-w-2xl mx-auto relative">
      <!-- Title Ambient Glow Aura -->
      <div class="heading-glow-section" />

      <h2 class="text-2xl sm:text-3xl font-bold tracking-tight text-slate-100 relative z-10">Direct Analiz</h2>
      <p class="mt-1.5 text-sm sm:text-base text-slate-400 relative z-10">Video doğrudan görsel-dil modeline (EVREN) gönderilerek analiz edilir.</p>
      <div v-if="statusLabel" class="mt-3 inline-flex items-center gap-2 text-xs bg-surface-2 px-3 py-1 rounded-full border border-edge relative z-10">
        <span
          class="status-dot"
          :class="store.isRunning ? 'bg-accent animate-pulse' : store.status === 'done' ? 'bg-risk-low' : store.status === 'error' ? 'bg-risk-crit' : 'bg-slate-600'"
        />
        <span class="uppercase tracking-wide font-semibold" :class="store.isRunning ? 'text-accent' : 'text-slate-300'">{{ statusLabel }}</span>
      </div>
    </div>

    <div v-if="backendHealth === 'offline'" class="mb-5 rounded-md border border-risk-crit/40 bg-risk-crit/10 px-4 py-2.5 text-sm text-risk-crit">
      Arka uca ulaşılamıyor. Analiz servisi şu anda kullanılamayabilir.
    </div>

    <!-- launch bar: bolt.new-style composer, see components/PromptLaunchBar.vue -->
    <PromptLaunchBar
      v-if="!hasRunAnalysis && !store.isRunning"
      v-model="userPrompt"
      :video-label="fileName"
      :can-submit="canSubmit"
      :submitting="store.submitting"
      :error="submitError"
      @pick-file="pickVideo"
      @submit="startAnalysis"
    />
    <!-- once an analysis has run (or is running), the composer collapses to a compact bar so the dashboard below takes over -->
    <div v-else class="card p-4 mb-5 flex flex-col md:flex-row md:items-end gap-3">
      <div class="flex-1 min-w-0">
        <div class="field-label">Video Kaynağı</div>
        <div class="text-sm text-slate-300 truncate font-mono">{{ videoPath || 'Henüz seçilmedi' }}</div>
      </div>
      <div class="flex-1 min-w-0">
        <label class="field-label" for="vlm-direct-prompt">Kullanıcı İstemi</label>
        <input id="vlm-direct-prompt" v-model="userPrompt" class="field-input" />
      </div>
      <div class="flex items-center gap-2 shrink-0">
        <button type="button" class="btn-ghost" :disabled="store.isRunning" @click="pickVideo">{{ videoPath ? 'Videoyu Değiştir' : 'Video Seç' }}</button>
        <button type="button" class="btn-primary" :disabled="!canSubmit" @click="startAnalysis">
          {{ store.submitting ? 'Başlatılıyor…' : 'Yeni Analiz' }}
        </button>
      </div>
    </div>
    <p v-if="submitError && (hasRunAnalysis || store.isRunning)" class="-mt-3 mb-5 text-sm text-risk-crit">{{ submitError }}</p>
    <p v-if="store.status === 'error'" class="-mt-3 mb-5 text-sm text-risk-crit">Analiz tamamlanamadı{{ store.error ? `: ${store.error}` : '.' }}</p>

    <!-- step-by-step VLM progress (video parçalanıyor/gönderiliyor) — bkz. StageCard.vue'daki eşdeğeri -->
    <div v-if="vlmProgress" class="mb-5 rounded-md border border-accent/30 bg-accent/10 px-4 py-3 flex items-center gap-3">
      <span class="inline-block w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin shrink-0 motion-reduce:animate-none" />
      <span class="text-sm text-slate-100 flex-1">{{ vlmStage?.summary }}</span>
      <span v-if="vlmProgress.total_chunks && vlmProgress.total_chunks > 1" class="text-xs font-mono text-slate-400 shrink-0">
        {{ vlmProgress.chunk_index ?? '—' }} / {{ vlmProgress.total_chunks }}
      </span>
    </div>

    <div class="mb-5">
      <VlmStatCards :summary="summary" :duration-seconds="effectiveDuration" />
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">
      <div class="lg:col-span-2">
        <VlmVideoPanel
          :video-url="videoUrl"
          :file-name="fileName"
          :events="events"
          :duration="effectiveDuration"
          :current-time="currentTime"
          :active-event-id="activeEventId"
          @pick-file="pickVideo"
          @time-update="(t: number) => (currentTime = t)"
          @duration-change="(d: number) => (duration = d)"
          @select-event="onSelectEvent"
        />
      </div>
      <div class="lg:col-span-1">
        <VlmRiskCharts
          :risk-counts="riskCounts"
          :type-counts="typeCounts"
          :report-ready="reportReady"
          :export-phase="exportPhase"
          :export-message="exportMessage"
          @export-json="doExportJson"
          @export-html="doExportHtml"
          @export-pdf="doExportPdf"
        />
      </div>
    </div>

    <div v-if="hasRunAnalysis && store.status === 'done' && !events.length" class="mt-5 card p-8 text-center">
      <p class="text-sm font-semibold text-slate-200">Kritik olay tespit edilmedi</p>
      <p class="mt-1 text-sm text-slate-500">Analiz tamamlandı. Bu videoda tanımlı güvenlik eşiklerini aşan bir olay bulunamadı.</p>
    </div>
    <div v-else class="mt-5">
      <VlmEventList :events="events" :active-event-id="activeEventId" @select="onSelectEvent" />
    </div>

    <!-- nihai JSON raporu (şartname formatı) — "Olay Türü Dağılımı"
         panelindeki JSON düğmesine basılınca burada, dashboard'un altında
         açılır; indirilen dosyayla BİREBİR AYNI içerik (bkz. buildSartnameJson). -->
    <div v-if="showJsonReport && store.report" class="mt-5 card p-4">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-sm font-semibold text-slate-100">Nihai JSON Raporu (Şartname Formatı)</h3>
        <button type="button" class="btn-ghost text-xs px-2 py-1" @click="showJsonReport = false">Kapat</button>
      </div>
      <pre class="text-[11px] font-mono text-slate-400 bg-surface-2 border border-edge rounded-md p-3 max-h-96 overflow-auto">{{ JSON.stringify(buildSartnameJson(store.report), null, 2) }}</pre>
    </div>
  </div>
</template>
