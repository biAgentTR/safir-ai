<script setup lang="ts">
// SAFİR Analysis Workspace. Central place the operator works after starting an
// analysis: cumulative pipeline timeline (left) + selected-stage detail (right),
// risk summary, KPI, and tabbed Evidence / Timeline / Report at the bottom.
// All state comes from the Pinia store; SSE + polling feed it.
const route = useRoute()
const jobId = computed(() => String(route.params.jobId))

const store = useAnalysisStore()
const stream = useAnalysisStream()
const { maybeAlarm } = useAlarm()
const { state: backendHealth } = useBackendHealth()

type Tab = 'evidence' | 'timeline' | 'report'
const tab = ref<Tab>('report')

// connection presentation
const connLabel = computed(
  () =>
    ({ idle: 'Bağlanılıyor…', connecting: 'Bağlanılıyor…', open: 'Canlı (SSE)', closed: 'Tamamlandı', error: 'Bağlantı hatası' })[
      store.connection
    ],
)
const running = computed(() => store.isRunning && store.status !== 'error')
const overall = computed(() => store.status ?? store.endStatus ?? 'queued')

// play the critical alarm once per (job, score) when the report arrives
watch(
  () => store.report?.risk_score,
  () => {
    if (store.report && store.jobId) maybeAlarm(`${store.jobId}:${store.report.risk_score}`)
  },
)

function start() {
  // if navigating to a different job, reset first
  if (store.jobId !== jobId.value) {
    store.resetJob()
    store.jobId = jobId.value
    store.status = 'queued'
  }
  stream.start(jobId.value)
  store.pollUntilDone(jobId.value).catch(() => {})
}

onMounted(start)
onBeforeUnmount(() => stream.stop())
</script>

<template>
  <div class="px-6 py-5 space-y-5 max-w-[1500px] mx-auto">
    <!-- header -->
    <div class="flex items-center gap-4">
      <NuxtLink to="/new-analysis" class="btn-ghost">← Yeni</NuxtLink>
      <div class="min-w-0">
        <div class="text-[11px] uppercase tracking-wide text-slate-500">Job</div>
        <div class="font-mono text-sm text-slate-300 truncate">{{ jobId }}</div>
      </div>
      <div class="ml-auto flex items-center gap-4 text-sm">
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full" :class="running ? 'bg-accent animate-pulse' : overall === 'done' ? 'bg-risk-low' : overall === 'error' ? 'bg-risk-crit' : 'bg-slate-600'" />
          <span class="uppercase tracking-wide text-xs" :class="running ? 'text-accent' : 'text-slate-300'">
            {{ running ? 'Analysis running' : overall === 'done' ? 'Completed' : overall === 'error' ? 'Error' : overall }}
          </span>
        </div>
        <span class="text-xs text-slate-500">{{ connLabel }} · {{ store.completedCount }}/7</span>
      </div>
    </div>

    <!-- degraded: backend offline -->
    <div v-if="backendHealth === 'offline'" class="rounded-md border border-risk-crit/40 bg-risk-crit/10 px-4 py-2.5 text-sm text-risk-crit">
      Backend'e ulaşılamıyor. Analiz servisi çevrimdışı olabilir.
    </div>

    <!-- degraded: analysis error -->
    <div v-if="store.status === 'error'" class="rounded-md border border-risk-crit/40 bg-risk-crit/10 px-4 py-3 text-sm text-slate-200">
      Analiz tamamlanamadı.
      <details v-if="store.error" class="mt-1">
        <summary class="cursor-pointer text-xs text-slate-400">Teknik detay</summary>
        <p class="mt-1 text-xs text-slate-500">{{ store.error }}</p>
      </details>
    </div>

    <!-- risk summary (appears with the report) -->
    <RiskSummary />

    <!-- KPI -->
    <KpiPanel />

    <!-- main: pipeline rail + selected stage detail -->
    <div class="grid grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)] gap-5 items-start">
      <PipelineTimeline />
      <StageCard @open-frame="() => { tab = 'evidence' }" />
    </div>

    <!-- bottom tabs -->
    <div class="card">
      <div class="flex items-center gap-1 border-b border-edge px-3">
        <button
          v-for="t in (['report','evidence','timeline'] as Tab[])"
          :key="t"
          class="px-4 py-2.5 text-sm border-b-2 -mb-px transition-colors"
          :class="tab === t ? 'border-accent text-slate-100' : 'border-transparent text-slate-400 hover:text-slate-200'"
          @click="tab = t"
        >
          {{ t === 'report' ? 'Rapor' : t === 'evidence' ? 'Kanıt Galerisi' : 'Zaman Çizelgesi' }}
        </button>
      </div>
      <div class="p-5">
        <FinalReport v-show="tab === 'report'" />
        <EvidenceGallery v-show="tab === 'evidence'" />
        <EventTimeline v-show="tab === 'timeline'" />
      </div>
    </div>

    <!-- stream error footnote -->
    <p v-if="store.streamError" class="text-xs text-risk-crit">Canlı akış: {{ store.streamError }}</p>
  </div>
</template>
