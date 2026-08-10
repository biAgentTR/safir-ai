<script setup lang="ts">
// Analysis workspace. On mount it (1) opens the real SSE stream for the job and
// (2) polls GET /analyze/jobs/{id} for the authoritative final SafirReport.
// The pipeline view here is intentionally the basic technical StageList; the
// rich Pipeline Timeline is ADIM 4.
import type { EvidenceFrameRef, SamplerStageData } from '~/types/api'

const route = useRoute()
const jobId = computed(() => String(route.params.jobId))

const store = useAnalysisStore()
const stream = useAnalysisStream()
const { getFrameUrl } = useSafirApi()

const connLabel = computed(
  () =>
    ({
      idle: 'Bağlanılıyor…',
      connecting: 'Bağlanılıyor…',
      open: 'Canlı (SSE)',
      closed: 'Kapandı',
      error: 'Bağlantı hatası',
    })[stream.connection.value],
)
const connTone = computed(
  () =>
    ({
      idle: 'text-slate-400',
      connecting: 'text-slate-400',
      open: 'text-risk-low',
      closed: 'text-slate-400',
      error: 'text-risk-crit',
    })[stream.connection.value],
)

const overallStatus = computed(() => store.status ?? stream.endStatus.value ?? 'queued')

// Evidence frames referenced by the sampler stage -> proven via frame endpoint.
const evidenceFrames = computed<EvidenceFrameRef[]>(() => {
  const ev = stream.events.value.find((e) => e.stage === 'sampler')
  return (ev?.data as SamplerStageData | undefined)?.evidence_frames ?? []
})

function start() {
  stream.start(jobId.value)
  // Fire-and-forget poll for the final report; errors are surfaced via store.
  store.pollUntilDone(jobId.value).catch(() => {})
}

onMounted(start)
onBeforeUnmount(() => stream.stop())
</script>

<template>
  <div class="max-w-6xl mx-auto px-6 py-6 space-y-5">
    <!-- header -->
    <div class="flex items-center gap-4">
      <NuxtLink to="/new-analysis" class="btn-ghost">← Yeni</NuxtLink>
      <div class="min-w-0">
        <div class="text-xs uppercase tracking-wide text-slate-500">Job</div>
        <div class="font-mono text-sm text-slate-200 truncate">{{ jobId }}</div>
      </div>
      <div class="ml-auto flex items-center gap-2 text-sm">
        <span class="w-2 h-2 rounded-full" :class="stream.connection.value === 'open' ? 'bg-risk-low animate-pulse' : 'bg-slate-600'" />
        <span :class="connTone">{{ connLabel }}</span>
      </div>
    </div>

    <!-- status strip -->
    <div class="grid grid-cols-4 gap-4">
      <div class="card p-4">
        <div class="text-xs uppercase tracking-wide text-slate-500">Durum</div>
        <div class="mt-1 text-lg font-semibold text-slate-100">{{ overallStatus }}</div>
      </div>
      <div class="card p-4">
        <div class="text-xs uppercase tracking-wide text-slate-500">Aktif Aşama</div>
        <div class="mt-1 text-lg font-semibold text-slate-100">
          {{ stream.latestEvent.value?.metadata.label ?? '—' }}
        </div>
      </div>
      <div class="card p-4">
        <div class="text-xs uppercase tracking-wide text-slate-500">Tamamlanan</div>
        <div class="mt-1 text-lg font-semibold text-slate-100">
          {{ stream.completedCount.value }} / 7
        </div>
      </div>
      <div class="card p-4">
        <div class="text-xs uppercase tracking-wide text-slate-500">Risk</div>
        <div class="mt-1 text-lg font-semibold" :class="store.report ? 'text-slate-100' : 'text-slate-600'">
          {{ store.report ? `${store.report.risk_level} (${store.report.risk_score})` : '—' }}
        </div>
      </div>
    </div>

    <!-- pipeline (technical list — ADIM 4 upgrades this) -->
    <div class="card p-5">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-semibold text-slate-200">Boru Hattı (canlı trace)</h3>
        <span class="text-xs text-slate-600">{{ stream.events.value.length }} olay</span>
      </div>
      <StageList :events="stream.events.value" :connected="stream.connection.value === 'open'" />
      <p v-if="stream.errorDetail.value" class="mt-3 text-xs text-risk-crit">
        {{ stream.errorDetail.value }}
      </p>
    </div>

    <!-- evidence frames via the frame endpoint (no base64 in SSE) -->
    <div v-if="evidenceFrames.length" class="card p-5">
      <h3 class="text-sm font-semibold text-slate-200 mb-3">Kanıt Kareleri</h3>
      <div class="flex flex-wrap gap-3">
        <figure
          v-for="f in evidenceFrames"
          :key="f.frame_id"
          class="w-40 border border-edge rounded-md overflow-hidden bg-surface-2"
        >
          <img
            :src="getFrameUrl(jobId, f.frame_id)"
            :alt="f.frame_id"
            class="w-full h-24 object-cover"
            loading="lazy"
          />
          <figcaption class="px-2 py-1 text-[11px] text-slate-400 font-mono">
            {{ f.timestamp_str }} · Δ{{ f.change_score }}
          </figcaption>
        </figure>
      </div>
    </div>

    <!-- raw event feed (debug/technical) -->
    <details class="card p-5">
      <summary class="text-sm font-semibold text-slate-200 cursor-pointer">Ham Trace Olayları (debug)</summary>
      <pre class="mt-3 max-h-80 overflow-auto text-[11px] leading-relaxed text-slate-400 font-mono">{{ JSON.stringify(stream.events.value, null, 2) }}</pre>
    </details>
  </div>
</template>
