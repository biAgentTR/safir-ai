<script setup lang="ts">
// Compact, real Event Timeline for the dashboard: the most recent completed
// analysis that has a persisted report. Shows the analysis duration, evidence
// frame markers, and — when the pipeline trace was persisted for that job —
// the VLM-derived event intervals with a real per-event risk level. Nothing
// here is synthetic: severity coloring only appears when a numeric
// risk_score backs it (VlmEvent.risk_score); SafirReport.timeline entries
// carry no severity field, so those render as neutral markers.
// Selecting any marker jumps to that analysis's full report.
import type { HistoryListItem, TimelineEntry, VlmEvent } from '~/types/api'

const props = defineProps<{ recent: HistoryListItem[] }>()

const api = useSafirApi()
const router = useRouter()

const candidate = computed(
  () => props.recent.find((h) => h.status === 'completed') ?? null,
)

const loading = ref(false)
const error = ref<string | null>(null)
const timeline = ref<TimelineEntry[]>([])
const vlmEvents = ref<VlmEvent[]>([])
const evidenceCount = ref(0)
const loadedJobId = ref<string | null>(null)

async function load(jobId: string) {
  loading.value = true
  error.value = null
  try {
    const detail = await api.getHistoryItem(jobId)
    timeline.value = detail.report?.timeline ?? []
    evidenceCount.value = detail.report?.evidence_frames?.length ?? 0
    const vlmStage = detail.trace_events?.find((e) => e.stage === 'vlm')
    vlmEvents.value = ((vlmStage?.data as { structured_events?: VlmEvent[] } | undefined)?.structured_events) ?? []
    loadedJobId.value = jobId
  } catch (e) {
    error.value = (e as { data?: { detail?: string } })?.data?.detail ?? (e as Error)?.message ?? 'Zaman çizelgesi yüklenemedi.'
  } finally {
    loading.value = false
  }
}

watch(
  candidate,
  (c) => {
    if (c && c.job_id !== loadedJobId.value) load(c.job_id)
  },
  { immediate: true },
)

const hasData = computed(() => timeline.value.length > 0 || vlmEvents.value.length > 0)

// Span of the timeline: latest of any timestamp we have (VLM event ends,
// timeline entries) — no separate "video duration" field exists on the
// history/report contract, so the last known event marks the extent.
const spanSec = computed(() => {
  const ends = [
    ...timeline.value.map((e) => e.timestamp),
    ...vlmEvents.value.map((e) => e.end_time),
  ]
  return ends.length ? Math.max(...ends) : 0
})

function pct(t: number): number {
  if (spanSec.value <= 0) return 0
  return Math.min(100, Math.max(0, (t / spanSec.value) * 100))
}

/** VlmEvent.risk_score (0-100) -> the same 4-tier vocabulary used everywhere
 * else in the app (mirrors CRITICAL_ALARM_THRESHOLD=70 for the top tier). */
function scoreTone(score: number | null): 'crit' | 'high' | 'mid' | 'low' {
  if (score == null) return 'low'
  if (score >= 70) return 'crit'
  if (score >= 50) return 'high'
  if (score >= 25) return 'mid'
  return 'low'
}
const toneLabel: Record<string, string> = { crit: 'Kritik', high: 'Yüksek', mid: 'Dikkat', low: 'Normal' }

interface ListEntry {
  key: string
  time: number
  tone: 'crit' | 'high' | 'mid' | 'low' | null
  description: string
}
const listEntries = computed<ListEntry[]>(() =>
  vlmEvents.value.length
    ? vlmEvents.value.map((ev) => ({ key: ev.event_id, time: ev.start_time, tone: scoreTone(ev.risk_score), description: ev.description }))
    : timeline.value.map((e, i) => ({ key: `t${i}`, time: e.timestamp, tone: null, description: e.description })),
)

function openReport() {
  if (candidate.value) router.push(`/reports/${candidate.value.job_id}`)
}
</script>

<template>
  <div class="card p-5">
    <div class="flex items-center justify-between mb-3">
      <h3 class="text-sm font-semibold text-slate-100">Olay Zaman Çizelgesi</h3>
      <button v-if="candidate" type="button" class="text-xs text-accent hover:underline" @click="openReport">
        Tam raporu aç →
      </button>
    </div>

    <div v-if="!candidate" class="py-8 text-center text-sm text-slate-500">
      Henüz tamamlanmış analiz yok — zaman çizelgesi ilk analiz tamamlandığında burada görünecek.
    </div>
    <div v-else-if="loading" class="py-8 text-center text-sm text-slate-500">Yükleniyor…</div>
    <div v-else-if="error" class="py-6 text-center text-sm text-risk-crit">{{ error }}</div>
    <div v-else-if="!hasData" class="py-8 text-center text-sm text-slate-500">
      Seçilen analiz için zaman çizelgesi verisi kaydedilmemiş.
    </div>

    <div v-else>
      <div class="flex items-center justify-between text-[11px] text-slate-500 mb-2 font-mono">
        <span>00:00</span>
        <span>{{ mmss(spanSec) }} · {{ evidenceCount }} kanıt karesi</span>
      </div>

      <!-- VLM event intervals (real risk_score -> tone), when the pipeline trace was persisted -->
      <div v-if="vlmEvents.length" class="relative h-9 rounded-md bg-surface-2 border border-edge mb-1">
        <button
          v-for="ev in vlmEvents"
          :key="ev.event_id"
          type="button"
          class="group absolute top-1/2 -translate-y-1/2 h-4 rounded-sm focus-visible:z-10 hover:opacity-80"
          :class="RISK_BG[scoreTone(ev.risk_score)]"
          :style="{ left: pct(ev.start_time) + '%', width: Math.max(1.5, pct(ev.end_time) - pct(ev.start_time)) + '%' }"
          :aria-label="`${mmss(ev.start_time)}–${mmss(ev.end_time)} · ${toneLabel[scoreTone(ev.risk_score)]}: ${ev.description}`"
          :title="`${mmss(ev.start_time)}–${mmss(ev.end_time)} · ${toneLabel[scoreTone(ev.risk_score)]} — ${ev.description}`"
          @click="openReport"
        />
      </div>
      <div v-else class="relative h-9 rounded-md bg-surface-2 border border-edge mb-1">
        <button
          v-for="(e, i) in timeline"
          :key="i"
          type="button"
          class="group absolute top-1/2 -translate-y-1/2 -translate-x-1/2 focus-visible:z-10"
          :style="{ left: pct(e.timestamp) + '%' }"
          :aria-label="`${mmss(e.timestamp)}: ${e.description}`"
          :title="`${mmss(e.timestamp)} — ${e.description}`"
          @click="openReport"
        >
          <span class="block w-2.5 h-2.5 rounded-full border-2 border-surface-1 bg-accent group-hover:scale-125 transition-transform" />
        </button>
      </div>

      <div v-if="vlmEvents.length" class="flex flex-wrap gap-x-3 gap-y-1 mb-3 text-[10px] text-slate-500">
        <span v-for="t in ['low', 'mid', 'high', 'crit']" :key="t" class="inline-flex items-center gap-1">
          <span class="w-2 h-2 rounded-sm" :class="RISK_BG[t]" aria-hidden="true" />{{ toneLabel[t] }}
        </span>
      </div>

      <ul class="mt-2 space-y-1.5 max-h-40 overflow-y-auto">
        <li v-for="entry in listEntries.slice(0, 6)" :key="entry.key" class="flex items-center gap-2 text-xs">
          <span class="w-1.5 h-1.5 rounded-full shrink-0" :class="entry.tone ? RISK_BG[entry.tone] : 'bg-accent'" aria-hidden="true" />
          <span class="font-mono text-slate-500 shrink-0">{{ mmss(entry.time) }}</span>
          <span v-if="entry.tone" class="text-slate-400 shrink-0">{{ toneLabel[entry.tone] }}</span>
          <span class="text-slate-300 truncate">{{ entry.description }}</span>
        </li>
      </ul>
      <p v-if="listEntries.length > 6" class="mt-1.5 text-[11px] text-slate-600">
        + {{ listEntries.length - 6 }} olay daha — tam rapor içinde
      </p>
    </div>
  </div>
</template>
