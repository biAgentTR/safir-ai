<script setup lang="ts">
// Compact detail overlay for a single timeline event (EventTimeline.vue).
// Everything shown here is derived from the already-loaded SafirReport —
// the timeline entry itself, the nearest report.evidence_frames entry by
// timestamp (base64 image is already embedded in the persisted report, so
// this works identically in live and history Workspace), and the report's
// overall risk. No new endpoint, no new data model.
import type { EvidenceFrameOut, TimelineEntry } from '~/types/api'

const props = defineProps<{ entry: TimelineEntry | null }>()
const emit = defineEmits<{ (e: 'close'): void }>()

const store = useAnalysisStore()

const nearestFrame = computed<{ frame: EvidenceFrameOut; deltaSec: number } | null>(() => {
  const entry = props.entry
  const frames = store.report?.evidence_frames ?? []
  if (!entry || !frames.length) return null
  let best: EvidenceFrameOut | null = null
  let bestDelta = Infinity
  for (const f of frames) {
    const delta = Math.abs(f.timestamp_sec - entry.timestamp)
    if (delta < bestDelta) {
      best = f
      bestDelta = delta
    }
  }
  return best ? { frame: best, deltaSec: bestDelta } : null
})

const isUnknownRisk = computed(() => store.report?.risk_status === 'unknown' || store.report?.risk_score == null)
const tone = computed(() => (isUnknownRisk.value ? 'unknown' : riskTone(store.report?.risk_level)))
</script>

<template>
  <div
    v-if="entry"
    class="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-6"
    @click="emit('close')"
  >
    <div class="card max-w-lg w-full p-5" @click.stop>
      <div class="flex items-start justify-between gap-4 mb-4">
        <div class="min-w-0">
          <div class="text-xs font-mono text-slate-400">
            {{ mmss(entry.timestamp) }} <span class="text-slate-600">({{ entry.timestamp.toFixed(1) }}s)</span>
          </div>
          <div class="text-sm text-slate-100 mt-1 leading-relaxed">{{ entry.description }}</div>
        </div>
        <button type="button" class="btn-ghost shrink-0 px-2 py-1 text-xs" @click="emit('close')">Kapat</button>
      </div>

      <div v-if="store.report" class="mb-4 flex items-center gap-2 text-xs">
        <span class="text-slate-500">Analiz risk seviyesi:</span>
        <span class="font-semibold" :class="RISK_TEXT[tone]">{{ isUnknownRisk ? 'Belirsiz' : `${store.report.risk_score} / 100 · ${store.report.risk_level}` }}</span>
      </div>

      <div v-if="nearestFrame">
        <div class="field-label">En yakın kanıt karesi (Δ{{ nearestFrame.deltaSec.toFixed(1) }}s)</div>
        <img
          :src="nearestFrame.frame.base64_image"
          :alt="nearestFrame.frame.timestamp_str"
          class="w-full max-h-64 object-contain rounded-md border border-edge bg-surface-2"
        />
        <div class="mt-2 text-[11px] text-slate-500 font-mono">
          {{ nearestFrame.frame.timestamp_str }} · değişim skoru {{ nearestFrame.frame.change_score }}
          <span v-if="nearestFrame.frame.is_fallback" class="text-risk-mid"> · fallback</span>
        </div>
      </div>
      <div v-else class="text-sm text-slate-500 py-4 text-center">
        Bu olay için ilişkili kanıt karesi bulunamadı.
      </div>
    </div>
  </div>
</template>
