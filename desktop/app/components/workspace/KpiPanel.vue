<script setup lang="ts">
// Field KPI metric cards from REAL sampler stats + report. Mirrors the
// Streamlit KpiPanel (frames scanned/evaluated, evidence, GPU savings, duration,
// ms/frame, active models).
import type { SamplerStageData } from '~/types/api'

const store = useAnalysisStore()

// Prefer the final report's sampler_stats; fall back to the live sampler trace.
const stats = computed(() => {
  const rep = store.report?.sampler_stats
  if (rep) return rep
  const s = store.eventForStage('sampler')?.data as SamplerStageData | undefined
  return s?.stats
})

const msPerFrame = computed(() => {
  const st = stats.value
  if (!st?.total_frames_scanned || !st.elapsed_sec) return 0
  return (st.elapsed_sec / st.total_frames_scanned) * 1000
})
const evidenceCount = computed(
  () => store.report?.evidence_frames?.length ?? stats.value?.evidence_frame_count ?? 0,
)
</script>

<template>
  <div v-if="stats" class="instrument-strip">
    <div class="instrument-cell">
      <div class="eyebrow">Taranan Kare</div>
      <div class="mt-0.5 text-lg font-semibold text-slate-100 tabular-nums">{{ stats.total_frames_scanned ?? 0 }}</div>
    </div>
    <div class="instrument-cell">
      <div class="eyebrow">Değerlendirilen</div>
      <div class="mt-0.5 text-lg font-semibold text-slate-100 tabular-nums">{{ stats.sampled_frames_evaluated ?? 0 }}</div>
    </div>
    <div class="instrument-cell">
      <div class="eyebrow">Kanıt Karesi</div>
      <div class="mt-0.5 text-lg font-semibold text-slate-100 tabular-nums">{{ evidenceCount }}</div>
    </div>
    <div class="instrument-cell">
      <div class="eyebrow">GPU Tasarrufu</div>
      <div class="mt-0.5 text-lg font-semibold text-accent tabular-nums">%{{ (stats.gpu_savings_ratio_pct ?? 0).toFixed?.(1) ?? stats.gpu_savings_ratio_pct }}</div>
    </div>
    <div class="instrument-cell">
      <div class="eyebrow">Süre</div>
      <div class="mt-0.5 text-lg font-semibold text-slate-100 tabular-nums font-mono">{{ stats.elapsed_sec ?? 0 }}s</div>
    </div>
    <div class="instrument-cell">
      <div class="eyebrow">ms / Kare</div>
      <div class="mt-0.5 text-lg font-semibold text-slate-100 tabular-nums font-mono">{{ msPerFrame.toFixed(1) }}</div>
    </div>
  </div>
</template>
