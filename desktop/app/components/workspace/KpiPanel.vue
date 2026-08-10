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
  <div v-if="stats" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
    <MetricCell label="Frames scanned" :value="stats.total_frames_scanned ?? 0" />
    <MetricCell label="Frames evaluated" :value="stats.sampled_frames_evaluated ?? 0" />
    <MetricCell label="Evidence frames" :value="evidenceCount" />
    <MetricCell label="GPU savings" :value="`%${(stats.gpu_savings_ratio_pct ?? 0).toFixed?.(1) ?? stats.gpu_savings_ratio_pct}`" />
    <MetricCell label="Duration" :value="`${(stats.elapsed_sec ?? 0)}s`" />
    <MetricCell label="ms / frame" :value="msPerFrame.toFixed(1)" />
  </div>
</template>
