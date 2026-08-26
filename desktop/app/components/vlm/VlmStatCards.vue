<script setup lang="ts">
// Summary KPI strip at the top of the VLM Direct dashboard.
import type { VlmAnalysisSummary } from '~/types/vlm'

const props = defineProps<{ summary: VlmAnalysisSummary; durationSeconds: number }>()

const scoreTone = computed(() => riskTone(
  props.summary.overallRiskScore >= 85 ? 'kritik'
    : props.summary.overallRiskScore >= 60 ? 'yuksek'
    : props.summary.overallRiskScore >= 30 ? 'orta'
    : 'dusuk',
))
</script>

<template>
  <div class="instrument-strip">
    <div class="instrument-cell">
      <div class="eyebrow">Video Süresi</div>
      <div class="mt-1 text-xl font-bold text-slate-100 tabular-nums font-mono">{{ durationSeconds ? mmss(durationSeconds) : '—' }}</div>
    </div>
    <div class="instrument-cell">
      <div class="eyebrow">Tespit Edilen Olay</div>
      <div class="mt-1 text-xl font-bold text-slate-100 tabular-nums">{{ summary.totalEvents }}</div>
    </div>
    <div class="instrument-cell">
      <div class="eyebrow">Kritik Olay</div>
      <div class="mt-1 text-xl font-bold tabular-nums" :class="summary.criticalEvents ? 'text-risk-crit' : 'text-slate-100'">{{ summary.criticalEvents }}</div>
    </div>
    <div class="instrument-cell">
      <div class="eyebrow">Genel Risk Skoru</div>
      <div class="mt-1 text-xl font-bold tabular-nums" :class="RISK_TEXT[scoreTone]">{{ summary.totalEvents ? summary.overallRiskScore : '—' }}</div>
    </div>
  </div>
</template>
