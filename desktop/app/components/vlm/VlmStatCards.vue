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
  <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
    <div class="card px-4 py-3">
      <div class="text-[11px] uppercase tracking-wide text-slate-500">Video Süresi</div>
      <div class="mt-1 text-xl font-semibold text-slate-100">{{ durationSeconds ? mmss(durationSeconds) : '—' }}</div>
    </div>
    <div class="card px-4 py-3">
      <div class="text-[11px] uppercase tracking-wide text-slate-500">Tespit Edilen Olay</div>
      <div class="mt-1 text-xl font-semibold text-slate-100">{{ summary.totalEvents }}</div>
    </div>
    <div class="card px-4 py-3">
      <div class="text-[11px] uppercase tracking-wide text-slate-500">Kritik Olay</div>
      <div class="mt-1 text-xl font-semibold" :class="summary.criticalEvents ? 'text-risk-crit' : 'text-slate-100'">{{ summary.criticalEvents }}</div>
    </div>
    <div class="card px-4 py-3">
      <div class="text-[11px] uppercase tracking-wide text-slate-500">Genel Risk Skoru</div>
      <div class="mt-1 text-xl font-semibold" :class="RISK_TEXT[scoreTone]">{{ summary.totalEvents ? summary.overallRiskScore : '—' }}</div>
    </div>
  </div>
</template>
