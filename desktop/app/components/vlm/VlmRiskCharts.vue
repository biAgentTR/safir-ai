<script setup lang="ts">
// Risk-level and event-type distribution, as simple horizontal bar charts.
// No charting library in this project (see desktop/package.json) — plain
// divs sized by percentage keep this dependency-free and on-brand (uses the
// same risk color tokens as the rest of SAFIR).
import type { VlmRiskLevel } from '~/types/vlm'

const props = defineProps<{
  riskCounts: Record<VlmRiskLevel, number>
  typeCounts: { type: string; count: number }[]
  reportReady?: boolean
}>()

const emit = defineEmits<{
  (e: 'export-json' | 'export-html' | 'export-pdf'): void
}>()

const RISK_LABEL: Record<VlmRiskLevel, string> = { crit: 'Kritik', high: 'Yüksek', mid: 'Orta', low: 'Düşük' }
const RISK_ORDER: VlmRiskLevel[] = ['crit', 'high', 'mid', 'low']

const riskTotal = computed(() => Object.values(props.riskCounts).reduce((a, b) => a + b, 0) || 1)
const maxTypeCount = computed(() => Math.max(1, ...props.typeCounts.map((t) => t.count)))
</script>

<template>
  <div class="grid grid-cols-1 gap-4">
    <div class="card p-4">
      <h3 class="text-sm font-semibold text-slate-100 mb-3">Risk Seviyesi Dağılımı</h3>
      <div class="space-y-2.5">
        <div v-for="lvl in RISK_ORDER" :key="lvl" class="flex items-center gap-2">
          <span class="w-14 shrink-0 text-xs text-slate-400">{{ RISK_LABEL[lvl] }}</span>
          <div class="flex-1 h-2.5 rounded-full bg-surface-2 overflow-hidden">
            <div class="h-full rounded-full" :class="RISK_BG[lvl]" :style="{ width: (riskCounts[lvl] / riskTotal) * 100 + '%' }" />
          </div>
          <span class="w-5 shrink-0 text-right text-xs font-mono text-slate-400">{{ riskCounts[lvl] }}</span>
        </div>
      </div>
    </div>

    <div class="card p-4">
      <h3 class="text-sm font-semibold text-slate-100 mb-3">Olay Türü Dağılımı</h3>
      <div v-if="!typeCounts.length" class="text-sm text-slate-500 py-4 text-center">Henüz olay yok.</div>
      <div v-else class="space-y-2.5">
        <div v-for="t in typeCounts" :key="t.type" class="flex items-center gap-2">
          <span class="w-40 shrink-0 text-xs text-slate-400 truncate" :title="t.type">{{ t.type }}</span>
          <div class="flex-1 h-2.5 rounded-full bg-surface-2 overflow-hidden">
            <div class="h-full rounded-full bg-accent" :style="{ width: (t.count / maxTypeCount) * 100 + '%' }" />
          </div>
          <span class="w-5 shrink-0 text-right text-xs font-mono text-slate-400">{{ t.count }}</span>
        </div>
      </div>

      <div class="mt-4 pt-3 border-t border-edge flex flex-wrap items-center gap-2">
        <span class="text-[11px] text-slate-500">Rapor:</span>
        <button
          type="button"
          class="btn-ghost text-xs px-2 py-1"
          :disabled="!reportReady"
          :title="reportReady ? 'JSON raporu indir ve altta görüntüle' : 'Rapor henüz hazır değil'"
          @click="emit('export-json')"
        >JSON</button>
        <button
          type="button"
          class="btn-ghost text-xs px-2 py-1"
          :disabled="!reportReady"
          :title="reportReady ? 'HTML raporu indir' : 'Rapor henüz hazır değil'"
          @click="emit('export-html')"
        >HTML</button>
        <button
          type="button"
          class="btn-ghost text-xs px-2 py-1"
          :disabled="!reportReady"
          :title="reportReady ? 'PDF raporu indir' : 'Rapor henüz hazır değil'"
          @click="emit('export-pdf')"
        >PDF</button>
      </div>
    </div>
  </div>
</template>
