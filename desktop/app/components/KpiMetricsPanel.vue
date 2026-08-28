<script setup lang="ts">
// Şartnamedeki "Ölçümleme ve KPI Tanımlama" maddesinin karşılığı: sistemin
// başarısını ölçen beş KPI. Hesaplama ve tanımlar useKpiMetrics.ts içinde,
// sadece gerçek analiz verisinden; her ölçütün formülü kartın `title`
// ipucunda durur. Görsel dil ve ölçü AI Metrikleri paneliyle aynıdır.
const { kpis, hasData } = useKpiMetrics()
const { closeDeck } = useMetricsDeck()

const TONE_TEXT: Record<string, string> = {
  good: 'text-risk-low',
  warn: 'text-risk-mid',
  bad: 'text-risk-crit',
  neutral: 'text-slate-100',
  muted: 'text-slate-500',
}

// İlk dördü yüzde tabanlı ölçüt (2x2 kart ızgarası), sonuncusu süre.
const cardKpis = computed(() => kpis.value.slice(0, 4))
const timeKpi = computed(() => kpis.value[kpis.value.length - 1])
</script>

<template>
  <div class="h-full">
    <div class="flex flex-col h-full">
      <!-- Header -->
      <div class="flex items-center justify-between pb-3 border-b border-edge/50">
        <div class="flex items-center gap-2">
          <div class="w-4 h-4 text-accent flex items-center justify-center">
            <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4 20V10m6 10V4m6 16v-6" />
            </svg>
          </div>
          <span class="font-semibold text-sm tracking-wide text-slate-100">KPI Metrikleri</span>
        </div>
        <div class="flex items-center gap-3">
          <span
            class="inline-flex items-center gap-1 text-[11px] font-mono font-medium"
            :class="hasData ? 'text-emerald-400/90' : 'text-slate-500'"
          >
            <span
              class="w-1.5 h-1.5 rounded-full"
              :class="hasData ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'"
            />
            {{ hasData ? 'ölçüldü' : 'veri yok' }}
          </span>
          <button
            type="button"
            class="text-slate-400 hover:text-slate-200 p-1 rounded-md transition-colors"
            @click="closeDeck"
          >
            <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      <!-- 4 oran KPI'ı (2x2) — kutunun boyunu doldurur -->
      <div class="grid grid-cols-2 grid-rows-2 gap-2.5 my-3.5 flex-1 min-h-0">
        <div
          v-for="k in cardKpis"
          :key="k.key"
          class="rounded-xl border border-edge/60 bg-surface-2/80 p-3 flex flex-col justify-between"
          :title="k.formula"
        >
          <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider leading-snug">
            {{ trUpper(k.label) }}
          </div>
          <div class="text-base sm:text-lg font-bold tabular-nums mt-1" :class="TONE_TEXT[k.tone]">
            {{ k.display }}
          </div>
          <div class="text-[10px] text-slate-500 mt-0.5 leading-snug">{{ k.detail }}</div>
        </div>
      </div>

      <!-- İşlem süresi -->
      <div
        v-if="timeKpi"
        class="flex items-center justify-between text-[11px] text-slate-400 tabular-nums py-1 border-b border-edge/40"
        :title="timeKpi.formula"
      >
        <span>{{ timeKpi.label.toLocaleLowerCase('tr-TR') }}: <span class="text-slate-200 font-mono">{{ timeKpi.display }}</span></span>
        <span>{{ timeKpi.detail }}</span>
      </div>

      <!-- Footer -->
      <div class="mt-3.5 pt-2.5 border-t border-edge/50 text-[11px] text-slate-500">
        <span v-if="hasData">kaynak: <span class="font-mono text-slate-400">canlı analiz verisi</span></span>
        <span v-else>Analiz çalıştırıldığında KPI'lar gerçek veriden hesaplanır.</span>
      </div>
    </div>
  </div>
</template>
