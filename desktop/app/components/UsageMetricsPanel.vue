<script setup lang="ts">
import { durationMs, tokenCount } from '~/utils/format'

const { isPanelOpen, metrics, keyInfo, closePanel } = useUsageMetrics()

const isRefreshing = ref(false)
function triggerRefresh() {
  isRefreshing.value = true
  setTimeout(() => {
    isRefreshing.value = false
  }, 400)
}
</script>

<template>
  <Transition
    enter-active-class="transition duration-200 ease-out"
    enter-from-class="opacity-0 scale-95 translate-y-2"
    enter-to-class="opacity-100 scale-100 translate-y-0"
    leave-active-class="transition duration-150 ease-in"
    leave-from-class="opacity-100 scale-100 translate-y-0"
    leave-to-class="opacity-0 scale-95 translate-y-2"
  >
    <div
      v-if="isPanelOpen"
      class="fixed z-50 bottom-20 right-6 sm:bottom-24 sm:right-8 w-84 sm:w-96 rounded-2xl bg-surface-1/95 backdrop-blur-2xl border border-edge shadow-2xl p-4 sm:p-5 text-slate-100 transition-all duration-200 ring-1 ring-white/5"
    >
      <!-- Header -->
      <div class="flex items-center justify-between pb-3 border-b border-edge/50">
        <div class="flex items-center gap-2">
          <div class="w-4 h-4 text-accent flex items-center justify-center">
            <svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L2 12l10 10 10-10L12 2zm0 3.6L18.4 12 12 18.4 5.6 12 12 5.6z"/>
            </svg>
          </div>
          <span class="font-semibold text-sm tracking-wide text-slate-100">AI Metrikleri</span>
        </div>
        <div class="flex items-center gap-3">
          <span class="inline-flex items-center gap-1 text-[11px] text-emerald-400/90 font-mono font-medium">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            canlı
          </span>
          <button
            type="button"
            class="text-slate-400 hover:text-slate-200 p-1 rounded-md transition-colors"
            @click="closePanel"
          >
            <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>
      </div>

      <!-- 4 Top Cards (2x2 Grid) -->
      <div class="grid grid-cols-2 gap-2.5 my-3.5">
        <div class="rounded-xl border border-edge/60 bg-surface-2/80 p-3 flex flex-col justify-between">
          <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">GEÇEN SÜRE</div>
          <div class="text-base sm:text-lg font-bold text-slate-100 tabular-nums mt-1">
            {{ durationMs(metrics.elapsed_ms) }}
          </div>
        </div>

        <div class="rounded-xl border border-edge/60 bg-surface-2/80 p-3 flex flex-col justify-between">
          <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">TOPLAM ÇAĞRI</div>
          <div class="text-base sm:text-lg font-bold text-slate-100 tabular-nums mt-1">
            {{ metrics.total_calls }}
          </div>
        </div>

        <div class="rounded-xl border border-edge/60 bg-surface-2/80 p-3 flex flex-col justify-between">
          <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">GİDEN TOKEN</div>
          <div class="text-base sm:text-lg font-bold text-slate-100 tabular-nums mt-1">
            {{ tokenCount(metrics.prompt_tokens) }}
          </div>
        </div>

        <div class="rounded-xl border border-edge/60 bg-surface-2/80 p-3 flex flex-col justify-between">
          <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">GELEN TOKEN</div>
          <div class="text-base sm:text-lg font-bold text-slate-100 tabular-nums mt-1">
            {{ tokenCount(metrics.completion_tokens) }}
          </div>
        </div>
      </div>

      <!-- Latency & Total Subtext -->
      <div class="flex items-center justify-between text-[11px] text-slate-400 tabular-nums py-1 border-b border-edge/40">
        <span>son: {{ (metrics.last_latency_ms / 1000).toLocaleString('tr-TR', { minimumFractionDigits: 1 }) }} sn</span>
        <span>ort: {{ (metrics.avg_latency_ms / 1000).toLocaleString('tr-TR', { minimumFractionDigits: 1 }) }} sn</span>
        <span>toplam: {{ tokenCount(metrics.total_tokens) }} tk</span>
      </div>

      <!-- Categories Section -->
      <div class="mt-3">
        <div class="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
          ÇAĞRI TÜRÜNE GÖRE
        </div>
        <div class="space-y-0.5">
          <UsageMetricsRow
            v-for="cat in metrics.categories"
            :key="cat.key"
            :item="cat"
          />
        </div>
      </div>

      <!-- Footer -->
      <div class="mt-3.5 pt-2.5 border-t border-edge/50 flex items-center justify-between text-[11px] text-slate-500">
        <div>
          anahtar: <span class="font-mono text-slate-400">{{ keyInfo.key_name }}</span>
          · harcama: <span class="font-mono text-slate-400">${{ keyInfo.cost_usd.toFixed(4) }}</span>
        </div>
        <button
          type="button"
          class="hover:text-slate-300 p-1 transition-transform active:rotate-180"
          :class="isRefreshing ? 'animate-spin' : ''"
          title="Yenile"
          @click="triggerRefresh"
        >
          <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
          </svg>
        </button>
      </div>
    </div>
  </Transition>
</template>
