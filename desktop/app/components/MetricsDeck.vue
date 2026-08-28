<script setup lang="ts">
// Sol alt köşedeki ölçüm destesi: altta yan yana iki düğme (AI Metrikleri /
// KPI Metrikleri), üstünde tek pencerelik kart alanı. Basılan sayfa soldan
// kayarak gelir; iki sayfa arasında yatay scroll (veya Shift+tekerlek) ile de
// geçilir. Sayfanın genel yapısını bozmamak için burada hiç gölge kullanılmaz.
import type { MetricsSlideId } from '~/composables/useMetricsDeck'
import { METRICS_SLIDE_ORDER } from '~/composables/useMetricsDeck'

const { isDeckOpen, activeSlide, activeIndex, isTransitioning, setSlide, toggleSlide, closeDeck } = useMetricsDeck()

const SLIDES: { id: MetricsSlideId; label: string }[] = [
  { id: 'ai', label: 'AI Metrikleri' },
  { id: 'kpi', label: 'KPI Metrikleri' },
]

// ---- yatay scroll / trackpad ile sayfa geçişi (TripleDrawerSection ile aynı eşikler) ----
let wheelCooldown = false

function onWheel(e: WheelEvent) {
  if (!isDeckOpen.value) return
  const isHorizontal = Math.abs(e.deltaX) > Math.abs(e.deltaY) + 8 || e.shiftKey
  const delta = e.shiftKey ? e.deltaY : e.deltaX
  if (!isHorizontal || Math.abs(delta) < 18) return
  if (wheelCooldown || isTransitioning.value) return

  const idx = activeIndex.value
  const next = delta > 0 ? idx + 1 : idx - 1
  const target = METRICS_SLIDE_ORDER[next]
  if (!target) return

  wheelCooldown = true
  setSlide(target)
  setTimeout(() => {
    wheelCooldown = false
  }, 550)
}

// ---- dokunmatik kaydırma ----
let touchStartX = 0
let touchStartY = 0

function onTouchStart(e: TouchEvent) {
  if (!e.touches[0]) return
  touchStartX = e.touches[0].clientX
  touchStartY = e.touches[0].clientY
}

function onTouchEnd(e: TouchEvent) {
  if (isTransitioning.value || !e.changedTouches[0]) return
  const diffX = e.changedTouches[0].clientX - touchStartX
  const diffY = e.changedTouches[0].clientY - touchStartY
  if (Math.abs(diffX) <= Math.abs(diffY) || Math.abs(diffX) < 40) return
  const target = METRICS_SLIDE_ORDER[activeIndex.value + (diffX < 0 ? 1 : -1)]
  if (target) setSlide(target)
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && isDeckOpen.value) closeDeck()
}
onMounted(() => window.addEventListener('keydown', onKeydown))
onBeforeUnmount(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <div class="fixed z-40 bottom-6 left-6 flex flex-col items-start gap-2.5 pointer-events-none select-none">
    <!-- Kart penceresi: kapalıyken sola kayıp saydamlaşır, açılınca soldan gelir -->
    <div
      class="transition-all duration-500 ease-out motion-reduce:transition-none"
      :class="isDeckOpen ? 'opacity-100 translate-x-0 pointer-events-auto' : 'opacity-0 -translate-x-[130%] pointer-events-none'"
      :aria-hidden="!isDeckOpen"
      @wheel.passive="onWheel"
      @touchstart="onTouchStart"
      @touchend="onTouchEnd"
    >
      <div class="w-96 max-w-[calc(100vw-3rem)] h-[32rem] max-h-[calc(100vh-11rem)] overflow-hidden rounded-2xl border border-edge bg-surface-1/95 backdrop-blur-2xl">
        <div
          class="flex h-full transition-transform duration-500 motion-reduce:transition-none"
          :style="{
            transform: `translateX(-${activeIndex * 100}%)`,
            transitionTimingFunction: 'cubic-bezier(0.16, 1, 0.3, 1)',
          }"
        >
          <div
            v-for="s in SLIDES"
            :key="s.id"
            class="w-full shrink-0 h-full overflow-y-auto p-4 sm:p-5 text-slate-100"
          >
            <UsageMetricsPanel v-if="s.id === 'ai'" />
            <KpiMetricsPanel v-else />
          </div>
        </div>
      </div>
    </div>

    <!-- Yan yana düğmeler: aynı kutu içinde, hangisine basılırsa o sayfa gelir -->
    <div class="pointer-events-auto inline-flex p-1 rounded-xl bg-surface-1 border border-edge">
      <button
        v-for="s in SLIDES"
        :key="`btn-${s.id}`"
        type="button"
        class="relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors duration-200"
        :class="
          isDeckOpen && activeSlide === s.id
            ? 'bg-accent text-white font-semibold'
            : 'text-slate-400 hover:text-slate-100 hover:bg-surface-2'
        "
        :aria-pressed="isDeckOpen && activeSlide === s.id"
        @click="toggleSlide(s.id)"
      >
        <svg v-if="s.id === 'ai'" class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2L2 12l10 10 10-10L12 2zm0 3.6L18.4 12 12 18.4 5.6 12 12 5.6z" />
        </svg>
        <svg v-else class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 20V10m6 10V4m6 16v-6" />
        </svg>
        <span>{{ s.label }}</span>
      </button>
    </div>
  </div>
</template>
