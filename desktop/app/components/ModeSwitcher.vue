<script setup lang="ts">
// Compact segmented control for switching between the two independent
// analysis modes with skeleton loaders & smooth animated pill.
import type { AnalysisMode } from '~/composables/useAnalysisMode'

const { mode, setMode, isModeSwitching } = useAnalysisMode()
const { goToSection } = useSectionNav()

function switchTo(next: AnalysisMode) {
  if (mode.value === next) return
  setMode(next)
  setTimeout(() => {
    goToSection(next === 'vlm_direct' ? 'vlm-direct' : 'yeni-analiz')
  }, 80)
}
</script>

<template>
  <div class="relative inline-flex items-center rounded-lg border border-edge/80 bg-surface-2 p-0.5 text-xs shadow-inner select-none" role="tablist" aria-label="Analiz modu">
    <!-- Animated Active Background Pill -->
    <div
      class="absolute top-0.5 bottom-0.5 w-[calc(50%-2px)] rounded-md bg-surface-3 border border-edge-strong shadow-sm transition-transform duration-250 ease-out"
      :style="{
        transform: mode === 'vlm_direct' ? 'translateX(calc(100% + 2px))' : 'translateX(0px)',
        transitionTimingFunction: 'cubic-bezier(0.16, 1, 0.3, 1)'
      }"
    />

    <button
      type="button"
      role="tab"
      :aria-selected="mode === 'low_budget'"
      class="relative z-10 px-3 py-1 rounded-md transition-colors duration-150 text-center font-medium"
      :class="mode === 'low_budget' ? 'text-slate-100' : 'text-slate-400 hover:text-slate-200'"
      title="Lite Analiz (Kare Örnekleme)"
      @click="switchTo('low_budget')"
    >
      Lite
    </button>
    <button
      type="button"
      role="tab"
      :aria-selected="mode === 'vlm_direct'"
      class="relative z-10 px-3 py-1 rounded-md transition-colors duration-150 text-center font-medium"
      :class="mode === 'vlm_direct' ? 'text-slate-100' : 'text-slate-400 hover:text-slate-200'"
      title="Direct Analiz (Doğrudan VLM)"
      @click="switchTo('vlm_direct')"
    >
      Direct
    </button>
  </div>
</template>
