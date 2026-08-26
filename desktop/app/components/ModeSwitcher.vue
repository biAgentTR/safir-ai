<script setup lang="ts">
// Compact segmented control for switching between the two independent
// analysis modes without closing the app. Lives in AppTopbar. Switching
// navigates to that mode's landing page (AppSidebar's nav updates to match).
import type { AnalysisMode } from '~/composables/useAnalysisMode'

const { mode, setMode } = useAnalysisMode()
const router = useRouter()

function switchTo(next: AnalysisMode) {
  if (mode.value === next) return
  setMode(next)
  router.push(next === 'vlm_direct' ? '/vlm-direct' : '/')
}
</script>

<template>
  <div class="inline-flex items-center rounded-md border border-edge bg-surface-2 p-0.5 text-xs" role="tablist" aria-label="Analiz modu">
    <button
      type="button"
      role="tab"
      :aria-selected="mode === 'low_budget'"
      class="px-2.5 py-1 rounded transition-colors"
      :class="mode === 'low_budget' ? 'bg-accent text-white' : 'text-slate-400 hover:text-slate-200'"
      title="Düşük Bütçeli Analiz"
      @click="switchTo('low_budget')"
    >
      Düşük Bütçeli
    </button>
    <button
      type="button"
      role="tab"
      :aria-selected="mode === 'vlm_direct'"
      class="px-2.5 py-1 rounded transition-colors"
      :class="mode === 'vlm_direct' ? 'bg-accent text-white' : 'text-slate-400 hover:text-slate-200'"
      title="VLM Direct Analysis"
      @click="switchTo('vlm_direct')"
    >
      VLM Direct
    </button>
  </div>
</template>
