<script setup lang="ts">
import { computed } from "vue"

const props = defineProps<{
  recommendedAction: string | null
  actions: string[]
}>()

const filteredActions = computed(() => {
  if (!props.actions) return []
  // Filter out any action that exactly matches the recommendedAction
  const rec = props.recommendedAction?.trim().toLowerCase()
  return props.actions.filter(a => a.trim().toLowerCase() !== rec)
})
</script>

<template>
  <div class="rounded-[18px] border border-white/10 bg-white/5 p-6 h-full flex flex-col">
    <div class="flex items-center gap-3 mb-6 shrink-0">
      <div class="w-8 h-8 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
      </div>
      <h2 class="text-lg font-medium text-white">Önerilen Operatör Aksiyonları</h2>
    </div>

    <div v-if="recommendedAction || filteredActions.length > 0" class="flex-1 space-y-4 overflow-y-auto pr-2">
      <div v-if="recommendedAction" class="p-4 rounded-xl border border-indigo-500/30 bg-indigo-500/10 flex gap-4">
        <div class="mt-0.5 shrink-0">
          <div class="w-6 h-6 rounded-full bg-indigo-500 text-white flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
          </div>
        </div>
        <div>
          <h3 class="text-xs font-bold text-indigo-400 uppercase tracking-wider mb-1">Birincil Aksiyon</h3>
          <p class="text-sm text-indigo-50 leading-relaxed">{{ recommendedAction }}</p>
        </div>
      </div>
      
      <div v-if="filteredActions.length > 0" class="space-y-3 mt-4">
        <h3 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Diğer Adımlar</h3>
        <div v-for="(action, idx) in filteredActions" :key="idx" class="flex items-start gap-4 p-3 rounded-xl bg-white/5 border border-white/5">
          <div class="w-6 h-6 rounded-full bg-black/40 text-slate-400 text-xs flex items-center justify-center font-mono shrink-0 border border-white/10">
            {{ idx + 1 }}
          </div>
          <p class="text-sm text-slate-300 leading-relaxed mt-0.5">{{ action }}</p>
        </div>
      </div>
    </div>
    
    <div v-else class="flex-1 flex flex-col items-center justify-center text-center p-6 text-slate-500">
      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="mb-3 opacity-50"><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <h3 class="font-medium text-slate-400 mb-1">Operatör aksiyonu bulunmuyor</h3>
      <p class="text-sm">Bu analiz için uygulanabilir bir operatör aksiyonu oluşturulmadı.</p>
    </div>
  </div>
</template>
