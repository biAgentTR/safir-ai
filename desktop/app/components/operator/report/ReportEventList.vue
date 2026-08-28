<script setup lang="ts">
import type { EventSummary } from "~/types/api"
import { ref } from "vue"

defineProps<{
  events: EventSummary[]
}>()

const expandedEvent = ref<number | null>(null)

function toggleEvent(idx: number) {
  if (expandedEvent.value === idx) {
    expandedEvent.value = null
  } else {
    expandedEvent.value = idx
  }
}

function formatEventName(name: string) {
  if (!name) return "Bilinmeyen Olay"
  return name.split("_").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ")
}
</script>

<template>
  <div class="rounded-[18px] border border-white/10 bg-white/5 p-6 h-full flex flex-col">
    <div class="flex items-center gap-3 mb-6 shrink-0">
      <div class="w-8 h-8 rounded-lg bg-rose-500/20 text-rose-400 flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      </div>
      <h2 class="text-lg font-medium text-white">Tespit Edilen Olaylar</h2>
    </div>

    <div v-if="events && events.length > 0" class="flex-1 overflow-y-auto pr-2 space-y-3">
      <div
        v-for="(event, idx) in events"
        :key="idx"
        class="border rounded-xl transition-colors overflow-hidden"
        :class="expandedEvent === idx ? `bg-white/10 border-white/20` : `bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20`"
      >
        <button
          class="w-full text-left p-4 flex items-center justify-between focus:outline-none"
          @click="toggleEvent(idx)"
        >
          <div class="flex items-center gap-3">
            <div 
              class="w-2 h-2 rounded-full"
              :class="event.risk_level === `critical` || event.risk_level === `kritik` ? `bg-red-500` : event.risk_level === `high` || event.risk_level === `yuksek` ? `bg-rose-500` : event.risk_level === `medium` || event.risk_level === `orta` ? `bg-amber-500` : `bg-emerald-500`"
            ></div>
            <span class="font-medium text-white">{{ formatEventName(event.event_name) }}</span>
          </div>
          <div class="flex items-center gap-3">
            <span v-if="event.risk_score !== null && event.risk_score !== undefined" class="text-xs font-mono px-2 py-1 bg-black/40 rounded text-slate-300">
              Skor: {{ event.risk_score }}
            </span>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-slate-500 transition-transform" :class="[expandedEvent === idx ? 'rotate-180' : '']"><polyline points="6 9 12 15 18 9"/></svg>
          </div>
        </button>
        
        <div v-show="expandedEvent === idx" class="p-4 pt-0 text-sm text-slate-300 border-t border-white/5 bg-black/20">
          <div v-if="event.event_type" class="mb-3 mt-3">
            <span class="text-xs text-slate-500 uppercase tracking-wider block mb-1">Olay Tipi</span>
            <span>{{ event.event_type }}</span>
          </div>
          
          <div v-if="event.keywords && event.keywords.length > 0" class="mb-3 mt-3">
            <span class="text-xs text-slate-500 uppercase tracking-wider block mb-2">Anahtar Kelimeler</span>
            <div class="flex flex-wrap gap-2">
              <span v-for="(kw, kIdx) in event.keywords" :key="kIdx" class="px-2 py-1 rounded-md bg-white/10 text-xs text-slate-300 border border-white/5">
                {{ kw }}
              </span>
            </div>
          </div>
          
          <div v-if="event.rule_ids && event.rule_ids.length > 0" class="mt-3">
            <span class="text-xs text-slate-500 uppercase tracking-wider block mb-1">Ä°hlal Edilen Kurallar</span>
            <div class="text-xs font-mono text-slate-400">
              {{ event.rule_ids.join(", ") }}
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div v-else class="flex-1 flex flex-col items-center justify-center text-center p-6 text-slate-500">
      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="mb-3 opacity-50 text-emerald-400"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
      <h3 class="font-medium text-emerald-400/80 mb-1">Kritik olay tespit edilmedi</h3>
      <p class="text-sm">Analiz tamamlandÄ±. Bu videoda tanÄ±mlÄ± gÃ¼venlik eÅŸiklerini aÅŸan bir olay bulunamadÄ±.</p>
    </div>
  </div>
</template>
