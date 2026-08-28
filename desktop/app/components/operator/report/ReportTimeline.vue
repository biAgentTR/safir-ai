<script setup lang="ts">
import type { TimelineEntry } from "~/types/api"
import { computed } from "vue"

const props = defineProps<{
  timeline: TimelineEntry[]
  selectedTimestamp: number | null
}>()

const emit = defineEmits<{
  (e: "seek", timestamp: number): void
}>()

const sortedTimeline = computed(() => {
  if (!props.timeline) return []
  return [...props.timeline].sort((a, b) => a.timestamp - b.timestamp)
})

function formatTime(sec: number) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}
</script>

<template>
  <div class="rounded-[18px] border border-white/10 bg-white/5 p-6 flex flex-col h-full">
    <div class="flex items-center gap-3 mb-6 shrink-0">
      <div class="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      </div>
      <h2 class="text-lg font-medium text-white">Zaman Çizelgesi</h2>
    </div>
    
    <div v-if="sortedTimeline.length > 0" class="flex-1 overflow-y-auto pr-2 space-y-4">
      <button
        v-for="(item, idx) in sortedTimeline"
        :key="idx"
        @click="emit(`seek`, item.timestamp)"
        class="w-full text-left flex items-start gap-4 p-3 rounded-xl transition-colors border group"
        :class="[
          selectedTimestamp === item.timestamp 
            ? `bg-cyan-500/10 border-cyan-500/30` 
            : `bg-white/5 border-transparent hover:bg-white/10 hover:border-white/10`
        ]"
      >
        <div class="shrink-0 mt-0.5">
          <span class="inline-flex items-center px-2 py-1 rounded bg-black/40 text-xs font-mono" :class="selectedTimestamp === item.timestamp ? `text-cyan-400` : `text-slate-300 group-hover:text-white`">
            {{ formatTime(item.timestamp) }}
          </span>
        </div>
        <div class="text-sm text-slate-300 group-hover:text-white leading-relaxed flex-1">
          {{ item.description }}
        </div>
      </button>
    </div>
    
    <div v-else class="flex-1 flex flex-col items-center justify-center text-center p-6 text-slate-500">
      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="mb-3 opacity-50"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7h4"/><path d="M3 11h4"/><path d="M3 15h4"/><path d="M3 19h4"/><path d="M17 3v18"/><path d="M17 7h4"/><path d="M17 11h4"/><path d="M17 15h4"/><path d="M17 19h4"/></svg>
      <h3 class="font-medium text-slate-400 mb-1">Zaman çizelgesi bulunmuyor</h3>
      <p class="text-sm">Bu analiz kaydında zaman damgalı olay açıklaması oluşturulmadı.</p>
    </div>
  </div>
</template>
