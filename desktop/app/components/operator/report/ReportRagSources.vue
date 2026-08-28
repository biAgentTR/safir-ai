<script setup lang="ts">
import type { RagContext } from "~/types/api"

defineProps<{
  sources: RagContext[]
}>()

function isUrlSecure(url: string | null | undefined) {
  if (!url) return false
  try {
    const parsed = new URL(url)
    return parsed.protocol === "http:" || parsed.protocol === "https:"
  } catch {
    return false
  }
}
</script>

<template>
  <div class="rounded-[18px] border border-white/10 bg-white/5 p-6 h-full flex flex-col">
    <div class="flex items-center gap-3 mb-6 shrink-0">
      <div class="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
      </div>
      <h2 class="text-lg font-medium text-white">Mevzuat ve Bilgi Kaynakları</h2>
    </div>

    <div v-if="sources && sources.length > 0" class="flex-1 space-y-4 overflow-y-auto pr-2">
      <div v-for="(source, idx) in sources" :key="idx" class="p-4 rounded-xl border border-white/10 bg-black/20 hover:bg-white/5 hover:border-white/20 transition-colors">
        <div class="flex items-start justify-between gap-4 mb-2">
          <h3 class="font-medium text-white">{{ source.rule_title || "İsimsiz Kaynak" }}</h3>
          <div v-if="source.score !== null && source.score !== undefined" class="shrink-0 text-xs font-mono px-2 py-1 bg-white/10 rounded text-slate-300 flex items-center gap-1" title="Semantik Eşleşme">
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
            {{ source.score.toFixed(3) }}
          </div>
        </div>
        
        <div v-if="source.article_number" class="text-xs font-mono text-emerald-400 mb-2">
          Madde: {{ source.article_number }}
        </div>
        
        <p class="text-sm text-slate-300 leading-relaxed mb-3">
          {{ source.content }}
        </p>
        
        <div v-if="isUrlSecure(source.source_url)">
          <a :href="source.source_url || undefined" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1.5 text-xs font-medium text-cyan-400 hover:text-cyan-300 transition-colors bg-cyan-500/10 px-3 py-1.5 rounded-md hover:bg-cyan-500/20">
            Kaynağa Git
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
          </a>
        </div>
      </div>
    </div>
    
    <div v-else class="flex-1 flex flex-col items-center justify-center text-center p-6 text-slate-500">
      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="mb-3 opacity-50"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
      <h3 class="font-medium text-slate-400 mb-1">Kaynak bulunamadı</h3>
      <p class="text-sm">Bu analiz için ilişkili mevzuat kaynağı bulunamadı.</p>
    </div>
  </div>
</template>
