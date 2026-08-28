<script setup lang="ts">
defineProps<{
  naturalLanguageSummary: string | null
  summary: string | null
}>()
</script>

<template>
  <div class="rounded-[18px] border border-white/10 bg-white/5 p-6">
    <h2 class="text-lg font-medium text-white mb-4">Analiz Özeti</h2>
    
    <div v-if="naturalLanguageSummary && summary && naturalLanguageSummary !== summary" class="space-y-4">
      <div>
        <h3 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Genel Değerlendirme</h3>
        <p class="text-sm text-slate-300 leading-relaxed">{{ naturalLanguageSummary }}</p>
      </div>
      <div>
        <h3 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Kısa Özet</h3>
        <p class="text-sm text-slate-300 leading-relaxed">{{ summary }}</p>
      </div>
    </div>
    
    <div v-else-if="naturalLanguageSummary || summary">
      <p class="text-sm text-slate-300 leading-relaxed">{{ naturalLanguageSummary || summary }}</p>
    </div>
    
    <div v-else class="text-sm text-slate-500 italic">
      Bu analiz için metinsel özet bulunmuyor.
    </div>
  </div>
</template>
