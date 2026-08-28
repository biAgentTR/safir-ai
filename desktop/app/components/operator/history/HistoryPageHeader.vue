<template>
  <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
    <div>
      <h1 class="text-2xl font-semibold text-white tracking-tight">Analiz geçmişi</h1>
      <p class="text-gray-400 text-sm mt-1">Başlatılan video analizlerini, işlem durumlarını ve risk değerlendirmelerini görüntüleyin.</p>
    </div>
    <div class="flex items-center gap-3">
      <button 
        @click="$emit('refresh')"
        :disabled="isRefreshing"
        class="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border border-gray-700 bg-gray-800/50 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 disabled:opacity-50"
        aria-label="Listeyi yenile"
      >
        <svg :class="{'animate-spin': isRefreshing}" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
        <span>Yenile</span>
      </button>
      <NuxtLink 
        to="/"
        class="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-cyan-600 hover:bg-cyan-500 text-white transition-colors shadow-lg shadow-cyan-900/20 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 focus:ring-offset-gray-900"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M12 5v14"/></svg>
        <span>Yeni analiz</span>
      </NuxtLink>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  isRefreshing: boolean
}>()

defineEmits<{
  (e: 'refresh'): void
}>()
</script>