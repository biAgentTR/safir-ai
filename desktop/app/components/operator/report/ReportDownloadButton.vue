<template>
  <button
    @click="handleDownload"
    :disabled="isDownloading"
    class="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
    :class="[
      isDownloading 
        ? 'bg-gray-800 text-gray-400 border border-gray-700 cursor-not-allowed'
        : status === 'success'
          ? 'bg-emerald-600 text-white hover:bg-emerald-500 shadow-lg shadow-emerald-900/20'
          : 'bg-cyan-600 text-white hover:bg-cyan-500 shadow-lg shadow-cyan-900/20'
    ]"
    :aria-label="ariaLabel"
  >
    <!-- Icons -->
    <svg v-if="status === 'success'" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
    <svg v-else-if="isDownloading" class="animate-spin text-cyan-500" xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
    <svg v-else xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
    
    <!-- Text -->
    <span>{{ buttonText }}</span>
  </button>
  
  <!-- Accessibility / Toast Message -->
  <div v-if="status === 'error' && errorMessage" class="fixed bottom-4 right-4 bg-red-900/90 text-white px-4 py-3 rounded-lg border border-red-700 shadow-2xl z-50 animate-fade-in" aria-live="assertive">
    <div class="flex items-center gap-3">
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <p class="text-sm font-medium">{{ errorMessage }}</p>
    </div>
  </div>
  
  <div v-if="status === 'success'" class="fixed bottom-4 right-4 bg-emerald-900/90 text-white px-4 py-3 rounded-lg border border-emerald-700 shadow-2xl z-50 animate-fade-in" aria-live="polite">
    <div class="flex items-center gap-3">
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
      <p class="text-sm font-medium">PDF raporu kaydedildi</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useReportDownload } from '~/composables/useReportDownload'

const props = defineProps<{
  jobId: string
  videoName?: string
}>()

const { getStatus, errorMessage, downloadReport } = useReportDownload()

const status = computed(() => getStatus(props.jobId))

const isDownloading = computed(() => {
  const s = status.value
  return s === 'fetching' || s === 'choosing-location' || s === 'saving'
})

const buttonText = computed(() => {
  switch (status.value) {
    case 'fetching': return 'PDF hazırlanıyor...'
    case 'choosing-location': return 'Konum seçiliyor...'
    case 'saving': return 'Kaydediliyor...'
    case 'success': return 'İndirildi'
    default: return 'PDF raporu indir'
  }
})

const ariaLabel = computed(() => {
  const name = props.videoName ? props.videoName + ' için ' : ''
  return `${name}PDF raporunu indir`
})

async function handleDownload() {
  await downloadReport(props.jobId)
}
</script>

<style scoped>
.animate-fade-in {
  animation: fadeIn 0.3s ease-out forwards;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>