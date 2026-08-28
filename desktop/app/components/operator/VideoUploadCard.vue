<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { useAnalysisStore } from '~/stores/analysis'
import { isTauri, convertFileSrc } from '@tauri-apps/api/core'
import { open } from '@tauri-apps/plugin-dialog'

const store = useAnalysisStore()

const fileInput = ref<HTMLInputElement | null>(null)
const objectUrl = ref<string | null>(null)

function cleanupUrl() {
  if (objectUrl.value) {
    // If it's a blob URL, revoke it
    if (objectUrl.value.startsWith('blob:')) {
      URL.revokeObjectURL(objectUrl.value)
    }
    objectUrl.value = null
  }
}

async function openFileDialog() {
  store.setValidationError(null)
  
  if (isTauri()) {
    try {
      const selected = await open({
        multiple: false,
        filters: [{
          name: 'Video',
          extensions: ['mp4', 'avi', 'mov', 'mkv', 'webm']
        }]
      })
      
      if (selected && typeof selected === 'string') {
        const filePath = selected
        const fileName = filePath.split(/[/\\]/).pop() || 'video'
        const extension = fileName.split('.').pop() || ''
        
        cleanupUrl()
        objectUrl.value = convertFileSrc(filePath)
        
        store.setSelectedVideo({
          name: fileName,
          size: 0, // native dialog doesn't provide size easily without fs, we can skip size
          type: `video/${extension}`,
          extension,
          durationSeconds: null,
          absolutePath: filePath
        }, null)
      }
    } catch (err) {
      console.error(err)
      store.setValidationError('Video seçilemedi. Lütfen dosyayı yeniden seçin.')
    }
  } else {
    // Web ortamı
    fileInput.value?.click()
  }
}

async function onFileSelect(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (file) {
    store.setValidationError(null)
    
    // Web'de yalnızca önizleme/demo. Gerçek analiz backend'i web'den direkt çalışmaz (absolute path yok)
    cleanupUrl()
    objectUrl.value = URL.createObjectURL(file)
    
    store.setSelectedVideo(
      {
        name: file.name,
        size: file.size,
        type: file.type,
        extension: file.name.split('.').pop() || '',
        durationSeconds: null,
        // absolutePath is missing, so start will be disabled
      },
      file
    )
    store.setValidationError('Yerel video seçimi masaüstü uygulamasında kullanılabilir. Bu dosya sadece önizleme içindir.')
  }
}

function removeFile() {
  cleanupUrl()
  store.clearSelectedVideo()
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

function formatBytes(bytes: number) {
  if (bytes === 0) return ''
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

function formatDuration(sec: number) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

onUnmounted(() => {
  cleanupUrl()
})
</script>

<template>
  <div class="w-full h-full flex flex-col items-center justify-center text-center">
    
    <!-- Empty / Dropzone -->
    <div
      v-if="!store.selectedVideoMetadata"
      class="w-full flex flex-col items-center justify-center group cursor-pointer"
      @click="openFileDialog"
      @keydown.enter="openFileDialog"
      @keydown.space.prevent="openFileDialog"
      tabindex="0"
      role="button"
    >
      <input
        ref="fileInput"
        type="file"
        accept="video/*"
        class="hidden"
        @change="onFileSelect"
      />
      
      <div class="w-[72px] h-[72px] rounded-[20px] border border-[var(--color-primary)]/20 border-dashed bg-[var(--color-surface)] flex items-center justify-center text-[var(--color-primary)] mb-[22px] group-hover:bg-[var(--color-surface-hover)] transition-colors shadow-[0_0_15px_rgba(103,232,249,0.05)]">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
      </div>
      
      <h3 class="text-[13px] font-medium text-[var(--color-text)] mb-1">Analiz edilecek videoyu bilgisayarınızdan seçin</h3>
      <p class="text-[11px] text-[var(--color-text-muted)] mb-5">Seçmek için tıklayın</p>
      
      <button class="mb-4 px-5 py-2 bg-[var(--color-surface-elevated)] hover:bg-[var(--color-surface-hover)] border border-[var(--color-border)] text-[var(--color-text-secondary)] rounded-md transition-colors text-xs font-medium">
        Video seç
      </button>
      
      <div v-if="store.validationError" class="mt-2 text-[11px] text-rose-400">
        {{ store.validationError }}
      </div>
    </div>
    
    <!-- Selected Video -->
    <div
      v-else
      class="w-full flex flex-col items-center justify-center"
    >
      <div class="w-[72px] h-[72px] rounded-[20px] border border-[var(--color-primary)]/20 border-dashed bg-[var(--color-surface)] flex items-center justify-center text-[var(--color-primary)] mb-[22px] shadow-[0_0_15px_rgba(103,232,249,0.05)]">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
      </div>
      
      <div class="text-[15px] font-medium text-[var(--color-text)] mb-2 truncate max-w-[280px]" :title="store.selectedVideoMetadata.name">
        {{ store.selectedVideoMetadata.name }}
      </div>
      
      <div class="text-[11px] text-[var(--color-text-secondary)] mb-6 flex items-center justify-center gap-1.5">
        <span v-if="store.selectedVideoMetadata.durationSeconds">{{ formatDuration(store.selectedVideoMetadata.durationSeconds) }} - </span>
        <span v-if="store.selectedVideoMetadata.size">{{ formatBytes(store.selectedVideoMetadata.size) }}</span>
      </div>
      
      <div class="flex items-center gap-1.5 mb-6 text-[11px] font-bold text-[var(--color-success)] tracking-wider">
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
        VİDEO HAZIR
      </div>
      
      <div class="flex items-center gap-4">
        <button @click="openFileDialog" class="text-[11px] text-[var(--color-text-secondary)] hover:text-white transition-colors uppercase tracking-wide font-medium">Değiştir</button>
        <span class="text-[var(--color-border)]">|</span>
        <button @click="removeFile" class="text-[11px] text-rose-400/80 hover:text-rose-400 transition-colors uppercase tracking-wide font-medium">Kaldır</button>
      </div>
    </div>
  </div>
</template>
