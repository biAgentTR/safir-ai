<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from "vue"
import { isTauri, convertFileSrc } from "@tauri-apps/api/core"

const props = defineProps<{
  videoPath: string
}>()

const videoRef = ref<HTMLVideoElement | null>(null)
const objectUrl = ref<string | null>(null)
const loadError = ref(false)

function setVideoSource() {
  loadError.value = false
  if (objectUrl.value) {
    if (objectUrl.value.startsWith("blob:")) URL.revokeObjectURL(objectUrl.value)
    objectUrl.value = null
  }

  if (isTauri()) {
    try {
      objectUrl.value = convertFileSrc(props.videoPath)
    } catch (e) {
      loadError.value = true
    }
  } else {
    // Browser fallback handled in template
  }
}

watch(() => props.videoPath, () => {
  setVideoSource()
})

onMounted(() => {
  setVideoSource()
})

onUnmounted(() => {
  if (objectUrl.value && objectUrl.value.startsWith("blob:")) {
    URL.revokeObjectURL(objectUrl.value)
  }
})
</script>

<template>
  <div class="relative w-full h-full flex items-center justify-center bg-[#05090c]">
    <template v-if="isTauri()">
      <video
        v-if="!loadError && objectUrl"
        ref="videoRef"
        :src="objectUrl"
        class="absolute inset-0 w-full h-full object-contain"
        controls
        preload="metadata"
        @error="loadError = true"
      ></video>
      <div v-if="loadError" class="text-center p-6 text-rose-400">
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="mx-auto mb-3"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
        <p class="font-medium">Video yüklenemedi</p>
        <p class="text-xs text-rose-500/70 mt-1">Dosya yolu geçersiz veya erişilemiyor olabilir.</p>
      </div>
    </template>
    
    <div v-else class="text-center p-6 text-[var(--color-text-muted)]">
      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="mx-auto mb-3"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7h4"/><path d="M3 11h4"/><path d="M3 15h4"/><path d="M3 19h4"/><path d="M17 3v18"/><path d="M17 7h4"/><path d="M17 11h4"/><path d="M17 15h4"/><path d="M17 19h4"/></svg>
      <p class="font-medium">Video masaüstü uygulamasında görüntülenebilir</p>
      <p class="text-xs mt-1">Lokal video kaynağı yalnızca SAFİR masaüstü uygulaması içinde açılabilir.</p>
    </div>
  </div>
</template>