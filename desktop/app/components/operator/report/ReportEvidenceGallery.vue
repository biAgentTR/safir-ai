<script setup lang="ts">
import type { EvidenceFrameOut } from "~/types/api"
import { ref, computed } from "vue"

const props = defineProps<{
  frames: EvidenceFrameOut[]
}>()

const emit = defineEmits<{
  (e: "seek", timestamp: number): void
}>()

const selectedImage = ref<string | null>(null)

function viewImage(base64: string) {
  selectedImage.value = base64
}

function closeImage() {
  selectedImage.value = null
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === "Escape" && selectedImage.value) {
    closeImage()
  }
}
</script>

<template>
  <div class="rounded-[18px] border border-white/10 bg-white/5 p-6 h-full flex flex-col" @keydown.window="onKeydown">
    <div class="flex items-center gap-3 mb-6 shrink-0">
      <div class="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
      </div>
      <h2 class="text-lg font-medium text-white">Kritik Kareler</h2>
    </div>

    <div v-if="frames && frames.length > 0" class="flex-1 overflow-x-auto">
      <div class="flex gap-4 pb-2 h-full">
        <div v-for="(frame, idx) in frames" :key="idx" class="w-64 shrink-0 flex flex-col group rounded-xl border border-white/10 bg-black/40 overflow-hidden">
          <button class="relative w-full aspect-video block focus:outline-none focus:ring-2 focus:ring-cyan-500" @click="viewImage(frame.base64_image)">
            <!-- Make sure we have valid data URL prefix if backend omitted it, though it usually comes with it -->
            <img 
              :src="isSafeEvidenceImageSource(frame.base64_image) ? (frame.base64_image.startsWith('data:') ? frame.base64_image : 'data:image/jpeg;base64,' + frame.base64_image) : ''"
              loading="lazy" 
              class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" 
              alt="Kritik kare"
            />
            <div class="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-md"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
            </div>
            <div v-if="frame.is_fallback" class="absolute top-2 left-2 px-2 py-0.5 rounded text-[10px] uppercase font-bold bg-black/60 text-slate-300 backdrop-blur-sm border border-white/10">
              Yedek Kare
            </div>
          </button>
          <div class="p-3 flex items-center justify-between flex-1 bg-white/5">
            <div>
              <span class="text-xs text-slate-400 block mb-1">Zaman</span>
              <button 
                class="text-sm font-mono text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1 focus:outline-none"
                @click="emit(`seek`, frame.timestamp_sec)"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                {{ frame.timestamp_str || frame.timestamp_sec + `s` }}
              </button>
            </div>
            <div class="text-right" v-if="frame.change_score !== null && frame.change_score !== undefined">
              <span class="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">Değişim Skoru</span>
              <span class="text-sm font-mono text-slate-300">{{ frame.change_score.toFixed(3) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div v-else class="flex-1 flex flex-col items-center justify-center text-center p-6 text-slate-500">
      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="mb-3 opacity-50"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
      <h3 class="font-medium text-slate-400 mb-1">Kare bulunamadı</h3>
      <p class="text-sm">Bu analiz kaydında kritik kare oluşturulmadı.</p>
    </div>

    <!-- Lightbox Modal -->
    <div v-if="selectedImage" class="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-8 bg-black/90 backdrop-blur-sm" @click="closeImage">
      <button class="absolute top-6 right-6 text-white/50 hover:text-white transition-colors p-2" @click="closeImage" aria-label="Kapat">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
      <img 
        :src="isSafeEvidenceImageSource(selectedImage) ? (selectedImage.startsWith('data:') ? selectedImage : 'data:image/jpeg;base64,' + selectedImage) : ''"
        class="max-w-full max-h-full object-contain rounded-lg border border-white/10 shadow-2xl" 
        alt="Kritik kare büyük boy"
        @click.stop
      />
    </div>
  </div>
</template>
