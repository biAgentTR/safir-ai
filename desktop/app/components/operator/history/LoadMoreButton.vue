<template>
  <div class="flex flex-col items-center justify-center mt-6 py-4 border-t border-white/5">
    <button 
      v-if="hasMore"
      @click="$emit('load')" 
      :disabled="loading"
      class="px-6 py-2 rounded-lg font-medium text-sm transition-all duration-200 border"
      :class="[
        loading 
          ? 'bg-gray-800 text-gray-400 border-gray-700 cursor-not-allowed' 
          : 'bg-[#1E2532] text-cyan-400 border-cyan-900/50 hover:bg-[#252D3D] hover:border-cyan-800 focus:outline-none focus:ring-2 focus:ring-cyan-500/50'
      ]"
    >
      <div class="flex items-center gap-2">
        <svg v-if="loading" class="animate-spin h-4 w-4 text-cyan-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <span>{{ loading ? 'Yükleniyor...' : 'Daha fazla göster' }}</span>
      </div>
    </button>
    <div v-else class="text-sm text-gray-500">
      {{ totalLoaded }} kayıt gösteriliyor (Tüm kayıtlar)
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  hasMore: boolean
  loading: boolean
  totalLoaded: number
}>()

defineEmits<{
  (e: 'load'): void
}>()
</script>