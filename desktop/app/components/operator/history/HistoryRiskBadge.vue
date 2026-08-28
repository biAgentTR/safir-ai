<template>
  <div v-if="score !== null && level" class="flex flex-col gap-1">
    <div class="flex items-center gap-2">
      <span class="inline-flex px-2 py-0.5 text-xs font-semibold rounded" :class="colorConfig.bg">
        {{ colorConfig.label }}
      </span>
      <span class="text-xs text-gray-400 font-medium">{{ score }} / 100</span>
    </div>
  </div>
  <div v-else class="text-xs text-gray-500 italic">
    Henüz değerlendirilmedi
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  level: string | null
  score: number | null
}>()

const riskMap: Record<string, { label: string; bg: string }> = {
  dusuk: { label: 'Düşük', bg: 'bg-emerald-500/20 text-emerald-400' },
  orta: { label: 'Orta', bg: 'bg-yellow-500/20 text-yellow-400' },
  yuksek: { label: 'Yüksek', bg: 'bg-orange-500/20 text-orange-400' },
  kritik: { label: 'Kritik', bg: 'bg-red-500/20 text-red-400' }
}

const colorConfig = computed(() => {
  if (!props.level) return { label: 'Bilinmiyor', bg: 'bg-gray-500/20 text-gray-400' }
  return riskMap[props.level] || { label: 'Bilinmiyor', bg: 'bg-gray-500/20 text-gray-400' }
})
</script>