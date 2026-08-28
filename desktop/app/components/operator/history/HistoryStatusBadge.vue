<template>
  <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border" :class="badgeClasses">
    <span class="w-1.5 h-1.5 rounded-full" :class="dotClasses"></span>
    {{ label }}
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  status: string
}>()

const badgeConfig: Record<string, { label: string; text: string; bg: string; border: string; dot: string }> = {
  queued: {
    label: 'Sırada',
    text: 'text-gray-300',
    bg: 'bg-gray-500/10',
    border: 'border-gray-500/20',
    dot: 'bg-gray-400'
  },
  running: {
    label: 'Analiz ediliyor',
    text: 'text-cyan-400',
    bg: 'bg-cyan-500/10',
    border: 'border-cyan-500/20',
    dot: 'bg-cyan-400 animate-pulse'
  },
  done: {
    label: 'Tamamlandı',
    text: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/20',
    dot: 'bg-emerald-400'
  },
  error: {
    label: 'Başarısız',
    text: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/20',
    dot: 'bg-red-400'
  }
}

const currentConfig = computed(() => badgeConfig[props.status] || {
  label: 'Bilinmiyor',
  text: 'text-gray-400',
  bg: 'bg-gray-500/10',
  border: 'border-gray-500/20',
  dot: 'bg-gray-400'
})

const badgeClasses = computed(() => [
  currentConfig.value.text,
  currentConfig.value.bg,
  currentConfig.value.border
])

const dotClasses = computed(() => currentConfig.value.dot)
const label = computed(() => currentConfig.value.label)
</script>