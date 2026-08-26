<script setup lang="ts">
// Top bar with a route-derived title and a live backend health indicator.
const { state, system } = useBackendHealth()
const route = useRoute()

const title = computed(() => {
  const p = route.path
  if (p === '/') return 'Genel Bakış'
  if (p.startsWith('/new-analysis')) return 'Yeni Analiz'
  if (p.startsWith('/workspace')) return 'Analiz Çalışma Alanı'
  if (p.startsWith('/history')) return 'Geçmiş'
  if (p.startsWith('/reports')) return 'Raporlar'
  if (p.startsWith('/assistant')) return 'SAFİR Asistan'
  if (p.startsWith('/system')) return 'Sistem Verileri'
  if (p.startsWith('/vlm-direct')) return 'VLM Direct Analiz'
  return 'SAFİR'
})

const label = computed(() => {
  if (state.value === 'online') return `Sistem Hazır${system.value ? ` · ${system.value}` : ''}`
  if (state.value === 'offline') return 'Arka Uca Ulaşılamıyor'
  return 'Kontrol ediliyor…'
})
const dot = computed(() => ({
  online: 'bg-risk-low',
  offline: 'bg-risk-crit',
  checking: 'bg-slate-500',
}[state.value]))
</script>

<template>
  <header class="h-14 shrink-0 bg-surface-1 border-b border-edge flex items-center px-5">
    <h1 class="text-sm font-medium text-slate-200">{{ title }}</h1>
    <div class="ml-auto flex items-center gap-4">
      <ModeSwitcher />
      <div class="flex items-center gap-2 text-xs">
        <span
          class="inline-block w-2 h-2 rounded-full"
          :class="[dot, state === 'online' ? 'animate-pulse motion-reduce:animate-none' : '']"
        />
        <span class="text-slate-400">{{ label }}</span>
      </div>
      <ThemeToggle />
    </div>
  </header>
</template>
