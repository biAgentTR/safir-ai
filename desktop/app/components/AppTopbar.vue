<script setup lang="ts">
// Top bar with a route-derived title and a live backend health indicator.
const { state, system } = useBackendHealth()
const route = useRoute()
const auth = useAuthStore()
onMounted(() => auth.init())

const initials = computed(() => {
  const name = auth.username ?? ''
  const parts = name.split(/[@.\s]+/).filter(Boolean)
  return (parts[0]?.[0] ?? 'Y').toUpperCase() + (parts[1]?.[0] ?? '').toUpperCase()
})

// Most panels now live as scroll-synced sections on the hub ('/') — see
// AppTabNav.vue, which shows which one you're on. Only routes that are
// genuinely their own screen (a workspace run, a single report, admin login)
// get their own title here.
const title = computed(() => {
  const p = route.path
  if (p.startsWith('/workspace')) return 'Analiz Çalışma Alanı'
  if (p.startsWith('/reports/')) return 'Rapor Detayı'
  if (p.startsWith('/admin/login')) return 'Yönetici Girişi'
  return 'SAFİR'
})

const label = computed(() => {
  if (state.value === 'online') return `Bağlı${system.value ? ` · ${system.value}` : ''}`
  if (state.value === 'offline') return 'Arka Uca Ulaşılamıyor'
  return 'Kontrol Ediliyor'
})
const dot = computed(() => ({
  online: 'bg-risk-low',
  offline: 'bg-risk-crit',
  checking: 'bg-slate-500',
}[state.value]))
</script>

<template>
  <header class="h-14 shrink-0 bg-surface-1 border-b border-edge flex items-center px-5">
    <h1 class="text-sm font-semibold tracking-wide text-slate-100">{{ title }}</h1>
    <div class="ml-auto flex items-center gap-3">
      <ModeSwitcher />
      <div class="flex items-center gap-2 rounded-md border border-edge bg-surface-2 px-2.5 py-1.5 text-xs">
        <span
          class="status-dot"
          :class="[dot, state === 'online' ? 'animate-pulse motion-reduce:animate-none' : '']"
        />
        <span class="text-slate-400 font-mono tracking-tight">{{ label }}</span>
      </div>
      <ThemeToggle />

      <NuxtLink v-if="!auth.isAuthenticated" to="/admin/login" class="btn-ghost">
        <span aria-hidden="true">🔒</span>
        <span class="hidden md:inline">Yönetici girişi</span>
      </NuxtLink>
      <NuxtLink
        v-else
        to="/#sistem"
        class="w-8 h-8 rounded-full bg-accent-soft border border-accent/30 text-accent text-xs font-bold flex items-center justify-center shrink-0"
        :title="auth.username ?? 'Yönetici'"
      >
        {{ initials }}
      </NuxtLink>
    </div>
  </header>
</template>
