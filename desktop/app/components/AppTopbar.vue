<script setup lang="ts">
// Top bar with route-derived title, section tabs, and status/auth controls.
const { state, system } = useBackendHealth()
const route = useRoute()
const auth = useAuthStore()
const { goToSection } = useSectionNav()
const { togglePanel, isPanelOpen } = useUsageMetrics()
onMounted(() => auth.init())

const isHidden = ref(false)
let lastScrollTop = 0
const MIN_DELTA = 6

function onScroll(e: Event) {
  const el = e.target as HTMLElement | null
  if (!el) return
  const current = el.scrollTop

  // Near the top of the page: always show
  if (current <= 10) {
    isHidden.value = false
    lastScrollTop = current
    return
  }

  const diff = current - lastScrollTop

  if (diff > MIN_DELTA) {
    // Scrolling down -> hide header
    isHidden.value = true
  } else if (diff < -MIN_DELTA) {
    // Scrolling up -> reveal header
    isHidden.value = false
  }

  lastScrollTop = current
}

onMounted(() => {
  nextTick(() => {
    const region = document.getElementById('app-scroll-region')
    region?.addEventListener('scroll', onScroll, { passive: true })
  })
})

onBeforeUnmount(() => {
  const region = document.getElementById('app-scroll-region')
  region?.removeEventListener('scroll', onScroll)
})

const initials = computed(() => {
  const name = auth.username ?? ''
  const parts = name.split(/[@.\s]+/).filter(Boolean)
  return (parts[0]?.[0] ?? 'Y').toUpperCase() + (parts[1]?.[0] ?? '').toUpperCase()
})

const title = computed(() => {
  const p = route.path
  if (p.startsWith('/workspace')) return 'Analiz Çalışma Alanı'
  if (p.startsWith('/reports/')) return 'Rapor Detayı'
  if (p.startsWith('/admin/login')) return 'Giriş'
  return 'SAFİR'
})

const label = computed(() => {
  if (state.value === 'online') return 'Bağlı'
  if (state.value === 'offline') return 'Çevrimdışı'
  return 'Kontrol Ediliyor'
})
const dot = computed(() => ({
  online: 'bg-risk-low',
  offline: 'bg-risk-crit',
  checking: 'bg-slate-500',
}[state.value]))
</script>

<template>
  <header
    class="fixed top-0 left-0 right-0 h-14 bg-surface-1/95 backdrop-blur-md border-b border-edge flex items-center px-5 gap-6 z-30 transition-transform duration-300 ease-out transform"
    :class="isHidden ? '-translate-y-full' : 'translate-y-0 shadow-sm'"
  >
    <div class="flex items-center gap-6 h-full min-w-0">
      <h1 class="text-sm font-semibold tracking-wide text-slate-100 shrink-0">SAFİR</h1>
      <AppTabNav />
    </div>

    <div class="ml-auto flex items-center gap-3 shrink-0">
      <!-- Live status indicator on the left side of topbar controls -->
      <div class="flex items-center gap-1.5 pr-2 text-xs text-slate-400">
        <span
          class="status-dot"
          :class="dot"
        />
        <span class="font-normal">{{ label }}</span>
      </div>

      <button
        type="button"
        class="btn-ghost !py-1.5 !px-2.5 text-xs flex items-center gap-1.5 transition-all"
        :class="isPanelOpen ? 'text-accent border-accent/40 bg-accent/10' : ''"
        title="AI Metrikleri"
        @click="togglePanel"
      >
        <svg class="w-3.5 h-3.5 text-accent" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2L2 12l10 10 10-10L12 2zm0 3.6L18.4 12 12 18.4 5.6 12 12 5.6z"/>
        </svg>
        <span class="hidden sm:inline">AI Metrikleri</span>
      </button>

      <ModeSwitcher />
      <ThemeToggle />

      <NuxtLink v-if="!auth.isAuthenticated" to="/admin/login" class="btn-ghost !py-1.5 !px-3 text-xs">
        <span>Giriş</span>
      </NuxtLink>
      <button
        v-else
        type="button"
        class="w-8 h-8 rounded-full bg-accent-soft border border-accent/30 text-accent text-xs font-bold flex items-center justify-center shrink-0"
        :title="auth.username ?? 'Yönetici'"
        @click="goToSection('sistem')"
      >
        {{ initials }}
      </button>
    </div>
  </header>
</template>
