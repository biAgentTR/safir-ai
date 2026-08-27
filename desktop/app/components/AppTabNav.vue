<script setup lang="ts">
// Horizontal, scroll-synced tab bar that transforms into a sleek left-side
// vertical floating dock when the operator scrolls down.
interface TabItem {
  id: string
  label: string
  icon: string
}

const sharedItems: TabItem[] = [
  { id: 'gecmis', label: 'Geçmiş', icon: '≡' },
  { id: 'asistan', label: 'SAFİR Asistan', icon: '◆' },
  { id: 'raporlar', label: 'Raporlar', icon: '▦' },
  { id: 'sistem', label: 'Sistem Verileri', icon: '⛁' },
]
const lowBudgetItems: TabItem[] = [
  { id: 'ana-sayfa', label: 'Ana Sayfa', icon: '▤' },
  ...sharedItems,
]
const vlmDirectItems: TabItem[] = [
  { id: 'ana-sayfa', label: 'Ana Sayfa', icon: '▤' },
  { id: 'vlm-direct', label: 'Direct', icon: '◆' },
  ...sharedItems,
]

const { mode } = useAnalysisMode()
const items = computed(() => (mode.value === 'vlm_direct' ? vlmDirectItems : lowBudgetItems))

const route = useRoute()
const onHub = computed(() => route.path === '/')
const activeId = ref<string | null>(null)
const { scrollToId, goToSection } = useSectionNav()
const { activeSlide, setSlide } = useDrawerDeck()

function onTabClick(id: string) {
  if (['gecmis', 'asistan', 'raporlar'].includes(id)) {
    setSlide(id as any)
  }
  if (onHub.value) activeId.value = id
  goToSection(id)
}

watch(activeSlide, (newVal) => {
  if (['gecmis', 'asistan', 'raporlar'].includes(activeId.value ?? '')) {
    activeId.value = newVal
  }
})

// Scroll state: when scrolled down, the dock on the left activates
const isScrolled = ref(false)
const SCROLL_THRESHOLD = 60

function onScroll(e: Event) {
  const el = e.target as HTMLElement | null
  if (!el) return
  isScrolled.value = el.scrollTop > SCROLL_THRESHOLD
}

// ---- scroll-spy: highlight the tab for whichever section is at the top of
// the scroll region (layouts/default.vue's <main id="app-scroll-region">) ----
let observer: IntersectionObserver | null = null

function teardownObserver() {
  observer?.disconnect()
  observer = null
}

function setupObserver() {
  teardownObserver()
  if (!onHub.value) return
  const root = document.getElementById('app-scroll-region')
  if (!root) return
  const ids = items.value.map((i) => i.id)
  observer = new IntersectionObserver(
    (entries) => {
      const visible = entries.filter((e) => e.isIntersecting)
      if (!visible.length) return
      visible.sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
      const id = visible[0]!.target.id
      if (ids.includes(id)) activeId.value = id
    },
    { root, rootMargin: '-10% 0px -70% 0px', threshold: 0 },
  )
  for (const id of ids) {
    const el = document.getElementById(id)
    if (el) observer.observe(el)
  }
}

function scrollToHash() {
  if (route.hash) scrollToId(route.hash.slice(1))
}

onMounted(() => {
  nextTick(() => {
    setupObserver()
    scrollToHash()
    const region = document.getElementById('app-scroll-region')
    region?.addEventListener('scroll', onScroll, { passive: true })
  })
})

onBeforeUnmount(() => {
  teardownObserver()
  const region = document.getElementById('app-scroll-region')
  region?.removeEventListener('scroll', onScroll)
})

watch([() => route.path, items], () => nextTick(setupObserver))
watch(
  () => route.hash,
  () => onHub.value && nextTick(scrollToHash),
)
</script>

<template>
  <!-- Topbar horizontal tab list -->
  <nav class="h-full flex items-center gap-0.5 no-scrollbar shrink-0" aria-label="Bölümler">
    <button
      v-for="item in items"
      :key="item.id"
      type="button"
      class="relative shrink-0 px-3 h-14 flex items-center text-xs sm:text-sm whitespace-nowrap border-b-2 transition-colors duration-150"
      :class="
        onHub && activeId === item.id
          ? 'border-accent text-slate-100 font-medium'
          : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-edge-strong'
      "
      :aria-current="onHub && activeId === item.id ? 'true' : undefined"
      @click="onTabClick(item.id)"
    >
      {{ item.label }}
    </button>
  </nav>

  <!-- Left-side vertical floating dock on scroll down: tucked into edge, slides out on hover -->
  <Teleport to="body">
    <aside
      v-if="onHub"
      class="fixed left-0 top-1/2 -translate-y-1/2 z-40 flex items-center transition-all duration-300 ease-out transform group/dock"
      :class="isScrolled ? 'translate-x-0 opacity-100' : '-translate-x-full opacity-0 pointer-events-none'"
      aria-label="Sol hızlı menü"
    >
      <div
        class="flex flex-col gap-1.5 p-1.5 pr-2 rounded-r-2xl border border-l-0 border-edge bg-surface-1/95 backdrop-blur-md shadow-2xl transition-transform duration-300 ease-out transform -translate-x-[calc(100%-14px)] group-hover/dock:translate-x-0"
      >
        <button
          v-for="item in items"
          :key="item.id"
          type="button"
          class="relative w-10 h-10 rounded-xl flex items-center justify-center text-sm transition-all duration-150 group/btn"
          :class="
            activeId === item.id
              ? 'bg-accent text-white shadow-md'
              : 'text-slate-400 hover:text-slate-100 hover:bg-surface-2'
          "
          :title="item.label"
          :aria-label="item.label"
          @click.stop="onTabClick(item.id)"
        >
          <span class="text-base font-semibold">{{ item.icon }}</span>
          <!-- Individual tooltip on hovering this specific button only -->
          <span class="pointer-events-none absolute left-full ml-3 px-2.5 py-1 rounded-md bg-surface-3 text-slate-100 text-xs font-medium whitespace-nowrap opacity-0 shadow-xl border border-edge transition-opacity duration-150 group-hover/btn:opacity-100 z-50">
            {{ item.label }}
          </span>
        </button>
      </div>
    </aside>
  </Teleport>
</template>
