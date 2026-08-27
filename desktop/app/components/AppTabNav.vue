<script setup lang="ts">
// Horizontal, scroll-synced tab bar — replaces the old vertical AppSidebar.
// All 6 (or 7, VLM Direct mode) panels now live stacked as sections on the
// single hub page ('/', see pages/index.vue + components/sections/*). This
// bar just tells you which section you're scrolled to, and clicking a tab
// scrolls you there (navigating to '/' first if you're elsewhere — e.g. from
// a workspace/report detail page).
interface TabItem {
  id: string
  label: string
}

const sharedItems: TabItem[] = [
  { id: 'gecmis', label: 'Geçmiş' },
  { id: 'raporlar', label: 'Raporlar' },
  { id: 'asistan', label: 'SAFİR Asistan' },
  { id: 'sistem', label: 'Sistem Verileri' },
]
const lowBudgetItems: TabItem[] = [
  { id: 'ana-sayfa', label: 'Ana Sayfa' },
  { id: 'yeni-analiz', label: 'Yeni Analiz' },
  ...sharedItems,
]
const vlmDirectItems: TabItem[] = [
  { id: 'ana-sayfa', label: 'Ana Sayfa' },
  { id: 'vlm-direct', label: 'VLM Direct Analiz' },
  { id: 'yeni-analiz', label: 'Yeni Analiz' },
  ...sharedItems,
]

const { mode } = useAnalysisMode()
const items = computed(() => (mode.value === 'vlm_direct' ? vlmDirectItems : lowBudgetItems))

const route = useRoute()
const onHub = computed(() => route.path === '/')
const activeId = ref<string | null>(null)
const { scrollToId, goToSection } = useSectionNav()

function onTabClick(id: string) {
  if (onHub.value) activeId.value = id
  goToSection(id)
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
  })
})
onBeforeUnmount(teardownObserver)
watch([() => route.path, items], () => nextTick(setupObserver))
watch(
  () => route.hash,
  () => onHub.value && nextTick(scrollToHash),
)
</script>

<template>
  <nav class="h-11 shrink-0 bg-surface-1 border-b border-edge flex items-center px-2 overflow-x-auto" aria-label="Bölümler">
    <button
      v-for="item in items"
      :key="item.id"
      type="button"
      class="relative shrink-0 px-3.5 h-full text-sm whitespace-nowrap border-b-2 transition-colors duration-150"
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
</template>
