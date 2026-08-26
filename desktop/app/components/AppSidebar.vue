<script setup lang="ts">
// Desktop navigation rail. Every entry here is a real, working feature — no
// "coming soon" placeholders. Reports is a dedicated report center (list of
// completed analyses -> single-document report view), distinct from History
// (the full analysis run log, including running/failed/queued).
//
// The two analysis modes are independent systems (see useAnalysisMode.ts),
// but Geçmiş/Raporlar/SAFİR Asistan/Sistem Verileri are mode-agnostic — they
// read the same backend regardless of which mode produced an analysis — so
// both modes' nav includes them. Only the primary landing screen differs
// (VLM Direct Analiz vs. Genel Bakış/Yeni Analiz). Switching mode is done via
// the ModeSwitcher in AppTopbar, not from here.
interface NavItem {
  label: string
  to: string
  glyph: string
}

const sharedItems: NavItem[] = [
  { label: 'Geçmiş', to: '/history', glyph: '≡' },
  { label: 'Raporlar', to: '/reports', glyph: '▦' },
  { label: 'SAFİR Asistan', to: '/assistant', glyph: '◆' },
  { label: 'Sistem Verileri', to: '/system', glyph: '⛁' },
]

const lowBudgetItems: NavItem[] = [
  { label: 'Genel Bakış', to: '/', glyph: '▤' },
  { label: 'Yeni Analiz', to: '/new-analysis', glyph: '＋' },
  ...sharedItems,
]

const vlmDirectItems: NavItem[] = [
  { label: 'VLM Direct Analiz', to: '/vlm-direct', glyph: '▤' },
  { label: 'Genel Bakış', to: '/', glyph: '◈' },
  { label: 'Yeni Analiz', to: '/new-analysis', glyph: '＋' },
  ...sharedItems,
]

const { mode } = useAnalysisMode()
const items = computed(() => (mode.value === 'vlm_direct' ? vlmDirectItems : lowBudgetItems))

// Brief welcome overlay on panel switch (PanelTransition.vue, mounted in the
// layout). Only fires for an actual panel change — clicking the panel
// you're already on shouldn't retrigger it. Navigation itself is untouched;
// NuxtLink below still does the real route change.
const route = useRoute()
const { trigger } = usePanelTransition()
function onNavClick(to: string) {
  if (route.path !== to) trigger(to)
}
</script>

<template>
  <aside class="w-56 shrink-0 bg-surface-1 border-r border-edge flex flex-col">
    <div class="h-14 flex items-center gap-2.5 px-4 border-b border-edge">
      <img src="~/assets/images/logo.png" alt="SAFİR" class="w-6 h-6 object-contain shrink-0" />
      <div class="min-w-0">
        <div class="text-sm font-bold tracking-[0.24em] text-slate-100 leading-none">SAFİR</div>
        <div class="text-[9px] tracking-wide text-slate-600 leading-none mt-1">SAHA ANALİZ SİSTEMİ</div>
      </div>
    </div>

    <nav class="flex-1 px-2 py-3 space-y-0.5">
      <NuxtLink
        v-for="item in items"
        :key="item.to"
        :to="item.to"
        class="group relative flex items-center gap-3 rounded-md pl-3 pr-3 py-2 text-sm text-slate-400 border-l-2 border-transparent hover:text-slate-200 hover:bg-surface-2/70 transition-colors duration-150"
        active-class="!border-accent !text-slate-100 bg-surface-2"
        @click="onNavClick(item.to)"
      >
        <span class="w-4 text-center text-slate-600 group-hover:text-slate-400 [.router-link-active_&]:text-accent">{{ item.glyph }}</span>
        <span>{{ item.label }}</span>
      </NuxtLink>
    </nav>

    <div class="px-4 py-3 border-t border-edge text-[10px] leading-relaxed text-slate-600">
      <div class="text-slate-500 tracking-wide">TEKNOFEST 2026</div>
      <div>Saha Analiz ve Farkındalık İçin Yapay Zekâ Destekli Karar Sistemi</div>
    </div>
  </aside>
</template>
