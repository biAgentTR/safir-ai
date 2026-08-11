<script setup lang="ts">
// Desktop navigation rail. History / Reports / Settings are placeholders in
// this step (ADIM 3) — navigation shell only, no functionality behind them.
interface NavItem {
  label: string
  to: string
  icon: string
  enabled: boolean
}

const items: NavItem[] = [
  { label: 'Overview', to: '/', icon: '▤', enabled: true },
  { label: 'New Analysis', to: '/new-analysis', icon: '＋', enabled: true },
  { label: 'History', to: '/history', icon: '≡', enabled: true },
  { label: 'Reports', to: '/reports', icon: '⎗', enabled: false },
  { label: 'Settings', to: '/settings', icon: '⚙', enabled: false },
]
</script>

<template>
  <aside class="w-56 shrink-0 bg-surface-1 border-r border-edge flex flex-col">
    <div class="h-14 flex items-center gap-2 px-4 border-b border-edge">
      <span class="text-lg font-semibold tracking-[0.2em] text-slate-100">SAFİR</span>
    </div>

    <nav class="flex-1 px-2 py-3 space-y-1">
      <template v-for="item in items" :key="item.to">
        <NuxtLink
          v-if="item.enabled"
          :to="item.to"
          class="group flex items-center gap-3 rounded-md px-3 py-2 text-sm text-slate-300 hover:bg-surface-2 transition-colors"
          active-class="bg-accent-soft text-white"
        >
          <span class="w-4 text-center text-slate-400 group-hover:text-slate-200">{{ item.icon }}</span>
          <span>{{ item.label }}</span>
        </NuxtLink>
        <div
          v-else
          class="flex items-center gap-3 rounded-md px-3 py-2 text-sm text-slate-600 cursor-not-allowed select-none"
          :title="`${item.label} — bu adımda etkin değil`"
        >
          <span class="w-4 text-center">{{ item.icon }}</span>
          <span>{{ item.label }}</span>
          <span class="ml-auto text-[10px] uppercase tracking-wide text-slate-700">yakında</span>
        </div>
      </template>
    </nav>

    <div class="px-4 py-3 border-t border-edge text-[11px] text-slate-500">
      <div>TEKNOFEST 2026</div>
      <div class="text-slate-600">Saha Analiz &amp; İSG Karar Sistemi</div>
    </div>
  </aside>
</template>
