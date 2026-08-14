<script setup lang="ts">
// Desktop navigation rail. Every entry here is a real, working feature — no
// "coming soon" placeholders (a report is already reachable via Workspace's
// Final Report tab and via History, so there is no separate "Reports" item).
interface NavItem {
  label: string
  to: string
  icon: string
}

const items: NavItem[] = [
  { label: 'Overview', to: '/', icon: '▤' },
  { label: 'New Analysis', to: '/new-analysis', icon: '＋' },
  { label: 'History', to: '/history', icon: '≡' },
  { label: 'SAFİR Asistan', to: '/assistant', icon: '◆' },
]
</script>

<template>
  <aside class="w-56 shrink-0 bg-surface-1 border-r border-edge flex flex-col">
    <div class="h-14 flex items-center gap-2 px-4 border-b border-edge">
      <span class="text-lg font-semibold tracking-[0.2em] text-slate-100">SAFİR</span>
    </div>

    <nav class="flex-1 px-2 py-3 space-y-1">
      <NuxtLink
        v-for="item in items"
        :key="item.to"
        :to="item.to"
        class="group flex items-center gap-3 rounded-md px-3 py-2 text-sm text-slate-300 hover:bg-surface-2 transition-colors"
        active-class="bg-accent-soft text-white"
      >
        <span class="w-4 text-center text-slate-400 group-hover:text-slate-200">{{ item.icon }}</span>
        <span>{{ item.label }}</span>
      </NuxtLink>
    </nav>

    <div class="px-4 py-3 border-t border-edge text-[11px] text-slate-500">
      <div>TEKNOFEST 2026</div>
      <div class="text-slate-600">Saha Analiz ve Farkındalık İçin Yapay Zekâ Destekli Karar Sistemi</div>
    </div>
  </aside>
</template>
