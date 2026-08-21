<script setup lang="ts">
const isCollapsed = ref(false)
const { theme, toggleTheme, isDark } = useTheme()

const navItems = [
  { label: 'Ana Analiz', to: '/', icon: '🏠' },
  { label: 'Yeni Analiz', to: '/new-analysis', icon: '⚡' },
  { label: 'Olay & İnceleme Geçmişi', to: '/history', icon: '📑' },
  { label: 'Resmi Raporlar', to: '/reports', icon: '📊' },
  { label: 'SAFİR AI Asistan (RAG)', to: '/assistant', icon: '🤖' },
  { label: 'Sistem Telemetrisi', to: '/system', icon: '⚙️' },
]

function toggleSidebar() {
  isCollapsed.value = !isCollapsed.value
}
</script>

<template>
  <aside
    class="shrink-0 bg-surface-1 border-r border-edge flex flex-col justify-between select-none transition-all duration-300 relative z-30"
    :class="isCollapsed ? 'w-16' : 'w-64'"
  >
    <div>
      <!-- Brand & Hamburger Header -->
      <div class="p-3.5 border-b border-edge flex items-center justify-between gap-2 h-16">
        <div v-if="!isCollapsed" class="flex items-center gap-2.5 overflow-hidden">
          <div class="w-8 h-8 rounded-xl bg-blue-600 text-white font-black text-xs flex items-center justify-center shadow-md shrink-0">
            SF
          </div>
          <div class="min-w-0">
            <span class="text-base font-extrabold tracking-tight truncate block">
              SAFİR<span class="text-blue-500">AI</span>
            </span>
            <span class="block text-[9px] font-mono text-slate-400 truncate">Kritik Tesis Güvenliği</span>
          </div>
        </div>

        <button
          type="button"
          class="p-2 rounded-xl border border-edge bg-surface-2 hover:bg-surface-3 transition-colors text-sm flex items-center justify-center shrink-0"
          :title="isCollapsed ? 'Menüyü Genişlet' : 'Menüyü Daralt'"
          @click="toggleSidebar"
        >
          ☰
        </button>
      </div>

      <!-- Quick Action Button -->
      <div class="p-3">
        <NuxtLink
          to="/new-analysis"
          class="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs py-2.5 rounded-xl flex items-center justify-center gap-2 transition-all shadow-md shadow-blue-500/20"
          :title="isCollapsed ? 'Yeni Analiz Başlat' : ''"
        >
          <span>⚡</span>
          <span v-if="!isCollapsed">Yeni Analiz</span>
        </NuxtLink>
      </div>

      <!-- Navigation Links -->
      <nav class="p-2 space-y-1">
        <NuxtLink
          v-for="item in navItems"
          :key="item.label"
          :to="item.to"
          class="group flex items-center gap-3 rounded-xl px-3 py-2.5 text-xs font-semibold hover:bg-surface-2 transition-colors relative"
          :class="isCollapsed ? 'justify-center' : ''"
          active-class="bg-blue-500/10 text-blue-600 dark:text-blue-400 font-bold border-l-2 border-blue-500"
          :title="isCollapsed ? item.label : ''"
        >
          <span class="text-base opacity-90 group-hover:scale-110 transition-transform shrink-0">{{ item.icon }}</span>
          <span v-if="!isCollapsed" class="truncate">{{ item.label }}</span>
        </NuxtLink>
      </nav>
    </div>

    <!-- Bottom Controls: Theme Toggle & Session Info -->
    <div class="p-3 border-t border-edge space-y-2 text-xs">
      <!-- Theme Switcher Button -->
      <button
        type="button"
        class="w-full rounded-xl border border-edge bg-surface-2 hover:bg-surface-3 p-2.5 text-xs font-bold flex items-center justify-between transition-all"
        :title="isDark ? 'Açık Mod Geçiş Yap' : 'Koyu Mod Geçiş Yap'"
        @click="toggleTheme"
      >
        <div class="flex items-center gap-2">
          <span>{{ isDark ? '🌙' : '☀️' }}</span>
          <span v-if="!isCollapsed">{{ isDark ? 'Koyu Mod' : 'Açık Mod' }}</span>
        </div>
        <span v-if="!isCollapsed" class="text-[10px] font-mono text-slate-400 uppercase">
          {{ isDark ? 'DARK' : 'LIGHT' }}
        </span>
      </button>

      <!-- System Badge -->
      <div v-if="!isCollapsed" class="p-2.5 rounded-xl bg-surface-2/70 border border-edge space-y-1">
        <div class="flex items-center justify-between font-mono text-[10px] font-bold">
          <span>TEKNOFEST 2026</span>
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
        </div>
        <div class="text-[10px] text-blue-500 dark:text-blue-400 font-mono">Dual-Use AI Engine</div>
      </div>
    </div>
  </aside>
</template>
