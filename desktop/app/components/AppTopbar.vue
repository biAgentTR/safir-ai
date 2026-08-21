<script setup lang="ts">
const { state, system } = useBackendHealth()
const { isDark, toggleTheme } = useTheme()
</script>

<template>
  <header class="h-16 shrink-0 bg-surface-1 border-b border-edge flex items-center justify-between px-6 select-none transition-colors duration-200">
    <!-- Left Title / System Branding -->
    <div class="flex items-center gap-3">
      <span class="text-sm font-extrabold tracking-tight">SAFİR AI — Saha Analiz &amp; Karar Destek Platformu</span>
    </div>

    <!-- Right Topbar Controls & Status -->
    <div class="flex items-center gap-3 text-xs">
      <!-- System Health Indicator -->
      <div class="px-3 py-1.5 rounded-full bg-surface-2 border border-edge font-mono text-[11px] flex items-center gap-2">
        <span class="w-2.5 h-2.5 rounded-full" :class="state === 'online' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'" />
        <span>SİSTEM HAZIR {{ system ? `(${system})` : '' }}</span>
      </div>

      <!-- Quick RAG Assistant Link -->
      <NuxtLink
        to="/assistant"
        class="px-3.5 py-1.5 rounded-xl border border-edge bg-surface-2 hover:bg-surface-3 font-semibold text-xs flex items-center gap-2 transition-colors"
      >
        <span>🤖</span>
        <span>SAFİR RAG Asistanı</span>
      </NuxtLink>

      <!-- Theme Switcher Topbar Toggle -->
      <button
        type="button"
        class="p-2 rounded-xl border border-edge bg-surface-2 hover:bg-surface-3 text-sm flex items-center justify-center transition-colors"
        :title="isDark ? 'Açık Moda Geç' : 'Koyu Moda Geç'"
        @click="toggleTheme"
      >
        <span>{{ isDark ? '☀️' : '🌙' }}</span>
      </button>

      <!-- User Avatar Badge -->
      <div class="w-8 h-8 rounded-full bg-blue-600 text-white font-black text-xs flex items-center justify-center shadow-md shrink-0">
        SF
      </div>
    </div>
  </header>
</template>
