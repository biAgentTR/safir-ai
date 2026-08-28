<template>
  <div class="h-screen w-screen flex flex-col overflow-hidden font-sans bg-[var(--color-bg)] text-[var(--color-text)]">
    <header class="h-[72px] shrink-0 w-full border-b border-[var(--color-border)] bg-[var(--color-bg)]/90 backdrop-blur-md z-50">
      <div class="h-full max-w-[1440px] mx-auto px-6 flex items-center justify-between">
        
        <!-- Left: Logo -->
        <div class="flex items-center gap-8 w-1/3">
          <NuxtLink to="/" class="flex items-center gap-3">
            <div class="w-8 h-8 rounded bg-[var(--color-primary)]/10 flex items-center justify-center border border-[var(--color-primary)]/20 text-[var(--color-primary)]">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/></svg>
            </div>
            <div class="flex items-baseline gap-2">
              <div class="font-bold text-lg tracking-wider text-white">SAFİR</div>
              <div class="text-[9px] text-[var(--color-text-muted)] tracking-widest uppercase">Vision Intelligence</div>
            </div>
          </NuxtLink>
        </div>
        
        <!-- Center: Nav -->
        <nav class="hidden md:flex items-center justify-center gap-6 w-1/3 h-full">
          <NuxtLink to="/" class="h-full flex items-center text-sm font-medium transition-colors hover:text-white border-b-2 border-transparent" active-class="!text-[var(--color-primary)] !border-[var(--color-primary)]">Operasyon</NuxtLink>
          <NuxtLink to="/analizler" class="h-full flex items-center text-sm font-medium transition-colors hover:text-white border-b-2 border-transparent text-[var(--color-text-secondary)]" active-class="!text-[var(--color-primary)] !border-[var(--color-primary)]">Analizler</NuxtLink>
          <NuxtLink to="/raporlar" class="h-full flex items-center text-sm font-medium transition-colors hover:text-white border-b-2 border-transparent text-[var(--color-text-secondary)]" active-class="!text-[var(--color-primary)] !border-[var(--color-primary)]">Raporlar</NuxtLink>
        </nav>
        
        <!-- Right: Status & Admin -->
        <div class="flex items-center justify-end gap-4 w-1/3">
          <div class="flex items-center gap-2 px-3 py-1.5 rounded-full border border-[var(--color-border)] text-xs font-medium"
               :class="[
                 backendState === 'online' ? 'text-[var(--color-success)]' :
                 backendState === 'checking' ? 'text-[var(--color-text-secondary)]' :
                 'text-rose-400'
               ]">
            <div class="w-1.5 h-1.5 rounded-full"
                 :class="[
                   backendState === 'online' ? 'bg-[var(--color-success)] shadow-[0_0_8px_rgba(56,242,178,0.6)]' :
                   backendState === 'checking' ? 'bg-[var(--color-text-muted)] animate-pulse' :
                   'bg-rose-500'
                 ]"></div>
            {{ backendState === 'online' ? 'SİSTEM AKTİF' : backendState === 'checking' ? 'KONTROL EDİLİYOR' : 'SİSTEM ÇEVRİMDIŞI' }}
          </div>
          <NuxtLink to="/admin/login" class="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md border border-[var(--color-border)] hover:border-[var(--color-border-strong)] transition-colors text-[var(--color-text-secondary)] hover:text-white">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
            Yönetici girişi
          </NuxtLink>
          <div class="w-8 h-8 rounded-full bg-[var(--color-surface-hover)] border border-[var(--color-border)] flex items-center justify-center text-xs font-medium text-white">HS</div>
        </div>

      </div>
    </header>
    
    <main class="flex-1 overflow-y-auto overflow-x-hidden relative">
      <slot />
    </main>
  </div>
</template>
<script setup lang="ts">
import { useBackendHealth } from '~/composables/useBackendHealth'

const { state: backendState } = useBackendHealth()
</script>
<style>
body { background-color: #0b0e14; color: #e2e8f0; }
</style>
