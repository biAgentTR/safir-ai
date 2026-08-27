<script setup lang="ts">
// Compact labelled metric cell used across stage details and the KPI panel.
withDefaults(
  defineProps<{ label: string; value: string | number; mono?: boolean }>(),
  { mono: false },
)
</script>

<template>
  <div class="relative overflow-hidden group bg-surface-2 hover:bg-surface-3 border border-edge hover:border-accent/40 rounded-lg px-3 py-2.5 shadow-sm hover:shadow-[0_4px_16px_rgba(20,184,166,0.12)] transition-all duration-200">
    <div class="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-transparent via-accent/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
    <div class="eyebrow truncate text-[10px] text-slate-500 group-hover:text-accent transition-colors">{{ trUpper(label) }}</div>
    <div class="mt-1 text-base font-bold text-slate-100 truncate tabular-nums" :class="mono ? 'font-mono' : ''">{{ value }}</div>
  </div>
</template>
