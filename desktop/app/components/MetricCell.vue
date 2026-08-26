<script setup lang="ts">
// Compact labelled metric cell used across stage details and the KPI panel.
withDefaults(
  defineProps<{ label: string; value: string | number; mono?: boolean }>(),
  { mono: false },
)
</script>

<template>
  <div class="bg-surface-2 border border-edge rounded-md px-3 py-2">
    <div class="eyebrow truncate">{{ trUpper(label) }}</div>
    <div class="mt-0.5 text-sm text-slate-100 truncate tabular-nums" :class="mono ? 'font-mono' : ''">{{ value }}</div>
  </div>
</template>
