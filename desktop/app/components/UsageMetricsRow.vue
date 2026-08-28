<script setup lang="ts">
import type { UsageCategoryBreakdown } from '~/types/api'
import { durationMs, tokenCount } from '~/utils/format'

defineProps<{
  item: UsageCategoryBreakdown
}>()
</script>

<template>
  <div class="flex items-center justify-between text-xs py-1.5 border-b border-edge/40 last:border-0">
    <span class="text-slate-300 font-medium">{{ item.label }}</span>
    <div class="flex items-center gap-3 text-right">
      <span class="font-mono text-slate-400 tabular-nums">{{ item.count }}×</span>
      <span class="font-mono text-slate-300 tabular-nums min-w-[70px]">
        {{ item.tokens != null ? `${tokenCount(item.tokens)} tk` : '— tk' }}
      </span>
      <span class="font-mono text-slate-400 tabular-nums min-w-[65px]">
        {{ durationMs(item.duration_ms) }}
      </span>
    </div>
  </div>
</template>
