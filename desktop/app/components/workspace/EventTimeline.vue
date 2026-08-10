<script setup lang="ts">
// Final report event timeline (report.timeline). Each entry: timestamp +
// description. Detected event types (from the report) are shown as tags — no
// fabricated fields beyond what the backend returns.
const store = useAnalysisStore()

const timeline = computed(() => store.report?.timeline ?? [])
const types = computed(() => store.report?.detected_event_types ?? [])
</script>

<template>
  <div>
    <div v-if="types.length" class="mb-4 flex flex-wrap gap-2">
      <span
        v-for="(t, i) in types"
        :key="i"
        class="text-xs px-2 py-1 rounded border border-edge bg-surface-2 text-slate-300"
      >{{ t }}</span>
    </div>

    <div v-if="!timeline.length" class="text-sm text-slate-500 py-6 text-center">
      Zaman çizelgesinde kayıt yok.
    </div>

    <ol v-else class="relative border-l border-edge ml-2 space-y-4">
      <li v-for="(e, i) in timeline" :key="i" class="ml-4">
        <span class="absolute -left-[6px] mt-1.5 w-3 h-3 rounded-full bg-accent border-2 border-surface-1" />
        <div class="text-xs font-mono text-slate-400">{{ mmss(e.timestamp) }} <span class="text-slate-600">({{ e.timestamp.toFixed(1) }}s)</span></div>
        <div class="text-sm text-slate-200">{{ e.description }}</div>
      </li>
    </ol>
  </div>
</template>
