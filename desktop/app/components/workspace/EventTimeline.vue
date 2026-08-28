<script setup lang="ts">
// Final report event timeline (report.timeline). Each entry: timestamp +
// description. Detected event types (from the report) are shown as tags — no
// fabricated fields beyond what the backend returns. Clicking an entry opens
// EventDetailPanel (nearest evidence frame + report risk context), sourced
// entirely from the already-loaded report — works the same live or history.
import type { TimelineEntry } from '~/types/api'

const store = useAnalysisStore()

const timeline = computed(() => store.report?.timeline ?? [])
const types = computed(() => store.report?.detected_event_types ?? [])

const selected = ref<TimelineEntry | null>(null)
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
        <button
          type="button"
          class="w-full text-left rounded-md -ml-2 px-2 py-1 hover:bg-surface-2/60 transition-colors"
          @click="selected = e"
        >
          <div class="text-xs font-mono text-slate-400">{{ mmss(e.timestamp) }} <span class="text-slate-600">({{ e.timestamp.toFixed(1) }}s)</span></div>
          <div class="text-sm text-slate-200">{{ e.description }}</div>
        </button>
      </li>
    </ol>

    <EventDetailPanel :entry="selected" @close="selected = null" />
  </div>
</template>
