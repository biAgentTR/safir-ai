<script setup lang="ts">
// Cumulative pipeline rail. All stages (STAGE_META) are always shown, in canonical order;
// completed stages stay visible. Clicking a stage selects it for the detail
// panel. Real status comes from the store's trace events.
import type { TraceStage, TraceStatus } from '~/types/api'

const store = useAnalysisStore()

function status(stage: TraceStage): TraceStatus | 'pending' {
  const ev = store.eventForStage(stage)
  if (ev) return ev.status
  // If the job is running and every earlier stage is done, mark this as running.
  if (store.isRunning) {
    const idx = STAGE_META.findIndex((m) => m.stage === stage)
    const prevDone = STAGE_META.slice(0, idx).every((m) => store.eventForStage(m.stage)?.status === 'completed')
    const selfDone = !!store.eventForStage(stage)
    if (prevDone && !selfDone) return 'running'
  }
  return 'pending'
}

const glyph: Record<string, string> = {
  completed: '✓',
  running: '⟳',
  failed: '✕',
  pending: '',
  queued: '',
}
function flow(stage: TraceStage) {
  return stageFlow(stage, store.eventForStage(stage)?.data)
}
function tone(s: string): string {
  if (s === 'completed') return 'text-risk-low border-risk-low/40'
  if (s === 'failed') return 'text-risk-crit border-risk-crit/40'
  if (s === 'running') return 'text-accent border-accent/50'
  return 'text-slate-600 border-edge'
}
</script>

<template>
  <div class="card p-3">
    <div class="px-2 pb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">ANALİZ HATTI</div>
    <ol class="relative">
      <li
        v-for="(m, i) in STAGE_META"
        :key="m.stage"
        class="relative"
      >
        <!-- connector -->
        <span
          v-if="i < STAGE_META.length - 1"
          class="absolute left-[19px] top-8 h-[calc(100%-1.5rem)] w-px"
          :class="status(m.stage) === 'completed' ? 'bg-risk-low/30' : 'bg-edge'"
        />
        <button
          type="button"
          class="w-full flex items-center gap-3 rounded-md px-2 py-2 text-left transition-colors"
          :class="store.selectedStage === m.stage ? 'bg-surface-2' : 'hover:bg-surface-2/50'"
          @click="store.selectStage(m.stage)"
        >
          <span
            class="grid place-items-center w-6 h-6 shrink-0 rounded-full border text-xs font-bold"
            :class="[tone(status(m.stage)), status(m.stage) === 'running' ? 'animate-pulse' : '']"
          >
            <span v-if="glyph[status(m.stage)]">{{ glyph[status(m.stage)] }}</span>
            <span v-else class="text-[10px] text-slate-600">{{ i + 1 }}</span>
          </span>
          <span class="min-w-0">
            <span
              class="block text-sm truncate"
              :class="status(m.stage) === 'pending' ? 'text-slate-500' : 'text-slate-100'"
            >{{ m.label }}</span>
            <span class="block text-[11px] text-slate-500 truncate">
              {{ store.eventForStage(m.stage)?.summary ?? m.blurb }}
            </span>
            <!-- Input -> Output (real data when the stage has run) -->
            <span class="mt-0.5 flex items-center gap-1 text-[10px] font-mono text-slate-600 truncate">
              <span class="truncate">{{ flow(m.stage).in }}</span>
              <span class="text-slate-700">→</span>
              <span class="truncate" :class="store.eventForStage(m.stage) ? 'text-slate-500' : ''">{{ flow(m.stage).out }}</span>
            </span>
          </span>
          <span
            v-if="store.eventForStage(m.stage)?.duration_ms != null"
            class="ml-auto shrink-0 text-[10px] font-mono text-slate-600"
          >{{ store.eventForStage(m.stage)?.duration_ms }}ms</span>
        </button>
      </li>
    </ol>
  </div>
</template>
