<script setup lang="ts">
// Technical/debug list of pipeline stages (ADIM 3). This is deliberately the
// raw, information-dense view: [✓] Frame Sampling / [⟳] Context & RAG ...
// In ADIM 4 it becomes a proper modern Pipeline Timeline component.
import type { TraceEvent, TraceStage } from '~/types/api'

const props = defineProps<{
  events: TraceEvent[]
  connected: boolean
}>()

// Canonical order + labels mirror trace_serializer.STAGE_ORDER / STAGE_LABELS.
const STAGES: { stage: TraceStage; label: string }[] = [
  { stage: 'sampler', label: 'Frame Sampling' },
  { stage: 'vlm', label: 'Multimodal Analysis' },
  { stage: 'events', label: 'Event Analysis' },
  { stage: 'agent_context', label: 'Context & RAG' },
  { stage: 'decision', label: 'Agent Decision' },
  { stage: 'escalation', label: 'Risk Escalation' },
  { stage: 'report', label: 'Final Report' },
]

function eventFor(stage: TraceStage): TraceEvent | undefined {
  // last event for the stage (there is one per stage today)
  for (let i = props.events.length - 1; i >= 0; i--) {
    if (props.events[i].stage === stage) return props.events[i]
  }
  return undefined
}

function glyph(stage: TraceStage): string {
  const ev = eventFor(stage)
  if (!ev) return props.connected ? '⟳' : '·'
  if (ev.status === 'completed') return '✓'
  if (ev.status === 'failed') return '✕'
  return '⟳'
}

function toneClass(stage: TraceStage): string {
  const ev = eventFor(stage)
  if (!ev) return 'text-slate-600'
  if (ev.status === 'completed') return 'text-risk-low'
  if (ev.status === 'failed') return 'text-risk-crit'
  return 'text-accent'
}
</script>

<template>
  <ul class="font-mono text-sm divide-y divide-edge">
    <li
      v-for="s in STAGES"
      :key="s.stage"
      class="flex items-baseline gap-3 py-2"
    >
      <span class="w-5 text-center" :class="toneClass(s.stage)">[{{ glyph(s.stage) }}]</span>
      <span class="w-40 shrink-0 text-slate-200">{{ s.label }}</span>
      <span class="text-slate-400 truncate">
        {{ eventFor(s.stage)?.summary ?? '—' }}
      </span>
      <span
        v-if="eventFor(s.stage)?.duration_ms != null"
        class="ml-auto shrink-0 text-slate-600"
      >
        {{ eventFor(s.stage)?.duration_ms }} ms
      </span>
    </li>
  </ul>
</template>
