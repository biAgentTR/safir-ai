<script setup lang="ts">
// Final report event timeline (report.timeline). Each entry: timestamp +
// description. Includes visual markers for Onset timestamp (onset_timestamp_str),
// routine safe frames (safe_timestamps), and active incident frames (incident_timestamps).
import type { TimelineEntry } from '~/types/api'

const store = useAnalysisStore()

const r = computed(() => store.report)
const timeline = computed(() => r.value?.timeline ?? [])
const types = computed(() => r.value?.detected_event_types ?? [])
const onsetStr = computed(() => r.value?.onset_timestamp_str)
const safeTimes = computed(() => r.value?.safe_timestamps ?? [])
const incidentTimes = computed(() => r.value?.incident_timestamps ?? [])

const selected = ref<TimelineEntry | null>(null)
</script>

<template>
  <div class="space-y-5 select-none">
    <!-- Tactical Heatmap Scrubber Bar -->
    <div class="glass-card p-5 space-y-4 glow-accent">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-blue-400 animate-ping"></span>
          <h4 class="text-xs font-bold uppercase tracking-wider text-slate-200">Zaman Çizelgesi &amp; Erken Tespit Isı Haritası (Heatmap)</h4>
        </div>
        <div v-if="onsetStr" class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
          <span>📍 Olay Başlangıcı (Onset):</span>
          <span class="font-mono underline">{{ onsetStr }}</span>
        </div>
      </div>

      <!-- Visual Heatmap Scrubber Line -->
      <div class="space-y-1.5">
        <div class="h-3 rounded-full bg-surface-3 relative overflow-hidden flex border border-edge/80 shadow-inner">
          <div class="w-1/4 bg-emerald-500/80 h-full border-r border-slate-900/50" title="Safe Region"></div>
          <div class="w-1/4 bg-emerald-500/60 h-full border-r border-slate-900/50" title="Safe Region"></div>
          <div v-if="incidentTimes.length || store.isCritical" class="w-2/4 bg-gradient-to-r from-amber-500 via-rose-500 to-red-600 h-full animate-pulse shadow-lg shadow-rose-500/50" title="Critical Incident Zone"></div>
          <div v-else class="w-2/4 bg-emerald-500/80 h-full"></div>
        </div>
        <div class="flex justify-between text-[10px] font-mono text-slate-500">
          <span>00:00 (Başlangıç)</span>
          <span v-if="onsetStr" class="text-amber-400 font-bold">▲ {{ onsetStr }}</span>
          <span>Videolar Bitiş</span>
        </div>
      </div>

      <!-- Safe & Incident Timestamps Pills -->
      <div class="grid sm:grid-cols-2 gap-4 text-xs pt-1">
        <div class="p-3 rounded-xl bg-emerald-500/5 border border-emerald-500/20 space-y-2">
          <div class="text-emerald-400 flex items-center justify-between font-bold">
            <span class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-emerald-400"></span> Rutin Kareler (Safe)</span>
            <span class="font-mono text-[11px]">{{ safeTimes.length }} kare</span>
          </div>
          <div v-if="safeTimes.length" class="flex flex-wrap gap-1.5">
            <span
              v-for="(t, i) in safeTimes"
              :key="i"
              class="px-2 py-0.5 rounded-md font-mono text-[11px] bg-emerald-500/10 text-emerald-300 border border-emerald-500/30"
            >{{ t }}</span>
          </div>
          <div v-else class="text-slate-500 text-[11px] italic">Rutin kare tespiti bulunmuyor</div>
        </div>

        <div class="p-3 rounded-xl bg-rose-500/5 border border-rose-500/20 space-y-2">
          <div class="text-rose-400 flex items-center justify-between font-bold">
            <span class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span> Tehlike Kareleri (Incident)</span>
            <span class="font-mono text-[11px]">{{ incidentTimes.length }} kare</span>
          </div>
          <div v-if="incidentTimes.length" class="flex flex-wrap gap-1.5">
            <span
              v-for="(t, i) in incidentTimes"
              :key="i"
              class="px-2 py-0.5 rounded-md font-mono text-[11px] bg-rose-500/20 text-rose-200 border border-rose-500/40 shadow-sm"
            >{{ t }}</span>
          </div>
          <div v-else class="text-slate-500 text-[11px] italic">Aktif tehlike karesi tespit edilmedi</div>
        </div>
      </div>
    </div>

    <!-- Event Category Badges -->
    <div v-if="types.length" class="flex flex-wrap gap-2">
      <span
        v-for="(t, i) in types"
        :key="i"
        class="text-xs px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-300 font-mono font-semibold"
      >🏷️ {{ t }}</span>
    </div>

    <!-- Main Timeline List -->
    <div v-if="!timeline.length" class="text-xs text-slate-500 py-8 text-center glass-card">
      Zaman çizelgesinde olay kaydı bulunmuyor.
    </div>

    <ol v-else class="relative border-l-2 border-blue-500/40 ml-3 space-y-4 pt-1">
      <li v-for="(e, i) in timeline" :key="i" class="ml-5">
        <span class="absolute -left-[9px] mt-1.5 w-4 h-4 rounded-full bg-blue-500 border-2 border-surface-1 shadow-md shadow-blue-500/50" />
        <button
          type="button"
          class="w-full text-left rounded-xl p-3 bg-surface-2/60 border border-edge/80 hover:bg-surface-2 hover:border-blue-500/50 transition-all duration-200 shadow-md"
          @click="selected = e"
        >
          <div class="text-xs font-mono font-bold text-blue-400 flex items-center justify-between">
            <span>⏱️ {{ mmss(e.timestamp) }}</span>
            <span class="text-[10px] text-slate-500 font-normal">({{ e.timestamp.toFixed(1) }}s)</span>
          </div>
          <div class="text-xs font-medium text-slate-200 mt-1 leading-relaxed">{{ e.description }}</div>
        </button>
      </li>
    </ol>

    <EventDetailPanel :entry="selected" @close="selected = null" />
  </div>
</template>
