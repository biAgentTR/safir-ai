<script setup lang="ts">
// Main video panel for VLM Direct mode: the operator's chosen video, a
// native <video> element, and a custom timeline strip beneath it with
// risk-colored markers at every VLM-flagged timestamp. Clicking a marker (or
// an event elsewhere on the page, via activeEventId) seeks the video there.
//
// File selection itself is NOT handled here: the backend needs a real local
// filesystem path (POST /analyze/jobs video_source), not a browser `File`
// object, so the parent page owns the Tauri file dialog (same pattern as
// pages/new-analysis.vue) and just hands this component a playable `videoUrl`
// (a converted asset:// URL) plus the display `fileName`. This component only
// asks the parent to open that picker via 'pick-file'.
import type { VlmEvent, VlmRiskLevel } from '~/types/vlm'

const props = defineProps<{
  videoUrl: string | null
  fileName: string | null
  events: VlmEvent[]
  duration: number
  currentTime: number
  activeEventId: string | null
}>()

const emit = defineEmits<{
  (e: 'pick-file'): void
  (e: 'time-update', t: number): void
  (e: 'duration-change', d: number): void
  (e: 'select-event', id: string): void
}>()

const videoEl = ref<HTMLVideoElement | null>(null)

function onLoadedMetadata() {
  if (videoEl.value) emit('duration-change', videoEl.value.duration)
}
function onTimeUpdate() {
  if (videoEl.value) emit('time-update', videoEl.value.currentTime)
}

function seek(t: number) {
  if (videoEl.value) {
    videoEl.value.currentTime = t
    videoEl.value.play().catch(() => {})
  }
}
function onMarkerClick(ev: VlmEvent) {
  seek(ev.timestamp)
  emit('select-event', ev.id)
}

// external seek requests (e.g. clicking a row in VlmEventList)
watch(
  () => props.activeEventId,
  (id) => {
    const ev = props.events.find((e) => e.id === id)
    if (ev) seek(ev.timestamp)
  },
)

const RISK_MARKER: Record<VlmRiskLevel, string> = {
  low: 'bg-risk-low',
  mid: 'bg-risk-mid',
  high: 'bg-risk-high',
  crit: 'bg-risk-crit',
}

function markerPct(t: number): number {
  if (!props.duration) return 0
  return Math.min(100, Math.max(0, (t / props.duration) * 100))
}

function onScrub(e: MouseEvent) {
  const bar = e.currentTarget as HTMLElement
  const rect = bar.getBoundingClientRect()
  const pct = (e.clientX - rect.left) / rect.width
  seek(Math.min(props.duration, Math.max(0, pct * props.duration)))
}

const progressPct = computed(() => markerPct(props.currentTime))
</script>

<template>
  <div class="card p-4 flex flex-col gap-3">
    <div class="flex items-center justify-between gap-2">
      <h3 class="text-sm font-semibold text-slate-100">Video</h3>
      <div class="flex items-center gap-2">
        <span v-if="fileName" class="text-xs text-slate-500 truncate max-w-[14rem]">{{ fileName }}</span>
        <button type="button" class="btn-ghost !py-1.5 !px-2.5 text-xs" @click="emit('pick-file')">
          <span aria-hidden="true">⤒</span> {{ fileName ? 'Videoyu Değiştir' : 'Video Seç' }}
        </button>
      </div>
    </div>

    <!-- video surface -->
    <div class="relative rounded-md overflow-hidden bg-black aspect-video flex items-center justify-center">
      <video
        v-if="videoUrl"
        ref="videoEl"
        :src="videoUrl"
        controls
        class="w-full h-full"
        @loadedmetadata="onLoadedMetadata"
        @timeupdate="onTimeUpdate"
      />
      <div v-else-if="fileName" class="text-center px-6">
        <p class="text-sm text-slate-400">Video önizlemesi bu ortamda kullanılamıyor; analiz seçilen dosya üzerinden devam eder.</p>
      </div>
      <div v-else class="text-center px-6">
        <div class="text-4xl mb-3 text-slate-600" aria-hidden="true">▶</div>
        <p class="text-sm text-slate-400">Video seçtikten sonra burada oynatılacaktır.</p>
        <p class="mt-1 text-xs text-slate-600">Yukarıdaki "Video Seç" ile bir dosya seçin.</p>
      </div>
    </div>

    <!-- risk-marked timeline -->
    <div v-if="videoUrl && duration" class="pt-1">
      <div
        class="relative h-3 rounded-full bg-surface-2 border border-edge cursor-pointer"
        role="slider"
        aria-label="Video zaman çizelgesi"
        :aria-valuenow="Math.round(currentTime)"
        :aria-valuemax="Math.round(duration)"
        @click="onScrub"
      >
        <div class="absolute inset-y-0 left-0 bg-accent-soft rounded-full pointer-events-none" :style="{ width: progressPct + '%' }" />
        <div class="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-accent border-2 border-surface-1 pointer-events-none" :style="{ left: `calc(${progressPct}% - 5px)` }" />
        <button
          v-for="ev in events"
          :key="ev.id"
          type="button"
          class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-3.5 rounded-sm hover:scale-125 transition-transform"
          :class="[RISK_MARKER[ev.riskLevel], activeEventId === ev.id ? 'ring-2 ring-white/70' : '']"
          :style="{ left: markerPct(ev.timestamp) + '%' }"
          :title="`${mmss(ev.timestamp)} · ${ev.type}`"
          @click.stop="onMarkerClick(ev)"
        />
      </div>
      <div class="mt-1.5 flex items-center justify-between text-[11px] font-mono text-slate-500">
        <span>{{ mmss(currentTime) }}</span>
        <span>{{ mmss(duration) }}</span>
      </div>
      <p class="mt-1 text-[11px] text-slate-500">Kırmızı/turuncu/sarı işaretler VLM'in riskli bulduğu saniyelerdir — tıklayarak o ana gidin.</p>
    </div>
  </div>
</template>
