<script setup lang="ts">
// Evidence gallery: frame cards served from /frames/{frame_id} (no base64),
// click to open a lightbox, and Human-in-the-Loop TP/FP feedback wired to the
// real POST /events/{event_id}/feedback endpoint.
import type { EvidenceFrameRef, SamplerStageData } from '~/types/api'

const store = useAnalysisStore()
const { getFrameUrl } = useSafirApi()

const frames = computed<EvidenceFrameRef[]>(
  () => (store.eventForStage('sampler')?.data as SamplerStageData | undefined)?.evidence_frames ?? [],
)
const canFeedback = computed(() => store.report?.event_id != null)

const lightbox = ref<EvidenceFrameRef | null>(null)
function url(id: string) {
  return store.jobId ? getFrameUrl(store.jobId, id) : ''
}
</script>

<template>
  <div>
    <div v-if="!frames.length" class="text-sm text-slate-500 py-6 text-center">
      Bu analiz için kanıt karesi üretilmedi.
    </div>

    <div v-else class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
      <figure
        v-for="f in frames"
        :key="f.frame_id"
        class="border border-edge rounded-lg overflow-hidden bg-surface-2"
      >
        <button type="button" class="block w-full" @click="lightbox = f">
          <img :src="url(f.frame_id)" :alt="f.frame_id" class="w-full h-32 object-cover hover:opacity-90" loading="lazy" />
        </button>
        <figcaption class="px-3 py-2">
          <div class="text-xs font-mono text-slate-300">{{ f.timestamp_str }}</div>
          <div class="text-[11px] text-slate-500">değişim skoru {{ f.change_score }}<span v-if="f.is_fallback" class="text-risk-mid"> · yedek</span></div>
        </figcaption>
      </figure>
    </div>

    <!-- feedback (single event_id per analysis — matches backend) -->
    <div v-if="frames.length && canFeedback" class="mt-4 flex items-center gap-3">
      <span class="text-xs text-slate-500">Bu analiz doğru muydu?</span>
      <button
        class="btn-ghost !border-risk-low/40 !text-risk-low"
        :disabled="store.feedback.state === 'pending'"
        @click="store.submitFeedback('true_positive')"
      >Doğru İhbar</button>
      <button
        class="btn-ghost !border-risk-crit/40 !text-risk-crit"
        :disabled="store.feedback.state === 'pending'"
        @click="store.submitFeedback('false_positive')"
      >Yanlış Alarm</button>
      <span
        v-if="store.feedback.message"
        class="text-xs"
        :class="store.feedback.state === 'error' ? 'text-risk-crit' : 'text-risk-low'"
      >{{ store.feedback.message }}</span>
    </div>

    <!-- lightbox -->
    <div
      v-if="lightbox"
      class="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8"
      @click="lightbox = null"
    >
      <figure class="max-w-3xl" @click.stop>
        <img :src="url(lightbox.frame_id)" :alt="lightbox.frame_id" class="max-h-[70vh] w-auto rounded-lg border border-edge" />
        <figcaption class="mt-3 flex items-center gap-4 text-sm text-slate-300">
          <span class="font-mono">{{ lightbox.timestamp_str }}</span>
          <span class="text-slate-500">change score {{ lightbox.change_score }}</span>
          <button class="btn-ghost ml-auto" @click="lightbox = null">Kapat</button>
        </figcaption>
      </figure>
    </div>
  </div>
</template>
