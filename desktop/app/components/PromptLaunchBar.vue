<script setup lang="ts">
// Bolt.new-style "start here" composer — the first thing the operator sees
// on VLM Direct Analysis's landing section (vlm_direct mode). Adapted to
// SAFİR's actual job: attach a video + describe what to look for, not a
// code-gen prompt. Purely presentational — the parent owns submission
// (POST /analyze/jobs via the Pinia store) and what happens afterward.
const props = defineProps<{
  modelValue: string
  videoLabel: string | null
  canSubmit: boolean
  submitting: boolean
  error: string | null
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'pick-file'): void
  (e: 'submit'): void
}>()

const SUGGESTIONS = [
  'Sahnede riskli bir durum var mı değerlendir.',
  'Yaya-araç yakınlaşması var mı kontrol et.',
  'Kişisel koruyucu ekipman ihlali var mı?',
  'Yasak bölgeye giriş oldu mu?',
]

function applySuggestion(text: string) {
  emit('update:modelValue', text)
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && props.canSubmit) {
    e.preventDefault()
    emit('submit')
  }
}
</script>

<template>
  <section class="relative" aria-label="Yeni analiz başlat">
    <div class="ambient-rings" aria-hidden="true">
      <span class="ambient-ring ambient-ring-a" />
      <span class="ambient-ring ambient-ring-b" />
    </div>
    <div class="grid-texture" aria-hidden="true" />

    <div class="relative max-w-3xl mx-auto text-center px-4 py-10 sm:py-14">
      <div class="eyebrow !text-accent mb-3">Yapay zekâ destekli operasyonel farkındalık</div>
      <h1 class="text-3xl sm:text-4xl font-bold tracking-tight text-slate-100 leading-tight">
        Görüntüyü izlemeyin.<br class="hidden sm:block" />
        Ne olduğunu anlayın.
      </h1>
      <p class="mt-3 text-sm sm:text-base text-slate-400 max-w-xl mx-auto">
        Bir video seçin, ne aradığınızı yazın — SAFİR kritik anları, riski ve uygulanabilir operatör aksiyonlarını saniyeler içinde çıkarsın.
      </p>

      <form class="mt-7 glass-panel rounded-2xl p-2.5 text-left" @submit.prevent="$emit('submit')">
        <textarea
          :value="modelValue"
          rows="2"
          placeholder="Bu videoda ne olduğunu sorun… (örn. “Yaya-araç yakınlaşması var mı?”)"
          class="w-full resize-none bg-transparent px-3.5 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none"
          aria-label="Analiz isteminiz"
          @input="$emit('update:modelValue', ($event.target as HTMLTextAreaElement).value)"
          @keydown="onKeydown"
        />
        <div class="flex items-center gap-2 px-1.5 pb-1">
          <button type="button" class="btn-ghost !py-1.5 !px-3 text-xs" @click="$emit('pick-file')">
            <span aria-hidden="true">📎</span>
            {{ videoLabel ? 'Videoyu Değiştir' : 'Video Ekle' }}
          </button>
          <span v-if="videoLabel" class="min-w-0 flex-1 truncate text-xs font-mono text-slate-400">{{ videoLabel }}</span>
          <span v-else class="flex-1" />
          <button
            type="submit"
            class="btn-primary !rounded-full !p-0 w-9 h-9 shrink-0"
            :disabled="!canSubmit"
            :aria-label="submitting ? 'Analiz başlatılıyor' : 'Analizi başlat'"
          >
            <span v-if="submitting" class="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin motion-reduce:animate-none" />
            <span v-else aria-hidden="true">↑</span>
          </button>
        </div>
      </form>

      <p v-if="error" role="alert" class="mt-3 text-sm text-risk-crit flex items-center justify-center gap-1.5">
        <span aria-hidden="true">⚠</span>{{ error }}
      </p>

      <div class="mt-4 flex flex-wrap items-center justify-center gap-2">
        <button
          v-for="s in SUGGESTIONS"
          :key="s"
          type="button"
          class="rounded-full border border-edge bg-surface-2/60 px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 hover:border-edge-strong hover:bg-surface-2 transition-colors"
          @click="applySuggestion(s)"
        >
          {{ s }}
        </button>
      </div>
    </div>
  </section>
</template>
