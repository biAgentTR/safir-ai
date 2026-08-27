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
    <div class="relative max-w-3xl mx-auto text-center px-4 py-10 sm:py-14">
      <!-- Title Ambient Glow Aura -->
      <div class="heading-glow-hero" />

      <div class="eyebrow !text-accent mb-3 relative z-10">Yapay zekâ destekli operasyonel farkındalık</div>
      <h1 class="text-3xl sm:text-4xl font-bold tracking-tight text-slate-100 leading-tight relative z-10">
        Görüntüyü izlemeyin.<br class="hidden sm:block" />
        Ne olduğunu anlayın.
      </h1>
      <p class="mt-3 text-sm sm:text-base text-slate-400 max-w-xl mx-auto relative z-10">
        Bir video seçin, ne aradığınızı yazın — SAFİR kritik anları, riski ve uygulanabilir operatör aksiyonlarını saniyeler içinde çıkarsın.
      </p>

      <!-- Glass Composer Box with glowing border effects -->
      <form class="mt-7 relative overflow-hidden glass-panel rounded-2xl p-3 text-left border border-edge/90 hover:border-accent/50 shadow-2xl hover:shadow-[0_16px_36px_-8px_rgba(20,184,166,0.18)] transition-all duration-300" @submit.prevent="$emit('submit')">
        <!-- Top Luminous Hairline Accent -->
        <div class="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-transparent via-accent/60 to-transparent" />

        <textarea
          :value="modelValue"
          rows="2"
          placeholder="Bu videoda ne olduğunu sorun… (örn. “Yaya-araç yakınlaşması var mı?”)"
          class="w-full resize-none bg-transparent px-3.5 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none"
          aria-label="Analiz isteminiz"
          @input="$emit('update:modelValue', ($event.target as HTMLTextAreaElement).value)"
          @keydown="onKeydown"
        />
        <div class="flex items-center gap-2.5 px-1.5 pb-1">
          <button type="button" class="btn-ghost !py-1.5 !px-3 text-xs flex items-center gap-1.5" @click="$emit('pick-file')">
            <span aria-hidden="true">📎</span>
            {{ videoLabel ? 'Videoyu Değiştir' : 'Video Ekle' }}
          </button>
          <span v-if="videoLabel" class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-surface-2 border border-accent/30 text-xs font-mono text-accent truncate max-w-xs shadow-sm">
            <span class="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            <span class="truncate">{{ videoLabel }}</span>
          </span>
          <span v-else class="flex-1" />
          <button
            type="submit"
            class="btn-primary !rounded-full !p-0 w-9 h-9 shrink-0 shadow-[0_0_15px_rgba(20,184,166,0.3)] hover:scale-105 transition-transform"
            :disabled="!canSubmit"
            :aria-label="submitting ? 'Analiz başlatılıyor' : 'Analizi başlat'"
          >
            <span v-if="submitting" class="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin motion-reduce:animate-none" />
            <span v-else aria-hidden="true" class="text-base font-bold">↑</span>
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
          class="rounded-full border border-edge bg-surface-2/70 px-3.5 py-1.5 text-xs text-slate-400 hover:text-slate-100 hover:border-accent/40 hover:bg-surface-3 transition-all duration-200 shadow-sm flex items-center gap-1.5 cursor-pointer"
          @click="applySuggestion(s)"
        >
          <span class="text-accent/60 text-[10px]">✦</span>
          <span>{{ s }}</span>
        </button>
      </div>
    </div>
  </section>
</template>
