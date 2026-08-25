<script setup lang="ts">
// Ask SAFİR — context-aware decision-support assistant embedded in the
// Workspace (not a standalone chatbot). Sends the question (+ the current
// job_id when available) to POST /ask and renders the grounded answer with its
// real sources. Works in both live and history workspaces.
import type { AskSource } from '~/types/api'

const props = defineProps<{ jobId?: string | null }>()
const api = useSafirApi()

type Phase = 'idle' | 'loading' | 'success' | 'error'
const phase = ref<Phase>('idle')
const question = ref('')
const answer = ref('')
const sources = ref<AskSource[]>([])
const contextUsed = ref<string[]>([])
const errorMsg = ref('')
const showSources = ref(true)

const SUGGESTIONS = [
  'Bu analiz neden bu risk seviyesinde değerlendirildi?',
  'Operatör ne yapmalı?',
  'Hangi İSG mevzuatı ilgili?',
]

const canSend = computed(() => question.value.trim().length > 0 && phase.value !== 'loading')

// "Videoya sor" (EVREN prefix-cache follow-up, bkz. AskSafirAsistan sayfası
// ile aynı desen) — yalnızca bir job bağlamı varken anlamlıdır.
const useVideo = ref(false)

async function submit() {
  if (!canSend.value) return
  phase.value = 'loading'
  errorMsg.value = ''
  answer.value = ''
  sources.value = []
  contextUsed.value = []
  try {
    const res = await api.ask(question.value.trim(), props.jobId ?? null, useVideo.value && !!props.jobId)
    answer.value = res.answer
    sources.value = res.sources ?? []
    contextUsed.value = res.context_used ?? []
    phase.value = 'success'
  } catch (e: unknown) {
    const status = (e as { response?: { status?: number }; statusCode?: number })?.statusCode
    const detail = (e as { data?: { detail?: string } })?.data?.detail
    // Never surface a stack trace — show a safe, human message.
    errorMsg.value =
      status === 503
        ? 'SAFİR şu anda cevap oluşturamadı.'
        : detail ?? 'SAFİR şu anda cevap oluşturamadı.'
    phase.value = 'error'
  }
}

function useSuggestion(s: string) {
  question.value = s
  submit()
}
</script>

<template>
  <section class="card p-5">
    <div class="flex items-center gap-2 mb-3">
      <h3 class="text-sm font-semibold text-slate-100 tracking-wide">SAFİR'e Sor</h3>
      <span class="text-[11px] text-slate-500">— analiz &amp; İSG mevzuatı bağlamlı asistan</span>
      <span v-if="jobId" class="ml-auto text-[10px] font-mono text-slate-600">job: {{ jobId.slice(0, 8) }}</span>
    </div>

    <!-- input -->
    <div class="flex gap-2">
      <textarea
        v-model="question"
        rows="2"
        class="field-input resize-none flex-1"
        placeholder="Örn: Bu analiz neden orta risk olarak değerlendirildi?"
        @keydown.enter.exact.prevent="submit"
      />
      <button class="btn-primary self-stretch px-5" :disabled="!canSend" @click="submit">
        <span v-if="phase === 'loading'">…</span><span v-else>Sor</span>
      </button>
    </div>

    <!-- suggestion chips -->
    <div class="mt-2 flex flex-wrap items-center gap-2">
      <button
        v-for="s in SUGGESTIONS"
        :key="s"
        class="text-[11px] px-2 py-1 rounded border border-edge bg-surface-2 text-slate-400 hover:text-slate-200"
        :disabled="phase === 'loading'"
        @click="useSuggestion(s)"
      >{{ s }}</button>
      <button
        v-if="jobId"
        type="button"
        class="ml-auto flex items-center gap-1.5 text-[11px] px-2 py-1 rounded border transition-colors"
        :class="useVideo
          ? 'border-accent/50 bg-accent-soft text-accent'
          : 'border-edge bg-surface-2 text-slate-400 hover:text-slate-200'"
        :disabled="phase === 'loading'"
        @click="useVideo = !useVideo"
      >
        <span>🎥</span> Videoya sor
      </button>
    </div>

    <!-- loading -->
    <div v-if="phase === 'loading'" class="mt-4 flex items-center gap-2 text-sm text-slate-400">
      <span class="inline-block w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      SAFİR yanıtı hazırlıyor…
    </div>

    <!-- error (safe message; no stack trace) -->
    <div v-else-if="phase === 'error'" class="mt-4 rounded-md border border-risk-crit/30 bg-risk-crit/10 px-3 py-2 text-sm text-risk-crit">
      {{ errorMsg }}
    </div>

    <!-- answer -->
    <div v-else-if="phase === 'success'" class="mt-4 space-y-3">
      <div class="rounded-md border border-edge bg-surface-2 p-3">
        <div class="text-[10px] uppercase tracking-wide text-slate-500 mb-1">SAFİR</div>
        <p class="text-sm text-slate-100 leading-relaxed whitespace-pre-line">{{ answer }}</p>
      </div>

      <div v-if="sources.length">
        <button class="text-xs text-slate-400 hover:text-slate-200" @click="showSources = !showSources">
          {{ showSources ? '▾' : '▸' }} Kaynaklar ({{ sources.length }})
        </button>
        <ul v-if="showSources" class="mt-2 space-y-1.5">
          <li
            v-for="(s, i) in sources"
            :key="i"
            class="text-xs bg-surface-2 border border-edge rounded-md px-3 py-2 flex gap-2"
          >
            <span
              class="shrink-0 px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wide"
              :class="s.type === 'analysis' || s.type === 'video' ? 'bg-accent-soft text-accent' : 'bg-surface-3 text-slate-400'"
            >{{ s.type === 'analysis' ? 'ANALİZ' : s.type === 'video' ? 'VİDEO' : 'MEVZUAT' }}</span>
            <span class="text-slate-300 min-w-0">
              {{ s.label ?? s.text }}
              <span v-if="s.score != null" class="text-slate-600"> · skor {{ s.score }}</span>
            </span>
          </li>
        </ul>
      </div>

      <p v-if="contextUsed.length" class="text-[11px] text-slate-600">
        Kullanılan bağlam: {{ contextUsed.join(' · ') }}
      </p>
    </div>
  </section>
</template>
