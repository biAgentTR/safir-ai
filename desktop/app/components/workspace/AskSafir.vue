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

async function submit() {
  if (!canSend.value) return
  phase.value = 'loading'
  errorMsg.value = ''
  answer.value = ''
  sources.value = []
  contextUsed.value = []
  try {
    const res = await api.ask(question.value.trim(), props.jobId ?? null)
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
  <section class="glass-card p-6 border border-edge/80 space-y-4">
    <div class="flex items-center justify-between border-b border-edge/60 pb-3">
      <div class="flex items-center gap-2.5">
        <div class="w-7 h-7 rounded-lg bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400 text-xs font-bold">
          ◆
        </div>
        <div>
          <h3 class="text-sm font-bold text-white tracking-wide">SAFİR AI Asistan</h3>
          <p class="text-[10px] text-slate-400">Analiz &amp; Mevzuat Bağlamlı Karar Destek Asistanı</p>
        </div>
      </div>
      <span v-if="jobId" class="px-2.5 py-1 rounded-full bg-surface-2 border border-edge/80 text-[10px] font-mono text-blue-400">
        job: {{ jobId.slice(0, 8) }}
      </span>
    </div>

    <!-- input -->
    <div class="space-y-2">
      <div class="flex gap-2.5">
        <textarea
          v-model="question"
          rows="2"
          class="field-input resize-none flex-1 font-sans text-xs leading-relaxed"
          placeholder="Örn: Bu analiz neden orta risk olarak değerlendirildi veya hangi aksiyonlar alınmalı?"
          @keydown.enter.exact.prevent="submit"
        />
        <button class="btn-primary self-stretch px-6 text-xs font-bold" :disabled="!canSend" @click="submit">
          <span v-if="phase === 'loading'" class="animate-pulse">Soruluyor…</span>
          <span v-else>Sor 🚀</span>
        </button>
      </div>

      <!-- suggestion chips -->
      <div class="flex flex-wrap gap-2 pt-1">
        <button
          v-for="s in SUGGESTIONS"
          :key="s"
          class="text-[11px] px-3 py-1 rounded-full border border-blue-500/20 bg-blue-500/5 text-blue-300 hover:bg-blue-500/15 hover:border-blue-500/40 transition-all duration-200"
          :disabled="phase === 'loading'"
          @click="useSuggestion(s)"
        >
          {{ s }}
        </button>
      </div>
    </div>

    <!-- loading -->
    <div v-if="phase === 'loading'" class="p-4 rounded-xl border border-blue-500/30 bg-blue-500/10 flex items-center gap-3 text-xs text-blue-200 animate-pulse">
      <span class="inline-block w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
      SAFİR FAISS RAG belleğini sorguluyor ve yanıtı hazırlıyor…
    </div>

    <!-- error -->
    <div v-else-if="phase === 'error'" class="p-3.5 rounded-xl border border-rose-500/40 bg-rose-500/10 text-xs text-rose-300 font-medium">
      {{ errorMsg }}
    </div>

    <!-- answer -->
    <div v-else-if="phase === 'success'" class="space-y-3 pt-2">
      <div class="p-4 rounded-xl border border-blue-500/30 bg-surface-2/90 space-y-2">
        <div class="flex items-center gap-2">
          <span class="px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 text-[10px] font-mono font-bold">SAFİR AI</span>
          <span class="text-[10px] text-slate-400">Üretilen Akıllı Yanıt</span>
        </div>
        <p class="text-xs text-slate-100 leading-relaxed whitespace-pre-line">{{ answer }}</p>
      </div>

      <div v-if="sources.length" class="space-y-2">
        <button class="text-xs font-semibold text-slate-400 hover:text-slate-200 flex items-center gap-1.5" @click="showSources = !showSources">
          <span>{{ showSources ? '▾' : '▸' }}</span>
          <span>Kullanılan Kaynaklar &amp; Mevcut Bağlam ({{ sources.length }})</span>
        </button>
        <ul v-if="showSources" class="space-y-1.5">
          <li
            v-for="(s, i) in sources"
            :key="i"
            class="text-xs bg-surface-2/80 border border-edge/80 rounded-lg p-2.5 flex items-center justify-between gap-3"
          >
            <div class="flex items-center gap-2 min-w-0">
              <span
                class="shrink-0 px-2 py-0.5 rounded text-[10px] font-mono font-bold"
                :class="s.type === 'analysis' ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30' : 'bg-purple-500/20 text-purple-300 border border-purple-500/30'"
              >
                {{ s.type === 'analysis' ? 'Analiz Kaydı' : 'Mevzuat RAG' }}
              </span>
              <span class="text-slate-200 truncate font-medium">{{ s.label ?? s.text }}</span>
            </div>
            <span v-if="s.score != null" class="shrink-0 font-mono text-[10px] text-slate-400">
              skor {{ s.score }}
            </span>
          </li>
        </ul>
      </div>

      <p v-if="contextUsed.length" class="text-[10px] font-mono text-slate-500">
        Kullanılan bağlam bileşenleri: {{ contextUsed.join(' · ') }}
      </p>
    </div>
  </section>
</template>
