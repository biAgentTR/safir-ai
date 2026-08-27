<script setup lang="ts">
// Collapsible right-side chat drawer for asking SAFİR about the video just
// analyzed in VLM Direct Analiz. A lighter, multi-turn sibling of
// workspace/AskSafir.vue (which is single-turn, no running thread) — this
// one keeps a local message history so it reads like a real chat, but
// deliberately does NOT persist to the backend /conversations endpoints
// (unlike the full SAFİR Asistan page): it's a quick contextual side panel
// tied to the current job, not a saved conversation to revisit later.
import type { AskSource } from '~/types/api'

const props = defineProps<{ jobId: string | null }>()

const api = useSafirApi()
const askStream = useAskStream()

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  pending?: boolean
}

const isOpen = ref(false)
const messages = ref<ChatMessage[]>([])
const question = ref('')
const sending = ref(false)
const sendError = ref<string | null>(null)
const lastSources = ref<AskSource[]>([])
const showSources = ref(false)

const canSend = computed(() => question.value.trim().length > 0 && !sending.value)

const STATIC_SUGGESTIONS = [
  'Bu videoda ne oldu, özetler misin?',
  'Bu risk seviyesi neden bu şekilde değerlendirildi?',
  'Operatör şimdi ne yapmalı?',
]
const dynamicSuggestions = ref<string[]>([])
const suggestions = computed(() => (dynamicSuggestions.value.length ? dynamicSuggestions.value : STATIC_SUGGESTIONS))

watch(
  () => props.jobId,
  async (jobId) => {
    dynamicSuggestions.value = []
    if (!jobId) return
    try {
      dynamicSuggestions.value = await api.getAskSuggestions(jobId)
    } catch {
      dynamicSuggestions.value = [] // sessizce statik onerilere don - panel kirilmaz
    }
  },
  { immediate: true },
)

// Analiz tamamlanınca paneli kendiliğinden bir kez aç — operatör video
// hakkında sormak isteyebilir, ama istediğinde yine kapatabilir.
let autoOpened = false
watch(
  () => props.jobId,
  (jobId) => {
    if (jobId && !autoOpened) {
      autoOpened = true
      isOpen.value = true
    }
  },
)

const messageListEl = ref<HTMLElement | null>(null)
function scrollToBottom() {
  nextTick(() => {
    if (messageListEl.value) messageListEl.value.scrollTop = messageListEl.value.scrollHeight
  })
}

async function submit() {
  if (!canSend.value) return
  const q = question.value.trim()
  question.value = ''
  sendError.value = ''
  sending.value = true

  const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content: q }
  const assistantMsg: ChatMessage = { id: `a-${Date.now()}`, role: 'assistant', content: '', pending: true }
  messages.value = [...messages.value, userMsg, assistantMsg]
  scrollToBottom()

  try {
    let full = ''
    await new Promise<void>((resolve, reject) => {
      askStream.start(
        q,
        props.jobId,
        null, // kalıcı bir sohbet değil — bkz. dosya başı açıklama
        {
          onMeta: (meta) => {
            lastSources.value = meta.sources ?? []
          },
          onDelta: (delta) => {
            full += delta
            assistantMsg.content = full
            scrollToBottom()
          },
          onEnd: () => resolve(),
          onError: (detail) => reject(new Error(detail)),
        },
        !!props.jobId,
      )
    })
    assistantMsg.pending = false
  } catch (e) {
    sendError.value = e instanceof Error ? e.message : 'SAFİR şu anda cevap oluşturamadı.'
    messages.value = messages.value.filter((m) => m.id !== assistantMsg.id)
  } finally {
    sending.value = false
  }
}

function useSuggestion(s: string) {
  question.value = s
  submit()
}

onBeforeUnmount(() => askStream.stop())
</script>

<template>
  <!-- collapsed: a small edge tab, always reachable -->
  <button
    v-if="!isOpen"
    type="button"
    class="fixed right-0 top-1/2 -translate-y-1/2 z-40 flex items-center gap-2 rounded-l-md border border-r-0 border-edge bg-surface-1 px-3 py-2.5 text-sm text-slate-300 hover:text-slate-100 hover:bg-surface-2 shadow-lg"
    aria-label="SAFİR'e sor panelini aç"
    @click="isOpen = true"
  >
    <span aria-hidden="true">◆</span>
    <span class="[writing-mode:vertical-rl] rotate-180">SAFİR'e Sor</span>
  </button>

  <!-- expanded: fixed right-side chat drawer -->
  <div
    v-else
    class="fixed right-0 top-24 bottom-4 z-40 w-full max-w-sm glass-panel rounded-l-lg shadow-2xl flex flex-col"
    role="complementary"
    aria-label="SAFİR'e Sor"
  >
    <div class="flex items-center gap-2 px-4 py-3 border-b border-edge shrink-0">
      <span class="text-accent" aria-hidden="true">◆</span>
      <h3 class="text-sm font-semibold text-slate-100">SAFİR'e Sor</h3>
      <span v-if="jobId" class="ml-auto text-[10px] font-mono text-slate-600">job: {{ jobId.slice(0, 8) }}</span>
      <button
        type="button"
        class="shrink-0 w-7 h-7 rounded-md text-slate-400 hover:text-slate-100 hover:bg-surface-2 flex items-center justify-center"
        aria-label="Paneli kapat"
        @click="isOpen = false"
      >
        ✕
      </button>
    </div>

    <!-- messages -->
    <div v-if="!messages.length" class="flex-1 flex flex-col items-center justify-center text-center px-6">
      <p class="text-sm text-slate-300">Bu video hakkında SAFİR'e sorun</p>
      <p class="mt-1 text-xs text-slate-500">Kritik olaylar, risk gerekçesi veya operatör aksiyonları hakkında sorabilirsiniz.</p>
      <div class="mt-4 flex flex-col gap-2 w-full">
        <button
          v-for="s in suggestions"
          :key="s"
          type="button"
          class="text-left text-xs px-2.5 py-1.5 rounded border border-edge bg-surface-2 text-slate-400 hover:text-slate-200"
          @click="useSuggestion(s)"
        >
          {{ s }}
        </button>
      </div>
    </div>
    <div v-else ref="messageListEl" class="flex-1 overflow-y-auto px-4 py-3 space-y-2.5">
      <div
        v-for="m in messages"
        :key="m.id"
        class="relative rounded-md border-l-2 pl-3 pr-3 py-2 text-sm leading-relaxed whitespace-pre-line break-words"
        :class="m.role === 'user' ? 'border-l-slate-600 bg-surface-2/40' : 'border-l-accent bg-surface-2'"
      >
        <div class="eyebrow mb-0.5" :class="m.role === 'user' ? 'text-slate-500' : 'text-accent'">{{ m.role === 'user' ? 'Siz' : 'SAFİR' }}</div>
        <span v-if="m.pending && !m.content" class="text-slate-500">yanıt oluşturuluyor…</span>
        <span v-else class="text-slate-100">{{ m.content }}</span>
      </div>

      <div v-if="lastSources.length" class="pt-1">
        <button type="button" class="text-[11px] text-slate-500 hover:text-slate-300" @click="showSources = !showSources">
          {{ showSources ? '▾' : '▸' }} Kaynaklar ({{ lastSources.length }})
        </button>
        <ul v-if="showSources" class="mt-1.5 space-y-1">
          <li v-for="(s, i) in lastSources" :key="i" class="text-[11px] bg-surface-2 border border-edge rounded px-2 py-1.5">
            {{ s.label ?? s.text }}
          </li>
        </ul>
      </div>
    </div>

    <p v-if="sendError" class="mx-4 mb-2 text-xs text-risk-crit shrink-0">{{ sendError }}</p>

    <!-- input -->
    <div class="p-3 border-t border-edge shrink-0 flex gap-2">
      <textarea
        v-model="question"
        rows="2"
        class="field-input resize-none flex-1 text-sm"
        placeholder="Video hakkında bir soru sorun…"
        :disabled="sending"
        @keydown.enter.exact.prevent="submit"
      />
      <button type="button" class="btn-primary self-stretch px-4" :disabled="!canSend" @click="submit">
        <span v-if="sending">…</span><span v-else>↑</span>
      </button>
    </div>
  </div>
</template>
