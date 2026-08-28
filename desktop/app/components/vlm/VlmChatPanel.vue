<script setup lang="ts">
// Collapsible right-side chat drawer for asking SAFİR about the video just
// analyzed in VLM Direct Analiz. A lighter, multi-turn sibling of
// workspace/AskSafir.vue (which is single-turn, no running thread) — this
// one keeps a local message history so it reads like a real chat, but
// deliberately does NOT persist to the backend /conversations endpoints
// (unlike the full SAFİR Asistan page): it's a quick contextual side panel
// tied to the current job, not a saved conversation to revisit later.
import type { AskSource } from '~/types/api'

const props = withDefaults(
  defineProps<{ jobId?: string | null }>(),
  { jobId: null }
)

const api = useSafirApi()
const askStream = useAskStream()
const analysisStore = useAnalysisStore()

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

// Video analizi hâlâ CALIŞIYORKEN gönderilen bir soru, backend'e HEMEN
// atılmaz — analiz tamamlanana kadar burada bekletilir, ardından otomatik
// gönderilir (bkz. `submit()`/watch aşağıda). Aynı anda yalnızca TEK bir
// bekleyen soru tutulur.
const queuedQuestion = ref<string | null>(null)
const waitingForAnalysis = computed(() => queuedQuestion.value !== null)

const canSend = computed(() => question.value.trim().length > 0 && !sending.value && !waitingForAnalysis.value)

watch(
  () => analysisStore.isRunning,
  (isRunning, wasRunning) => {
    if (wasRunning && !isRunning && queuedQuestion.value) {
      const q = queuedQuestion.value
      queuedQuestion.value = null
      sendNow(q, { skipUserBubble: true })
    }
  },
)

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

  if (analysisStore.isRunning) {
    // Video hâlâ analiz edilirken /ask'a gitmek hem EVREN üzerinde gereksiz
    // yük oluşturur hem de henüz tamamlanmamış bir analize dayanan, tutarsız
    // bir cevap üretebilir - soru burada bekletilir, analiz bitince (bkz.
    // yukarıdaki `watch`) otomatik gönderilir.
    queuedQuestion.value = q
    messages.value = [
      ...messages.value,
      { id: `u-${Date.now()}`, role: 'user', content: q },
      {
        id: `sys-${Date.now()}`,
        role: 'assistant',
        content: 'Analiz devam ediyor — bu soru, analiz tamamlanınca otomatik olarak gönderilecek.',
      },
    ]
    scrollToBottom()
    return
  }

  await sendNow(q)
}

async function sendNow(q: string, opts: { skipUserBubble?: boolean } = {}) {
  sending.value = true

  const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content: q }
  const assistantMsg: ChatMessage = { id: `a-${Date.now()}`, role: 'assistant', content: '', pending: true }
  messages.value = opts.skipUserBubble ? [...messages.value, assistantMsg] : [...messages.value, userMsg, assistantMsg]
  scrollToBottom()

  try {
    let full = ''
    await new Promise<void>((resolve, reject) => {
      askStream.start(
        q,
        props.jobId ?? null,
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
  <div>
    <!-- collapsed: button with smooth slide transition, beacon ping and glowing border -->
    <Transition
      enter-active-class="transition-transform duration-400 [transition-timing-function:cubic-bezier(0.16,1,0.3,1)]"
      enter-from-class="translate-x-full"
      enter-to-class="translate-x-0"
      leave-active-class="transition-transform duration-250 [transition-timing-function:cubic-bezier(0.4,0,0.2,1)]"
      leave-from-class="translate-x-0"
      leave-to-class="translate-x-full"
    >
      <button
        v-if="!isOpen"
        type="button"
        class="fixed right-0 top-1/2 -translate-y-1/2 z-[100] flex items-center gap-2 rounded-l-xl border border-r-0 glow-tab bg-surface-1/95 backdrop-blur-md px-3 py-3 text-sm text-slate-200 hover:text-white hover:bg-surface-2 transition-colors duration-200 shadow-2xl cursor-pointer group"
        aria-label="SAFİR'e sor panelini aç"
        @click="isOpen = true"
      >
        <span class="relative flex h-2 w-2 items-center justify-center">
          <span class="animate-[ping_4.5s_cubic-bezier(0,0,0.2,1)_infinite] absolute inline-flex h-full w-full rounded-full bg-accent opacity-60" />
          <span class="relative inline-flex rounded-full h-1.5 w-1.5 bg-accent" />
        </span>
        <span class="[writing-mode:vertical-rl] rotate-180 font-semibold tracking-wider select-none text-xs text-slate-200 group-hover:text-accent transition-colors">SAFİR'e Sor</span>
      </button>
    </Transition>

    <!-- expanded: fixed right-side chat drawer with silky smooth sliding animation (no background blur) -->
    <Transition
      enter-active-class="transition-transform duration-500 [transition-timing-function:cubic-bezier(0.16,1,0.3,1)]"
      enter-from-class="translate-x-full"
      enter-to-class="translate-x-0"
      leave-active-class="transition-transform duration-350 [transition-timing-function:cubic-bezier(0.4,0,0.2,1)]"
      leave-from-class="translate-x-0"
      leave-to-class="translate-x-full"
    >
      <div
        v-if="isOpen"
        class="fixed right-0 top-16 bottom-4 z-[100] w-full max-w-sm glass-panel rounded-l-2xl shadow-2xl flex flex-col border border-r-0 border-edge"
        role="complementary"
        aria-label="SAFİR'e Sor"
      >
        <div class="flex items-center gap-2 px-4 py-3 border-b border-edge shrink-0">
          <span class="text-accent" aria-hidden="true">◆</span>
          <h3 class="text-sm font-semibold text-slate-100">SAFİR'e Sor</h3>
          <span v-if="jobId" class="ml-auto text-[10px] font-mono text-slate-600">job: {{ jobId.slice(0, 8) }}</span>
          <button
            type="button"
            class="shrink-0 w-7 h-7 rounded-md text-slate-400 hover:text-slate-100 hover:bg-surface-2 flex items-center justify-center transition-colors"
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
              class="text-left text-xs px-2.5 py-1.5 rounded border border-edge bg-surface-2 text-slate-400 hover:text-slate-200 transition-colors"
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
        <p v-if="waitingForAnalysis" class="mx-4 mb-2 text-xs text-accent shrink-0">Analiz bitince otomatik gönderilecek…</p>

        <!-- input -->
        <div class="p-3 border-t border-edge shrink-0 flex gap-2">
          <textarea
            v-model="question"
            rows="2"
            class="field-input resize-none flex-1 text-sm"
            :placeholder="waitingForAnalysis ? 'Soru analiz bitince gönderilecek…' : 'Video hakkında bir soru sorun…'"
            :disabled="sending || waitingForAnalysis"
            @keydown.enter.exact.prevent="submit"
          />
          <button type="button" class="btn-primary self-stretch px-4" :disabled="!canSend" @click="submit">
            <span v-if="sending || waitingForAnalysis">…</span><span v-else>↑</span>
          </button>
        </div>
      </div>
    </Transition>
  </div>
</template>
