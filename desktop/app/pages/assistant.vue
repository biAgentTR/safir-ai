<script setup lang="ts">
// SAFİR Asistan — merkezi, kalıcı sohbet geçmişine sahip asistan sayfası.
// Workspace içindeki AskSafir.vue (tek-soru, stateless, iş bağlamında hızlı
// soru) DEĞİŞMEDİ ve çalışmaya devam ediyor; bu sayfa onun yanında, ayrı bir
// merkezi deneyimdir (Workspace = analiz bağlamında hızlı soru, SAFİR Asistan
// = kalıcı sohbet geçmişi).
//
// State bilerek page-local (Pinia store DEĞİL): history.vue ile aynı deseni
// izler — bu state tek bir sayfaya özgü, başka bileşenler arasında paylaşılmıyor.
//
// Konuşma geçmişi yalnızca UI/kalıcılık içindir; backend'e (GET /ask/stream)
// yalnızca sınırlı sayıda (bkz. backend `_MAX_HISTORY_MESSAGES`) önceki mesaj
// takip sorularını anlamak için gönderilir — SAFIR kendi önceki cevabını
// kanıt olarak kabul etmez (bkz. backend `ASK_SYSTEM_PROMPT`).
import type { AskSource, AskStreamMeta, Conversation, ConversationContext, ConversationDocument, HistoryListItem } from '~/types/api'

const MAX_NOTE_LEN = 4000
const MAX_LABEL_LEN = 80
const ALLOWED_DOC_EXTENSIONS = ['pdf', 'txt', 'docx']
const MAX_DOC_SIZE_BYTES = 10 * 1024 * 1024

const api = useSafirApi()
const askStream = useAskStream()
const { state: backendHealth } = useBackendHealth()

// ---- sohbet listesi ----
const conversations = ref<Conversation[]>([])
const conversationsLoading = ref(true)
const conversationsError = ref<string | null>(null)

async function loadConversations() {
  conversationsLoading.value = true
  conversationsError.value = null
  try {
    conversations.value = await api.getConversations(100)
  } catch (e) {
    conversationsError.value = humanError(e, 'Sohbetler yüklenemedi.')
  } finally {
    conversationsLoading.value = false
  }
}

// ---- analiz seçici icin History (mevcut veri yeniden kullanilir, ikinci kez saklanmaz) ----
const historyItems = ref<HistoryListItem[]>([])
async function loadHistoryForSelector() {
  try {
    historyItems.value = await api.getHistory(100)
  } catch {
    historyItems.value = [] // seçici boş kalır; sayfa kırılmaz
  }
}

function historyLabel(jobId: string | null): string | null {
  if (!jobId) return null
  const item = historyItems.value.find((h) => h.job_id === jobId)
  if (!item) return jobId.slice(0, 8)
  const base = item.video_source ? item.video_source.split(/[\\/]/).pop() : null
  return base || jobId.slice(0, 8)
}

function conversationTag(c: Conversation): string {
  return historyLabel(c.job_id) ?? 'Genel SAFİR'
}

const activeConversation = computed(() =>
  conversations.value.find((c) => c.conversation_id === activeConversationId.value) ?? null,
)

const contextDisplay = computed(() => {
  const name = activeJobId.value ? historyLabel(activeJobId.value) : 'Genel SAFİR'
  const created = activeConversation.value?.created_at
  return created ? `${name} — ${fmtDate(created)}` : name
})

// ---- aktif sohbet ----
interface ChatMessage {
  id: string | number
  role: 'user' | 'assistant'
  content: string
  pending?: boolean
}

const activeConversationId = ref<string | null>(null)
const activeJobId = ref<string | null>(null) // aktif/az önce oluşturulan sohbetin bağlı olduğu analiz
const messages = ref<ChatMessage[]>([])
const messagesLoading = ref(false)
const messagesError = ref<string | null>(null)
const lastSources = ref<AskSource[]>([])
const showSources = ref(false)

// ---- ek kullanıcı bağlamı (Adım 3: yalnızca metin/not; dosya Adım 4'te) ----
// SAFIR'e ASLA sistem talimatı veya analiz kanıtı olarak sunulmaz — yalnızca
// yardımcı bilgi (bkz. backend AskService._prepare / ASK_SYSTEM_PROMPT).
const contextNotes = ref<ConversationContext[]>([])
const showNoteForm = ref(false)
const noteLabel = ref('')
const noteContent = ref('')
const noteSaving = ref(false)
const noteError = ref<string | null>(null)
const removingContextId = ref<number | null>(null)

function openNoteForm() {
  showNoteForm.value = true
  noteLabel.value = ''
  noteContent.value = ''
  noteError.value = null
}
function cancelNoteForm() {
  showNoteForm.value = false
  noteError.value = null
}

// ---- ek kullanıcı bağlamı: yüklenen belgeler (PDF/TXT/DOCX) ----
const documents = ref<ConversationDocument[]>([])
const fileInputEl = ref<HTMLInputElement | null>(null)
const uploading = ref(false)
const uploadError = ref<string | null>(null)
const removingDocumentId = ref<string | null>(null)

function openFilePicker() {
  uploadError.value = null
  fileInputEl.value?.click()
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

async function pollDocumentUntilDone(conversationId: string, documentId: string) {
  const deadline = Date.now() + 30_000
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 800))
    if (activeConversationId.value !== conversationId) return // kullanıcı başka sohbete geçti
    try {
      const detail = await api.getConversation(conversationId)
      const doc = detail.documents.find((d) => d.document_id === documentId)
      if (!doc) return
      const idx = documents.value.findIndex((d) => d.document_id === documentId)
      if (idx !== -1) documents.value[idx] = doc
      if (doc.status !== 'processing') return
    } catch {
      return // sessizce dur — kullanıcı sayfayı yenileyip son durumu görebilir
    }
  }
}

async function onFileSelected(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = '' // aynı dosya art arda seçilebilsin
  if (!file) return

  const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
  if (!ALLOWED_DOC_EXTENSIONS.includes(ext)) {
    uploadError.value = `Desteklenmeyen dosya türü: .${ext || '?'} (yalnızca PDF/TXT/DOCX)`
    return
  }
  if (file.size > MAX_DOC_SIZE_BYTES) {
    uploadError.value = `Dosya çok büyük (azami ${MAX_DOC_SIZE_BYTES / (1024 * 1024)} MB).`
    return
  }

  uploading.value = true
  uploadError.value = null
  try {
    const conversationId = await ensureConversation()
    const created = await api.uploadConversationDocument(conversationId, file)
    documents.value = [...documents.value, created]
    if (created.status === 'processing') {
      void pollDocumentUntilDone(conversationId, created.document_id)
    }
  } catch (e) {
    uploadError.value = humanError(e, 'Dosya yüklenemedi.')
  } finally {
    uploading.value = false
  }
}

async function removeDocument(documentId: string) {
  if (!activeConversationId.value) return
  removingDocumentId.value = documentId
  try {
    await api.removeConversationDocument(activeConversationId.value, documentId)
    documents.value = documents.value.filter((d) => d.document_id !== documentId)
  } catch (e) {
    uploadError.value = humanError(e, 'Belge kaldırılamadı.')
  } finally {
    removingDocumentId.value = null
  }
}

// ---- otomatik kaydırma: yeni içerik geldiğinde en alta in, ama kullanıcı
// yukarı kaydırdıysa (eski mesajları okuyorsa) araya girme ----
const messageListEl = ref<HTMLElement | null>(null)
const stickToBottom = ref(true)

function onMessageListScroll() {
  const el = messageListEl.value
  if (!el) return
  stickToBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight < 80
}

function scrollToBottom() {
  nextTick(() => {
    const el = messageListEl.value
    if (el && stickToBottom.value) el.scrollTop = el.scrollHeight
  })
}

const selectedJobId = ref<string | null>(null) // yalnızca YENİ (henüz oluşturulmamış) sohbet kurulurken kullanılır

function startNewChat() {
  askStream.stop()
  activeConversationId.value = null
  activeJobId.value = null
  selectedJobId.value = null
  messages.value = []
  messagesError.value = null
  sendError.value = null
  lastSources.value = []
  question.value = ''
  contextNotes.value = []
  showNoteForm.value = false
  documents.value = []
  uploadError.value = null
}

async function openConversation(id: string) {
  if (sending.value) return
  askStream.stop()
  activeConversationId.value = id
  messagesLoading.value = true
  messagesError.value = null
  sendError.value = null
  lastSources.value = []
  showNoteForm.value = false
  uploadError.value = null
  try {
    const detail = await api.getConversation(id)
    activeJobId.value = detail.job_id
    messages.value = detail.messages.map((m) => ({ id: m.id, role: m.role, content: m.content }))
    contextNotes.value = detail.context ?? []
    documents.value = detail.documents ?? []
    // Sayfa yenilenmeden/yeniden açılmadan önce hâlâ işlenmekte olan bir belge
    // varsa (ör. yükleme sonrası hemen sayfa değiştirildiyse) izlemeye devam et.
    for (const doc of documents.value) {
      if (doc.status === 'processing') void pollDocumentUntilDone(id, doc.document_id)
    }
    stickToBottom.value = true
    scrollToBottom()
  } catch (e) {
    messagesError.value = humanError(e, 'Sohbet yüklenemedi.')
    messages.value = []
  } finally {
    messagesLoading.value = false
  }
}

/**
 * İlk mesajda VEYA ilk not eklendiğinde sohbeti "tembel" (lazy) oluşturur —
 * boş kayıt birikmez. `titleHint` verilirse (yalnızca ilk sorudan) başlık
 * olarak kullanılır; not-önce akışında (henüz soru yokken) başlıksız oluşur.
 */
async function ensureConversation(titleHint?: string): Promise<string> {
  if (activeConversationId.value) return activeConversationId.value
  const title = titleHint ? truncateTitle(titleHint) : null
  const created = await api.createConversation({ title, job_id: selectedJobId.value })
  activeConversationId.value = created.conversation_id
  activeJobId.value = created.job_id
  void loadConversations()
  return created.conversation_id
}

async function addNote() {
  const content = noteContent.value.trim()
  if (!content) return
  noteSaving.value = true
  noteError.value = null
  try {
    const conversationId = await ensureConversation()
    const label = noteLabel.value.trim() || null
    const created = await api.addConversationContext(conversationId, { content, label })
    contextNotes.value = [...contextNotes.value, created]
    showNoteForm.value = false
  } catch (e) {
    noteError.value = humanError(e, 'Not eklenemedi.')
  } finally {
    noteSaving.value = false
  }
}

async function removeNote(contextId: number) {
  if (!activeConversationId.value) return
  removingContextId.value = contextId
  try {
    await api.removeConversationContext(activeConversationId.value, contextId)
    contextNotes.value = contextNotes.value.filter((c) => c.id !== contextId)
  } catch (e) {
    noteError.value = humanError(e, 'Not kaldırılamadı.')
  } finally {
    removingContextId.value = null
  }
}

function truncateTitle(text: string): string {
  let t = text.trim().replace(/\?+$/, '').trim()
  const MAX = 48
  if (t.length <= MAX) return t
  t = t.slice(0, MAX)
  const lastSpace = t.lastIndexOf(' ')
  if (lastSpace > 20) t = t.slice(0, lastSpace)
  return t + '…'
}

const SUGGESTIONS = [
  'SAFİR nasıl çalışır?',
  'Bu analizdeki en kritik olay hangisi?',
  'Hangi İSG mevzuatı ilgili?',
]

function useSuggestion(s: string) {
  question.value = s
  submit()
}

// ---- soru gönderme (streaming) ----
const question = ref('')
const sending = ref(false)
const sendError = ref<string | null>(null)

const canSend = computed(() => question.value.trim().length > 0 && !sending.value)
const textareaRows = computed(() => Math.min(6, Math.max(2, question.value.split('\n').length)))

async function submit() {
  if (!canSend.value) return
  const q = question.value.trim()
  question.value = ''
  sendError.value = null
  sending.value = true

  const userMsg: ChatMessage = { id: `local-u-${Date.now()}`, role: 'user', content: q }
  const assistantMsg: ChatMessage = { id: `local-a-${Date.now()}`, role: 'assistant', content: '', pending: true }
  messages.value = [...messages.value, userMsg, assistantMsg]
  stickToBottom.value = true
  scrollToBottom()

  try {
    // İlk soruda sohbeti "tembel" (lazy) oluştur (veya bir not eklenerek zaten
    // oluşturulmuşsa onu kullan) — boş "Yeni Sohbet" kayıtları birikmez.
    const conversationId = await ensureConversation(q)

    let full = ''
    await new Promise<void>((resolve, reject) => {
      askStream.start(q, activeJobId.value, conversationId, {
        onMeta: (meta: AskStreamMeta) => {
          lastSources.value = meta.sources ?? []
        },
        onDelta: (delta: string) => {
          full += delta
          assistantMsg.content = full
          scrollToBottom()
        },
        onEnd: () => resolve(),
        onError: (detail: string) => reject(new Error(detail)),
      })
    })

    assistantMsg.pending = false
    // Yalnızca TAMAMLANMIŞ çift kaydedilir — akış sırasında hiçbir parça tek tek yazılmaz.
    await api.addConversationMessage(conversationId, 'user', q)
    const saved = await api.addConversationMessage(conversationId, 'assistant', full)
    assistantMsg.id = saved.id
    void loadConversations() // liste sırası/başlık tazelensin
  } catch (e) {
    assistantMsg.pending = false
    sendError.value = e instanceof Error ? e.message : 'SAFİR şu anda cevap oluşturamadı.'
    // Boş/başarısız asistan balonunu kaldır — kullanıcı sorusu ekranda kalır, tekrar denenebilir.
    messages.value = messages.value.filter((m) => m.id !== assistantMsg.id)
  } finally {
    sending.value = false
  }
}

function humanError(e: unknown, fallback: string): string {
  return (
    (e as { data?: { detail?: string } })?.data?.detail ??
    (e as Error)?.message ??
    fallback
  )
}

function fmtDate(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('tr-TR', { dateStyle: 'medium', timeStyle: 'short' })
}

onMounted(() => {
  loadConversations()
  loadHistoryForSelector()
})
onBeforeUnmount(() => askStream.stop())
</script>

<template>
  <div class="max-w-6xl mx-auto px-6 py-8">
    <div class="mb-5">
      <h2 class="text-xl font-semibold text-slate-100">SAFİR Asistan</h2>
      <p class="mt-1 text-sm text-slate-500">Analiz sonuçları ve ilgili mevzuat hakkında SAFİR'e soru sorun.</p>
    </div>

    <div v-if="backendHealth === 'offline'" class="mb-4 rounded-md border border-risk-crit/40 bg-risk-crit/10 px-4 py-2.5 text-sm text-risk-crit">
      Arka uca ulaşılamıyor. SAFİR Asistan şu anda kullanılamayabilir.
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-[280px_minmax(0,1fr)] gap-5 items-start">
      <!-- sohbet listesi -->
      <div class="card p-3">
        <button type="button" class="btn-primary w-full mb-3" @click="startNewChat">
          <span>＋</span> Yeni Sohbet
        </button>

        <div class="px-1 mb-1 text-[11px] uppercase tracking-wide text-slate-500">Sohbetler</div>

        <div v-if="conversationsLoading && !conversations.length" class="text-sm text-slate-500 text-center py-6">
          Yükleniyor…
        </div>
        <div v-else-if="conversationsError && !conversations.length" class="text-sm text-risk-crit px-1">
          {{ conversationsError }}
        </div>
        <div v-else-if="!conversations.length" class="text-sm text-slate-500 text-center py-6">
          Henüz sohbet yok.
        </div>
        <div v-else class="space-y-1 max-h-[65vh] overflow-y-auto">
          <button
            v-for="c in conversations"
            :key="c.conversation_id"
            type="button"
            class="w-full text-left rounded-md px-3 py-2 transition-colors"
            :class="activeConversationId === c.conversation_id ? 'bg-surface-2 ring-1 ring-accent/40' : 'hover:bg-surface-2/60'"
            @click="openConversation(c.conversation_id)"
          >
            <div class="text-sm text-slate-100 truncate">{{ c.title || 'Yeni sohbet' }}</div>
            <div class="mt-0.5 flex items-center gap-2 text-[11px] text-slate-500">
              <span class="truncate">{{ conversationTag(c) }}</span>
              <span class="shrink-0">·</span>
              <span class="shrink-0">{{ fmtDate(c.updated_at) }}</span>
            </div>
          </button>
        </div>
      </div>

      <!-- aktif sohbet -->
      <div class="card p-5 flex flex-col min-h-[65vh]">
        <!-- bağlam göstergesi -->
        <div class="flex items-center gap-2 pb-3 mb-3 border-b border-edge text-sm">
          <span class="text-slate-500">Bağlam:</span>
          <template v-if="activeConversationId">
            <span class="text-slate-200">{{ contextDisplay }}</span>
            <span class="text-[11px] text-slate-600">(bu sohbet için sabit)</span>
          </template>
          <template v-else>
            <select v-model="selectedJobId" class="field-input !w-auto !py-1 text-sm">
              <option :value="null">Genel SAFİR</option>
              <option v-for="h in historyItems" :key="h.job_id" :value="h.job_id">
                {{ h.video_source ? h.video_source.split(/[\\/]/).pop() : h.job_id.slice(0, 8) }}
              </option>
            </select>
            <span class="text-[11px] text-slate-600">İlk mesajla birlikte kesinleşir</span>
          </template>
        </div>

        <!-- ek kullanıcı bağlamı: operatörün kendi eklediği notlar/belgeler (analiz kanıtı DEĞİLDİR) -->
        <div class="pb-3 mb-3 border-b border-edge">
          <div class="flex items-center justify-between">
            <span class="text-xs text-slate-500">Ek Bağlam</span>
            <div class="flex gap-2">
              <button
                v-if="!showNoteForm"
                type="button"
                class="text-[11px] px-2 py-1 rounded border border-edge bg-surface-2 text-slate-400 hover:text-slate-200"
                @click="openNoteForm"
              >＋ Not ekle</button>
              <button
                type="button"
                class="text-[11px] px-2 py-1 rounded border border-edge bg-surface-2 text-slate-400 hover:text-slate-200 disabled:opacity-50"
                :disabled="uploading"
                @click="openFilePicker"
              >{{ uploading ? 'Yükleniyor…' : '＋ Dosya ekle' }}</button>
              <input
                ref="fileInputEl"
                type="file"
                accept=".pdf,.txt,.docx"
                class="hidden"
                @change="onFileSelected"
              />
            </div>
          </div>
          <p v-if="uploadError" class="mt-1.5 text-xs text-risk-crit">{{ uploadError }}</p>

          <!-- not ekleme formu -->
          <div v-if="showNoteForm" class="mt-2 space-y-2">
            <input
              v-model="noteLabel"
              :maxlength="MAX_LABEL_LEN"
              class="field-input text-sm"
              placeholder="Kısa etiket (opsiyonel, ör. Tesis bilgisi)"
            />
            <textarea
              v-model="noteContent"
              :maxlength="MAX_NOTE_LEN"
              rows="3"
              class="field-input text-sm resize-none"
              placeholder="Örn: Bu tesiste CO2 bazlı yangın söndürme sistemi kullanılmaktadır."
            />
            <div class="flex items-center justify-between text-[11px] text-slate-600">
              <span>{{ noteContent.length }} / {{ MAX_NOTE_LEN }} karakter</span>
              <div class="flex gap-2">
                <button type="button" class="btn-ghost !px-2 !py-1 text-xs" @click="cancelNoteForm">Vazgeç</button>
                <button
                  type="button"
                  class="btn-primary !px-3 !py-1 text-xs"
                  :disabled="!noteContent.trim() || noteSaving"
                  @click="addNote"
                >{{ noteSaving ? 'Ekleniyor…' : 'Ekle' }}</button>
              </div>
            </div>
            <p v-if="noteError" class="text-xs text-risk-crit">{{ noteError }}</p>
          </div>

          <!-- eklenen notlar -->
          <div v-if="contextNotes.length" class="mt-2 space-y-1.5">
            <div
              v-for="note in contextNotes"
              :key="note.id"
              class="flex items-start gap-2 bg-surface-2 border border-edge rounded-md px-2.5 py-2 text-xs"
            >
              <span class="shrink-0 text-slate-500">📝</span>
              <div class="min-w-0 flex-1">
                <div v-if="note.label" class="text-slate-300 font-medium">{{ note.label }}</div>
                <div class="text-slate-400 whitespace-pre-line break-words">{{ note.content }}</div>
              </div>
              <button
                type="button"
                class="shrink-0 text-slate-500 hover:text-risk-crit disabled:opacity-50"
                :disabled="removingContextId === note.id"
                @click="removeNote(note.id)"
              >{{ removingContextId === note.id ? '…' : 'Kaldır' }}</button>
            </div>
          </div>

          <!-- yüklenen belgeler -->
          <div v-if="documents.length" class="mt-2 space-y-1.5">
            <div
              v-for="doc in documents"
              :key="doc.document_id"
              class="flex items-start gap-2 bg-surface-2 border border-edge rounded-md px-2.5 py-2 text-xs"
            >
              <span class="shrink-0 text-slate-500">📄</span>
              <div class="min-w-0 flex-1">
                <div class="text-slate-300 font-medium truncate">{{ doc.filename }}</div>
                <div class="text-slate-500">
                  {{ formatFileSize(doc.file_size_bytes) }}<span v-if="doc.page_count"> · {{ doc.page_count }} sayfa</span>
                </div>
                <div v-if="doc.status === 'processing'" class="mt-0.5 text-slate-400">Metin çıkarılıyor…</div>
                <div v-else-if="doc.status === 'ready'" class="mt-0.5 text-risk-low">✓ Bağlama dahil</div>
                <div v-else class="mt-0.5 text-risk-crit">Dosya işlenemedi{{ doc.error_message ? `: ${doc.error_message}` : '' }}</div>
              </div>
              <button
                type="button"
                class="shrink-0 text-slate-500 hover:text-risk-crit disabled:opacity-50"
                :disabled="removingDocumentId === doc.document_id"
                @click="removeDocument(doc.document_id)"
              >{{ removingDocumentId === doc.document_id ? '…' : 'Kaldır' }}</button>
            </div>
          </div>

          <p v-if="!showNoteForm && !contextNotes.length && !documents.length" class="mt-1.5 text-[11px] text-slate-600">
            Henüz ek bağlam eklenmedi.
          </p>
        </div>

        <!-- mesaj gecmisi -->
        <div v-if="messagesLoading" class="flex-1 flex items-center justify-center text-sm text-slate-500">
          Sohbet yükleniyor…
        </div>
        <div v-else-if="messagesError" class="flex-1 flex items-center justify-center text-sm text-risk-crit text-center px-6">
          {{ messagesError }}
        </div>
        <div v-else-if="!messages.length" class="flex-1 flex flex-col items-center justify-center text-center px-6">
          <p class="text-sm text-slate-300">SAFİR Asistan'a bir soru sorun</p>
          <p class="mt-1 text-xs text-slate-500 max-w-sm">
            Seçili analiz hakkında veya genel İSG/SAFİR konularında soru sorabilirsiniz.
          </p>
          <div class="mt-4 flex flex-wrap justify-center gap-2">
            <button
              v-for="s in SUGGESTIONS"
              :key="s"
              type="button"
              class="text-[11px] px-2 py-1 rounded border border-edge bg-surface-2 text-slate-400 hover:text-slate-200"
              @click="useSuggestion(s)"
            >{{ s }}</button>
          </div>
        </div>
        <div v-else ref="messageListEl" class="flex-1 overflow-y-auto space-y-3 pr-1" @scroll="onMessageListScroll">
          <div v-for="m in messages" :key="m.id" class="flex" :class="m.role === 'user' ? 'justify-end' : 'justify-start'">
            <div
              class="max-w-[75%] rounded-md border px-3 py-2 text-sm leading-relaxed whitespace-pre-line break-words"
              :class="m.role === 'user'
                ? 'bg-accent/10 border-accent/30 text-slate-100'
                : 'bg-surface-2 border-edge text-slate-100'"
            >
              <span v-if="m.pending && !m.content" class="text-slate-500">SAFİR yanıtı oluşturuyor…</span>
              <span v-else>{{ m.content }}</span>
            </div>
          </div>
        </div>

        <!-- kaynaklar (son cevaba ait, varsa) -->
        <div v-if="lastSources.length" class="pt-3">
          <button type="button" class="text-xs text-slate-400 hover:text-slate-200" @click="showSources = !showSources">
            {{ showSources ? '▾' : '▸' }} Kaynaklar ({{ lastSources.length }})
          </button>
          <ul v-if="showSources" class="mt-2 space-y-1.5">
            <li
              v-for="(s, i) in lastSources"
              :key="i"
              class="text-xs bg-surface-2 border border-edge rounded-md px-3 py-2 flex gap-2"
            >
              <span
                class="shrink-0 px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wide"
                :class="s.type === 'analysis' ? 'bg-accent-soft text-accent' : s.type === 'document' ? 'bg-surface-3 text-slate-300' : 'bg-surface-3 text-slate-400'"
              >{{ s.type === 'analysis' ? 'ANALİZ' : s.type === 'document' ? 'BELGE' : 'MEVZUAT' }}</span>
              <span class="text-slate-300 min-w-0">
                {{ s.label ?? s.text }}
                <span v-if="s.score != null" class="text-slate-600"> · skor {{ s.score }}</span>
              </span>
            </li>
          </ul>
        </div>

        <!-- hata -->
        <div v-if="sendError" class="mt-3 rounded-md border border-risk-crit/30 bg-risk-crit/10 px-3 py-2 text-sm text-risk-crit">
          {{ sendError }}
        </div>

        <!-- giris -->
        <div class="mt-4 flex gap-2">
          <textarea
            v-model="question"
            :rows="textareaRows"
            class="field-input resize-none flex-1"
            placeholder="SAFİR'e bir soru sorun…"
            :disabled="sending"
            @keydown.enter.exact.prevent="submit"
          />
          <button type="button" class="btn-primary self-stretch px-5" :disabled="!canSend" @click="submit">
            <span v-if="sending">…</span><span v-else>Gönder</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
