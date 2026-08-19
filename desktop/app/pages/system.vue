<script setup lang="ts">
// Sistem Verileri (Data Center) — operatore, SAFİR'in SQLite/disk depolarinda
// GERCEKTEN neyin saklandigini gosteren, TAMAMEN read-only bir sayfa. SQL
// calistirma/DELETE/UPDATE YOKTUR — yalnizca mevcut GET /system/overview,
// GET /history, GET /history/{id}, GET /conversations, GET /conversations/{id}
// uc noktalarini okur. Yeni bir DB soyutlamasi/tablosu yoktur.
import type { Conversation, ConversationDetail, HistoryDetail, HistoryListItem, TraceEvent } from '~/types/api'

const api = useSafirApi()

type Tab = 'analizler' | 'konusmalar' | 'pipeline' | 'kanitlar'
const tab = ref<Tab>('analizler')
const TABS: { key: Tab; label: string }[] = [
  { key: 'analizler', label: 'Analizler' },
  { key: 'konusmalar', label: 'Konuşmalar' },
  { key: 'pipeline', label: 'Pipeline / Trace' },
  { key: 'kanitlar', label: 'Depolanan Kanıtlar' },
]

// ---- KPI özeti ----
const overview = ref<Awaited<ReturnType<typeof api.getSystemOverview>> | null>(null)
const overviewLoading = ref(true)
const overviewError = ref<string | null>(null)

async function loadOverview() {
  overviewLoading.value = true
  overviewError.value = null
  try {
    overview.value = await api.getSystemOverview()
  } catch (e) {
    overviewError.value = humanError(e, 'Sistem özeti yüklenemedi.')
  } finally {
    overviewLoading.value = false
  }
}

// ---- Analizler ----
const analyses = ref<HistoryListItem[]>([])
const analysesLoading = ref(true)
const analysesError = ref<string | null>(null)
const expandedJobId = ref<string | null>(null)

async function loadAnalyses() {
  analysesLoading.value = true
  analysesError.value = null
  try {
    analyses.value = await api.getHistory(100)
  } catch (e) {
    analysesError.value = humanError(e, 'Analizler yüklenemedi.')
  } finally {
    analysesLoading.value = false
  }
}

function toggleExpand(jobId: string) {
  expandedJobId.value = expandedJobId.value === jobId ? null : jobId
}

function durationLabel(createdAt: string, updatedAt: string): string {
  const start = new Date(createdAt).getTime()
  const end = new Date(updatedAt).getTime()
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) return '—'
  const sec = Math.round((end - start) / 1000)
  if (sec < 60) return `${sec}s`
  return `${Math.floor(sec / 60)}dk ${sec % 60}s`
}

// ---- Konuşmalar ----
const conversations = ref<Conversation[]>([])
const conversationsLoading = ref(true)
const conversationsError = ref<string | null>(null)
const expandedConversationId = ref<string | null>(null)
const conversationDetail = ref<ConversationDetail | null>(null)
const conversationDetailLoading = ref(false)
const conversationDetailError = ref<string | null>(null)

async function loadConversations() {
  conversationsLoading.value = true
  conversationsError.value = null
  try {
    conversations.value = await api.getConversations(100)
  } catch (e) {
    conversationsError.value = humanError(e, 'Konuşmalar yüklenemedi.')
  } finally {
    conversationsLoading.value = false
  }
}

async function toggleConversation(id: string) {
  if (expandedConversationId.value === id) {
    expandedConversationId.value = null
    return
  }
  expandedConversationId.value = id
  conversationDetail.value = null
  conversationDetailError.value = null
  conversationDetailLoading.value = true
  try {
    conversationDetail.value = await api.getConversation(id)
  } catch (e) {
    conversationDetailError.value = humanError(e, 'Sohbet detayı yüklenemedi.')
  } finally {
    conversationDetailLoading.value = false
  }
}

// ---- Pipeline/Trace + Depolanan Kanıtlar: ortak analiz seçici ----
const selectedJobId = ref<string | null>(null)
const selectedDetail = ref<HistoryDetail | null>(null)
const selectedLoading = ref(false)
const selectedError = ref<string | null>(null)

async function selectAnalysis(jobId: string | null) {
  selectedJobId.value = jobId
  selectedDetail.value = null
  selectedError.value = null
  if (!jobId) return
  selectedLoading.value = true
  try {
    selectedDetail.value = await api.getHistoryItem(jobId)
  } catch (e) {
    selectedError.value = humanError(e, 'Analiz detayı yüklenemedi.')
  } finally {
    selectedLoading.value = false
  }
}

const traceEvents = computed<TraceEvent[]>(() => selectedDetail.value?.trace_events ?? [])

interface StoredFrameRow {
  event_id: number
  selection_reason: string
  timestamp_str: string
  frame_id: string
}
const storedFrames = computed<StoredFrameRow[]>(() => {
  const samplerEvent = traceEvents.value.find((e) => e.stage === 'sampler')
  if (!samplerEvent) return []
  const data = samplerEvent.data as {
    event_groups?: { event_id: number; representative_frames: { selection_reason: string; timestamp_str: string; frame_id: string }[] }[]
  }
  const rows: StoredFrameRow[] = []
  for (const eg of data.event_groups ?? []) {
    for (const rf of eg.representative_frames ?? []) {
      rows.push({ event_id: eg.event_id, selection_reason: rf.selection_reason, timestamp_str: rf.timestamp_str, frame_id: rf.frame_id })
    }
  }
  return rows
})

function humanError(e: unknown, fallback: string): string {
  return (e as { data?: { detail?: string } })?.data?.detail ?? (e as Error)?.message ?? fallback
}
function fmtDate(iso: string | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('tr-TR', { dateStyle: 'medium', timeStyle: 'short' })
}
function basename(src: string | null | undefined): string {
  if (!src) return '—'
  return src.split(/[\\/]/).pop() || src
}

onMounted(() => {
  loadOverview()
  loadAnalyses()
  loadConversations()
})
</script>

<template>
  <div class="max-w-6xl mx-auto px-6 py-8">
    <div class="mb-5">
      <h2 class="text-xl font-semibold text-slate-100">Sistem Verileri</h2>
      <p class="mt-1 text-sm text-slate-500">
        SAFİR'in kalıcı olarak sakladığı gerçek veriler — salt okunur. SQL çalıştırma veya veri değiştirme imkanı yoktur.
      </p>
    </div>

    <!-- KPI özeti -->
    <div v-if="overviewLoading" class="card p-6 text-center text-sm text-slate-500 mb-5">Özet yükleniyor…</div>
    <div v-else-if="overviewError" class="card p-4 border-risk-crit/30 text-sm text-risk-crit mb-5">
      {{ overviewError }}
      <button class="btn-ghost ml-2" @click="loadOverview">Tekrar dene</button>
    </div>
    <div v-else-if="overview" class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3 mb-6">
      <MetricCell label="Toplam analiz" :value="overview.totals.total_analyses" />
      <MetricCell label="Tamamlanan" :value="overview.totals.completed_analyses" />
      <MetricCell label="Başarısız" :value="overview.totals.failed_analyses" />
      <MetricCell label="Toplam konuşma" :value="overview.totals.total_conversations" />
      <MetricCell label="Saklanan pipeline izi" :value="overview.totals.analyses_with_trace" />
      <MetricCell label="Saklanan kanıt karesi" :value="overview.totals.stored_representative_frame_count" />
    </div>

    <!-- tabs -->
    <div class="card">
      <div class="flex items-center gap-1 border-b border-edge px-3 overflow-x-auto">
        <button
          v-for="t in TABS"
          :key="t.key"
          class="px-4 py-2.5 text-sm border-b-2 -mb-px transition-colors whitespace-nowrap"
          :class="tab === t.key ? 'border-accent text-slate-100' : 'border-transparent text-slate-400 hover:text-slate-200'"
          @click="tab = t.key"
        >{{ t.label }}</button>
      </div>

      <div class="p-5">
        <!-- Analizler -->
        <div v-if="tab === 'analizler'">
          <div v-if="analysesLoading" class="text-sm text-slate-500 text-center py-8">Yükleniyor…</div>
          <div v-else-if="analysesError" class="text-sm text-risk-crit text-center py-8">
            {{ analysesError }}
            <button class="btn-ghost ml-2" @click="loadAnalyses">Tekrar dene</button>
          </div>
          <div v-else-if="!analyses.length" class="text-sm text-slate-500 text-center py-8">Henüz kayıtlı analiz yok.</div>
          <div v-else class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] uppercase tracking-wide text-slate-500 border-b border-edge">
                  <th class="py-2 pr-3">Job ID</th>
                  <th class="py-2 pr-3">Tarih</th>
                  <th class="py-2 pr-3">Video</th>
                  <th class="py-2 pr-3">Durum</th>
                  <th class="py-2 pr-3">Risk</th>
                  <th class="py-2 pr-3">Süre</th>
                </tr>
              </thead>
              <tbody>
                <template v-for="a in analyses" :key="a.job_id">
                  <tr
                    class="border-b border-edge/60 hover:bg-surface-2/60 cursor-pointer"
                    @click="toggleExpand(a.job_id)"
                  >
                    <td class="py-2 pr-3 font-mono text-xs text-slate-300">{{ a.job_id.slice(0, 8) }}</td>
                    <td class="py-2 pr-3 text-slate-400">{{ fmtDate(a.created_at) }}</td>
                    <td class="py-2 pr-3 text-slate-300 max-w-[220px] truncate">{{ basename(a.video_source) }}</td>
                    <td class="py-2 pr-3 text-slate-400">{{ a.status }}</td>
                    <td class="py-2 pr-3">
                      <span v-if="a.risk_score != null" :class="RISK_TEXT[riskTone(a.risk_level)]">{{ a.risk_score }} · {{ a.risk_level }}</span>
                      <span v-else class="text-slate-600">—</span>
                    </td>
                    <td class="py-2 pr-3 text-slate-400">{{ durationLabel(a.created_at, a.updated_at) }}</td>
                  </tr>
                  <tr v-if="expandedJobId === a.job_id" class="border-b border-edge/60 bg-surface-2/40">
                    <td colspan="6" class="py-3 px-3 text-xs text-slate-400 space-y-1">
                      <div><span class="text-slate-500">Tam job_id:</span> <span class="font-mono">{{ a.job_id }}</span></div>
                      <div><span class="text-slate-500">Video kaynağı:</span> {{ a.video_source ?? '—' }}</div>
                      <div><span class="text-slate-500">Oluşturulma:</span> {{ fmtDate(a.created_at) }} · <span class="text-slate-500">Son güncelleme:</span> {{ fmtDate(a.updated_at) }}</div>
                      <div><span class="text-slate-500">Özet:</span> {{ a.summary || '—' }}</div>
                      <div class="pt-1">
                        <NuxtLink :to="`/reports/${a.job_id}`" class="text-accent hover:underline">Raporu aç →</NuxtLink>
                      </div>
                    </td>
                  </tr>
                </template>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Konuşmalar -->
        <div v-else-if="tab === 'konusmalar'">
          <div v-if="conversationsLoading" class="text-sm text-slate-500 text-center py-8">Yükleniyor…</div>
          <div v-else-if="conversationsError" class="text-sm text-risk-crit text-center py-8">
            {{ conversationsError }}
            <button class="btn-ghost ml-2" @click="loadConversations">Tekrar dene</button>
          </div>
          <div v-else-if="!conversations.length" class="text-sm text-slate-500 text-center py-8">Henüz kayıtlı sohbet yok.</div>
          <div v-else class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] uppercase tracking-wide text-slate-500 border-b border-edge">
                  <th class="py-2 pr-3">Sohbet ID</th>
                  <th class="py-2 pr-3">Başlık</th>
                  <th class="py-2 pr-3">Oluşturulma</th>
                  <th class="py-2 pr-3">Son güncelleme</th>
                  <th class="py-2 pr-3">Mesaj sayısı</th>
                </tr>
              </thead>
              <tbody>
                <template v-for="c in conversations" :key="c.conversation_id">
                  <tr
                    class="border-b border-edge/60 hover:bg-surface-2/60 cursor-pointer"
                    @click="toggleConversation(c.conversation_id)"
                  >
                    <td class="py-2 pr-3 font-mono text-xs text-slate-300">{{ c.conversation_id.slice(0, 8) }}</td>
                    <td class="py-2 pr-3 text-slate-200 max-w-[220px] truncate">{{ c.title || '(başlıksız)' }}</td>
                    <td class="py-2 pr-3 text-slate-400">{{ fmtDate(c.created_at) }}</td>
                    <td class="py-2 pr-3 text-slate-400">{{ fmtDate(c.updated_at) }}</td>
                    <td class="py-2 pr-3 text-slate-300">{{ c.message_count }}</td>
                  </tr>
                  <tr v-if="expandedConversationId === c.conversation_id" class="border-b border-edge/60 bg-surface-2/40">
                    <td colspan="5" class="py-3 px-3 text-xs text-slate-400">
                      <div class="mb-2">
                        <span class="text-slate-500">Tam sohbet ID:</span> <span class="font-mono">{{ c.conversation_id }}</span>
                        <span v-if="c.job_id"> · <span class="text-slate-500">bağlı analiz:</span> <span class="font-mono">{{ c.job_id.slice(0, 8) }}</span></span>
                      </div>
                      <div v-if="conversationDetailLoading">Mesajlar yükleniyor…</div>
                      <div v-else-if="conversationDetailError" class="text-risk-crit">{{ conversationDetailError }}</div>
                      <ul v-else-if="conversationDetail" class="space-y-1.5 max-h-64 overflow-y-auto">
                        <li v-for="m in conversationDetail.messages" :key="m.id" class="bg-surface-3/40 rounded px-2 py-1.5">
                          <span class="uppercase text-[10px] text-slate-500">{{ m.role }}</span>
                          <p class="text-slate-300 whitespace-pre-line">{{ m.content }}</p>
                        </li>
                      </ul>
                    </td>
                  </tr>
                </template>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Pipeline / Trace + Depolanan Kanıtlar: ortak analiz seçici -->
        <div v-else>
          <div class="flex items-center gap-2 mb-4 text-sm">
            <span class="text-slate-500">Analiz:</span>
            <select v-model="selectedJobId" class="field-input !w-auto !py-1" @change="selectAnalysis(selectedJobId)">
              <option :value="null">— seçin —</option>
              <option v-for="a in analyses" :key="a.job_id" :value="a.job_id">
                {{ basename(a.video_source) }} · {{ a.job_id.slice(0, 8) }}
              </option>
            </select>
          </div>

          <div v-if="!selectedJobId" class="text-sm text-slate-500 text-center py-8">
            İncelemek için yukarıdan bir analiz seçin.
          </div>
          <div v-else-if="selectedLoading" class="text-sm text-slate-500 text-center py-8">Yükleniyor…</div>
          <div v-else-if="selectedError" class="text-sm text-risk-crit text-center py-8">{{ selectedError }}</div>

          <!-- Pipeline / Trace -->
          <template v-else-if="tab === 'pipeline'">
            <div v-if="!traceEvents.length" class="text-sm text-slate-500 text-center py-8">
              Bu analiz için kalıcı pipeline (trace) kaydı bulunmuyor.
            </div>
            <div v-else class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="text-left text-[11px] uppercase tracking-wide text-slate-500 border-b border-edge">
                    <th class="py-2 pr-3">Stage</th>
                    <th class="py-2 pr-3">Durum</th>
                    <th class="py-2 pr-3">Zaman</th>
                    <th class="py-2 pr-3">Süre</th>
                    <th class="py-2 pr-3">Özet</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(e, i) in traceEvents" :key="i" class="border-b border-edge/60">
                    <td class="py-2 pr-3 text-slate-200">{{ stageLabel(e.stage) }}</td>
                    <td class="py-2 pr-3" :class="e.status === 'failed' ? 'text-risk-crit' : e.status === 'completed' ? 'text-risk-low' : 'text-slate-400'">{{ e.status }}</td>
                    <td class="py-2 pr-3 text-slate-400 font-mono text-xs">{{ fmtDate(e.timestamp) }}</td>
                    <td class="py-2 pr-3 text-slate-400">{{ e.duration_ms != null ? `${e.duration_ms.toFixed(0)} ms` : '—' }}</td>
                    <td class="py-2 pr-3 text-slate-300">{{ e.summary }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </template>

          <!-- Depolanan Kanıtlar -->
          <template v-else>
            <div v-if="!storedFrames.length" class="text-sm text-slate-500 text-center py-8">
              Bu analiz için kalıcı olarak saklanmış kanıt karesi bulunmuyor.
            </div>
            <div v-else class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
              <figure v-for="f in storedFrames" :key="f.frame_id" class="border border-edge rounded-lg overflow-hidden bg-surface-2">
                <img :src="api.getFrameUrl(selectedJobId!, f.frame_id)" :alt="f.frame_id" class="w-full h-28 object-cover" loading="lazy" />
                <figcaption class="px-2 py-1.5 text-[11px] text-slate-400">
                  <div class="text-slate-300">Olay #{{ f.event_id }} · {{ f.selection_reason }}</div>
                  <div class="font-mono">{{ f.timestamp_str }} · {{ f.frame_id }}</div>
                </figcaption>
              </figure>
            </div>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>
