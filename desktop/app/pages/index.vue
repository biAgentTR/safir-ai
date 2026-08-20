<script setup lang="ts">
// Overview / Dashboard — the operator's entry point. Surfaces real backend
// state only: GET /health (topbar), GET /system/overview (status strip),
// GET /history (critical events + recent analyses + timeline source).
// No fabricated metrics — a section with no backing data renders an honest
// empty state instead of a placeholder number.
import type { HistoryListItem } from '~/types/api'

const api = useSafirApi()
const router = useRouter()
const { state: backendState } = useBackendHealth()

// ---- system status strip (GET /system/overview) ----
const overview = ref<Awaited<ReturnType<typeof api.getSystemOverview>> | null>(null)
const overviewLoading = ref(true)
const overviewError = ref<string | null>(null)

async function loadOverview() {
  overviewLoading.value = true
  overviewError.value = null
  try {
    overview.value = await api.getSystemOverview()
  } catch (e) {
    overviewError.value = humanError(e, 'Sistem özeti alınamadı.')
  } finally {
    overviewLoading.value = false
  }
}

// ---- recent analyses (GET /history) ----
const RECENT_LIMIT = 20
const history = ref<HistoryListItem[]>([])
const historyLoading = ref(true)
const historyError = ref<string | null>(null)

async function loadHistory() {
  historyLoading.value = true
  historyError.value = null
  try {
    history.value = await api.getHistory(RECENT_LIMIT)
  } catch (e) {
    historyError.value = humanError(e, 'Analizler alınamadı.')
  } finally {
    historyLoading.value = false
  }
}

function humanError(e: unknown, fallback: string): string {
  return (e as { data?: { detail?: string } })?.data?.detail ?? (e as Error)?.message ?? fallback
}

function refreshAll() {
  loadOverview()
  loadHistory()
}

onMounted(refreshAll)

// ---- search + filters (client-side, over the loaded recent-analyses window) ----
const searchQuery = ref('')
const riskFilter = ref<'all' | 'crit' | 'high' | 'mid' | 'low' | 'unknown'>('all')
const statusFilter = ref<'all' | 'completed' | 'running' | 'queued' | 'failed'>('all')
const dateFilter = ref<'all' | '24h' | '7d' | '30d'>('all')

function basename(src: string | null): string {
  if (!src) return '—'
  return src.split(/[\\/]/).pop() || src
}

function withinDate(iso: string): boolean {
  if (dateFilter.value === 'all') return true
  const ms = Date.now() - new Date(iso).getTime()
  const day = 86_400_000
  if (dateFilter.value === '24h') return ms <= day
  if (dateFilter.value === '7d') return ms <= 7 * day
  return ms <= 30 * day
}

const filteredHistory = computed(() => {
  const q = searchQuery.value.trim().toLowerCase()
  return history.value.filter((h) => {
    if (statusFilter.value !== 'all' && h.status !== statusFilter.value) return false
    const tone = h.risk_status === 'unknown' || h.risk_score == null ? 'unknown' : riskTone(h.risk_level)
    if (riskFilter.value !== 'all' && tone !== riskFilter.value) return false
    if (!withinDate(h.created_at)) return false
    if (q) {
      const hay = `${basename(h.video_source)} ${h.summary ?? ''}`.toLowerCase()
      if (!hay.includes(q)) return false
    }
    return true
  })
})

const hasActiveFilters = computed(
  () => !!searchQuery.value || riskFilter.value !== 'all' || statusFilter.value !== 'all' || dateFilter.value !== 'all',
)

function clearFilters() {
  searchQuery.value = ''
  riskFilter.value = 'all'
  statusFilter.value = 'all'
  dateFilter.value = 'all'
}

// ---- critical / high-risk analyses needing operator attention ----
const criticalRecent = computed(() =>
  history.value
    .filter((h) => h.risk_status !== 'unknown' && h.risk_score != null && ['crit', 'high'].includes(riskTone(h.risk_level)))
    .slice(0, 5),
)

// ---- running/queued count for the status strip ----
const runningCount = computed(() => history.value.filter((h) => h.status === 'running' || h.status === 'queued').length)

function fmtDateTime(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('tr-TR', { dateStyle: 'medium', timeStyle: 'short' })
}
function fmtRelative(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime()
  if (Number.isNaN(ms)) return '—'
  const min = Math.floor(ms / 60_000)
  if (min < 1) return 'az önce'
  if (min < 60) return `${min} dk önce`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr} sa önce`
  return `${Math.floor(hr / 24)} gün önce`
}

function openJob(h: HistoryListItem) {
  router.push(`/workspace/${h.job_id}?history=1`)
}

const statusMeta: Record<string, { label: string; tone: string }> = {
  completed: { label: 'Tamamlandı', tone: 'text-risk-low' },
  running: { label: 'Devam ediyor', tone: 'text-accent' },
  queued: { label: 'Kuyrukta', tone: 'text-slate-400' },
  failed: { label: 'Başarısız', tone: 'text-risk-crit' },
}

// ---- fullscreen operation mode ----
const isFullscreen = ref(false)
function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen()
  else document.documentElement.requestFullscreen().catch(() => {})
}
function onFullscreenChange() {
  isFullscreen.value = !!document.fullscreenElement
}
onMounted(() => document.addEventListener('fullscreenchange', onFullscreenChange))
onBeforeUnmount(() => document.removeEventListener('fullscreenchange', onFullscreenChange))

// ---- keyboard shortcuts: / search, n new analysis, f fullscreen, r refresh ----
const searchInput = ref<HTMLInputElement | null>(null)
function onKeydown(e: KeyboardEvent) {
  const target = e.target as HTMLElement | null
  const typing = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
  if (typing) return
  if (e.key === '/') {
    e.preventDefault()
    searchInput.value?.focus()
  } else if (e.key === 'n') {
    router.push('/new-analysis')
  } else if (e.key === 'f') {
    toggleFullscreen()
  } else if (e.key === 'r') {
    refreshAll()
  }
}
onMounted(() => window.addEventListener('keydown', onKeydown))
onBeforeUnmount(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <div class="max-w-6xl mx-auto px-6 py-6">
    <!-- backend offline: blocks everything below with a clear, actionable banner -->
    <div
      v-if="backendState === 'offline'"
      class="mb-5 rounded-md border border-risk-crit/40 bg-risk-crit/10 px-4 py-3 flex items-center gap-3"
      role="alert"
    >
      <span class="text-risk-crit text-lg shrink-0" aria-hidden="true">⚠</span>
      <div class="text-sm text-slate-200">
        <span class="font-medium text-risk-crit">Arka uç sunucusuna ulaşılamıyor.</span>
        Canlı durum, geçmiş analizler ve yeni analiz başlatma şu anda kullanılamıyor.
      </div>
      <button class="btn-ghost ml-auto shrink-0" @click="refreshAll">Tekrar dene</button>
    </div>

    <!-- header: title, quick action, search + filters -->
    <div class="mb-5 flex flex-wrap items-start justify-between gap-4">
      <div>
        <h2 class="text-xl font-semibold text-slate-100">Genel Bakış</h2>
        <p class="mt-1 text-sm text-slate-500">Saha analizlerinin ve kritik olayların operasyon özeti.</p>
      </div>
      <div class="flex items-center gap-2 shrink-0">
        <button type="button" class="btn-ghost" :aria-pressed="isFullscreen" title="Tam ekran (f)" @click="toggleFullscreen">
          <span aria-hidden="true">{{ isFullscreen ? '⤡' : '⤢' }}</span>
          <span class="hidden md:inline">{{ isFullscreen ? 'Tam Ekrandan Çık' : 'Tam Ekran' }}</span>
        </button>
        <NuxtLink to="/new-analysis" class="btn-primary"><span aria-hidden="true">＋</span> Yeni Analiz</NuxtLink>
      </div>
    </div>

    <!-- critical / high-risk attention section — top priority, decision-relevant -->
    <section v-if="criticalRecent.length" class="mb-6" aria-label="Dikkat gerektiren son analizler">
      <h3 class="text-xs font-semibold uppercase tracking-wide text-risk-crit mb-2">Dikkat Gerektiren Son Analizler</h3>
      <div class="space-y-2">
        <button
          v-for="h in criticalRecent"
          :key="h.job_id"
          type="button"
          class="w-full text-left rounded-md border px-4 py-3 flex items-center gap-4 transition-colors"
          :class="riskTone(h.risk_level) === 'crit'
            ? 'border-risk-crit/40 bg-risk-crit/10 hover:bg-risk-crit/15'
            : 'border-risk-high/40 bg-risk-high/10 hover:bg-risk-high/15'"
          @click="openJob(h)"
        >
          <span class="text-lg shrink-0" :class="RISK_TEXT[riskTone(h.risk_level)]" aria-hidden="true">⚠</span>
          <div class="min-w-0 flex-1">
            <div class="text-sm text-slate-100 truncate">{{ basename(h.video_source) }}</div>
            <div class="text-xs text-slate-400 truncate mt-0.5">{{ h.summary || 'Özet yok' }}</div>
          </div>
          <div class="text-right shrink-0">
            <div class="text-lg font-bold" :class="RISK_TEXT[riskTone(h.risk_level)]">{{ h.risk_score }}</div>
            <div class="text-[10px] uppercase tracking-wide" :class="RISK_TEXT[riskTone(h.risk_level)]">{{ h.risk_level }}</div>
          </div>
          <span class="text-xs text-slate-500 shrink-0 w-16 text-right">{{ fmtRelative(h.created_at) }}</span>
        </button>
      </div>
    </section>

    <!-- status strip: real counts from /system/overview, compact -->
    <section class="mb-6" aria-label="Canlı sistem özeti">
      <div v-if="overviewLoading" class="card p-4 text-sm text-slate-500 text-center">Sistem özeti yükleniyor…</div>
      <div v-else-if="overviewError" class="card p-4 border-risk-crit/30 flex items-center gap-3">
        <span class="text-sm text-risk-crit flex-1">{{ overviewError }}</span>
        <button class="btn-ghost shrink-0" @click="loadOverview">Tekrar dene</button>
      </div>
      <div v-else-if="overview" class="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div class="card px-4 py-3">
          <div class="text-[11px] uppercase tracking-wide text-slate-500">Toplam Analiz</div>
          <div class="mt-1 text-xl font-semibold text-slate-100">{{ overview.totals.total_analyses }}</div>
        </div>
        <div class="card px-4 py-3">
          <div class="text-[11px] uppercase tracking-wide text-slate-500">Devam Eden</div>
          <div class="mt-1 text-xl font-semibold" :class="runningCount ? 'text-accent' : 'text-slate-100'">
            {{ overview.totals.running_or_queued_analyses }}
          </div>
        </div>
        <div class="card px-4 py-3">
          <div class="text-[11px] uppercase tracking-wide text-slate-500">Tamamlanan</div>
          <div class="mt-1 text-xl font-semibold text-risk-low">{{ overview.totals.completed_analyses }}</div>
        </div>
        <div class="card px-4 py-3">
          <div class="text-[11px] uppercase tracking-wide text-slate-500">Başarısız</div>
          <div class="mt-1 text-xl font-semibold" :class="overview.totals.failed_analyses ? 'text-risk-crit' : 'text-slate-100'">
            {{ overview.totals.failed_analyses }}
          </div>
        </div>
      </div>
    </section>

    <!-- main content: recent analyses (search/filter) + compact timeline -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">
      <section class="lg:col-span-2 card p-5" aria-label="Son analizler">
        <div class="flex flex-wrap items-center gap-2 mb-4">
          <h3 class="text-sm font-semibold text-slate-100 mr-auto">Son Analizler</h3>
          <label class="sr-only" for="dashboard-search">Analizlerde ara</label>
          <input
            id="dashboard-search"
            ref="searchInput"
            v-model="searchQuery"
            type="search"
            placeholder="Ara ( / )"
            class="field-input !w-40 !py-1.5"
          />
          <select v-model="riskFilter" class="field-input !w-auto !py-1.5" aria-label="Risk filtresi">
            <option value="all">Tüm riskler</option>
            <option value="crit">Kritik</option>
            <option value="high">Yüksek</option>
            <option value="mid">Orta</option>
            <option value="low">Düşük</option>
            <option value="unknown">Belirsiz</option>
          </select>
          <select v-model="statusFilter" class="field-input !w-auto !py-1.5" aria-label="Durum filtresi">
            <option value="all">Tüm durumlar</option>
            <option value="completed">Tamamlandı</option>
            <option value="running">Devam ediyor</option>
            <option value="queued">Kuyrukta</option>
            <option value="failed">Başarısız</option>
          </select>
          <select v-model="dateFilter" class="field-input !w-auto !py-1.5" aria-label="Tarih filtresi">
            <option value="all">Tüm zamanlar</option>
            <option value="24h">Son 24 saat</option>
            <option value="7d">Son 7 gün</option>
            <option value="30d">Son 30 gün</option>
          </select>
        </div>

        <!-- loading -->
        <div v-if="historyLoading && !history.length" class="py-10 text-center text-sm text-slate-500">
          <div class="inline-block w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin motion-reduce:animate-none" />
          <div class="mt-3">Analizler yükleniyor…</div>
        </div>

        <!-- error (no data at all) -->
        <div v-else-if="historyError && !history.length" class="py-10 text-center">
          <p class="text-sm text-risk-crit">{{ historyError }}</p>
          <button class="btn-ghost mt-3" @click="loadHistory">Tekrar dene</button>
        </div>

        <!-- first-run: no analyses at all -->
        <div v-else-if="!history.length" class="py-10 text-center">
          <p class="text-sm text-slate-400">Henüz analiz yapılmadı.</p>
          <NuxtLink to="/new-analysis" class="btn-primary mt-4 inline-flex">İlk analizi başlat</NuxtLink>
        </div>

        <!-- empty filter/search result -->
        <div v-else-if="!filteredHistory.length" class="py-10 text-center">
          <p class="text-sm text-slate-400">Bu filtrelerle eşleşen analiz yok.</p>
          <button v-if="hasActiveFilters" class="btn-ghost mt-3" @click="clearFilters">Filtreleri temizle</button>
        </div>

        <!-- results -->
        <ul v-else class="divide-y divide-edge">
          <li v-for="h in filteredHistory" :key="h.job_id">
            <button type="button" class="w-full text-left py-2.5 flex items-center gap-3 hover:bg-surface-2/60 -mx-2 px-2 rounded-md transition-colors" @click="openJob(h)">
              <div class="min-w-0 flex-1">
                <div class="flex items-center gap-2">
                  <span class="text-sm text-slate-100 truncate">{{ basename(h.video_source) }}</span>
                  <span class="text-[11px]" :class="statusMeta[h.status]?.tone ?? 'text-slate-400'">{{ statusMeta[h.status]?.label ?? h.status }}</span>
                </div>
                <div class="text-xs text-slate-500 truncate mt-0.5">{{ h.summary || 'Özet yok' }} · {{ fmtDateTime(h.created_at) }}</div>
              </div>
              <div class="text-right shrink-0 w-20">
                <template v-if="h.risk_status === 'unknown' || h.risk_score == null">
                  <span class="text-sm text-slate-500">Belirsiz</span>
                </template>
                <template v-else>
                  <span class="text-sm font-semibold" :class="RISK_TEXT[riskTone(h.risk_level)]">{{ h.risk_score }} · {{ h.risk_level }}</span>
                </template>
              </div>
            </button>
          </li>
        </ul>

        <p class="mt-4 text-[11px] text-slate-600">Kısayollar: <kbd class="font-mono">/</kbd> ara · <kbd class="font-mono">n</kbd> yeni analiz · <kbd class="font-mono">f</kbd> tam ekran · <kbd class="font-mono">r</kbd> yenile</p>
      </section>

      <div class="space-y-5">
        <DashboardTimeline :recent="history" />
      </div>
    </div>
  </div>
</template>
