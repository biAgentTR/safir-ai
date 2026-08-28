<script setup lang="ts">
// Analysis History — real persisted analyses from GET /history. Clicking a row
// reuses the existing Workspace in history mode (/workspace/{id}?history=1).
import type { HistoryListItem } from '~/types/api'

const api = useSafirApi()
const router = useRouter()
const { goToSection } = useSectionNav()
const { mode } = useAnalysisMode()
const newAnalysisSectionId = computed(() => (mode.value === 'vlm_direct' ? 'vlm-direct' : 'yeni-analiz'))

const items = ref<HistoryListItem[]>([])
const loading = ref(true)
const error = ref<string | null>(null)

// Card pagination: 10 rows
const PAGE_SIZE = 10
const currentPage = ref(1)
const isPageTransitioning = ref(false)
let pageTimer: ReturnType<typeof setTimeout> | null = null

const totalPages = computed(() => Math.max(1, Math.ceil(items.value.length / PAGE_SIZE)))
const paginatedItems = computed(() => {
  const start = (currentPage.value - 1) * PAGE_SIZE
  return items.value.slice(start, start + PAGE_SIZE)
})

function triggerPageChange(newPage: number) {
  if (isPageTransitioning.value || newPage === currentPage.value) return
  isPageTransitioning.value = true
  currentPage.value = newPage

  if (pageTimer) clearTimeout(pageTimer)
  pageTimer = setTimeout(() => {
    isPageTransitioning.value = false
  }, 220)
}

function prevPage() {
  if (currentPage.value > 1) triggerPageChange(currentPage.value - 1)
}

function nextPage() {
  if (currentPage.value < totalPages.value) triggerPageChange(currentPage.value + 1)
}

async function load() {
  loading.value = true
  error.value = null
  try {
    items.value = await api.getHistory(100, 0)
  } catch (e: unknown) {
    error.value =
      (e as { data?: { detail?: string } })?.data?.detail ?? (e as Error)?.message ?? 'Geçmiş yüklenemedi.'
  } finally {
    loading.value = false
  }
}

function open(item: HistoryListItem) {
  router.push(`/workspace/${item.job_id}?history=1`)
}

function fmtDate(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('tr-TR', { dateStyle: 'medium', timeStyle: 'short' })
}
function basename(src: string | null): string {
  if (!src) return '—'
  return src.split(/[\\/]/).pop() || src
}
const statusBadge: Record<string, string> = {
  completed: 'badge-low',
  failed: 'badge-crit',
  running: 'badge-accent',
  queued: 'badge-neutral',
}
const statusLabel: Record<string, string> = {
  completed: 'Tamamlandı',
  failed: 'Başarısız',
  running: 'Devam ediyor',
  queued: 'Kuyrukta',
}

onMounted(() => load())
</script>

<template>
  <div class="h-full flex flex-col justify-between">
    <!-- Header -->
    <div>
      <div class="flex items-center justify-between mb-4">
        <div>
          <div class="flex items-center gap-2">
            <span class="text-accent text-sm">≡</span>
            <h2 class="text-lg font-bold tracking-tight text-slate-100">Analiz Geçmişi</h2>
          </div>
          <p class="text-xs text-slate-500 mt-0.5">Kalıcı olarak saklanmış tüm analizler.</p>
        </div>
        <div class="flex items-center gap-2">
          <!-- < > Pagination Controls in Header -->
          <div v-if="totalPages > 1" class="flex items-center gap-1.5 bg-surface-2 px-2 py-1 rounded-lg border border-edge">
            <button
              type="button"
              class="w-6 h-6 rounded flex items-center justify-center text-xs font-bold transition-colors"
              :class="currentPage > 1 ? 'text-slate-200 hover:bg-surface-3' : 'text-slate-600 cursor-not-allowed'"
              :disabled="currentPage <= 1"
              aria-label="Önceki Sayfa"
              @click.stop="prevPage"
            >
              ❮
            </button>
            <span class="text-xs text-slate-400 font-mono px-1">{{ currentPage }} / {{ totalPages }}</span>
            <button
              type="button"
              class="w-6 h-6 rounded flex items-center justify-center text-xs font-bold transition-colors"
              :class="currentPage < totalPages ? 'text-slate-200 hover:bg-surface-3' : 'text-slate-600 cursor-not-allowed'"
              :disabled="currentPage >= totalPages"
              aria-label="Sonraki Sayfa"
              @click.stop="nextPage"
            >
              ❯
            </button>
          </div>
          <button type="button" class="btn-primary !py-1.5 !px-3 text-xs" @click.stop="goToSection(newAnalysisSectionId)">Yeni Analiz</button>
        </div>
      </div>

      <!-- loading -->
      <div v-if="loading && !items.length" class="card p-10 text-center text-slate-500">
        <div class="inline-block w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        <div class="mt-3 text-sm">Geçmiş yükleniyor…</div>
      </div>

      <!-- error -->
      <div v-else-if="error && !items.length" class="card p-6 border-risk-crit/30">
        <p class="text-sm text-risk-crit">{{ error }}</p>
        <button class="btn-ghost mt-3 text-xs" @click.stop="load()">Tekrar dene</button>
      </div>

      <!-- empty -->
      <div v-else-if="!items.length" class="card p-10 text-center">
        <p class="text-sm text-slate-400">Henüz kayıtlı analiz yok.</p>
        <button type="button" class="btn-primary mt-4 inline-flex text-xs" @click.stop="goToSection(newAnalysisSectionId)">İlk analizi başlat</button>
      </div>

      <!-- list with transition & skeleton loader -->
      <div v-else class="card overflow-hidden min-h-[440px]">
        <Transition name="page-crossfade" mode="out-in">
          <!-- Page Change Skeleton Rows -->
          <div v-if="isPageTransitioning" key="skeleton" class="p-4 space-y-3 animate-pulse">
            <div
              v-for="i in PAGE_SIZE"
              :key="i"
              class="flex items-center justify-between py-2 border-b border-edge/30 last:border-0"
            >
              <div class="space-y-1.5 w-2/5">
                <div class="h-3.5 w-32 rounded bg-surface-2" />
                <div class="h-2.5 w-48 rounded bg-surface-2/60" />
              </div>
              <div class="h-5 w-20 rounded bg-surface-2/80" />
              <div class="h-3 w-28 rounded bg-surface-2/60" />
              <div class="h-4 w-16 rounded bg-surface-2/80" />
            </div>
          </div>

          <!-- Real Paginated Table -->
          <div v-else :key="currentPage" class="overflow-x-auto">
            <table class="op-table">
              <thead>
                <tr>
                  <th>Kaynak</th>
                  <th>Durum</th>
                  <th>Tarih</th>
                  <th class="text-right">Risk</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="item in paginatedItems" :key="item.job_id" data-clickable @click.stop="open(item)">
                  <td class="min-w-0 max-w-0 w-full">
                    <div class="text-sm text-slate-100 truncate">{{ basename(item.video_source) }}</div>
                    <div class="text-xs text-slate-500 mt-0.5 truncate">{{ item.summary || 'Özet yok' }}</div>
                  </td>
                  <td class="whitespace-nowrap">
                    <span class="badge" :class="statusBadge[item.status] ?? 'badge-neutral'">{{ statusLabel[item.status] ?? item.status }}</span>
                  </td>
                  <td class="whitespace-nowrap font-mono text-xs text-slate-500">{{ fmtDate(item.created_at) }}</td>
                  <td class="whitespace-nowrap text-right">
                    <template v-if="item.risk_status === 'unknown' || item.risk_score == null">
                      <span class="text-sm text-slate-500">Belirsiz</span>
                    </template>
                    <template v-else>
                      <span class="text-sm font-bold tabular-nums" :class="RISK_TEXT[riskTone(item.risk_level)]">{{ item.risk_score }} <span class="font-normal text-xs uppercase tracking-wide">{{ trUpper(item.risk_level) }}</span></span>
                    </template>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </Transition>
      </div>
    </div>

    <!-- Bottom Pagination Footer -->
    <div v-if="totalPages > 1" class="pt-4 flex items-center justify-between border-t border-edge text-xs text-slate-500">
      <span>Toplam {{ items.length }} kayıt</span>
      <div class="flex items-center gap-2">
        <button
          type="button"
          class="px-2.5 py-1 rounded bg-surface-2 hover:bg-surface-3 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          :disabled="currentPage <= 1"
          @click.stop="prevPage"
        >
          ❮ Önceki
        </button>
        <span class="font-mono text-slate-400">{{ currentPage }} / {{ totalPages }}</span>
        <button
          type="button"
          class="px-2.5 py-1 rounded bg-surface-2 hover:bg-surface-3 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          :disabled="currentPage >= totalPages"
          @click.stop="nextPage"
        >
          Sonraki ❯
        </button>
      </div>
    </div>
  </div>
</template>
