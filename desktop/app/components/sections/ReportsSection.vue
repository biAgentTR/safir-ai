<script setup lang="ts">
// Reports — report center. Distinct from History (which lists ALL analyses,
// including running/failed/queued): this page lists only COMPLETED analyses
// (real reports), framed around the report itself rather than the run log.
import type { HistoryListItem } from '~/types/api'

const api = useSafirApi()
const router = useRouter()
const { goToSection } = useSectionNav()
const { mode } = useAnalysisMode()
const newAnalysisSectionId = computed(() => (mode.value === 'vlm_direct' ? 'vlm-direct' : 'yeni-analiz'))

const items = ref<HistoryListItem[]>([])
const loading = ref(true)
const error = ref<string | null>(null)

const reports = computed(() => items.value.filter((i) => i.status === 'completed'))

// Card pagination: 8 reports (4 rows x 2 cols) to fully fill the box
const PAGE_SIZE = 8
const currentPage = ref(1)
const isPageTransitioning = ref(false)
let pageTimer: ReturnType<typeof setTimeout> | null = null

const totalPages = computed(() => Math.max(1, Math.ceil(reports.value.length / PAGE_SIZE)))
const paginatedReports = computed(() => {
  const start = (currentPage.value - 1) * PAGE_SIZE
  return reports.value.slice(start, start + PAGE_SIZE)
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
      (e as { data?: { detail?: string } })?.data?.detail ?? (e as Error)?.message ?? 'Raporlar yüklenemedi.'
  } finally {
    loading.value = false
  }
}

function open(item: HistoryListItem) {
  router.push(`/reports/${item.job_id}`)
}

function fmtDate(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('tr-TR', { dateStyle: 'medium', timeStyle: 'short' })
}
function basename(src: string | null): string {
  if (!src) return '—'
  return src.split(/[\\/]/).pop() || src
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
            <span class="text-accent text-sm">▦</span>
            <h2 class="text-lg font-bold tracking-tight text-slate-100">Raporlar</h2>
          </div>
          <p class="text-xs text-slate-500 mt-0.5">Tamamlanmış analizlerin detaylı risk ve özet raporları.</p>
        </div>
        <div class="flex items-center gap-2">
          <!-- < > Pagination Controls in Header -->
          <div v-if="totalPages > 1" class="flex items-center gap-1.5 bg-surface-2 px-2.5 py-1 rounded-lg border border-edge">
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
        </div>
      </div>

      <!-- loading -->
      <div v-if="loading && !items.length" class="card p-10 text-center text-slate-500">
        <div class="inline-block w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        <div class="mt-3 text-sm">Raporlar yükleniyor…</div>
      </div>

      <!-- error -->
      <div v-else-if="error && !items.length" class="card p-6 border-risk-crit/30">
        <p class="text-sm text-risk-crit">{{ error }}</p>
        <button class="btn-ghost mt-3 text-xs" @click.stop="load()">Tekrar dene</button>
      </div>

      <!-- empty -->
      <div v-else-if="!reports.length" class="card p-10 text-center">
        <p class="text-sm text-slate-400">Henüz tamamlanmış bir rapor yok.</p>
        <button type="button" class="btn-primary mt-4 inline-flex text-xs" @click.stop="goToSection(newAnalysisSectionId)">İlk analizi başlat</button>
      </div>

      <!-- list with transition & skeleton loader -->
      <div v-else class="min-h-[440px]">
        <Transition name="page-crossfade" mode="out-in">
          <!-- Page Change Skeleton Cards -->
          <div v-if="isPageTransitioning" key="skeleton" class="grid grid-cols-1 sm:grid-cols-2 gap-3 animate-pulse">
            <div
              v-for="i in PAGE_SIZE"
              :key="i"
              class="card p-4 border border-edge/60 space-y-3 flex flex-col justify-between"
            >
              <div>
                <div class="flex items-start justify-between gap-3">
                  <div class="space-y-1.5">
                    <div class="h-4 w-36 rounded bg-surface-2" />
                    <div class="h-2.5 w-24 rounded bg-surface-2/60" />
                  </div>
                  <div class="h-6 w-12 rounded bg-surface-2" />
                </div>
                <div class="mt-3 space-y-1.5">
                  <div class="h-3 w-full rounded bg-surface-2/60" />
                  <div class="h-3 w-4/5 rounded bg-surface-2/40" />
                </div>
              </div>
              <div class="pt-2 border-t border-edge/40 flex items-center justify-between">
                <div class="h-3 w-16 rounded bg-surface-2/60" />
                <div class="h-3 w-20 rounded bg-accent/30" />
              </div>
            </div>
          </div>

          <!-- Real Paginated Reports Grid -->
          <div v-else :key="currentPage" class="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <button
              v-for="item in paginatedReports"
              :key="item.job_id"
              type="button"
              class="text-left card p-4 hover:bg-surface-2 transition-colors border border-edge/60 hover:border-accent/40 flex flex-col justify-between"
              @click.stop="open(item)"
            >
              <div>
                <div class="flex items-start justify-between gap-3">
                  <div class="min-w-0">
                    <div class="text-sm font-semibold text-slate-100 truncate">{{ basename(item.video_source) }}</div>
                    <div class="text-[11px] text-slate-500 mt-0.5 font-mono">{{ fmtDate(item.created_at) }}</div>
                  </div>
                  <div v-if="item.risk_status === 'unknown' || item.risk_score == null" class="text-right shrink-0">
                    <div class="text-lg font-bold text-slate-400">—</div>
                    <div class="text-[9px] uppercase tracking-wide text-slate-400">BELİRSİZ</div>
                  </div>
                  <div v-else class="text-right shrink-0">
                    <div class="text-lg font-bold tabular-nums" :class="RISK_TEXT[riskTone(item.risk_level)]">{{ item.risk_score }}</div>
                    <div class="text-[9px] uppercase tracking-wide font-medium" :class="RISK_TEXT[riskTone(item.risk_level)]">{{ trUpper(item.risk_level) }}</div>
                  </div>
                </div>
                <p class="text-xs text-slate-400 mt-2.5 line-clamp-2 leading-relaxed">{{ item.summary || 'Özet yok' }}</p>
              </div>
              <div class="mt-3 pt-2 border-t border-edge/40 flex items-center justify-between text-[11px] text-slate-500">
                <span class="font-mono text-slate-600">ID: {{ item.job_id.slice(0, 8) }}</span>
                <span class="text-accent group-hover:underline">Raporu Gör →</span>
              </div>
            </button>
          </div>
        </Transition>
      </div>
    </div>

    <!-- Bottom Pagination Footer -->
    <div v-if="totalPages > 1" class="pt-4 flex items-center justify-between border-t border-edge text-xs text-slate-500">
      <span>Toplam {{ reports.length }} tamamlanmış rapor</span>
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
