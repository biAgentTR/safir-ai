<script setup lang="ts">
// Reports — report center. Distinct from History (which lists ALL analyses,
// including running/failed/queued): this page lists only COMPLETED analyses
// (real reports), framed around the report itself rather than the run log.
// Reuses GET /history (no new endpoint) — just a different filter/framing.
import type { HistoryListItem } from '~/types/api'

const api = useSafirApi()
const router = useRouter()

const items = ref<HistoryListItem[]>([])
const loading = ref(true)
const error = ref<string | null>(null)
const done = ref(false)
const PAGE = 25

const reports = computed(() => items.value.filter((i) => i.status === 'completed'))

async function load(initial = false) {
  loading.value = true
  error.value = null
  try {
    const batch = await api.getHistory(PAGE, initial ? 0 : items.value.length)
    if (initial) items.value = batch
    else items.value = [...items.value, ...batch]
    if (batch.length < PAGE) done.value = true
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

onMounted(() => load(true))
</script>

<template>
  <div class="max-w-5xl mx-auto px-6 py-8">
    <div class="flex items-center justify-between mb-5">
      <div>
        <h2 class="text-xl font-semibold text-slate-100">Raporlar</h2>
        <p class="text-sm text-slate-500 mt-1">Tamamlanmış analizlerin raporları — özet, risk, zaman çizelgesi, kanıt kareleri ve dışa aktarma tek ekranda.</p>
      </div>
      <NuxtLink to="/history" class="btn-ghost">Tüm Geçmiş →</NuxtLink>
    </div>

    <!-- loading (initial) -->
    <div v-if="loading && !items.length" class="card p-10 text-center text-slate-500">
      <div class="inline-block w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      <div class="mt-3 text-sm">Raporlar yükleniyor…</div>
    </div>

    <!-- error -->
    <div v-else-if="error && !items.length" class="card p-6 border-risk-crit/30">
      <p class="text-sm text-risk-crit">{{ error }}</p>
      <button class="btn-ghost mt-3" @click="load(true)">Tekrar dene</button>
    </div>

    <!-- truly empty: no more pages left AND nothing completed found -->
    <div v-else-if="!reports.length && done" class="card p-10 text-center">
      <p class="text-sm text-slate-400">Henüz tamamlanmış bir rapor yok.</p>
      <NuxtLink to="/new-analysis" class="btn-primary mt-4 inline-flex">İlk analizi başlat</NuxtLink>
    </div>

    <!-- loaded page(s) had no completed analyses yet, but more pages remain -->
    <div v-else-if="!reports.length" class="card p-10 text-center">
      <p class="text-sm text-slate-400">Bu sayfada tamamlanmış rapor yok, sonraki kayıtlarda olabilir.</p>
      <button class="btn-ghost mt-4" :disabled="loading" @click="load(false)">
        {{ loading ? 'Yükleniyor…' : 'Daha fazla yükle' }}
      </button>
    </div>

    <!-- list -->
    <div v-else class="grid sm:grid-cols-2 gap-3">
      <button
        v-for="item in reports"
        :key="item.job_id"
        type="button"
        class="text-left card p-4 hover:bg-surface-2 transition-colors"
        @click="open(item)"
      >
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0">
            <div class="text-sm text-slate-100 truncate">{{ basename(item.video_source) }}</div>
            <div class="text-[11px] text-slate-600 mt-0.5 font-mono">{{ fmtDate(item.created_at) }} · {{ item.job_id.slice(0, 8) }}</div>
          </div>
          <div v-if="item.risk_status === 'unknown' || item.risk_score == null" class="text-right shrink-0">
            <div class="text-xl font-bold text-slate-400">—</div>
            <div class="text-[10px] uppercase tracking-wide text-slate-400">Belirsiz</div>
          </div>
          <div v-else class="text-right shrink-0">
            <div class="text-xl font-bold" :class="RISK_TEXT[riskTone(item.risk_level)]">{{ item.risk_score }}</div>
            <div class="text-[10px] uppercase tracking-wide" :class="RISK_TEXT[riskTone(item.risk_level)]">{{ item.risk_level }}</div>
          </div>
        </div>
        <p class="text-xs text-slate-400 mt-2 line-clamp-2">{{ item.summary || 'Özet yok' }}</p>
      </button>

      <div v-if="!done" class="sm:col-span-2 pt-2 text-center">
        <button class="btn-ghost" :disabled="loading" @click="load(false)">
          {{ loading ? 'Yükleniyor…' : 'Daha fazla' }}
        </button>
      </div>
    </div>
  </div>
</template>
