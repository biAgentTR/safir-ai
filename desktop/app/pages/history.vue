<script setup lang="ts">
// Analysis History — real persisted analyses from GET /history. Clicking a row
// reuses the existing Workspace in history mode (/workspace/{id}?history=1).
import type { HistoryListItem } from '~/types/api'

const api = useSafirApi()
const router = useRouter()

const items = ref<HistoryListItem[]>([])
const loading = ref(true)
const error = ref<string | null>(null)
const done = ref(false)
const PAGE = 25

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
const statusTone: Record<string, string> = {
  completed: 'text-risk-low border-risk-low/30 bg-risk-low/10',
  failed: 'text-risk-crit border-risk-crit/30 bg-risk-crit/10',
  running: 'text-accent border-accent/30 bg-accent/10',
  queued: 'text-slate-400 border-edge bg-surface-2',
}

onMounted(() => load(true))
</script>

<template>
  <div class="max-w-5xl mx-auto px-6 py-8">
    <div class="flex items-center justify-between mb-5">
      <div>
        <h2 class="text-xl font-semibold text-slate-100">Analiz Geçmişi</h2>
        <p class="text-sm text-slate-500 mt-1">Kalıcı olarak saklanmış tüm analizler (en yeni önce).</p>
      </div>
      <NuxtLink to="/new-analysis" class="btn-primary"><span>＋</span> Yeni Analiz</NuxtLink>
    </div>

    <!-- loading (initial) -->
    <div v-if="loading && !items.length" class="card p-10 text-center text-slate-500">
      <div class="inline-block w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      <div class="mt-3 text-sm">Geçmiş yükleniyor…</div>
    </div>

    <!-- error -->
    <div v-else-if="error && !items.length" class="card p-6 border-risk-crit/30">
      <p class="text-sm text-risk-crit">{{ error }}</p>
      <button class="btn-ghost mt-3" @click="load(true)">Tekrar dene</button>
    </div>

    <!-- empty -->
    <div v-else-if="!items.length" class="card p-10 text-center">
      <p class="text-sm text-slate-400">Henüz kayıtlı analiz yok.</p>
      <NuxtLink to="/new-analysis" class="btn-primary mt-4 inline-flex">İlk analizi başlat</NuxtLink>
    </div>

    <!-- list -->
    <div v-else class="space-y-2">
      <button
        v-for="item in items"
        :key="item.job_id"
        type="button"
        class="w-full text-left card px-4 py-3 hover:bg-surface-2 transition-colors flex items-center gap-4"
        @click="open(item)"
      >
        <div class="min-w-0 flex-1">
          <div class="flex items-center gap-2">
            <span class="text-sm text-slate-100 truncate">{{ basename(item.video_source) }}</span>
            <span class="text-[11px] px-2 py-0.5 rounded border" :class="statusTone[item.status] ?? statusTone.queued">{{ item.status }}</span>
          </div>
          <div class="text-xs text-slate-500 mt-0.5 truncate">{{ item.summary || 'Özet yok' }}</div>
          <div class="text-[11px] text-slate-600 mt-0.5 font-mono">{{ fmtDate(item.created_at) }} · {{ item.job_id.slice(0, 8) }}</div>
        </div>
        <div v-if="item.risk_status === 'unknown' || item.risk_score == null" class="text-right shrink-0">
          <div class="text-lg font-bold text-slate-400">—</div>
          <div class="text-[10px] uppercase tracking-wide text-slate-400">Belirsiz</div>
        </div>
        <div v-else class="text-right shrink-0">
          <div class="text-lg font-bold" :class="RISK_TEXT[riskTone(item.risk_level)]">{{ item.risk_score }}</div>
          <div class="text-[10px] uppercase tracking-wide" :class="RISK_TEXT[riskTone(item.risk_level)]">{{ item.risk_level }}</div>
        </div>
        <span class="text-slate-600 shrink-0">›</span>
      </button>

      <div v-if="!done" class="pt-2 text-center">
        <button class="btn-ghost" :disabled="loading" @click="load(false)">
          {{ loading ? 'Yükleniyor…' : 'Daha fazla' }}
        </button>
      </div>
    </div>
  </div>
</template>
