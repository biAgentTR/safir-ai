<script setup lang="ts">
// Analysis History — real persisted analyses from GET /history. Clicking a row
// reuses the existing Workspace in history mode (/workspace/{id}?history=1).
import type { HistoryListItem } from '~/types/api'

const api = useSafirApi()
const router = useRouter()
const { goToSection } = useSectionNav()
const { mode } = useAnalysisMode()
// Yeni Analiz (the low_budget form) isn't mounted in vlm_direct mode — its
// own composer lives in the VLM Direct Analiz section instead. See pages/index.vue.
const newAnalysisSectionId = computed(() => (mode.value === 'vlm_direct' ? 'vlm-direct' : 'yeni-analiz'))

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

onMounted(() => load(true))
</script>

<template>
  <div id="gecmis" class="scroll-mt-16 max-w-5xl mx-auto px-6 py-8">
    <div class="flex items-center justify-between mb-5">
      <div>
        <h2 class="text-xl font-bold tracking-tight text-slate-100">Analiz Geçmişi</h2>
        <p class="text-sm text-slate-500 mt-1">Kalıcı olarak saklanmış tüm analizler (en yeni önce).</p>
      </div>
      <button type="button" class="btn-primary" @click="goToSection(newAnalysisSectionId)">Yeni Analiz</button>
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
      <button type="button" class="btn-primary mt-4 inline-flex" @click="goToSection(newAnalysisSectionId)">İlk analizi başlat</button>
    </div>

    <!-- list -->
    <div v-else class="card overflow-hidden">
      <div class="overflow-x-auto">
        <table class="op-table">
          <thead>
            <tr>
              <th>Kaynak</th>
              <th>Durum</th>
              <th>İş Kimliği / Tarih</th>
              <th class="text-right">Risk</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in items" :key="item.job_id" data-clickable @click="open(item)">
              <td class="min-w-0 max-w-0 w-full">
                <div class="text-sm text-slate-100 truncate">{{ basename(item.video_source) }}</div>
                <div class="text-xs text-slate-500 mt-0.5 truncate">{{ item.summary || 'Özet yok' }}</div>
              </td>
              <td class="whitespace-nowrap">
                <span class="badge" :class="statusBadge[item.status] ?? 'badge-neutral'">{{ statusLabel[item.status] ?? item.status }}</span>
              </td>
              <td class="whitespace-nowrap font-mono text-xs text-slate-500">{{ fmtDate(item.created_at) }} · {{ item.job_id.slice(0, 8) }}</td>
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

      <div v-if="!done" class="p-3 text-center border-t border-edge">
        <button class="btn-ghost" :disabled="loading" @click="load(false)">
          {{ loading ? 'Yükleniyor…' : 'Daha fazla' }}
        </button>
      </div>
    </div>
  </div>
</template>
