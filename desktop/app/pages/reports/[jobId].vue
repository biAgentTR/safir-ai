<script setup lang="ts">
// Report detail — a document-style, single-scroll view of one completed
// analysis. Deliberately NOT a copy of Workspace's tabbed/technical layout:
// no pipeline rail, no "connection" status, no tabs — just the report itself,
// composed entirely from EXISTING, already-working components (RiskSummary,
// KpiPanel, FinalReport, EventTimeline, EvidenceGallery, AskSafir), all of
// which are already store-driven and already work identically in history
// mode (see workspace/[jobId].vue). Same data source as Workspace's history
// mode (store.loadHistory) — no new backend endpoint, no duplicated logic.
const route = useRoute()
const jobId = computed(() => String(route.params.jobId))

const store = useAnalysisStore()

onMounted(async () => {
  await store.loadHistory(jobId.value)
})

function fmtDate(iso: string | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('tr-TR', { dateStyle: 'medium', timeStyle: 'short' })
}
function basename(src: string | null | undefined): string {
  if (!src) return '—'
  return src.split(/[\\/]/).pop() || src
}
</script>

<template>
  <div class="px-6 py-8 max-w-4xl mx-auto space-y-6">
    <!-- header -->
    <div class="flex items-center gap-4">
      <NuxtLink to="/#raporlar" class="btn-ghost">← Raporlar</NuxtLink>
      <div class="min-w-0">
        <h2 class="text-lg font-semibold text-slate-100 truncate">{{ basename(store.lastRequest?.video_source) }}</h2>
        <div class="text-[11px] text-slate-500 font-mono">{{ jobId.slice(0, 8) }}</div>
      </div>
    </div>

    <!-- loading / error / no-report states -->
    <div v-if="store.historyLoading" class="card p-10 text-center text-slate-500 text-sm">Rapor yükleniyor…</div>
    <div v-else-if="store.historyError" class="rounded-md border border-risk-crit/40 bg-risk-crit/10 px-4 py-3 text-sm text-risk-crit">
      {{ store.historyError }}
    </div>
    <div v-else-if="store.status === 'error'" class="rounded-md border border-risk-crit/40 bg-risk-crit/10 px-4 py-3 text-sm text-slate-200">
      Bu analiz başarısız durumda tamamlandı — kalıcı rapor bulunmuyor.
    </div>
    <div v-else-if="!store.report" class="card p-10 text-center text-slate-500 text-sm">
      Bu analiz için henüz bir rapor yok.
    </div>

    <!-- report document -->
    <template v-else>
      <RiskSummary />
      <KpiPanel />

      <section class="card p-5">
        <div class="field-label mb-3">Rapor</div>
        <FinalReport />
      </section>

      <section class="card p-5">
        <div class="field-label mb-3">Zaman Çizelgesi</div>
        <EventTimeline />
      </section>

      <section class="card p-5">
        <div class="field-label mb-3">Kanıt Kareleri</div>
        <EvidenceGallery v-if="store.traceEvents.length" />
        <p v-else class="text-sm text-slate-500 py-4 text-center">
          Bu analiz için kalıcı kanıt karesi kaydı bulunmuyor (pipeline izi saklanmadan tamamlanmış eski bir analiz olabilir).
        </p>
      </section>

      <AskSafir :job-id="jobId" />
    </template>
  </div>
</template>
