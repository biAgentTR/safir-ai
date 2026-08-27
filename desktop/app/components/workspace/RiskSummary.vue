<script setup lang="ts">
// Prominent risk summary. Appears once the final report (risk) is available.
// Includes the critical banner (>=70) and the Human-on-the-Loop acknowledge
// action for auto-dispatched alerts.
const store = useAnalysisStore()
const ackNote = ref('')
const pendingReviewNote = ref('')

const isUnknownRisk = computed(() => store.report?.risk_status === 'unknown' || store.report?.risk_score == null)
const tone = computed(() => (isUnknownRisk.value ? 'unknown' : riskTone(store.report?.risk_level)))
const headline = computed(() => {
  const t = tone.value
  if (t === 'unknown') return 'RİSK BELİRSİZ'
  if (t === 'crit') return 'KRİTİK RİSK'
  if (t === 'high') return 'YÜKSEK RİSK'
  if (t === 'mid') return 'ORTA RİSK'
  return 'DÜŞÜK RİSK'
})

// Tone-colored top rule on the status band. Built from the same CSS custom
// properties as the Tailwind risk-* classes (see main.css) via inline style —
// deliberately NOT a dynamically-built Tailwind class name, which the JIT
// content scanner can't see (it only picks up literal class strings).
const RISK_VAR: Record<string, string> = {
  low: '--c-risk-low',
  mid: '--c-risk-mid',
  high: '--c-risk-high',
  crit: '--c-risk-crit',
  unknown: '--c-slate-500',
}
const bandBorderStyle = computed(() => ({ borderTopColor: `rgb(var(${RISK_VAR[tone.value]}))` }))

// Şartname "Açıklanabilir Çıktı" gerekliliği: nihai risk skoru, deterministik
// (RuleEngine + risk_model) skor ile Ajan'ın (LLM) taslak skorunun ortalaması
// olarak hesaplanır — her iki bileşen de operatöre AYRI AYRI gösterilir.
const hasRiskBreakdown = computed(
  () => store.report?.deterministic_score != null || store.report?.llm_proposed_score != null,
)
</script>

<template>
  <div v-if="store.report" class="space-y-2.5">
    <!-- critical banner (>=70) — a controlled pulse on the accent bar only, not the whole block -->
    <div
      v-if="store.isCritical"
      class="relative overflow-hidden rounded-md border border-risk-crit/40 bg-risk-crit/10 pl-4 pr-4 py-2.5 text-sm font-medium text-risk-crit flex items-center gap-2"
    >
      <span class="absolute inset-y-0 left-0 w-1 bg-risk-crit animate-pulse motion-reduce:animate-none" />
      KRİTİK RİSK TESPİT EDİLDİ — otomatik saha alarmı tetiklendi.
    </div>

    <!-- unknown-risk banner: analysis failed to produce a reliable decision -->
    <div
      v-if="isUnknownRisk"
      class="rounded-md border border-slate-600/50 bg-slate-500/10 px-4 py-2.5 text-sm font-medium text-slate-300"
    >
      Risk değerlendirilemedi — analiz güvenilir bir karar üretemedi. Manuel inceleme gerekli.
    </div>

    <!-- Human-on-the-Loop: pending_review - risk_status belirsiz VEYA deterministik
         (RuleEngine) kanit yokken hicbir otomatik islem/alarm yapilmadi; operatorun
         ACIK kararini bekler (bkz. src/decision/escalation.py). -->
    <div
      v-if="store.needsHumanReview"
      class="rounded-md border border-risk-mid/40 bg-risk-mid/10 px-4 py-2.5 text-sm font-medium text-risk-mid flex flex-col gap-2"
    >
      <div>Operatör onayı bekleniyor — bu durumda hiçbir otomatik bildirim/alarm tetiklenmedi.</div>
      <p v-if="store.report?.risk_explanation" class="text-xs font-normal text-slate-400">
        {{ store.report.risk_explanation }}
      </p>
      <div class="flex flex-col sm:flex-row gap-2 mt-1">
        <input v-model="pendingReviewNote" class="field-input" placeholder="Karar notu (opsiyonel)" />
        <button
          class="btn-primary shrink-0"
          :disabled="store.manualAlert.state === 'pending'"
          @click="store.triggerManualAlert(pendingReviewNote)"
        >Saha Alarmını Manuel Tetikle</button>
      </div>
      <p v-if="store.manualAlert.message" class="text-xs" :class="store.manualAlert.state === 'error' ? 'text-risk-crit' : 'text-risk-low'">
        {{ store.manualAlert.message }}
      </p>
    </div>

    <!-- primary status band: the #1 thing an operator must read in 2-3s -->
    <div class="panel-band p-5 flex flex-col md:flex-row md:items-stretch gap-5" :style="bandBorderStyle">
      <div class="flex items-center gap-5">
        <div class="text-center shrink-0 w-20">
          <div class="text-5xl font-bold leading-none tabular-nums" :class="RISK_TEXT[tone]">{{ isUnknownRisk ? '—' : store.report.risk_score }}</div>
          <div class="mt-1 text-[11px] uppercase tracking-wide text-slate-500">{{ isUnknownRisk ? 'belirsiz' : '/ 100' }}</div>
        </div>
        <div class="border-l border-edge pl-5">
          <div class="text-lg font-bold tracking-wide" :class="RISK_TEXT[tone]">{{ headline }}</div>
          <div class="text-xs uppercase tracking-wide text-slate-500 mt-0.5">{{ store.report.risk_level }}</div>
          <p class="mt-2 text-sm text-slate-300 max-w-xl">
            {{ store.report.recommended_action || 'Operatör değerlendirmesi önerilir.' }}
          </p>
        </div>
      </div>

      <!-- acknowledge (only for auto-dispatched alerts) -->
      <div v-if="store.hasAutoAlert" class="md:ml-auto md:w-72 shrink-0 md:border-l md:border-edge md:pl-5 space-y-2">
        <div class="eyebrow">Uyarı Kimliği</div>
        <div class="text-xs text-slate-500 font-mono truncate -mt-1">{{ store.report.alert_id }}</div>
        <input v-model="ackNote" class="field-input" placeholder="Operatör denetim notu (opsiyonel)" />
        <button
          class="btn-primary w-full"
          :disabled="store.ack.state === 'pending' || store.ack.state === 'ok'"
          @click="store.acknowledgeAlert(ackNote)"
        >
          {{ store.ack.state === 'ok' ? '✓ Onaylandı' : store.ack.state === 'pending' ? 'Gönderiliyor…' : 'Alarmı Denetle / Onayla' }}
        </button>
        <p v-if="store.ack.message" class="text-xs" :class="store.ack.state === 'error' ? 'text-risk-crit' : 'text-risk-low'">
          {{ store.ack.message }}
        </p>
      </div>
    </div>

    <!-- Risk doğruluğu / hesaplama şeffaflığı: deterministik (RuleEngine) skor
         ile Ajan'ın (LLM) taslak skoru AYRI AYRI gösterilir; nihai skor bu
         ikisinin ortalamasıdır (bkz. src/event_analysis/risk_resolver.py). -->
    <div v-if="hasRiskBreakdown" class="card p-4">
      <div class="eyebrow mb-2">Risk Doğruluğu — Hesaplama Detayı</div>
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
        <div class="rounded-md border border-edge bg-panel/40 p-3">
          <div class="text-[11px] uppercase tracking-wide text-slate-500">Deterministik Skor</div>
          <div class="mt-1 text-xl font-bold tabular-nums text-slate-200">
            {{ store.report?.deterministic_score ?? '—' }}<span class="text-xs text-slate-500"> / 100</span>
          </div>
          <div class="text-xs text-slate-500 mt-0.5">{{ store.report?.deterministic_level ?? '—' }} (RuleEngine)</div>
        </div>
        <div class="rounded-md border border-edge bg-panel/40 p-3">
          <div class="text-[11px] uppercase tracking-wide text-slate-500">Ajan (LLM) Taslak Skoru</div>
          <div class="mt-1 text-xl font-bold tabular-nums text-slate-200">
            {{ store.report?.llm_proposed_score ?? '—' }}<span class="text-xs text-slate-500"> / 100</span>
          </div>
          <div class="text-xs text-slate-500 mt-0.5">Doğrulanmamış taslak tahmin</div>
        </div>
        <div class="rounded-md border p-3" :class="RISK_TEXT[tone]" :style="{ borderColor: `rgb(var(${RISK_VAR[tone]}) / 0.4)` }">
          <div class="text-[11px] uppercase tracking-wide opacity-70">Nihai (Ortalama) Skor</div>
          <div class="mt-1 text-xl font-bold tabular-nums">
            {{ store.report?.risk_score ?? '—' }}<span class="text-xs opacity-70"> / 100</span>
          </div>
          <div class="text-xs opacity-70 mt-0.5">{{ store.report?.risk_level ?? '—' }}</div>
        </div>
      </div>
      <p v-if="store.report?.risk_explanation" class="mt-3 text-xs text-slate-400">
        {{ store.report.risk_explanation }}
      </p>
    </div>
  </div>
</template>
