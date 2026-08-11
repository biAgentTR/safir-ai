<script setup lang="ts">
// User-friendly final report: executive summary, risk recap, recommended
// actions, RAG regulations, escalation (+ manual override trigger), technical
// metrics, and export (JSON / HTML / PDF, client-side from real data).
// Timeline and Evidence live in their own tabs.
const store = useAnalysisStore()
const { exportJson, exportHtml, exportPdf } = useReportExport()
const manualNote = ref('')

const r = computed(() => store.report)
const tone = computed(() => riskTone(r.value?.risk_level))
</script>

<template>
  <div v-if="r" class="space-y-6">
    <!-- executive summary -->
    <section>
      <div class="field-label">Executive summary</div>
      <p class="text-sm text-slate-200 leading-relaxed">{{ r.summary || r.natural_language_summary || '—' }}</p>
    </section>

    <div class="grid md:grid-cols-2 gap-6">
      <!-- risk + actions -->
      <section class="space-y-4">
        <div>
          <div class="field-label">Risk</div>
          <div class="text-2xl font-bold" :class="RISK_TEXT[tone]">{{ r.risk_score }} / 100 · <span class="uppercase text-base">{{ r.risk_level }}</span></div>
        </div>
        <div>
          <div class="field-label">Önerilen aksiyonlar</div>
          <ol v-if="r.actions?.length" class="list-decimal list-inside space-y-1 text-sm text-slate-200">
            <li v-for="(a, i) in r.actions" :key="i">{{ a }}</li>
          </ol>
          <p v-else class="text-sm text-slate-400">{{ r.recommended_action || '—' }}</p>
        </div>
      </section>

      <!-- RAG regulations -->
      <section>
        <div class="field-label">İlgili İSG mevzuatı (RAG / FAISS)</div>
        <ul v-if="r.relevant_regulations?.length" class="space-y-1 text-sm text-slate-200">
          <li v-for="(reg, i) in r.relevant_regulations" :key="i" class="bg-surface-2 border border-edge rounded-md px-3 py-2">{{ reg }}</li>
        </ul>
        <p v-else class="text-sm text-slate-400">Bu analiz için ilgili mevzuat maddesi bulunamadı.</p>
      </section>
    </div>

    <!-- escalation + manual override -->
    <section class="card p-4 bg-surface-2/40">
      <div class="field-label">Eskalasyon (Human-on-the-Loop)</div>
      <p class="text-sm text-slate-200">
        Kademe: <b>{{ r.escalation_tier ?? '—' }}</b>
        · otomatik tetik: {{ r.auto_dispatched ? 'evet' : 'hayır' }}
        <span v-if="r.alert_id" class="font-mono text-slate-400"> · {{ r.alert_id }}</span>
      </p>
      <details class="mt-3">
        <summary class="cursor-pointer text-xs text-slate-400">Manuel saha alarmı tetikle (override)</summary>
        <div class="mt-2 flex flex-col sm:flex-row gap-2">
          <input v-model="manualNote" class="field-input" placeholder="Manuel alarm notu (opsiyonel)" />
          <button
            class="btn-ghost shrink-0"
            :disabled="store.manualAlert.state === 'pending'"
            @click="store.triggerManualAlert(manualNote)"
          >Manuel Alarm Tetikle</button>
        </div>
        <p v-if="store.manualAlert.message" class="mt-2 text-xs" :class="store.manualAlert.state === 'error' ? 'text-risk-crit' : 'text-risk-low'">
          {{ store.manualAlert.message }}
        </p>
      </details>
    </section>

    <!-- technical metrics -->
    <section>
      <div class="field-label">Teknik metrikler</div>
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCell label="VLM" :value="r.vlm_model ?? '—'" mono />
        <MetricCell label="LLM" :value="r.llm_model ?? '—'" mono />
        <MetricCell label="Event id" :value="r.event_id ?? '—'" mono />
        <MetricCell label="Auto dispatched" :value="r.auto_dispatched ? 'evet' : 'hayır'" />
      </div>
    </section>

    <!-- technical JSON (real SafirReport; no raw_response/secret/reasoning) -->
    <section>
      <details>
        <summary class="cursor-pointer text-xs text-slate-400">Technical JSON (tam SafirReport)</summary>
        <pre class="mt-2 text-[11px] font-mono text-slate-400 bg-surface-2 border border-edge rounded-md p-3 max-h-96 overflow-auto">{{ JSON.stringify(r, null, 2) }}</pre>
      </details>
    </section>

    <!-- export -->
    <section class="flex items-center gap-3 pt-2 border-t border-edge">
      <span class="text-xs text-slate-500">Dışa aktar:</span>
      <button class="btn-ghost" @click="exportJson(r)">JSON</button>
      <button class="btn-ghost" @click="exportHtml(r)">HTML</button>
      <button class="btn-ghost" @click="exportPdf(r)">PDF</button>
      <span class="text-[11px] text-slate-600 ml-2">(gerçek rapor verisinden, istemci tarafında)</span>
    </section>
  </div>

  <div v-else class="text-sm text-slate-500 py-8 text-center">
    Nihai rapor henüz hazır değil.
  </div>
</template>
