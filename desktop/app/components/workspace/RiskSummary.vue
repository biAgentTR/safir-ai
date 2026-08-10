<script setup lang="ts">
// Prominent risk summary. Appears once the final report (risk) is available.
// Includes the critical banner (>=70) and the Human-on-the-Loop acknowledge
// action for auto-dispatched alerts.
const store = useAnalysisStore()
const ackNote = ref('')

const tone = computed(() => riskTone(store.report?.risk_level))
const headline = computed(() => {
  const t = tone.value
  if (t === 'crit') return 'CRITICAL RISK'
  if (t === 'high') return 'HIGH RISK'
  if (t === 'mid') return 'ELEVATED RISK'
  return 'LOW RISK'
})
</script>

<template>
  <div v-if="store.report" class="space-y-3">
    <!-- critical flashing banner (>=70), mirrors Streamlit -->
    <div
      v-if="store.isCritical"
      class="rounded-md border border-risk-crit/50 bg-risk-crit/15 px-4 py-2.5 text-sm font-medium text-risk-crit flex items-center gap-2 animate-pulse"
    >
      🚨 KRİTİK RİSK TESPİT EDİLDİ — otomatik saha alarmı tetiklendi.
    </div>

    <div class="card p-5 flex flex-col md:flex-row md:items-center gap-5">
      <div class="flex items-center gap-5">
        <div class="text-center">
          <div class="text-5xl font-bold leading-none" :class="RISK_TEXT[tone]">{{ store.report.risk_score }}</div>
          <div class="mt-1 text-xs text-slate-500">/ 100</div>
        </div>
        <div>
          <div class="text-lg font-semibold tracking-wide" :class="RISK_TEXT[tone]">{{ headline }}</div>
          <div class="text-sm text-slate-400 mt-0.5">{{ store.report.risk_level }}</div>
          <p class="mt-2 text-sm text-slate-300 max-w-xl">
            {{ store.report.recommended_action || 'Operatör değerlendirmesi önerilir.' }}
          </p>
        </div>
      </div>

      <!-- acknowledge (only for auto-dispatched alerts) -->
      <div v-if="store.hasAutoAlert" class="md:ml-auto md:w-72 shrink-0 space-y-2">
        <div class="text-[11px] text-slate-500 font-mono truncate">alert_id: {{ store.report.alert_id }}</div>
        <input v-model="ackNote" class="field-input" placeholder="Operatör denetim notu (opsiyonel)" />
        <button
          class="btn-primary w-full"
          :disabled="store.ack.state === 'pending' || store.ack.state === 'ok'"
          @click="store.acknowledgeAlert(ackNote)"
        >
          {{ store.ack.state === 'ok' ? '✓ Onaylandı' : store.ack.state === 'pending' ? 'Gönderiliyor…' : '👁 Alarmı Denetle / Onayla' }}
        </button>
        <p v-if="store.ack.message" class="text-xs" :class="store.ack.state === 'error' ? 'text-risk-crit' : 'text-risk-low'">
          {{ store.ack.message }}
        </p>
      </div>
    </div>
  </div>
</template>
