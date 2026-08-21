<script setup lang="ts">
// Prominent risk summary. Appears once the final report (risk) is available.
// Includes the critical banner (>=70), unclassified warning banner (8-ISG rule exception),
// and the Human-on-the-Loop acknowledge action for auto-dispatched alerts.
const store = useAnalysisStore()
const ackNote = ref('')

const r = computed(() => store.report)

const isUnclassified = computed(() =>
  r.value?.risk_status === 'unclassified' ||
  r.value?.detected_event_types?.includes('siniflandirilamadi')
)

const isUnknownRisk = computed(() =>
  !isUnclassified.value && (r.value?.risk_status === 'unknown' || r.value?.risk_score == null)
)

const tone = computed(() => {
  if (isUnclassified.value) return 'unclassified'
  if (isUnknownRisk.value) return 'unknown'
  return riskTone(r.value?.risk_level)
})

const headline = computed(() => {
  if (isUnclassified.value) return 'SINIFLANDIRILAMAYAN RİSK'
  if (isUnknownRisk.value) return 'RİSK BELİRSİZ'
  const t = tone.value
  if (t === 'crit') return 'CRITICAL RISK'
  if (t === 'high') return 'HIGH RISK'
  if (t === 'mid') return 'ELEVATED RISK'
  return 'LOW RISK'
})
const api = useSafirApi()
const showManualModal = ref(false)
const manualNote = ref('')
const manualTriggerState = reactive({
  pending: false,
  message: null as string | null,
  alertId: null as string | null,
})

async function triggerManualAlert() {
  manualTriggerState.pending = true
  manualTriggerState.message = null
  try {
    const res = await api.triggerAlert({
      risk_score: 95,
      risk_level: 'kritik',
      recommended_action: 'Operatör tarafından manuel saha alarmı tetiklendi. Alanı derhal kontrol edin.',
      operator_note: manualNote.value || 'Operatör manuel müdahalesi',
    })
    manualTriggerState.alertId = res.alert_id
    manualTriggerState.message = `✓ Manuel saha alarmı tetiklendi (ID: ${res.alert_id})`
  } catch (e: unknown) {
    manualTriggerState.message = (e as Error)?.message || 'Alarm tetiklenemedi.'
  } finally {
    manualTriggerState.pending = false
  }
}
</script>

<template>
  <div v-if="r" class="space-y-3">
    <!-- critical flashing banner (>=70) -->
    <div
      v-if="store.isCritical"
      class="rounded-xl border border-rose-500/80 bg-gradient-to-r from-rose-950/80 to-red-900/60 backdrop-blur-xl px-5 py-3 text-sm font-bold text-rose-200 flex items-center gap-3 shadow-xl glow-crit animate-pulse"
    >
      <span class="text-xl">🚨</span>
      <span>KRİTİK RİSK TESPİT EDİLDİ — Otomatik saha alarmı yayına alındı.</span>
    </div>

    <!-- unclassified risk banner: event detected but not fitting the 8 core ISG rules -->
    <div
      v-if="isUnclassified"
      class="rounded-xl border border-amber-500/60 bg-gradient-to-r from-amber-950/80 to-amber-900/40 backdrop-blur-xl px-5 py-3.5 text-sm font-medium text-amber-200 flex items-start gap-3 shadow-xl glow-amber"
    >
      <span class="text-xl shrink-0">🏷️</span>
      <div>
        <div class="font-bold text-amber-100 text-base">8 Ana İSG Kuralı Dışında Şüpheli Risk / İnceleme Bekliyor</div>
        <p class="text-amber-200/90 text-xs mt-1 leading-relaxed">
          Sahada anormallik/şüpheli olay tespit edilmiştir ancak tanımlı 8 temel İSG mevzuat kategorisinden birine eşleştirilememiştir. Risk skoru <b>null (belirtilmedi)</b> olarak işaretlenmiş olup 0 (risk yok/rutin) durumundan ayrılmıştır. Manuel inceleme önerilir.
        </p>
      </div>
    </div>

    <!-- unknown-risk banner: decision parsing failed -->
    <div
      v-else-if="isUnknownRisk"
      class="rounded-xl border border-slate-500/50 bg-surface-2/80 px-4 py-3 text-sm font-medium text-slate-300 flex items-center gap-2"
    >
      ⚠️ Risk değerlendirilemedi — analiz güvenilir bir karar üretemedi. Manuel inceleme gerekli.
    </div>

    <div
      class="card p-6 flex flex-col md:flex-row md:items-center gap-6 shadow-2xl"
      :class="[
        store.isCritical ? 'glow-crit border-rose-500/60 bg-rose-950/20' : '',
        isUnclassified ? 'glow-amber border-amber-500/40 bg-amber-500/5' : '',
        tone === 'low' ? 'glow-emerald border-emerald-500/30' : ''
      ]"
    >
      <div class="flex items-center gap-6">
        <div class="text-center shrink-0 p-4 rounded-2xl bg-surface-2/80 border border-edge/80 min-w-[100px]">
          <div
            class="text-5xl font-black leading-none font-mono"
            :class="isUnclassified ? 'text-amber-400' : RISK_TEXT[tone]"
          >
            {{ isUnclassified ? 'null' : (isUnknownRisk ? '—' : r.risk_score) }}
          </div>
          <div class="mt-1.5 text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold">
            {{ isUnclassified ? 'skor yok' : (isUnknownRisk ? 'belirsiz' : '/ 100') }}
          </div>
        </div>

        <div>
          <div
            class="text-lg font-semibold tracking-wide"
            :class="isUnclassified ? 'text-amber-300' : RISK_TEXT[tone]"
          >
            {{ headline }}
          </div>
          <div class="flex items-center gap-2 mt-0.5">
            <span class="text-sm text-slate-400 font-medium">{{ r.risk_level }}</span>
            <span
              v-if="r.confidence"
              class="text-[11px] px-2 py-0.5 rounded font-mono font-semibold uppercase tracking-wider border"
              :class="
                r.confidence === 'yuksek'
                  ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400'
                  : r.confidence === 'orta'
                  ? 'border-amber-500/40 bg-amber-500/10 text-amber-400'
                  : 'border-rose-500/40 bg-rose-500/10 text-rose-400'
              "
            >
              Güven: {{ r.confidence }}
            </span>
          </div>
          <p class="mt-2 text-sm text-slate-300 max-w-xl">
            {{ r.recommended_action || 'Operatör değerlendirmesi önerilir.' }}
          </p>
        </div>
      </div>

      <!-- acknowledge (only for auto-dispatched alerts) -->
      <div v-if="store.hasAutoAlert" class="md:ml-auto md:w-72 shrink-0 space-y-2">
        <div class="text-[11px] text-slate-500 font-mono truncate">alert_id: {{ r.alert_id }}</div>
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

      <!-- manual override trigger button (when no auto alert) -->
      <div v-else class="md:ml-auto md:w-72 shrink-0 space-y-2">
        <button
          type="button"
          class="w-full px-3 py-2 rounded-md border border-rose-500/40 bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors"
          @click="showManualModal = !showManualModal"
        >
          🚨 Manuel Saha Alarmı Tetikle (Override)
        </button>
        <div v-if="showManualModal" class="p-3 rounded-md border border-edge bg-surface-2 space-y-2 text-xs">
          <input v-model="manualNote" class="field-input text-xs" placeholder="Alarm nedeni / Operatör notu..." />
          <button
            type="button"
            class="btn-primary w-full bg-rose-600 hover:bg-rose-500 text-xs"
            :disabled="manualTriggerState.pending || !!manualTriggerState.alertId"
            @click="triggerManualAlert"
          >
            {{ manualTriggerState.alertId ? '✓ Alarm Gönderildi' : manualTriggerState.pending ? 'Tetikleniyor...' : 'Saha Alarmını Yayına Al' }}
          </button>
          <p v-if="manualTriggerState.message" class="text-[11px] text-emerald-400 font-mono">
            {{ manualTriggerState.message }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>
