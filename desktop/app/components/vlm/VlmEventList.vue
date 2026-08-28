<script setup lang="ts">
// Risky-event list for VLM Direct mode: filter by risk level/type, sort by
// timestamp/risk/confidence, and click a row to seek the video (bound to
// VlmVideoPanel via the parent's activeEventId/select-event wiring).
import type { VlmEvent, VlmRiskLevel } from '~/types/vlm'

const props = defineProps<{
  events: VlmEvent[]
  activeEventId: string | null
}>()
const emit = defineEmits<{ (e: 'select', id: string): void }>()

const riskFilter = ref<'all' | VlmRiskLevel>('all')
const typeFilter = ref<'all' | string>('all')
const sortBy = ref<'time' | 'risk' | 'confidence'>('time')

const types = computed(() => [...new Set(props.events.map((e) => e.type))])

const RISK_ORDER: Record<VlmRiskLevel, number> = { crit: 3, high: 2, mid: 1, low: 0 }

const filtered = computed(() => {
  let list = props.events
  if (riskFilter.value !== 'all') list = list.filter((e) => e.riskLevel === riskFilter.value)
  if (typeFilter.value !== 'all') list = list.filter((e) => e.type === typeFilter.value)
  list = [...list]
  if (sortBy.value === 'time') list.sort((a, b) => a.timestamp - b.timestamp)
  else if (sortBy.value === 'risk') list.sort((a, b) => RISK_ORDER[b.riskLevel] - RISK_ORDER[a.riskLevel])
  else list.sort((a, b) => b.confidence - a.confidence)
  return list
})

const RISK_LABEL: Record<VlmRiskLevel, string> = { low: 'Düşük', mid: 'Orta', high: 'Yüksek', crit: 'Kritik' }
</script>

<template>
  <div class="card p-4">
    <div class="flex flex-wrap items-center gap-2 mb-3">
      <h3 class="text-sm font-semibold text-slate-100 mr-auto">Riskli Olaylar ({{ filtered.length }})</h3>
      <select v-model="riskFilter" class="field-input !w-auto !py-1.5" aria-label="Risk seviyesi filtresi">
        <option value="all">Tüm risk seviyeleri</option>
        <option value="crit">Kritik</option>
        <option value="high">Yüksek</option>
        <option value="mid">Orta</option>
        <option value="low">Düşük</option>
      </select>
      <select v-model="typeFilter" class="field-input !w-auto !py-1.5" aria-label="Olay türü filtresi">
        <option value="all">Tüm türler</option>
        <option v-for="t in types" :key="t" :value="t">{{ t }}</option>
      </select>
      <select v-model="sortBy" class="field-input !w-auto !py-1.5" aria-label="Sıralama">
        <option value="time">Zamana göre</option>
        <option value="risk">Risk seviyesine göre</option>
        <option value="confidence">Güven skoruna göre</option>
      </select>
    </div>

    <div v-if="!filtered.length" class="py-8 text-center text-sm text-slate-500">Bu filtrelerle eşleşen olay yok.</div>

    <!-- table (md+) -->
    <div v-else class="hidden md:block overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-[11px] uppercase tracking-wide text-slate-500 border-b border-edge">
            <th class="py-2 pr-3 font-medium">Zaman</th>
            <th class="py-2 pr-3 font-medium">Tür</th>
            <th class="py-2 pr-3 font-medium">Açıklama</th>
            <th class="py-2 pr-3 font-medium">Risk</th>
            <th class="py-2 pr-3 font-medium">Güven</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-edge">
          <tr
            v-for="ev in filtered"
            :key="ev.id"
            class="cursor-pointer hover:bg-surface-2/60 transition-colors"
            :class="activeEventId === ev.id ? 'bg-accent-soft/40' : ''"
            @click="emit('select', ev.id)"
          >
            <td class="py-2 pr-3 font-mono text-xs text-slate-400 whitespace-nowrap">{{ mmss(ev.timestamp) }}</td>
            <td class="py-2 pr-3 text-slate-200 whitespace-nowrap">{{ ev.type }}</td>
            <td class="py-2 pr-3 text-slate-400 max-w-md truncate">{{ ev.description }}</td>
            <td class="py-2 pr-3 whitespace-nowrap">
              <span class="text-xs font-semibold" :class="RISK_TEXT[ev.riskLevel]">{{ RISK_LABEL[ev.riskLevel] }}</span>
            </td>
            <td class="py-2 pr-3 text-xs text-slate-400 whitespace-nowrap">%{{ ev.confidence }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- cards (mobile) -->
    <ul v-if="filtered.length" class="md:hidden space-y-2">
      <li v-for="ev in filtered" :key="ev.id">
        <button
          type="button"
          class="w-full text-left rounded-md border border-edge px-3 py-2.5 transition-colors"
          :class="activeEventId === ev.id ? 'bg-accent-soft/40 border-accent/40' : 'hover:bg-surface-2/60'"
          @click="emit('select', ev.id)"
        >
          <div class="flex items-center justify-between gap-2">
            <span class="text-xs font-mono text-slate-400">{{ mmss(ev.timestamp) }}</span>
            <span class="text-xs font-semibold" :class="RISK_TEXT[ev.riskLevel]">{{ RISK_LABEL[ev.riskLevel] }}</span>
          </div>
          <div class="mt-1 text-sm text-slate-200">{{ ev.type }}</div>
          <div class="mt-0.5 text-xs text-slate-400">{{ ev.description }}</div>
          <div class="mt-1 text-[11px] text-slate-500">Güven: %{{ ev.confidence }}</div>
        </button>
      </li>
    </ul>
  </div>
</template>
