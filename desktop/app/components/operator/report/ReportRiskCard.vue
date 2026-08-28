<script setup lang="ts">
import { computed } from "vue"

const props = defineProps<{
  score: number | null
  level: string | null
  explanation: string | null
  summary: string | null
  title?: string | null
  recommendedAction: string | null
  actions: string[]
}>()

const emit = defineEmits<{
  (e: 'ask-safir'): void
}>()

const levelLabel = computed(() => {
  const l = props.level?.toLowerCase()
  if (l === "dusuk" || l === "low") return "Düşük"
  if (l === "orta" || l === "medium") return "Orta"
  if (l === "yuksek" || l === "high") return "Yüksek"
  if (l === "kritik" || l === "critical") return "Kritik"
  return "Bilinmiyor"
})

const isDanger = computed(() => {
  const l = levelLabel.value
  return l === "Yüksek" || l === "Kritik"
})

const isMedium = computed(() => {
  return levelLabel.value === "Orta"
})

const isLow = computed(() => {
  return levelLabel.value === "Düşük"
})

const actionList = computed(() => {
  const list: string[] = []
  if (props.recommendedAction) {
    list.push(props.recommendedAction)
  }
  if (props.actions && props.actions.length > 0) {
    props.actions.forEach(a => {
      if (!list.includes(a)) {
        list.push(a)
      }
    })
  }
  return list.slice(0, 3)
})

</script>

<template>
  <div 
    class="flex flex-col h-full rounded-[16px] border p-6 shadow-sm overflow-hidden"
    :class="[
      isDanger 
        ? 'bg-[#0f0709] border-[#ff7f91]/30' 
        : isMedium 
          ? 'bg-[#0f0a05] border-amber-500/30'
          : isLow
            ? 'bg-[#050f0a] border-emerald-500/30'
            : 'bg-[var(--color-surface)] border-[var(--color-border)]'
    ]"
  >
    <!-- Header: Title & Severity Badge -->
    <div class="flex items-start justify-between mb-4">
      <div class="flex flex-col">
        <div class="text-[11px] font-bold text-[var(--color-text-muted)] tracking-wider uppercase mb-1">
          Genel Risk Değerlendirmesi
        </div>
        <h2 class="text-xl font-bold text-[var(--color-text)]">
          {{ title || 'Olay Tespit Edildi' }}
        </h2>
      </div>
      <div 
        class="px-2.5 py-1 text-xs font-bold rounded-md border"
        :class="[
          isDanger 
            ? 'bg-[#ff7f91]/10 text-[#ff7f91] border-[#ff7f91]/20' 
            : isMedium 
              ? 'bg-amber-500/10 text-amber-500 border-amber-500/20'
              : isLow
                ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
                : 'bg-slate-500/10 text-slate-400 border-slate-500/20'
        ]"
      >
        {{ levelLabel }}
      </div>
    </div>

    <!-- Summary (Line clamped) -->
    <div class="mb-5 text-[13px] leading-relaxed text-[var(--color-text-secondary)] line-clamp-3 lg:line-clamp-4">
      {{ summary || explanation || 'Detaylı analiz açıklaması bulunamadı.' }}
    </div>

    <!-- Actions -->
    <div class="mt-auto pt-5 border-t border-[var(--color-border)]/50">
      <div class="text-[10px] font-bold text-[var(--color-text-muted)] tracking-wider uppercase mb-3">
        Operatör Aksiyonu
      </div>
      <div class="space-y-2 mb-4">
        <template v-if="actionList.length > 0">
          <div 
            v-for="(action, idx) in actionList" 
            :key="idx"
            class="flex items-start gap-2.5 text-[13px]"
          >
            <div 
              class="w-5 h-5 rounded flex items-center justify-center shrink-0 mt-0.5 text-[10px] font-bold"
              :class="isDanger ? 'bg-[#ff7f91]/10 text-[#ff7f91]' : 'bg-[var(--color-primary)]/10 text-[var(--color-primary)]'"
            >
              {{ idx + 1 }}
            </div>
            <span class="text-[var(--color-text)]">{{ action }}</span>
          </div>
        </template>
        <template v-else>
          <div class="text-sm text-[var(--color-text-secondary)] italic">
            Önerilen aksiyon bulunmuyor.
          </div>
        </template>
      </div>
      
      <!-- Ask SAFIR Button -->
      <button 
        @click="emit('ask-safir')"
        class="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-colors"
        :class="isDanger ? 'bg-[#ff7f91]/10 text-[#ff7f91] hover:bg-[#ff7f91]/20' : 'bg-[var(--color-primary)]/10 text-[var(--color-primary)] hover:bg-[var(--color-primary)]/20'"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        SAFİR'e Sor
      </button>
    </div>
  </div>
</template>