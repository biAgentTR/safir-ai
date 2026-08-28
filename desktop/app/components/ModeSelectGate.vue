<script setup lang="ts">
// First-launch, full-screen mode picker. Blocks the entire app (rendered
// instead of NuxtLayout/NuxtPage by app.vue) until the operator has chosen
// an analysis mode at least once. After that, useAnalysisMode.hasChosen
// stays true (persisted in localStorage) and this never shows again — the
// AppTopbar ModeSwitcher is how the operator changes mode afterwards.
import type { AnalysisMode } from '~/composables/useAnalysisMode'

const { setMode } = useAnalysisMode()
const router = useRouter()

function choose(mode: AnalysisMode) {
  setMode(mode)
  router.push({ path: '/', hash: mode === 'vlm_direct' ? '#vlm-direct' : '#ana-sayfa' })
}

const options: { mode: AnalysisMode; title: string; blurb: string; points: string[]; icon: string }[] = [
  {
    mode: 'low_budget',
    title: 'Lite Analiz',
    blurb: 'Uyarlanabilir kare örnekleme tabanlı hafif analiz sistemi.',
    points: [
      'Uyarlanabilir kare örnekleme (Adaptive Sampler)',
      'Düşük donanım ve kaynak gereksinimi',
      'Hızlı ve optimize analiz akışı',
    ],
    icon: '▤',
  },
  {
    mode: 'vlm_direct',
    title: 'Direct Analiz',
    blurb: 'Videonun doğrudan görsel-dil modeline (VLM) aktarıldığı analiz sistemi.',
    points: [
      'Video doğrudan görsel-dil modeline gönderilir',
      'Zaman çizelgesi üzerinde işaretlenmiş riskli anlar',
      'Detaylı olay listesi ve risk dağılım grafikleri',
    ],
    icon: '◆',
  },
]
</script>

<template>
  <div class="fixed inset-0 z-50 flex flex-col items-center justify-center bg-surface-0 px-6 py-10 overflow-y-auto">
    <div class="w-full max-w-3xl">
      <div class="text-center mb-8">
        <img src="~/assets/images/logo.png" alt="SAFİR" class="w-14 h-14 object-contain mx-auto mb-4" />
        <h1 class="text-2xl font-bold tracking-tight text-slate-100">Analiz Modu Seç</h1>
        <p class="mt-2 text-sm text-slate-500 max-w-md mx-auto">
          Bu seçimi istediğiniz zaman uygulama içinden (üst çubuktaki mod anahtarından) değiştirebilirsiniz.
        </p>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <button
          v-for="o in options"
          :key="o.mode"
          type="button"
          class="text-left card p-5 hover:border-accent/60 hover:bg-surface-2/60 transition-colors focus-visible:border-accent"
          @click="choose(o.mode)"
        >
          <div class="flex items-center gap-3 mb-2">
            <span class="w-9 h-9 rounded-md bg-accent-soft text-accent flex items-center justify-center text-lg shrink-0" aria-hidden="true">{{ o.icon }}</span>
            <h2 class="text-base font-semibold text-slate-100">{{ o.title }}</h2>
          </div>
          <p class="text-sm text-slate-400 mb-3">{{ o.blurb }}</p>
          <ul class="space-y-1.5">
            <li v-for="p in o.points" :key="p" class="flex items-start gap-2 text-xs text-slate-400">
              <span class="text-accent mt-0.5" aria-hidden="true">·</span>
              <span>{{ p }}</span>
            </li>
          </ul>
          <div class="mt-4 text-xs font-medium text-accent">Bu modu seç →</div>
        </button>
      </div>
    </div>
  </div>
</template>
