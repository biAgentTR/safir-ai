<script setup lang="ts">
// Ana Sayfa — a calm, fixed landing section at the very top of the hub page.
// Purely navigational (no forms, no data tables): a stable "home" the
// operator can always scroll/tab back to, distinct from the old Genel
// Bakış dashboard (removed) and from Yeni Analiz's actual submission form.
// "Yeni Analiz" deliberately has no card here — it's still there as its own
// tab, just not surfaced from the home screen.
const { mode } = useAnalysisMode()
const { requestNewAnalysis } = useVlmDirectReset()
const { goToSection } = useSectionNav()

interface QuickLink {
  id: string
  label: string
  blurb: string
  glyph: string
  /** VLM Direct's card should start a fresh analysis every time, not just scroll to whatever's left on screen. */
  onClick?: () => void
}

const links = computed<QuickLink[]>(() => {
  const base: QuickLink[] = [
    { id: 'gecmis', label: 'Geçmiş', blurb: 'Daha önce çalıştırılmış tüm analizler.', glyph: '≡' },
    { id: 'raporlar', label: 'Raporlar', blurb: 'Tamamlanmış analizlerin risk raporları.', glyph: '▦' },
    { id: 'asistan', label: 'SAFİR Asistan', blurb: 'Analizler ve mevzuat hakkında soru sorun.', glyph: '◆' },
  ]
  if (mode.value === 'vlm_direct') {
    base.unshift({
      id: 'vlm-direct',
      label: 'VLM Direct Analiz',
      blurb: 'Video doğrudan görsel-dil modeline gönderilir.',
      glyph: '▤',
      onClick: requestNewAnalysis,
    })
  }
  return base
})

// Not a plain <NuxtLink to="/#id"> — see composables/useSectionNav.ts: a link
// to a hash we're already sitting on is a silent no-op in Vue Router, which
// is exactly what made this card look broken (mode picker/tab clicks already
// leave the URL on #vlm-direct, so clicking here again did nothing).
function onCardClick(l: QuickLink) {
  l.onClick?.()
  goToSection(l.id)
}
</script>

<template>
  <div id="ana-sayfa" class="scroll-mt-16 relative">
    <div class="ambient-rings" aria-hidden="true">
      <span class="ambient-ring ambient-ring-a" />
      <span class="ambient-ring ambient-ring-b" />
    </div>
    <div class="grid-texture" aria-hidden="true" />

    <div class="relative max-w-4xl mx-auto px-6 py-16 sm:py-20 text-center">
      <img src="~/assets/images/logo.png" alt="" class="w-12 h-12 object-contain mx-auto mb-5" />
      <div class="eyebrow !text-accent mb-3">Yapay zekâ destekli operasyonel farkındalık</div>
      <h1 class="text-3xl sm:text-4xl font-bold tracking-tight text-slate-100 leading-tight">
        Görüntüyü izlemeyin.<br class="hidden sm:block" />
        Ne olduğunu anlayın.
      </h1>
      <p class="mt-3 text-sm sm:text-base text-slate-400 max-w-xl mx-auto">
        SAFİR, saha kamerası görüntülerini analiz eder; kritik anları, riskleri ve uygulanabilir operatör
        aksiyonlarını saniyeler içinde çıkarır. Aşağıdan başlayın.
      </p>

      <div class="mt-10 grid grid-cols-1 sm:grid-cols-2 gap-3 text-left">
        <button
          v-for="l in links"
          :key="l.id"
          type="button"
          class="glass-panel rounded-lg p-4 flex items-start gap-3 hover:border-accent/60 hover:bg-surface-2/40 transition-colors"
          @click="onCardClick(l)"
        >
          <span class="w-9 h-9 rounded-md bg-accent-soft text-accent flex items-center justify-center text-lg shrink-0" aria-hidden="true">{{ l.glyph }}</span>
          <span class="min-w-0">
            <span class="block text-sm font-semibold text-slate-100">{{ l.label }}</span>
            <span class="block mt-0.5 text-xs text-slate-500">{{ l.blurb }}</span>
          </span>
        </button>
      </div>
    </div>
  </div>
</template>
