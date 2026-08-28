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
    { id: 'asistan', label: 'SAFİR Asistan', blurb: 'Analizler ve mevzuat hakkında soru sorun.', glyph: '◆' },
    { id: 'raporlar', label: 'Raporlar', blurb: 'Tamamlanmış analizlerin risk raporları.', glyph: '▦' },
  ]
  if (mode.value === 'vlm_direct') {
    base.unshift({
      id: 'vlm-direct',
      label: 'Direct Analiz',
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
    <div class="relative max-w-4xl mx-auto px-6 py-16 sm:py-20 text-center">
      <!-- Hero Title Ambient Glow Aura -->
      <div class="pointer-events-none absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[520px] max-w-[90vw] h-[240px] rounded-full bg-gradient-to-r from-accent/25 via-cyan-500/20 to-teal-400/25 dark:from-accent/20 dark:via-cyan-500/15 dark:to-teal-500/20 blur-[100px] -z-10" />

      <img src="~/assets/images/logo.png" alt="" class="w-12 h-12 object-contain mx-auto mb-5 relative z-10" />
      <div class="eyebrow !text-accent mb-3 relative z-10">Yapay zekâ destekli operasyonel farkındalık</div>
      <h1 class="text-3xl sm:text-4xl font-bold tracking-tight text-slate-100 leading-tight relative z-10">
        Görüntüyü izlemeyin.<br class="hidden sm:block" />
        Ne olduğunu anlayın.
      </h1>
      <p class="mt-3 text-sm sm:text-base text-slate-400 max-w-xl mx-auto relative z-10">
        SAFİR, saha kamerası görüntülerini analiz eder; kritik anları, riskleri ve uygulanabilir operatör
        aksiyonlarını saniyeler içinde çıkarır. Aşağıdan başlayın.
      </p>

      <div class="mt-10 grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-left relative z-10">
        <button
          v-for="l in links"
          :key="l.id"
          type="button"
          class="relative overflow-hidden group rounded-xl p-4.5 flex items-start justify-between gap-3.5 bg-surface-1/90 backdrop-blur-md border border-edge/80 hover:border-accent/60 shadow-lg hover:shadow-[0_12px_28px_-6px_rgba(20,184,166,0.18)] hover:-translate-y-0.5 transition-all duration-300 cursor-pointer text-left"
          @click="onCardClick(l)"
        >
          <!-- Top Luminous Hairline Accent -->
          <div class="absolute inset-x-0 top-0 h-[1.5px] bg-gradient-to-r from-transparent via-accent/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

          <div class="flex items-start gap-3.5 min-w-0">
            <span class="w-10 h-10 rounded-lg bg-accent/15 border border-accent/30 text-accent flex items-center justify-center text-lg shrink-0 shadow-[0_0_12px_rgba(20,184,166,0.2)] group-hover:scale-105 group-hover:bg-accent/25 transition-all duration-200" aria-hidden="true">{{ l.glyph }}</span>
            <span class="min-w-0">
              <span class="block text-sm font-semibold text-slate-100 group-hover:text-accent transition-colors">{{ l.label }}</span>
              <span class="block mt-0.5 text-xs text-slate-400 group-hover:text-slate-300 transition-colors leading-relaxed">{{ l.blurb }}</span>
            </span>
          </div>

          <span class="text-xs text-slate-500 group-hover:text-accent group-hover:translate-x-1 transition-all duration-200 shrink-0 mt-0.5">→</span>
        </button>
      </div>
    </div>
  </div>
</template>
