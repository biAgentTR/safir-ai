<script setup lang="ts">
// Ana Sayfa — a calm, fixed landing section at the very top of the hub page.
// Purely navigational (no forms, no data tables): a stable "home" the
// operator can always scroll/tab back to, distinct from the old Genel
// Bakış dashboard (removed) and from Yeni Analiz's actual submission form.
const { mode } = useAnalysisMode()

interface QuickLink {
  href: string
  label: string
  blurb: string
  glyph: string
}

const links = computed<QuickLink[]>(() => {
  const base: QuickLink[] = [
    { href: '/#yeni-analiz', label: 'Yeni Analiz', blurb: 'Video seçin, ne aranacağını yazın, analizi başlatın.', glyph: '＋' },
    { href: '/#gecmis', label: 'Geçmiş', blurb: 'Daha önce çalıştırılmış tüm analizler.', glyph: '≡' },
    { href: '/#raporlar', label: 'Raporlar', blurb: 'Tamamlanmış analizlerin risk raporları.', glyph: '▦' },
    { href: '/#asistan', label: 'SAFİR Asistan', blurb: 'Analizler ve mevzuat hakkında soru sorun.', glyph: '◆' },
  ]
  if (mode.value === 'vlm_direct') {
    base.unshift({ href: '/#vlm-direct', label: 'VLM Direct Analiz', blurb: 'Video doğrudan görsel-dil modeline gönderilir.', glyph: '▤' })
  }
  return base
})
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
        <NuxtLink
          v-for="l in links"
          :key="l.href"
          :to="l.href"
          class="glass-panel rounded-lg p-4 flex items-start gap-3 hover:border-accent/60 hover:bg-surface-2/40 transition-colors"
        >
          <span class="w-9 h-9 rounded-md bg-accent-soft text-accent flex items-center justify-center text-lg shrink-0" aria-hidden="true">{{ l.glyph }}</span>
          <span class="min-w-0">
            <span class="block text-sm font-semibold text-slate-100">{{ l.label }}</span>
            <span class="block mt-0.5 text-xs text-slate-500">{{ l.blurb }}</span>
          </span>
        </NuxtLink>
      </div>
    </div>
  </div>
</template>
