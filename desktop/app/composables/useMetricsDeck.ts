/**
 * Sol alttaki ölçüm destesi (AI Metrikleri ↔ KPI Metrikleri) durumu.
 *
 * TripleDrawerSection'daki 3'lü çekmece deseninin iki kartlık karşılığı:
 * üstte segment anahtarı, altta yatay kayan kart rayı. Üst çubuktaki
 * "AI Metrikleri" / "KPI Metrikleri" düğmelerinden hangisine basılırsa deste
 * o kartla açılır; aynı düğmeye tekrar basmak desteyi kapatır.
 */
export type MetricsSlideId = 'ai' | 'kpi'

export const METRICS_SLIDE_ORDER: MetricsSlideId[] = ['ai', 'kpi']

const isDeckOpen = ref(false)
const activeSlide = ref<MetricsSlideId>('ai')
const isTransitioning = ref(false)

let transitionTimer: ReturnType<typeof setTimeout> | null = null

export function useMetricsDeck() {
  function setSlide(id: MetricsSlideId) {
    if (activeSlide.value === id) return
    isTransitioning.value = true
    activeSlide.value = id
    if (transitionTimer) clearTimeout(transitionTimer)
    transitionTimer = setTimeout(() => {
      isTransitioning.value = false
    }, 520)
  }

  /** Üst çubuk düğmeleri: basılan kart açılır, açıksa ve aynı kartsa kapanır. */
  function toggleSlide(id: MetricsSlideId) {
    if (isDeckOpen.value && activeSlide.value === id) {
      closeDeck()
      return
    }
    isDeckOpen.value = true
    setSlide(id)
  }

  function closeDeck() {
    isDeckOpen.value = false
    isTransitioning.value = false
    if (transitionTimer) clearTimeout(transitionTimer)
  }

  const activeIndex = computed(() => Math.max(0, METRICS_SLIDE_ORDER.indexOf(activeSlide.value)))

  function isActive(id: MetricsSlideId) {
    return isDeckOpen.value && activeSlide.value === id
  }

  return {
    isDeckOpen,
    activeSlide,
    activeIndex,
    isTransitioning,
    setSlide,
    toggleSlide,
    closeDeck,
    isActive,
  }
}
