/**
 * Robust "go to a hub section" navigation, used everywhere a link points at
 * one of the hub page's anchors (#yeni-analiz, #gecmis, #vlm-direct, ...).
 *
 * Why not just `<NuxtLink to="/#id">`: Vue Router only reacts to a route.hash
 * CHANGE. Once you're already sitting on some hash (e.g. #vlm-direct, set by
 * the mode picker or a previous click), clicking a link to that SAME hash
 * again is a no-op navigation — nothing changes, so nothing re-scrolls, and
 * the click silently does nothing. That's exactly what happened with Ana
 * Sayfa's "VLM Direct Analiz" card. goToSection() always scrolls directly
 * (never depends on a hash-diff to trigger it) and only touches the URL to
 * keep it bookmarkable/consistent with the tab bar.
 */
export function useSectionNav() {
  const route = useRoute()
  const router = useRouter()

  function prefersReducedMotion(): boolean {
    return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  }

  function scrollToId(id: string) {
    const el = document.getElementById(id)
    if (!el) return
    el.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'start' })
  }

  async function goToSection(id: string) {
    if (route.path !== '/') {
      await router.push({ path: '/', hash: `#${id}` })
      await nextTick()
      scrollToId(id)
      return
    }
    if (route.hash !== `#${id}`) router.replace({ hash: `#${id}` })
    scrollToId(id)
  }

  return { scrollToId, goToSection }
}
