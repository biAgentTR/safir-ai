/**
 * Shared state for the brief welcome/transition overlay shown when the
 * operator switches between sidebar panels (see AppSidebar.vue's nav click
 * handler and PanelTransition.vue, the overlay that renders this state).
 *
 * Purely presentational: the route change itself is untouched (NuxtLink
 * navigates normally) — this only times a short overlay on top of it, so
 * there is no double navigation and no change to routing/business logic.
 */
export interface PanelWelcome {
  title: string
  subtitle: string
}

/** Keyed by the exact `to` path used in AppSidebar.vue's nav items. */
const PANEL_WELCOME: Record<string, PanelWelcome> = {
  '/': {
    title: 'Genel Bakış’a Hoş Geldiniz',
    subtitle: 'Analizlerinizi ve kritik olayları tek ekrandan takip edin.',
  },
  '/new-analysis': {
    title: 'Yeni Analize Hoş Geldiniz',
    subtitle: 'Video kaynağınızı seçerek saha analizini başlatın.',
  },
  '/history': {
    title: 'Analiz Geçmişine Hoş Geldiniz',
    subtitle: 'Daha önce gerçekleştirilen analizleri görüntüleyin.',
  },
  '/reports': {
    title: 'Raporlara Hoş Geldiniz',
    subtitle: 'Analiz sonuçlarını ve ayrıntılı risk raporlarını inceleyin.',
  },
  '/assistant': {
    title: 'SAFİR Asistan’a Hoş Geldiniz',
    subtitle: 'Analizleriniz hakkında sorular sorun ve karar desteği alın.',
  },
  '/system': {
    title: 'Sistem Verilerine Hoş Geldiniz',
    subtitle: 'Sistem bileşenlerinin durumunu ve çalışma verilerini takip edin.',
  },
}

const DISPLAY_MS = 800

// Module-scoped (not useState): just an implementation-detail timer handle,
// shared across every usePanelTransition() call site by virtue of living
// outside the exported function, same as AppSplash.vue's local timer.
let timer: ReturnType<typeof setTimeout> | null = null

export function usePanelTransition() {
  const active = useState<boolean>('panel-transition-active', () => false)
  const welcome = useState<PanelWelcome | null>('panel-transition-welcome', () => null)

  /**
   * Call from a sidebar nav click with the link's target path. No-ops for
   * paths outside the 6 sidebar panels (e.g. a dynamic /workspace/:id link),
   * so it never fires anywhere routing wasn't explicitly wired for it.
   */
  function trigger(path: string) {
    const meta = PANEL_WELCOME[path]
    if (!meta) return
    if (timer) clearTimeout(timer)
    welcome.value = meta
    active.value = true
    timer = setTimeout(() => {
      active.value = false
    }, DISPLAY_MS)
  }

  return { active, welcome, trigger }
}
