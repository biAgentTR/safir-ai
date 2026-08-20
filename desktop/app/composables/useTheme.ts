/**
 * Shared dark/light theme controller. Backs the header's theme toggle and any
 * future consumer — one source of truth for the resolved theme and the
 * user's stored preference.
 *
 * - 'system' (default): follows prefers-color-scheme, live (no reload).
 * - 'light' / 'dark': explicit choice, persisted in localStorage and applied
 *   as [data-theme] on <html>, which is what app/assets/css/main.css keys off.
 *
 * The actual FIRST paint is handled by a blocking inline script in
 * nuxt.config.ts (runs before Vue mounts) so there is no flash of the wrong
 * theme; init() here just keeps this composable's state in sync afterwards
 * and reacts to OS theme changes while 'system' is selected.
 */
export type ThemePreference = 'system' | 'light' | 'dark'
export type ResolvedTheme = 'light' | 'dark'

const STORAGE_KEY = 'safir-theme'

function systemTheme(): ResolvedTheme {
  if (typeof window === 'undefined') return 'dark'
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

function readStoredPreference(): ThemePreference {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'light' || v === 'dark') return v
  } catch {
    // localStorage unavailable (e.g. privacy mode) — fall back to system.
  }
  return 'system'
}

export function useTheme() {
  const preference = useState<ThemePreference>('safir-theme-preference', () => 'system')
  const resolved = useState<ResolvedTheme>('safir-theme-resolved', () => 'dark')

  function apply(pref: ThemePreference) {
    resolved.value = pref === 'system' ? systemTheme() : pref
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-theme', resolved.value)
    }
  }

  function setPreference(pref: ThemePreference) {
    preference.value = pref
    try {
      if (pref === 'system') localStorage.removeItem(STORAGE_KEY)
      else localStorage.setItem(STORAGE_KEY, pref)
    } catch {
      // best-effort persistence only
    }
    apply(pref)
  }

  /** Cycle: system -> light -> dark -> system. Called by the header toggle. */
  function cycle() {
    const next: Record<ThemePreference, ThemePreference> = {
      system: 'light',
      light: 'dark',
      dark: 'system',
    }
    setPreference(next[preference.value])
  }

  function init() {
    if (typeof window === 'undefined') return
    preference.value = readStoredPreference()
    apply(preference.value)

    const mq = window.matchMedia('(prefers-color-scheme: light)')
    const onChange = () => {
      if (preference.value === 'system') apply('system')
    }
    mq.addEventListener('change', onChange)
    onBeforeUnmount(() => mq.removeEventListener('change', onChange))
  }

  return { preference, resolved, setPreference, cycle, init }
}
