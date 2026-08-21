import { computed, ref, useState } from '#imports'

export type ThemeMode = 'light' | 'dark'

export function useTheme() {
  const theme = useState<ThemeMode>('safir_theme', () => 'light')

  function initTheme() {
    if (process.client) {
      const saved = localStorage.getItem('safir_theme') as ThemeMode
      if (saved === 'dark' || saved === 'light') {
        theme.value = saved
      } else {
        theme.value = 'light' // Varsayılan tema: açık/beyaz
      }
      applyTheme(theme.value)
    }
  }

  function toggleTheme() {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
    if (process.client) {
      localStorage.setItem('safir_theme', theme.value)
      applyTheme(theme.value)
    }
  }

  function setTheme(newTheme: ThemeMode) {
    theme.value = newTheme
    if (process.client) {
      localStorage.setItem('safir_theme', newTheme)
      applyTheme(newTheme)
    }
  }

  function applyTheme(t: ThemeMode) {
    if (process.client) {
      const htmlEl = document.documentElement
      if (t === 'dark') {
        htmlEl.classList.add('dark')
        htmlEl.setAttribute('data-theme', 'dark')
      } else {
        htmlEl.classList.remove('dark')
        htmlEl.setAttribute('data-theme', 'light')
      }
    }
  }

  return {
    theme,
    initTheme,
    toggleTheme,
    setTheme,
    isDark: computed(() => theme.value === 'dark'),
  }
}
