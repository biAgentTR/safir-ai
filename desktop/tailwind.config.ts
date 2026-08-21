import type { Config } from 'tailwindcss'

// SAFIR Desktop Theme Configuration with class-based Dark Mode & CSS Variables
export default <Partial<Config>>{
  darkMode: 'class',
  content: [
    './app/components/**/*.{vue,ts}',
    './app/layouts/**/*.vue',
    './app/pages/**/*.vue',
    './app/composables/**/*.ts',
    './app/app.vue',
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          0: 'var(--bg-surface-0)',
          1: 'var(--bg-surface-1)',
          2: 'var(--bg-surface-2)',
          3: 'var(--bg-surface-3)',
        },
        edge: 'var(--border-edge)',
        accent: {
          DEFAULT: 'var(--accent-primary)',
          soft: 'var(--accent-hover)',
        },
        risk: {
          low: 'var(--risk-low)',
          mid: 'var(--risk-mid)',
          high: 'var(--risk-high)',
          crit: 'var(--risk-crit)',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
}
