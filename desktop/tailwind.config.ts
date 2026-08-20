import type { Config } from 'tailwindcss'

// SAFIR desktop theme: sober, technical, information-dense. Not a colorful
// admin dashboard — a slate/steel base with a single restrained accent.
//
// Every color below resolves through a CSS custom property (see
// app/assets/css/main.css) so the same class names (bg-surface-1,
// text-slate-400, text-risk-crit, ...) render correctly in both the dark
// (operations-room) and light (daylight/report-review) themes without any
// call site needing a `dark:` variant.
const token = (name: string) => `rgb(var(${name}) / <alpha-value>)`

export default <Partial<Config>>{
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
          0: token('--c-bg'),
          1: token('--c-surface-1'),
          2: token('--c-surface-2'),
          3: token('--c-surface-3'),
        },
        edge: token('--c-edge'),
        accent: {
          DEFAULT: token('--c-accent'),
          soft: token('--c-accent-soft'),
        },
        risk: {
          low: token('--c-risk-low'),
          mid: token('--c-risk-mid'),
          high: token('--c-risk-high'),
          crit: token('--c-risk-crit'),
        },
        // Overrides only the shades actually used across the app (100-700);
        // untouched shades (50, 800, 900) keep Tailwind's static defaults.
        slate: {
          100: token('--c-slate-100'),
          200: token('--c-slate-200'),
          300: token('--c-slate-300'),
          400: token('--c-slate-400'),
          500: token('--c-slate-500'),
          600: token('--c-slate-600'),
          700: token('--c-slate-700'),
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
}
