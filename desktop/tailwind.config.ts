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
          0: token('--c-bg'),
          1: token('--c-surface-1'),
          2: token('--c-surface-2'),
          3: token('--c-surface-3'),
        },
        edge: {
          DEFAULT: token('--c-edge'),
          strong: token('--c-edge-strong'),
        },
        // Brand/interactive accent — a controlled steel-teal, deliberately NOT
        // the generic blue/purple "AI SaaS" hue. Buttons, active nav, focus
        // rings, links, progress. See main.css for the actual values.
        accent: {
          DEFAULT: token('--c-accent'),
          soft: token('--c-accent-soft'),
        },
        // Semantic "informational" blue — reserved for neutral info tags/
        // banners (never for interactive elements — that's `accent` above).
        info: {
          DEFAULT: token('--c-info'),
          soft: token('--c-info-soft'),
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
      // Tightened, deliberately un-"bubbly" radius scale — an industrial
      // control surface reads sharper than a consumer SaaS app. Overriding
      // these two keys retunes every existing rounded-md/rounded-lg call
      // site app-wide with no per-component edits.
      borderRadius: {
        md: '5px',
        lg: '7px',
      },
    },
  },
}
