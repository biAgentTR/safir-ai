import type { Config } from 'tailwindcss'

// SAFIR desktop theme: sober, technical, information-dense. Not a colorful
// admin dashboard — a slate/steel base with a single restrained accent.
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
          0: '#0b0f14',
          1: '#0f141b',
          2: '#151c26',
          3: '#1c2531',
        },
        edge: '#26303d',
        accent: {
          DEFAULT: '#3b82f6',
          soft: '#1e3a5f',
        },
        risk: {
          low: '#22c55e',
          mid: '#eab308',
          high: '#f97316',
          crit: '#ef4444',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
}
