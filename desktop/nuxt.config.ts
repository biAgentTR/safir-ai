// SAFIR desktop (Tauri 2 + Nuxt 3) configuration.
//
// - SPA mode (ssr:false): Tauri serves the built frontend as static files.
// - srcDir 'app/': matches the requested desktop/app/{pages,components,...} layout.
// - Nitro devProxy: forwards /health and /analyze/** to the existing FastAPI
//   backend (http://localhost:8000) so the browser/webview talks same-origin.
//   This keeps EventSource (SSE) and <img> frame requests working WITHOUT any
//   CORS change on the backend — the backend is left completely untouched.
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  srcDir: 'app/',
  ssr: false,
  devtools: { enabled: false },

  modules: ['@nuxtjs/tailwindcss', '@pinia/nuxt'],

  experimental: {
    appManifest: false,
  },

  // Flat component names regardless of subfolder (StageCard, not WorkspaceStageCard).
  components: [{ path: '~/components', pathPrefix: false }],

  css: ['~/assets/css/main.css'],

  app: {
    head: {
      title: 'SAFIR',
      meta: [{ name: 'viewport', content: 'width=device-width, initial-scale=1' }],
    },
  },

  // Fixed dev server so Tauri's devUrl (http://localhost:3000) is stable.
  devServer: {
    host: '127.0.0.1',
    port: 3000,
  },

  runtimeConfig: {
    public: {
      // All API traffic is namespaced under /api so it never collides with a
      // Vue page route (e.g. the /history page vs the /history endpoint). The
      // Nitro dev proxy below maps /api/** -> FastAPI. Override in production
      // packaging (out of scope) via NUXT_PUBLIC_API_BASE, e.g. http://localhost:8000.
      apiBase: '/api',
    },
  },

  nitro: {
    // Single rule: Nitro strips the '/api' key, leaving the real backend path
    // (/api/history -> /history, /api/analyze/jobs -> /analyze/jobs, ...). This
    // keeps EventSource (SSE) and <img> frame requests same-origin, no CORS
    // change on the backend, and no page-route / endpoint collision.
    devProxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },

  // Tauri expects a fixed dev server and no clobbered HMR websocket.
  vite: {
    clearScreen: false,
    envPrefix: ['VITE_', 'TAURI_'],
    server: {
      strictPort: true,
    },
  },
})
