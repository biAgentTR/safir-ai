# SAFİR Desktop (Tauri 2 + Nuxt 3)

Modern desktop frontend for SAFİR. Runs **in parallel** with the existing
Streamlit dashboard and talks to the **unchanged** FastAPI backend over
HTTP + SSE.

> Scope of this step (ADIM 3): application skeleton + real backend connection.
> History, "Ask SAFİR" / Assistant, and the rich pipeline timeline are **not**
> implemented here.

## Stack

- Tauri 2 (Rust shell, native window + file dialog)
- Nuxt 3 / Vue 3 / TypeScript (SPA, `ssr: false`)
- Tailwind CSS
- Pinia

## Layout

```
desktop/
  nuxt.config.ts          # SPA + Nitro devProxy to FastAPI (no CORS change)
  tailwind.config.ts
  app/
    app.vue
    layouts/default.vue   # sidebar + topbar shell
    pages/
      index.vue           # Overview
      new-analysis.vue    # POST /analyze/jobs
      workspace/[jobId].vue  # live SSE trace + final report
    components/           # AppSidebar, AppTopbar, StageList
    composables/          # useSafirApi, useAnalysisStream, useBackendHealth
    stores/analysis.ts    # Pinia job lifecycle
    types/api.ts          # backend contract types
  src-tauri/              # Tauri 2 (Cargo.toml, tauri.conf.json, src/)
```

## How it talks to the backend (no backend change)

In dev, Nitro's `devProxy` forwards `/health` and `/analyze/**` to
`http://localhost:8000`. The browser/webview therefore makes **same-origin**
requests, so `fetch`, `EventSource` (SSE) and `<img>` frame requests all work
without adding CORS to FastAPI.

## Run (development)

Three processes; start the backend first.

```bash
# 1) FastAPI backend (from safir-ai/)
cd ../safir-ai
uvicorn src.main:app --host 0.0.0.0 --port 8000

# 2a) Web dev (browser) — Nuxt only
cd ../desktop
npm install
npm run dev            # http://localhost:3000

# 2b) OR full desktop app (Tauri window; starts Nuxt automatically)
npm run tauri:dev
```

Streamlit is untouched and can keep running on its own port.

## Checks

```bash
npm run build          # Nuxt production build (SPA)
npm run typecheck      # vue-tsc, no type errors
cargo build --manifest-path src-tauri/Cargo.toml   # Tauri Rust build
```

## Not in this step

History backend / persistence, Ask SAFİR, agent/prompt/VLM/RAG/pipeline/SSE
backend changes, Streamlit removal, production packaging (PyInstaller / sidecar).
