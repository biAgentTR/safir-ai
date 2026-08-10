<script setup lang="ts">
// New Analysis form -> POST /analyze/jobs (real backend, no mock). On success
// navigates to /workspace/{job_id} where the SSE stream is opened.
import type { AnalyzeRequest } from '~/types/api'

const store = useAnalysisStore()
const router = useRouter()

type SourceMode = 'file' | 'stream'
const mode = ref<SourceMode>('file')

const form = reactive({
  videoPath: '',
  streamUrl: '',
  userPrompt: 'Sahnede riskli bir durum var mi degerlendir.',
  sampleFps: 2 as number | null,
  minChangeThreshold: 0.01 as number | null,
})

const submitError = ref<string | null>(null)

const videoSource = computed(() =>
  mode.value === 'file' ? form.videoPath.trim() : form.streamUrl.trim(),
)
const canSubmit = computed(() => videoSource.value.length > 0 && !store.submitting)

// Desktop: pick a real local file path via the Tauri dialog when available;
// fall back to manual path entry in a plain browser (dev without Tauri).
async function pickFile() {
  try {
    // Dynamically imported so a non-Tauri browser build still compiles/runs.
    const dialog = await import('@tauri-apps/plugin-dialog')
    const selected = await dialog.open({
      multiple: false,
      filters: [{ name: 'Video', extensions: ['mp4', 'avi', 'mov', 'mkv', 'webm'] }],
    })
    if (typeof selected === 'string') form.videoPath = selected
  } catch {
    // Not running inside Tauri (or plugin unavailable) — manual entry stays.
  }
}

async function submit() {
  submitError.value = null
  const payload: AnalyzeRequest = {
    video_source: videoSource.value,
    user_prompt: form.userPrompt.trim() || undefined,
    sample_fps: form.sampleFps ?? null,
    min_change_threshold: form.minChangeThreshold ?? null,
  }
  try {
    const jobId = await store.createAnalysis(payload)
    router.push(`/workspace/${jobId}`)
  } catch (e: unknown) {
    submitError.value =
      (e as { data?: { detail?: string }; message?: string })?.data?.detail ??
      (e as Error)?.message ??
      'Analiz başlatılamadı.'
  }
}
</script>

<template>
  <div class="max-w-3xl mx-auto px-6 py-8">
    <form class="card p-6 space-y-6" @submit.prevent="submit">
      <div>
        <h2 class="text-lg font-semibold text-slate-100">Yeni Analiz</h2>
        <p class="mt-1 text-sm text-slate-500">
          Kaynağı ve parametreleri belirleyip analizi başlatın. İstek doğrudan
          <code class="text-slate-400">POST /analyze/jobs</code> uç noktasına gider.
        </p>
      </div>

      <!-- source mode toggle -->
      <div class="flex gap-2">
        <button
          type="button"
          class="btn-ghost"
          :class="mode === 'file' ? 'ring-1 ring-accent text-white' : ''"
          @click="mode = 'file'"
        >
          Video Dosyası
        </button>
        <button
          type="button"
          class="btn-ghost"
          :class="mode === 'stream' ? 'ring-1 ring-accent text-white' : ''"
          @click="mode = 'stream'"
        >
          RTSP / HTTP Akış
        </button>
      </div>

      <div v-if="mode === 'file'">
        <label class="field-label">Video Dosya Yolu</label>
        <div class="flex gap-2">
          <input
            v-model="form.videoPath"
            class="field-input"
            placeholder="/path/to/video.mp4"
          />
          <button type="button" class="btn-ghost shrink-0" @click="pickFile">Seç…</button>
        </div>
        <p class="mt-1 text-xs text-slate-600">
          Masaüstünde dosya seçici açılır; tarayıcıda yolu elle girebilirsiniz.
        </p>
      </div>

      <div v-else>
        <label class="field-label">Akış Kaynağı (RTSP / HTTP)</label>
        <input
          v-model="form.streamUrl"
          class="field-input"
          placeholder="rtsp://... veya http://.../stream.m3u8"
        />
      </div>

      <div>
        <label class="field-label">Kullanıcı İstemi (prompt)</label>
        <textarea v-model="form.userPrompt" rows="3" class="field-input resize-none" />
      </div>

      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="field-label">sample_fps (1–10)</label>
          <input
            v-model.number="form.sampleFps"
            type="number"
            min="1"
            max="10"
            class="field-input"
          />
        </div>
        <div>
          <label class="field-label">min_change_threshold (0.001–0.050)</label>
          <input
            v-model.number="form.minChangeThreshold"
            type="number"
            min="0.001"
            max="0.05"
            step="0.001"
            class="field-input"
          />
        </div>
      </div>

      <div v-if="submitError" class="text-sm text-risk-crit">{{ submitError }}</div>

      <div class="flex items-center gap-3">
        <button type="submit" class="btn-primary" :disabled="!canSubmit">
          <span v-if="store.submitting">Başlatılıyor…</span>
          <span v-else>Analizi Başlat</span>
        </button>
        <span class="text-xs text-slate-600 truncate">Kaynak: {{ videoSource || '—' }}</span>
      </div>
    </form>
  </div>
</template>
