<script setup lang="ts">
import type { AnalyzeRequest } from '~/types/api'

const store = useAnalysisStore()
const router = useRouter()

type SourceMode = 'file' | 'stream'
const mode = ref<SourceMode>('file')

const form = reactive({
  videoPath: '',
  streamUrl: '',
  userPrompt: 'Sahnede riskli bir durum var mi degerlendir.',
  sampleFps: 5 as number | null,
  minChangeThreshold: 0.001 as number | null,
})

const submitError = ref<string | null>(null)

const videoSource = computed(() =>
  mode.value === 'file' ? form.videoPath.trim() : form.streamUrl.trim(),
)
const canSubmit = computed(() => videoSource.value.length > 0 && !store.submitting && !isUploading.value)

const api = useSafirApi()
const fileInputRef = ref<HTMLInputElement | null>(null)
const isUploading = ref(false)
const uploadStatus = ref<string | null>(null)

async function pickFile(event?: Event) {
  if (event) {
    event.stopPropagation()
  }
  let success = false
  try {
    const dialog = await import('@tauri-apps/plugin-dialog')
    const selected = await dialog.open({
      multiple: false,
      filters: [{ name: 'Video', extensions: ['mp4', 'avi', 'mov', 'mkv', 'webm'] }],
    })
    if (typeof selected === 'string' && selected) {
      form.videoPath = selected
      success = true
    }
  } catch (e) {
    console.warn('Tauri dialog kullanılabilir değil, tarayıcı dosya seçiciye geçiliyor:', e)
  }

  if (!success && fileInputRef.value) {
    fileInputRef.value.click()
  }
}

async function onFileSelected(event: Event) {
  const target = event.target as HTMLInputElement
  if (target.files && target.files.length > 0) {
    const file = target.files[0]
    const nativePath = (file as any).path

    if (
      nativePath &&
      typeof nativePath === 'string' &&
      nativePath.length > 0 &&
      !nativePath.toLowerCase().includes('fakepath')
    ) {
      form.videoPath = nativePath
      return
    }

    isUploading.value = true
    uploadStatus.value = `"${file.name}" sunucuya yükleniyor...`
    submitError.value = null
    try {
      const res = await api.uploadVideo(file)
      form.videoPath = res.video_source
      uploadStatus.value = `✓ Sunucuya Yüklendi: ${res.filename} (${(res.size_bytes / 1024 / 1024).toFixed(1)} MB)`
    } catch (e: unknown) {
      uploadStatus.value = null
      submitError.value = (e as Error)?.message || 'Video sunucuya yüklenemedi.'
    } finally {
      isUploading.value = false
    }
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

function applyPreset(type: 'isg' | 'defense') {
  if (type === 'isg') {
    form.userPrompt = 'Sahnede yüksekte çalışma, baret/yelek eksikliği, forklift/yaya yakınlığı veya iş kazası riski var mı değerlendir.'
  } else {
    form.userPrompt = 'Çevre koruma hattında tel örgü ihlali, sızma, şüpheli paket/davranış veya izinsiz İHA/drone var mı değerlendir.'
  }
}
</script>

<template>
  <div class="max-w-4xl mx-auto px-6 py-8 select-none">
    <div class="glass-card p-8 space-y-6 glow-accent border-glow-accent">
      <div class="flex items-center justify-between border-b border-edge/80 pb-4">
        <div>
          <h2 class="text-xl font-bold text-white flex items-center gap-2">
            <span>📹</span> Yeni Video &amp; Kamera Analizi
          </h2>
          <p class="text-xs text-slate-400 mt-1">
            Analiz edilecek video dosyasını veya canlı RTSP/HTTP akışını belirleyin.
          </p>
        </div>
        <div class="px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-mono font-bold">
          v2.0 DUAL-USE
        </div>
      </div>

      <form class="space-y-6" @submit.prevent="submit">
        <!-- source mode toggle -->
        <div>
          <label class="field-label">Kaynak Türü</label>
          <div class="grid grid-cols-2 gap-3 p-1 rounded-xl bg-surface-2/80 border border-edge/80">
            <button
              type="button"
              class="py-2.5 rounded-lg text-xs font-bold tracking-wide transition-all duration-200"
              :class="mode === 'file' ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md shadow-blue-500/20' : 'text-slate-400 hover:text-slate-200'"
              @click="mode = 'file'"
            >
              📁 Video Dosyası
            </button>
            <button
              type="button"
              class="py-2.5 rounded-lg text-xs font-bold tracking-wide transition-all duration-200"
              :class="mode === 'stream' ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md shadow-blue-500/20' : 'text-slate-400 hover:text-slate-200'"
              @click="mode = 'stream'"
            >
              📡 Canlı RTSP / HTTP Akış
            </button>
          </div>
        </div>

        <div v-if="mode === 'file'" class="space-y-2">
          <label class="field-label">Video Dosya Yolu</label>
          <div class="flex gap-2">
            <input
              v-model="form.videoPath"
              class="field-input font-mono text-xs"
              placeholder="ör. data/duman_video.mp4 veya C:\videos\test.mp4"
            />
            <input
              ref="fileInputRef"
              type="file"
              accept="video/*,.mp4,.avi,.mov,.mkv,.webm"
              class="hidden"
              @change="onFileSelected"
              @click.stop
            />
            <button
              type="button"
              class="btn-ghost shrink-0"
              :disabled="isUploading"
              @click="pickFile"
            >
              <span v-if="isUploading" class="animate-pulse">Yükleniyor…</span>
              <span v-else>📁 Dosya Seç…</span>
            </button>
          </div>
          <p v-if="uploadStatus" class="text-xs text-emerald-400 font-mono font-semibold flex items-center gap-1">
            <span>✓</span> {{ uploadStatus }}
          </p>
        </div>

        <div v-else class="space-y-2">
          <label class="field-label">Akış Kaynağı (RTSP / HTTP)</label>
          <input
            v-model="form.streamUrl"
            class="field-input font-mono text-xs"
            placeholder="rtsp://admin:pass@192.168.1.100:554/live/ch0 veya http://.../stream.m3u8"
          />
        </div>

        <!-- Scenario Prompt & Presets -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="field-label">Analiz Odağı &amp; İstem (Prompt)</label>
            <div class="flex gap-2">
              <button
                type="button"
                class="px-2.5 py-1 rounded-md bg-blue-500/10 border border-blue-500/30 text-blue-300 text-[11px] font-semibold hover:bg-blue-500/20 transition-colors"
                @click="applyPreset('isg')"
              >
                🏭 Üretim &amp; İSG
              </button>
              <button
                type="button"
                class="px-2.5 py-1 rounded-md bg-purple-500/10 border border-purple-500/30 text-purple-300 text-[11px] font-semibold hover:bg-purple-500/20 transition-colors"
                @click="applyPreset('defense')"
              >
                🛡️ Savunma &amp; Çevre
              </button>
            </div>
          </div>
          <textarea v-model="form.userPrompt" rows="3" class="field-input resize-none text-xs leading-relaxed" />
        </div>

        <!-- Advanced Sampler Parameters -->
        <div class="p-4 rounded-xl border border-edge/80 bg-surface-2/40 space-y-3">
          <div class="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
            <span>⚙️</span> CPU Frame Sampler İnce Ayarları
          </div>
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="field-label text-[11px]">sample_fps (1–10)</label>
              <input
                v-model.number="form.sampleFps"
                type="number"
                min="1"
                max="10"
                class="field-input font-mono text-xs"
              />
            </div>
            <div>
              <label class="field-label text-[11px]">min_change_threshold ($\Delta$)</label>
              <input
                v-model.number="form.minChangeThreshold"
                type="number"
                min="0.001"
                max="0.05"
                step="0.001"
                class="field-input font-mono text-xs"
              />
            </div>
          </div>
        </div>

        <div v-if="submitError" class="p-3 rounded-lg border border-rose-500/50 bg-rose-500/10 text-xs text-rose-300 font-medium">
          {{ submitError }}
        </div>

        <div class="pt-2 flex items-center justify-between">
          <button type="submit" class="btn-primary w-full md:w-auto px-8 py-3 text-base shadow-xl shadow-blue-500/25" :disabled="!canSubmit">
            <span v-if="store.submitting" class="animate-pulse">Boru Hattı Başlatılıyor…</span>
            <span v-else>🚀 Analiz Boru Hattını Başlat</span>
          </button>
        </div>
      </form>
    </div>
  </div>
</template>
