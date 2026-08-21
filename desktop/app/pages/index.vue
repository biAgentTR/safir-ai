<script setup lang="ts">
import type { AnalyzeRequest } from '~/types/api'

const store = useAnalysisStore()
const router = useRouter()
const api = useSafirApi()

type SourceMode = 'file' | 'stream'
const mode = ref<SourceMode>('file')

const form = reactive({
  videoPath: '',
  streamUrl: '',
  userPrompt: '',
  sampleFps: 5 as number | null,
  minChangeThreshold: 0.001 as number | null,
})

const submitError = ref<string | null>(null)
const isUploading = ref(false)
const uploadStatus = ref<string | null>(null)
const isDragging = ref(false)
const fileInputRef = ref<HTMLInputElement | null>(null)
const previewObjectUrl = ref<string | null>(null)

const videoSource = computed(() =>
  mode.value === 'file' ? form.videoPath.trim() : form.streamUrl.trim(),
)
const canSubmit = computed(() => videoSource.value.length > 0 && !store.submitting && !isUploading.value)

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
      uploadStatus.value = `✓ Seçildi: ${selected.split(/[\/\\]/).pop()}`
      success = true
    }
  } catch (e) {
    console.warn('Tauri dialog varsayılan tarayıcı dosyasına düşüyor:', e)
  }

  if (!success && fileInputRef.value) {
    fileInputRef.value.click()
  }
}

async function handleFile(file: File) {
  const nativePath = (file as any).path

  if (file) {
    previewObjectUrl.value = URL.createObjectURL(file)
  }

  if (
    nativePath &&
    typeof nativePath === 'string' &&
    nativePath.length > 0 &&
    !nativePath.toLowerCase().includes('fakepath')
  ) {
    form.videoPath = nativePath
    uploadStatus.value = `✓ Dosya Seçildi: ${file.name}`
    return
  }

  isUploading.value = true
  uploadStatus.value = `"${file.name}" yükleniyor...`
  submitError.value = null
  try {
    const res = await api.uploadVideo(file)
    form.videoPath = res.video_source
    uploadStatus.value = `✓ Yüklendi: ${res.filename} (${(res.size_bytes / 1024 / 1024).toFixed(1)} MB)`
  } catch (e: unknown) {
    uploadStatus.value = null
    submitError.value = (e as Error)?.message || 'Video sunucuya yüklenemedi.'
  } finally {
    isUploading.value = false
  }
}

function onFileSelected(event: Event) {
  const target = event.target as HTMLInputElement
  if (target.files && target.files.length > 0) {
    handleFile(target.files[0])
  }
}

function onDrop(event: DragEvent) {
  isDragging.value = false
  if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
    handleFile(event.dataTransfer.files[0])
  }
}

async function submit() {
  if (!canSubmit.value) return
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
    form.userPrompt = 'Bu videoda özellikle yüksekte çalışma, baret/yelek eksikliği, forklift-yaya yakınlığı veya iş kazası riski var mı değerlendir.'
  } else {
    form.userPrompt = 'Bu videoda özellikle çevre koruma hattında tel örgü ihlali, izinsiz sızma, şüpheli paket/davranış veya İHA/drone var mı değerlendir.'
  }
}
</script>

<template>
  <div class="min-h-full flex items-center justify-center p-6 select-none">
    <!-- Hidden file input isolated with @click.stop -->
    <input
      ref="fileInputRef"
      type="file"
      accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
      class="hidden"
      @change="onFileSelected"
      @click.stop
    >

    <div class="w-full max-w-3xl space-y-6">
      <!-- App Header Branding Banner -->
      <div class="text-center space-y-2">
        <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-600 dark:text-blue-400 text-xs font-mono font-bold">
          🛡️ SAFİR AI — ÇİFT KULLANIMLI (DUAL-USE) OPERASYONEL RİSK PLATFORMU
        </div>
        <h1 class="text-3xl font-extrabold tracking-tight">
          Otonom Video &amp; Saha İncelemesi
        </h1>
        <p class="text-xs text-slate-500 dark:text-slate-400 max-w-xl mx-auto">
          İSG kuralları (Safety) ve Tesis Çevre Güvenliği (Security) olaylarını VLM ve BM25+FAISS RAG katmanı ile anında analiz edin.
        </p>
      </div>

      <!-- Main Central Card: Clean & Minimal UI -->
      <div class="glass-card p-8 space-y-6">
        <!-- Error Alert Banner -->
        <div v-if="submitError" class="p-4 rounded-xl border border-red-500/50 bg-red-500/10 text-xs text-red-600 dark:text-red-400 flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span>⚠️</span>
            <span>{{ submitError }}</span>
          </div>
          <button type="button" class="font-bold underline" @click="submitError = null">Kapat</button>
        </div>

        <form class="space-y-6" @submit.prevent="submit">
          <!-- 1. Video Input Area (File Drag & Drop vs Stream URL) -->
          <div class="space-y-3">
            <div class="flex items-center justify-between">
              <span class="field-label">1. Video Kaynağı Seçin</span>
              <div class="flex items-center gap-1 bg-surface-2 p-1 rounded-xl border border-edge text-xs">
                <button
                  type="button"
                  class="px-3 py-1 rounded-lg font-bold transition-all"
                  :class="mode === 'file' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-100'"
                  @click="mode = 'file'"
                >
                  📁 Dosya Yükle
                </button>
                <button
                  type="button"
                  class="px-3 py-1 rounded-lg font-bold transition-all"
                  :class="mode === 'stream' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-100'"
                  @click="mode = 'stream'"
                >
                  📡 Canlı Yayın Bağla
                </button>
              </div>
            </div>

            <!-- File Upload Drop Zone -->
            <div v-if="mode === 'file'" class="space-y-2">
              <div
                class="border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-200 relative overflow-hidden"
                :class="[
                  isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-edge bg-surface-2/50 hover:border-blue-500/60',
                  form.videoPath ? 'border-emerald-500/50 bg-emerald-500/5' : ''
                ]"
                @dragover.prevent="isDragging = true"
                @dragleave.prevent="isDragging = false"
                @drop.prevent="onDrop"
                @click="pickFile"
              >
                <div v-if="previewObjectUrl" class="mb-3 max-h-36 overflow-hidden rounded-xl mx-auto flex items-center justify-center" @click.stop>
                  <video :src="previewObjectUrl" class="h-32 rounded-xl object-cover shadow-md" controls muted />
                </div>

                <div class="space-y-2">
                  <div class="w-12 h-12 rounded-2xl bg-blue-500/10 text-blue-600 dark:text-blue-400 font-bold text-xl flex items-center justify-center mx-auto">
                    {{ form.videoPath ? '✅' : '📁' }}
                  </div>
                  <div>
                    <p class="text-sm font-bold">
                      {{ form.videoPath ? form.videoPath.split(/[\/\\]/).pop() : 'Videoyu buraya sürükleyin veya dosya seçin' }}
                    </p>
                    <p class="text-xs text-slate-400 mt-1">
                      Desteklenen formatlar: MP4, AVI, MOV, MKV (Örnek: data/test_videos/medium_motion_isg.mp4)
                    </p>
                  </div>
                </div>
              </div>

              <div v-if="uploadStatus" class="text-xs font-mono text-emerald-600 dark:text-emerald-400 text-center font-semibold">
                {{ uploadStatus }}
              </div>
            </div>

            <!-- RTSP Stream URL Input -->
            <div v-else class="space-y-2">
              <input
                v-model="form.streamUrl"
                type="text"
                class="field-input font-mono text-xs"
                placeholder="rtsp://192.168.1.100:554/live/ch0 veya http://cam.local/stream.m3u8"
              >
              <p class="text-xs text-slate-400">
                Canlı IP kamera akışı veya ağ yayın bağlantı adresini girin.
              </p>
            </div>
          </div>

          <!-- 2. Prompt & Special Instructions Input -->
          <div class="space-y-3">
            <div class="flex items-center justify-between">
              <span class="field-label">2. Özel Analiz Talimatı / Bağlam (Opsiyonel)</span>
              <div class="flex items-center gap-2">
                <button
                  type="button"
                  class="px-2.5 py-1 rounded-lg border border-edge bg-surface-2 hover:bg-surface-3 text-[11px] font-bold text-slate-600 dark:text-slate-300 transition-colors"
                  @click="applyPreset('isg')"
                >
                  🏭 İSG Önerisi
                </button>
                <button
                  type="button"
                  class="px-2.5 py-1 rounded-lg border border-edge bg-surface-2 hover:bg-surface-3 text-[11px] font-bold text-slate-600 dark:text-slate-300 transition-colors"
                  @click="applyPreset('defense')"
                >
                  🛡️ Savunma Önerisi
                </button>
              </div>
            </div>

            <textarea
              v-model="form.userPrompt"
              rows="3"
              class="field-input text-xs leading-relaxed"
              placeholder="Örnek: Bu videoda özellikle forklift ve personel hareketlerine odaklan, baret eksikliği veya yetkisiz tel örgü sızması var mı kontrol et..."
            />
          </div>

          <!-- 3. Prominent Start Button & Submit Options -->
          <div class="pt-2 flex items-center justify-between gap-4">
            <div class="text-xs text-slate-400 font-mono">
              CPU Sampler: <strong>Δ ≥ 0.001 (%85+ GPU Tasarrufu)</strong>
            </div>

            <button
              type="submit"
              class="btn-primary px-8 py-3 text-sm font-extrabold shadow-lg shadow-blue-500/25 flex items-center gap-2 shrink-0"
              :disabled="!canSubmit"
            >
              <span v-if="store.submitting" class="w-4 h-4 rounded-full border-2 border-white/30 border-t-white animate-spin" />
              <span>{{ store.submitting ? 'Analiz Başlatılıyor...' : '🚀 Analizi Başlat' }}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>
