<script setup lang="ts">
// First-launch welcome overlay: covers the shell briefly on cold boot.
// Enforces a strict max timeout so the splash screen NEVER gets stuck on a black frame.

const { state } = useBackendHealth()
const hasShownOnce = useState<boolean>('app-splash-shown-once', () => false)
const isDismissed = ref(false)

let minTimer: ReturnType<typeof setTimeout> | null = null
let maxTimer: ReturnType<typeof setTimeout> | null = null

onMounted(() => {
  if (hasShownOnce.value) {
    isDismissed.value = true
    return
  }

  // Hard safety timeout: Dismiss splash screen after at most 800ms regardless of backend state
  maxTimer = setTimeout(() => {
    isDismissed.value = true
    hasShownOnce.value = true
  }, 800)
})

watch(state, (newState) => {
  if (newState !== 'checking' && !isDismissed.value) {
    minTimer = setTimeout(() => {
      isDismissed.value = true
      hasShownOnce.value = true
    }, 400)
  }
})

onBeforeUnmount(() => {
  if (minTimer) clearTimeout(minTimer)
  if (maxTimer) clearTimeout(maxTimer)
})

const visible = computed(() => {
  if (hasShownOnce.value || isDismissed.value) return false
  return true
})
</script>

<template>
  <Transition name="splash-fade">
    <div
      v-if="visible"
      class="fixed inset-0 z-50 flex flex-col items-center justify-center gap-6 bg-surface-0"
      role="status"
      aria-live="polite"
    >
      <img src="~/assets/images/logo.png" alt="SAFİR" class="w-24 h-24 object-contain drop-shadow-lg" />
      <div class="text-center">
        <h1 class="text-2xl font-semibold tracking-[0.35em] text-slate-100">SAFİR</h1>
        <p class="mt-2 text-sm text-slate-500 max-w-sm px-6">
          Saha Analiz ve Farkındalık İçin Yapay Zekâ Destekli Karar Sistemi
        </p>
      </div>
      <div class="flex items-center gap-2 text-xs text-slate-500">
        <span class="inline-block w-3.5 h-3.5 border-2 border-accent border-t-transparent rounded-full animate-spin motion-reduce:animate-none" />
        <span>Hoş geldiniz — arka uç sunucusuna bağlanılıyor…</span>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.splash-fade-enter-active,
.splash-fade-leave-active {
  transition: opacity 0.35s ease;
}
.splash-fade-enter-from,
.splash-fade-leave-to {
  opacity: 0;
}
</style>
