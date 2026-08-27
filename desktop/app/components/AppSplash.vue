<script setup lang="ts">
// First-launch welcome overlay: covers the shell until the initial backend
// health probe resolves, so the app doesn't cold-open on a bare, empty frame.
// A short minimum display time avoids a one-frame flash when the backend is
// already warm and the probe resolves almost instantly.
//
// hasShownOnce is shared app-wide (useState) so this only ever blocks on the
// TRUE cold boot — without this guard it would re-run its own 700ms minimum
// display every time this component remounts.
const { state } = useBackendHealth()
const hasShownOnce = useState<boolean>('app-splash-shown-once', () => false)

const minTimeElapsed = ref(false)
let timer: ReturnType<typeof setTimeout> | null = null
onMounted(() => {
  if (hasShownOnce.value) return
  timer = setTimeout(() => {
    minTimeElapsed.value = true
  }, 700)
})
onBeforeUnmount(() => {
  if (timer) clearTimeout(timer)
})

const visible = computed(() => {
  if (hasShownOnce.value) return false
  const resolved = state.value !== 'checking' && minTimeElapsed.value
  if (resolved) hasShownOnce.value = true
  return !resolved
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
