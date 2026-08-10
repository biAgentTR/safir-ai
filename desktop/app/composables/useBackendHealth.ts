/**
 * Lightweight backend health probe for the topbar status indicator.
 * Polls GET /health; exposes an online/offline/checking signal.
 */
import { ref, onMounted, onBeforeUnmount } from 'vue'

export type HealthState = 'checking' | 'online' | 'offline'

export function useBackendHealth(intervalMs = 5000) {
  const { health } = useSafirApi()
  const state = ref<HealthState>('checking')
  const system = ref<string>('')
  let timer: ReturnType<typeof setInterval> | null = null

  async function probe() {
    try {
      const res = await health()
      state.value = res.status === 'ok' ? 'online' : 'offline'
      system.value = res.system ?? ''
    } catch {
      state.value = 'offline'
    }
  }

  onMounted(() => {
    probe()
    timer = setInterval(probe, intervalMs)
  })
  onBeforeUnmount(() => {
    if (timer) clearInterval(timer)
  })

  return { state, system, probe }
}
