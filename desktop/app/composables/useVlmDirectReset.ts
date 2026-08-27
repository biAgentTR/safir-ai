/**
 * Cross-component "start a fresh VLM Direct analysis" signal. HomeSection's
 * "VLM Direct Analiz" card bumps this before navigating there; VlmDirectSection
 * (which stays mounted for the whole session once vlm_direct mode is active —
 * clicking the card only scrolls to it, it doesn't remount) watches it and
 * clears its video/prompt/results state, so every click from Ana Sayfa really
 * does start a new analysis instead of resuming whatever was left on screen.
 */
export function useVlmDirectReset() {
  const trigger = useState<number>('vlm-direct-reset-trigger', () => 0)
  function requestNewAnalysis() {
    trigger.value++
  }
  return { trigger, requestNewAnalysis }
}
