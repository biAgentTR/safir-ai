/**
 * Analysis mode switch: which of the two independent analysis systems the
 * operator is currently using.
 *
 * - 'low_budget': the existing sampler/pipeline dashboard (untouched).
 * - 'vlm_direct': the video-to-VLM direct analysis dashboard (/vlm-direct).
 *
 * Persisted in localStorage so the choice survives a reload, and read
 * synchronously at first use (this app runs ssr:false / SPA, so the
 * useState initializer below runs client-side before first paint — no flash
 * of the mode-select gate on an already-chosen install, mirroring the
 * pattern in useTheme.ts).
 */
export type AnalysisMode = 'low_budget' | 'vlm_direct'

const STORAGE_KEY = 'safir-analysis-mode'

function readStored(): AnalysisMode | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'low_budget' || v === 'vlm_direct') return v
  } catch {
    // localStorage unavailable (e.g. privacy mode) — treat as unset.
  }
  return null
}

export function useAnalysisMode() {
  const mode = useState<AnalysisMode>('safir-analysis-mode', () => readStored() ?? 'low_budget')
  const hasChosen = useState<boolean>('safir-analysis-mode-chosen', () => readStored() !== null)

  function setMode(next: AnalysisMode) {
    mode.value = next
    hasChosen.value = true
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // best-effort persistence only
    }
  }

  return { mode, hasChosen, setMode }
}
