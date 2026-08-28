import { ref, onMounted, onBeforeUnmount } from "vue"
import { useSafirApi } from "~/composables/useSafirApi"
import type { SafirReport, HistoryDetail, JobStatusResponse } from "~/types/api"

export type AnalysisDetailState =
  | "loading"
  | "processing"
  | "completed"
  | "empty"
  | "not-found"
  | "service-unavailable"
  | "failed"

export function useAnalysisDetail(jobId: string) {
  const api = useSafirApi()
  const state = ref<AnalysisDetailState>("loading")
  const report = ref<SafirReport | null>(null)
  
  const jobStatus = ref<"queued" | "running" | "done" | "error" | null>(null)
  const step = ref<number>(0)
  const totalSteps = ref<number>(0)
  const errorMsg = ref<string | null>(null)

  let abortController: AbortController | null = null
  let retryTimer: ReturnType<typeof setTimeout> | null = null

  function cleanup() {
    if (abortController) {
      abortController.abort()
      abortController = null
    }
    if (retryTimer) {
      clearTimeout(retryTimer)
      retryTimer = null
    }
  }

  async function load(retryCount = 0) {
    cleanup()
    
    if (!jobId) {
      state.value = "not-found"
      return
    }

    abortController = new AbortController()
    const signal = abortController.signal

    try {
      state.value = "loading"
      
      try {
        const history: HistoryDetail = await api.getHistoryItem(jobId)
        
        if (history.report) {
          report.value = history.report
          const r = history.report
          const isEmpty = (!r.events || r.events.length === 0) &&
                          (!r.timeline || r.timeline.length === 0) &&
                          (!r.evidence_frames || r.evidence_frames.length === 0) &&
                          (!r.actions || r.actions.length === 0) &&
                          (!r.semantic_rag_sources || r.semantic_rag_sources.length === 0)
          state.value = isEmpty ? "empty" : "completed"
          return
        }
        
        if (history.status === "failed") {
          state.value = "failed"
          errorMsg.value = "Analiz başarısız olarak işaretlendi."
          return
        }
      } catch (err: any) {
        if (signal.aborted) return
        
        if (err.response?.status !== 404) {
          state.value = "service-unavailable"
          return
        }
      }

      if (signal.aborted) return

      try {
        const job: JobStatusResponse = await api.getJob(jobId, signal)
        
        jobStatus.value = job.status
        step.value = job.step
        totalSteps.value = job.total_steps
        errorMsg.value = job.error
        
        if (job.status === "queued" || job.status === "running") {
          state.value = "processing"
        } else if (job.status === "error") {
          state.value = "failed"
        } else if (job.status === "done") {
          if (retryCount < 3) {
            retryTimer = setTimeout(() => {
              load(retryCount + 1)
            }, 2000)
          } else {
            state.value = "failed"
            errorMsg.value = "Analiz tamamlandı ancak rapor verisi bulunamadı."
          }
        } else {
          state.value = "not-found"
        }
      } catch (err: any) {
        if (signal.aborted) return
        if (err.response?.status === 404) {
          state.value = "not-found"
        } else {
          state.value = "service-unavailable"
        }
      }
    } catch (err: any) {
      if (signal.aborted) return
      state.value = "service-unavailable"
    }
  }

  onMounted(() => {
    load()
  })

  onBeforeUnmount(() => {
    cleanup()
  })

  return {
    state,
    report,
    jobStatus,
    step,
    totalSteps,
    errorMsg,
    reload: load
  }
}

