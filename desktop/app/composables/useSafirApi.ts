/**
 * Central SAFIR API client. All HTTP access to the FastAPI backend goes through
 * here — components never call fetch directly.
 *
 * In dev, requests are same-origin ('' base) and Nitro's devProxy forwards
 * /health and /analyze/** to http://localhost:8000, so no CORS change is
 * required on the backend.
 */
import type {
  AnalyzeRequest,
  AnalyzeJobResponse,
  JobStatusResponse,
  FeedbackLabel,
  FeedbackResponse,
  AlertAcknowledgeResponse,
  AlertTriggerRequest,
  AlertTriggerResponse,
  HistoryListItem,
  HistoryDetail,
} from '~/types/api'

export function useSafirApi() {
  const base = useRuntimeConfig().public.apiBase || ''

  const url = (path: string) => `${base}${path}`

  /** GET /health -> { status, system } */
  async function health(): Promise<{ status: string; system: string }> {
    return await $fetch(url('/health'))
  }

  /** POST /analyze/jobs -> { job_id } */
  async function createAnalysis(
    payload: AnalyzeRequest,
  ): Promise<AnalyzeJobResponse> {
    return await $fetch(url('/analyze/jobs'), {
      method: 'POST',
      body: payload,
    })
  }

  /** GET /analyze/jobs/{job_id} -> JobStatusResponse */
  async function getJob(jobId: string): Promise<JobStatusResponse> {
    return await $fetch(url(`/analyze/jobs/${encodeURIComponent(jobId)}`))
  }

  /** Absolute (proxied) URL for the SSE stream of a job. */
  function streamJobUrl(jobId: string): string {
    return url(`/analyze/jobs/${encodeURIComponent(jobId)}/stream`)
  }

  /**
   * URL for a single evidence frame (image/jpeg). Base64 is never in the SSE
   * stream; the UI references frames via this endpoint.
   */
  function getFrameUrl(jobId: string, frameId: string): string {
    return url(
      `/analyze/jobs/${encodeURIComponent(jobId)}/frames/${encodeURIComponent(
        frameId,
      )}`,
    )
  }

  /** POST /events/{event_id}/feedback (Human-in-the-Loop TP/FP). */
  async function sendFeedback(
    eventId: number,
    feedback: FeedbackLabel,
  ): Promise<FeedbackResponse> {
    return await $fetch(url(`/events/${eventId}/feedback`), {
      method: 'POST',
      body: { feedback },
    })
  }

  /** POST /alerts/{alert_id}/acknowledge (Human-on-the-Loop review). */
  async function acknowledgeAlert(
    alertId: string,
    operatorNote = '',
  ): Promise<AlertAcknowledgeResponse> {
    return await $fetch(url(`/alerts/${encodeURIComponent(alertId)}/acknowledge`), {
      method: 'POST',
      body: { operator_note: operatorNote },
    })
  }

  /** POST /alerts/trigger (operator manual/override alarm). */
  async function triggerAlert(
    payload: AlertTriggerRequest,
  ): Promise<AlertTriggerResponse> {
    return await $fetch(url('/alerts/trigger'), {
      method: 'POST',
      body: payload,
    })
  }

  /** GET /history?limit&offset -> newest-first list. */
  async function getHistory(limit = 50, offset = 0): Promise<HistoryListItem[]> {
    return await $fetch(url('/history'), { query: { limit, offset } })
  }

  /** GET /history/{job_id} -> metadata + persisted SafirReport. */
  async function getHistoryItem(jobId: string): Promise<HistoryDetail> {
    return await $fetch(url(`/history/${encodeURIComponent(jobId)}`))
  }

  return {
    base,
    health,
    createAnalysis,
    getJob,
    streamJobUrl,
    getFrameUrl,
    sendFeedback,
    acknowledgeAlert,
    triggerAlert,
    getHistory,
    getHistoryItem,
  }
}
