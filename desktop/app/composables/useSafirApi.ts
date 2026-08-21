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
  AskResponse,
  Conversation,
  ConversationCreateRequest,
  ConversationDetail,
  ConversationMessage,
  ConversationContext,
  ConversationContextCreateRequest,
  ConversationDocument,
  SystemOverview,
} from '~/types/api'

export function useSafirApi() {
  const base = useRuntimeConfig().public.apiBase || ''

  const url = (path: string) => `${base}${path}`

  /** GET /health -> { status, system } */
  async function health(): Promise<{ status: string; system: string }> {
    return await $fetch(url('/health'))
  }

  /** POST /upload/video -> { video_source, filename, size_bytes } */
  async function uploadVideo(
    file: File,
  ): Promise<{ video_source: string; filename: string; size_bytes: number }> {
    const formData = new FormData()
    formData.append('file', file)
    return await $fetch(url('/upload/video'), {
      method: 'POST',
      body: formData,
    })
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

  /** Absolute (proxied) URL for the real, backend-generated PDF report. */
  function getReportPdfUrl(jobId: string): string {
    return url(`/history/${encodeURIComponent(jobId)}/report.pdf`)
  }

  /**
   * GET /history/{job_id}/report.pdf -> real PDF bytes (reportlab, embedded
   * evidence images — same `ReportExporter` the Operator Panel uses). Works
   * for a live job still in memory OR an already-persisted History record.
   */
  async function getReportPdf(jobId: string): Promise<Blob> {
    return await $fetch(getReportPdfUrl(jobId), { responseType: 'blob' })
  }

  /** POST /ask -> grounded answer + sources (context-aware assistant). */
  async function ask(question: string, jobId?: string | null): Promise<AskResponse> {
    return await $fetch(url('/ask'), {
      method: 'POST',
      body: { question, job_id: jobId ?? null },
    })
  }

  /** Absolute (proxied) URL for the streaming Ask SAFIR SSE endpoint (GET /ask/stream). */
  function askStreamUrl(question: string, jobId?: string | null, conversationId?: string | null): string {
    const params = new URLSearchParams({ question })
    if (jobId) params.set('job_id', jobId)
    if (conversationId) params.set('conversation_id', conversationId)
    return url(`/ask/stream?${params.toString()}`)
  }

  /** POST /conversations -> create a new SAFIR Asistan chat. */
  async function createConversation(payload: ConversationCreateRequest): Promise<Conversation> {
    return await $fetch(url('/conversations'), { method: 'POST', body: payload })
  }

  /** GET /conversations?limit&offset -> newest-updated-first list. */
  async function getConversations(limit = 50, offset = 0): Promise<Conversation[]> {
    return await $fetch(url('/conversations'), { query: { limit, offset } })
  }

  /** GET /conversations/{id} -> conversation summary + full message history. */
  async function getConversation(conversationId: string): Promise<ConversationDetail> {
    return await $fetch(url(`/conversations/${encodeURIComponent(conversationId)}`))
  }

  /**
   * POST /conversations/{id}/messages -> append a message (UI/history only —
   * never fed back into the LLM prompt, see backend docstring).
   */
  async function addConversationMessage(
    conversationId: string,
    role: 'user' | 'assistant',
    content: string,
  ): Promise<ConversationMessage> {
    return await $fetch(url(`/conversations/${encodeURIComponent(conversationId)}/messages`), {
      method: 'POST',
      body: { role, content },
    })
  }

  /** GET /system/overview -> Data Center ozet KPI'lari (read-only). */
  async function getSystemOverview(): Promise<SystemOverview> {
    return await $fetch(url('/system/overview'))
  }

  /** POST /conversations/{id}/context -> sohbete kullanıcı notu ekler (Adım 3). */
  async function addConversationContext(
    conversationId: string,
    payload: ConversationContextCreateRequest,
  ): Promise<ConversationContext> {
    return await $fetch(url(`/conversations/${encodeURIComponent(conversationId)}/context`), {
      method: 'POST',
      body: payload,
    })
  }

  /** DELETE /conversations/{id}/context/{context_id} -> eklenen bağlamı kaldırır. */
  async function removeConversationContext(conversationId: string, contextId: number): Promise<{ removed: boolean }> {
    return await $fetch(
      url(`/conversations/${encodeURIComponent(conversationId)}/context/${contextId}`),
      { method: 'DELETE' },
    )
  }

  /**
   * POST /conversations/{id}/documents -> belge yükler (PDF/TXT/DOCX, multipart).
   * Yanıt `status: "processing"` ile döner; metin çıkarımı arka planda sürer —
   * çağıran taraf `getConversation(id)` ile durumu izlemelidir (polling).
   */
  async function uploadConversationDocument(conversationId: string, file: File): Promise<ConversationDocument> {
    const form = new FormData()
    form.append('file', file)
    return await $fetch(url(`/conversations/${encodeURIComponent(conversationId)}/documents`), {
      method: 'POST',
      body: form,
    })
  }

  /** DELETE /conversations/{id}/documents/{document_id} -> yüklenen belgeyi kaldırır. */
  async function removeConversationDocument(
    conversationId: string,
    documentId: string,
  ): Promise<{ removed: boolean }> {
    return await $fetch(
      url(`/conversations/${encodeURIComponent(conversationId)}/documents/${encodeURIComponent(documentId)}`),
      { method: 'DELETE' },
    )
  }

  return {
    base,
    health,
    uploadVideo,
    createAnalysis,
    getJob,
    streamJobUrl,
    getFrameUrl,
    sendFeedback,
    acknowledgeAlert,
    triggerAlert,
    getHistory,
    getHistoryItem,
    getReportPdfUrl,
    getReportPdf,
    ask,
    askStreamUrl,
    createConversation,
    getConversations,
    getConversation,
    addConversationMessage,
    addConversationContext,
    removeConversationContext,
    uploadConversationDocument,
    removeConversationDocument,
    getSystemOverview,
  }
}
