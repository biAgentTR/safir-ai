/**
 * Central Pinia store for a single analysis job. Holds EVERYTHING the workspace
 * renders (spec #16): current job, live trace events, job status, final report,
 * selected stage, and alarm/feedback/acknowledge state. Components read from
 * here and act through `useSafirApi` / `useAnalysisStream` — never raw HTTP.
 */
import { defineStore } from 'pinia'
import type {
  AnalyzeRequest,
  FeedbackLabel,
  JobStatus,
  JobStatusResponse,
  SafirReport,
  TraceEvent,
  TraceStage,
  TraceStatus,
} from '~/types/api'
import type { StreamConnection } from '~/composables/useAnalysisStream'

export const STAGE_ORDER: TraceStage[] = [
  'sampler',
  'vlm',
  'events',
  'agent_context',
  'decision',
  'escalation',
  'report',
]

/** Critical alarm threshold — mirrors Streamlit theme.CRITICAL_ALARM_THRESHOLD. */
export const CRITICAL_ALARM_THRESHOLD = 70

type ActionState = 'idle' | 'pending' | 'ok' | 'error'

interface AnalysisState {
  jobId: string | null
  status: JobStatus | null
  stageName: string
  step: number
  totalSteps: number
  report: SafirReport | null
  error: string | null
  submitting: boolean
  lastRequest: AnalyzeRequest | null

  // live SSE trace
  traceEvents: TraceEvent[]
  connection: StreamConnection
  endStatus: JobStatus | null
  streamError: string | null

  // ui
  selectedStage: TraceStage | null
  alarmPlayedFor: string | null
  // history mode: a persisted analysis opened from /history (no live SSE/trace)
  historyMode: boolean
  historyLoading: boolean
  historyError: string | null

  // human-in/on-the-loop
  feedback: { state: ActionState; label: FeedbackLabel | null; message: string }
  ack: { state: ActionState; message: string }
  manualAlert: { state: ActionState; message: string; alertId: string | null }
}

export const useAnalysisStore = defineStore('analysis', {
  state: (): AnalysisState => ({
    jobId: null,
    status: null,
    stageName: '',
    step: 0,
    totalSteps: 3,
    report: null,
    error: null,
    submitting: false,
    lastRequest: null,

    traceEvents: [],
    connection: 'idle',
    endStatus: null,
    streamError: null,

    selectedStage: null,
    alarmPlayedFor: null,
    historyMode: false,
    historyLoading: false,
    historyError: null,

    feedback: { state: 'idle', label: null, message: '' },
    ack: { state: 'idle', message: '' },
    manualAlert: { state: 'idle', message: '', alertId: null },
  }),

  getters: {
    isTerminal: (s) => s.status === 'done' || s.status === 'error',
    isRunning: (s) => s.status === 'queued' || s.status === 'running',

    /** Last trace event for a stage (one per stage today). */
    eventForStage:
      (s) =>
      (stage: TraceStage): TraceEvent | undefined => {
        for (let i = s.traceEvents.length - 1; i >= 0; i--) {
          if (s.traceEvents[i].stage === stage) return s.traceEvents[i]
        }
        return undefined
      },

    /** Derived per-stage status for the cumulative timeline. */
    stageStatus() {
      return (stage: TraceStage): TraceStatus | 'pending' => {
        const ev = (this as unknown as { eventForStage: (x: TraceStage) => TraceEvent | undefined }).eventForStage(stage)
        if (ev) return ev.status
        // no event yet: running if the job is active and this is the next stage
        return 'pending'
      }
    },

    completedCount: (s) =>
      new Set(s.traceEvents.filter((e) => e.status === 'completed').map((e) => e.stage)).size,

    latestStage: (s): TraceStage | null =>
      s.traceEvents.length ? s.traceEvents[s.traceEvents.length - 1].stage : null,

    riskScore(): number | null {
      return this.report ? this.report.risk_score : null
    },
    riskLevel(): string | null {
      return this.report ? this.report.risk_level : null
    },
    isCritical(): boolean {
      return this.report != null && this.report.risk_score != null && this.report.risk_score >= CRITICAL_ALARM_THRESHOLD
    },
    hasAutoAlert(): boolean {
      return !!(this.report && this.report.auto_dispatched && this.report.alert_id)
    },
  },

  actions: {
    resetJob() {
      this.jobId = null
      this.status = null
      this.stageName = ''
      this.step = 0
      this.totalSteps = 3
      this.report = null
      this.error = null
      this.traceEvents = []
      this.connection = 'idle'
      this.endStatus = null
      this.streamError = null
      this.selectedStage = null
      this.alarmPlayedFor = null
      this.historyMode = false
      this.historyLoading = false
      this.historyError = null
      this.feedback = { state: 'idle', label: null, message: '' }
      this.ack = { state: 'idle', message: '' }
      this.manualAlert = { state: 'idle', message: '', alertId: null }
    },

    // ---- SSE trace ingestion (called by useAnalysisStream) ----
    pushTraceEvent(ev: TraceEvent) {
      // Capture the latest stage BEFORE appending: auto-follow should advance
      // only while the selection is still tracking the live head. If the user
      // manually picked an earlier stage, selectedStage no longer equals the
      // previous head, so we stop following and keep their choice.
      const prevLatest = this.latestStage
      this.traceEvents = [...this.traceEvents, ev]
      if (!this.selectedStage || this.selectedStage === prevLatest) {
        this.selectedStage = ev.stage
      }
    },
    setConnection(c: StreamConnection) {
      this.connection = c
    },
    setEndStatus(s: JobStatus) {
      this.endStatus = s
    },
    setStreamError(detail: string | null) {
      this.streamError = detail
    },
    selectStage(stage: TraceStage) {
      this.selectedStage = stage
    },
    markAlarmPlayed(signature: string) {
      this.alarmPlayedFor = signature
    },

    // ---- job lifecycle ----
    async createAnalysis(payload: AnalyzeRequest): Promise<string> {
      const api = useSafirApi()
      this.submitting = true
      try {
        this.resetJob()
        this.lastRequest = payload
        const { job_id } = await api.createAnalysis(payload)
        this.jobId = job_id
        this.status = 'queued'
        return job_id
      } finally {
        this.submitting = false
      }
    },

    applyStatus(s: JobStatusResponse) {
      this.status = s.status
      this.stageName = s.stage_name
      this.step = s.step
      this.totalSteps = s.total_steps
      this.report = s.result
      this.error = s.error
    },

    /**
     * Load a PERSISTED analysis from GET /history/{job_id} (history mode).
     * No SSE, no polling — the report AND (if persisted) the pipeline trace
     * come straight from the store on disk. traceEvents is filled with the
     * exact same TraceEvent[] shape the live SSE stream produces, so
     * PipelineTimeline / StageCard render unchanged in history mode.
     */
    async loadHistory(jobId: string) {
      const api = useSafirApi()
      this.resetJob()
      this.jobId = jobId
      this.historyMode = true
      this.historyLoading = true
      this.historyError = null
      try {
        const detail = await api.getHistoryItem(jobId)
        this.report = detail.report
        this.lastRequest = detail.video_source
          ? { video_source: detail.video_source, user_prompt: '' }
          : null
        // map persisted status -> JobStatus used across the workspace
        this.status = detail.status === 'completed' ? 'done' : detail.status === 'failed' ? 'error' : (detail.status as JobStatus)

        // Pipeline trace (older records or a persistence failure -> empty array;
        // PipelineTimeline already renders an all-"pending" rail for that case).
        this.traceEvents = detail.trace_events ?? []
        this.selectedStage = this.traceEvents.length
          ? this.traceEvents[this.traceEvents.length - 1].stage
          : null
      } catch (e) {
        this.historyError = humanError(e, 'Geçmiş analiz yüklenemedi.')
        this.status = 'error'
      } finally {
        this.historyLoading = false
      }
    },

    async pollUntilDone(jobId: string, intervalMs = 500, timeoutMs = 120_000) {
      const api = useSafirApi()
      const deadline = Date.now() + timeoutMs
      while (Date.now() < deadline) {
        const s = await api.getJob(jobId)
        this.applyStatus(s)
        if (s.status === 'done' || s.status === 'error') return
        await new Promise((r) => setTimeout(r, intervalMs))
      }
      throw new Error('Poll timeout')
    },

    // ---- human-in/on-the-loop actions (real endpoints) ----
    async submitFeedback(label: FeedbackLabel) {
      if (this.report?.event_id == null) return
      const api = useSafirApi()
      this.feedback = { state: 'pending', label, message: '' }
      try {
        const res = await api.sendFeedback(this.report.event_id, label)
        this.feedback = { state: 'ok', label, message: res.message }
      } catch (e) {
        this.feedback = { state: 'error', label, message: humanError(e, 'Geri bildirim kaydedilemedi.') }
      }
    },

    async acknowledgeAlert(note = '') {
      if (!this.report?.alert_id) return
      const api = useSafirApi()
      this.ack = { state: 'pending', message: '' }
      try {
        const res = await api.acknowledgeAlert(this.report.alert_id, note)
        this.ack = { state: 'ok', message: res.message }
      } catch (e) {
        this.ack = { state: 'error', message: humanError(e, 'Alarm onaylanamadı.') }
      }
    },

    async triggerManualAlert(note = '') {
      if (!this.report) return
      const api = useSafirApi()
      this.manualAlert = { state: 'pending', message: '', alertId: null }
      try {
        const res = await api.triggerAlert({
          // Manuel/operatör tetikli alarm: risk belirsiz (null) olsa bile operatör
          // bilinçli olarak override ediyor; 0 burada yalnızca dekoratif kayıt amaçlıdır,
          // otomatik eskalasyon kararı DEĞİLDİR.
          risk_score: this.report.risk_score ?? 0,
          risk_level: this.report.risk_level,
          recommended_action: this.report.actions?.[0] ?? this.report.recommended_action ?? '',
          operator_note: note,
        })
        this.manualAlert = { state: 'ok', message: res.message, alertId: res.alert_id }
      } catch (e) {
        this.manualAlert = { state: 'error', message: humanError(e, 'Manuel alarm tetiklenemedi.'), alertId: null }
      }
    },
  },
})

function humanError(e: unknown, fallback: string): string {
  return (
    (e as { data?: { detail?: string } })?.data?.detail ??
    (e as Error)?.message ??
    fallback
  )
}
