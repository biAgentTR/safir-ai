/**
 * Pinia store: analysis job lifecycle (create + poll for final report).
 *
 * Live pipeline trace comes from `useAnalysisStream` (SSE); this store owns the
 * job identity and the terminal poll result (SafirReport). Keeping the two
 * separated mirrors the backend: SSE = live observability, GET /analyze/jobs =
 * authoritative status + result.
 */
import { defineStore } from 'pinia'
import type {
  AnalyzeRequest,
  JobStatus,
  JobStatusResponse,
  SafirReport,
} from '~/types/api'

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
  }),

  getters: {
    isTerminal: (s) => s.status === 'done' || s.status === 'error',
    isRunning: (s) => s.status === 'queued' || s.status === 'running',
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
    },

    /** POST /analyze/jobs; returns the new job_id. */
    async createAnalysis(payload: AnalyzeRequest): Promise<string> {
      const api = useSafirApi()
      this.submitting = true
      this.error = null
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
     * Poll GET /analyze/jobs/{id} until terminal. SSE drives the live pipeline
     * view; this fetches the authoritative final SafirReport.
     */
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
  },
})
