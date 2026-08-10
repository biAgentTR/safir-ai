/**
 * SAFIR backend contract, transcribed into TypeScript from the REAL FastAPI /
 * pydantic models — no invented fields.
 *
 * Sources (do not drift from these):
 *   - src/main.py            AnalyzeRequest, AnalyzeJobResponse, JobStatusResponse
 *   - src/schemas/report.py  SafirReport, TimelineEntry, TimelineEvent,
 *                            EvidenceFrameOut, SamplerStats
 *   - src/observability/trace_serializer.py  TraceEvent + per-stage data
 *
 * The trace/SSE `data` payloads are intentionally typed per stage, matching the
 * serializers in trace_serializer.py.
 */

// ---------------------------------------------------------------- requests ---

/** POST /analyze and POST /analyze/jobs body (src/main.py: AnalyzeRequest). */
export interface AnalyzeRequest {
  video_source: string
  user_prompt?: string
  /** 1–10; operator FPS override. */
  sample_fps?: number | null
  /** 0.001–0.050; operator sensitivity override. */
  min_change_threshold?: number | null
}

/** POST /analyze/jobs response (src/main.py: AnalyzeJobResponse). */
export interface AnalyzeJobResponse {
  job_id: string
}

// -------------------------------------------------------- job status / poll ---

export type JobStatus = 'queued' | 'running' | 'done' | 'error'

/** GET /analyze/jobs/{job_id} response (src/main.py: JobStatusResponse). */
export interface JobStatusResponse {
  status: JobStatus
  stage_name: string
  step: number
  total_steps: number
  result: SafirReport | null
  error: string | null
}

// -------------------------------------------------------------- report ------

/** src/schemas/report.py: TimelineEntry. */
export interface TimelineEntry {
  timestamp: number
  description: string
}

/** src/schemas/report.py: TimelineEvent (TimelineEntry + optional severity). */
export interface TimelineEvent extends TimelineEntry {
  severity?: string | null
}

/** src/schemas/report.py: EvidenceFrameOut (final report — carries base64). */
export interface EvidenceFrameOut {
  event_id: number
  timestamp_sec: number
  timestamp_str: string
  change_score: number
  base64_image: string
  saved_path?: string | null
  is_fallback: boolean
}

/** src/schemas/report.py: SamplerStats. */
export interface SamplerStats {
  total_frames_scanned: number
  sampled_frames_evaluated: number
  evidence_frame_count: number
  eliminated_frame_count: number
  gpu_savings_ratio_pct: number
  elapsed_sec: number
}

export type RiskLevel = 'dusuk' | 'orta' | 'yuksek' | 'kritik'
export type EscalationTier = 'monitor' | 'notify' | 'alarm'

/** src/schemas/report.py: SafirReport (the polling `result`). */
export interface SafirReport {
  event_id?: number | null
  video_source: string
  generated_at: string
  natural_language_summary: string
  summary: string
  risk_score: number
  risk_level: RiskLevel | string
  recommended_action: string
  actions: string[]
  detected_event_types: string[]
  timeline: TimelineEntry[]
  evidence_frames: EvidenceFrameOut[]
  relevant_regulations: string[]
  escalation_tier?: EscalationTier | string | null
  auto_dispatched: boolean
  alert_id?: string | null
  sampler_stats?: SamplerStats | null
  vlm_model?: string | null
  llm_model?: string | null
}

// -------------------------------------------------------- trace / SSE -------

/** Internal stage keys (trace_serializer.STAGE_ORDER). */
export type TraceStage =
  | 'sampler'
  | 'vlm'
  | 'events'
  | 'agent_context'
  | 'decision'
  | 'escalation'
  | 'report'

export type TraceStatus = 'queued' | 'running' | 'completed' | 'failed'

export interface TraceMetadata {
  /** Presentation label (STAGE_LABELS), e.g. "Frame Sampling". */
  label: string
  /** Index in STAGE_ORDER. */
  order: number
  [k: string]: unknown
}

/**
 * A single serialized trace event (trace_serializer.make_event).
 * `data` shape depends on `stage`; typed loosely here and narrowed per stage
 * where needed by the UI.
 */
export interface TraceEvent<D = Record<string, unknown>> {
  stage: TraceStage
  status: TraceStatus
  /** ISO-8601 UTC with trailing "Z". */
  timestamp: string
  duration_ms: number | null
  summary: string
  data: D
  metadata: TraceMetadata
  error: string | null
}

// --- per-stage `data` payloads (from the serializers) ---

/** Frame reference inside sampler stage — points to the frame endpoint. */
export interface EvidenceFrameRef {
  frame_id: string
  timestamp_sec: number
  timestamp_str: string
  change_score: number
  is_fallback: boolean
  motion_bbox: number[] | null
  /** e.g. /analyze/jobs/{job_id}/frames/{frame_id} (no base64 in the stream). */
  thumbnail_url: string | null
}

export interface RepresentativeFrameRef {
  label: string
  timestamp_str: string
  frame_id: string
  thumbnail_url: string | null
}

export interface EventGroupRef {
  event_id: number
  start_time: number
  end_time: number
  total_candidate_frames: number
  representative_frames: RepresentativeFrameRef[]
}

export interface SamplerStageData {
  stats: Partial<SamplerStats>
  evidence_frames: EvidenceFrameRef[]
  event_groups: EventGroupRef[]
}

export interface VlmStageData {
  model_name: string
  frame_count: number
  latency_ms: number
  user_prompt: string
  frames_sent: number
  description: string
  structured_events: unknown[]
}

/** decision stage — NOTE: raw_response is intentionally absent server-side. */
export interface Decision {
  risk_score: number
  risk_level: string
  summary: string
  recommended_action: string
  actions: string[]
  events: unknown[]
}

export interface Escalation {
  tier: EscalationTier | string
  auto_dispatched: boolean
  alert_id: string | null
  reason: string
}

// -------------------------------------------------------- SSE envelope ------

/** Terminal SSE control event: `event: end` / `event: error`. */
export interface SseEndEvent {
  kind: 'end'
  status: JobStatus
}
export interface SseErrorEvent {
  kind: 'error'
  detail: string
}
export interface SseTraceEnvelope {
  kind: 'trace'
  event: TraceEvent
}
export type SseMessage = SseTraceEnvelope | SseEndEvent | SseErrorEvent
