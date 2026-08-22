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
  /** VLM event_id this frame belongs to (bkz. EVENTS_JSON.event_id) — string, not a positional label. */
  event_id: string
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

export type RiskLevel = 'dusuk' | 'orta' | 'yuksek' | 'kritik' | 'unknown'
export type EscalationTier = 'monitor' | 'notify' | 'alarm'
/**
 * 'assessed': risk_score/risk_level guvenilir sekilde hesaplandi (0 dahil gecerli deger).
 * 'unknown': VLM/LLM/ajan karar zincirinde hata olustu; risk_score=null, ASLA dusuk risk
 * olarak yorumlanmamali (bkz. src/agent/langgraph_agent.py: AgentDecision.risk_status).
 */
export type RiskStatus = 'assessed' | 'unknown'

/** src/schemas/report.py: SafirReport (the polling `result`). */
export interface SafirReport {
  event_id?: number | null
  video_source: string
  generated_at: string
  natural_language_summary: string
  summary: string
  risk_score: number | null
  risk_level: RiskLevel | string
  risk_status: RiskStatus | string
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

/**
 * Frame reference inside sampler stage — points to the frame endpoint.
 * The sampler no longer clusters: every threshold-passing frame is sent
 * here, in chronological order, with no positional role attached. Event
 * clustering happens downstream, in the VLM stage.
 */
export interface EvidenceFrameRef {
  evidence_id: string
  frame_id: string
  timestamp_sec: number
  timestamp_str: string
  change_score: number
  is_fallback: boolean
  motion_bbox: number[] | null
  /** e.g. /analyze/jobs/{job_id}/frames/{frame_id} (no base64 in the stream). */
  thumbnail_url: string | null
}

export interface SamplerStageData {
  stats: Partial<SamplerStats>
  evidence_frames: EvidenceFrameRef[]
}

/** A single VLM-clustered event from EVENTS_JSON (src/prompts/vlm_prompts.py). */
export interface VlmEvent {
  event_id: string
  type: string
  start_time: number
  end_time: number
  evidence_ids: string[]
  description: string
  risk_score: number | null
  confidence: number
}

export interface VlmStageData {
  model_name: string
  frame_count: number
  latency_ms: number
  user_prompt: string
  frames_sent: number
  description: string
  structured_events: VlmEvent[]
  vlm_status: string
}

/** decision stage — NOTE: raw_response is intentionally absent server-side. */
export interface Decision {
  risk_score: number | null
  risk_level: string
  risk_status: RiskStatus | string
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

/** events stage: detected_events[] (trace_serializer.serialize_events). */
export interface DetectedEvent {
  event_name: string
  event_type: string
  timestamp: number
  confidence: number
  matched_keywords: string[]
}
export interface TemporalEvent {
  event_type: string
  occurrence_count: number
  duration: number
  start_timestamp: number
  end_timestamp: number
  confidence: number
}
export interface RuleMatch {
  rule_id: string
  rule_description: string
  severity: string
  event_type: string
}
export interface EventsStageData {
  detected_events: DetectedEvent[]
  temporal_events: TemporalEvent[]
  rule_matches: RuleMatch[]
}

/** agent_context stage. */
export interface ContextStageData {
  prompt_block: string
  length: number
}

/** report stage (compact; full report comes from the polling endpoint). */
export interface ReportStageData {
  event_id: number | null
  risk_score: number | null
  risk_level: string
  risk_status: RiskStatus | string
  escalation_tier: string | null
  auto_dispatched: boolean
  alert_id: string | null
  detected_event_types: string[]
  vlm_model: string | null
  llm_model: string | null
  timeline: TimelineEntry[]
  sartname_json: Record<string, unknown>
}

// -------------------------------------------------- feedback / alerts -------

/** POST /events/{event_id}/feedback */
export type FeedbackLabel = 'true_positive' | 'false_positive'
export interface FeedbackRequest {
  feedback: FeedbackLabel
}
export interface FeedbackResponse {
  event_id: number
  feedback: string
  message: string
}

/** POST /alerts/{alert_id}/acknowledge */
export interface AlertAcknowledgeRequest {
  operator_note?: string
}
export interface AlertAcknowledgeResponse {
  alert_id: string
  acknowledged: boolean
  message: string
}

// -------------------------------------------------- history -----------------

/** GET /history item (src/main.py: HistoryListItem). */
export interface HistoryListItem {
  job_id: string
  created_at: string
  updated_at: string
  video_source: string | null
  status: 'queued' | 'running' | 'completed' | 'failed' | string
  risk_level: string | null
  risk_score: number | null
  risk_status: RiskStatus | string
  summary: string | null
}

/** GET /history/{job_id} (src/main.py: HistoryDetail). */
export interface HistoryDetail {
  job_id: string
  created_at: string
  updated_at: string
  status: 'queued' | 'running' | 'completed' | 'failed' | string
  video_source: string | null
  report: SafirReport | null
  /**
   * Persisted trace events — same TraceEvent schema the live SSE stream
   * produces (src/observability/trace_serializer.py::make_event). Empty for
   * analyses completed before this field existed, or if persistence failed
   * (best-effort on the backend) — the UI must treat that as "no pipeline
   * detail available", not an error.
   */
  trace_events: TraceEvent[]
}

// -------------------------------------------------- ask safir ---------------

/** POST /ask request (src/main.py: AskRequest). */
export interface AskRequest {
  question: string
  job_id?: string | null
}

/** A grounded source in the answer (src/assistant: Source). Only real fields. */
export interface AskSource {
  type: 'analysis' | 'regulation' | string
  text?: string | null
  score?: number | null
  label?: string | null
}

/** POST /ask response (src/main.py: AskResponse). */
export interface AskResponse {
  answer: string
  sources: AskSource[]
  job_id: string | null
  context_used: string[]
}

/** POST /alerts/trigger */
export interface AlertTriggerRequest {
  risk_score: number
  risk_level: string
  recommended_action: string
  operator_note?: string
}
export interface AlertTriggerResponse {
  acknowledged: boolean
  alert_id: string
  message: string
}

// -------------------------------------------------------- conversations -----
// SAFIR Asistan sohbet gecmisi. Yalnizca UI/gecmis icindir — bu mesajlar
// backend'de LLM prompt'una GERI BESLENMEZ (bkz. src/assistant/ask_service.py).

/** POST /conversations istek govdesi. */
export interface ConversationCreateRequest {
  title?: string | null
  job_id?: string | null
}

/** src/main.py: ConversationSummary (sohbet listesi/ozet, mesajlar HARIC). */
export interface Conversation {
  conversation_id: string
  created_at: string
  updated_at: string
  title: string | null
  job_id: string | null
  message_count: number
}

/** src/main.py: ConversationMessageOut. */
export interface ConversationMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

/** src/main.py: ConversationContextOut — kullanıcının sohbete eklediği metin/not (Adım 3). */
export interface ConversationContext {
  id: number
  kind: 'note' | string
  label: string | null
  content: string
  created_at: string
}

/** src/main.py: ConversationDocumentOut — yüklenen belge metadata + durumu (Adım 4). Ham dosya içeriği ASLA gelmez. */
export interface ConversationDocument {
  document_id: string
  filename: string
  file_ext: 'pdf' | 'txt' | 'docx' | string
  file_size_bytes: number
  page_count: number | null
  status: 'processing' | 'ready' | 'error'
  error_message: string | null
  created_at: string
}

/** GET /conversations/{id} yaniti: ozet + tam mesaj gecmisi + ek bağlamlar + belgeler. */
export interface ConversationDetail extends Conversation {
  messages: ConversationMessage[]
  context: ConversationContext[]
  documents: ConversationDocument[]
}

/** POST /conversations/{id}/messages istek govdesi. */
export interface ConversationMessageCreateRequest {
  role: 'user' | 'assistant'
  content: string
}

/** POST /conversations/{id}/context istek govdesi (Adım 3: yalnızca metin/not). */
export interface ConversationContextCreateRequest {
  content: string
  label?: string | null
}

/** GET /ask/stream ilk SSE olayi (meta) — sonraki olaylar {delta: string}. */
export interface AskStreamMeta {
  sources: AskSource[]
  job_id: string | null
  context_used: string[]
}

// -------------------------------------------------- system (Data Center) ----

/** GET /system/overview: src/main.py SystemOverviewTotals — gercek DB/disk sayimlari. */
export interface SystemOverviewTotals {
  total_analyses: number
  completed_analyses: number
  failed_analyses: number
  running_or_queued_analyses: number
  total_conversations: number
  total_messages: number
  analyses_with_trace: number
  stored_representative_frame_count: number
}

/** GET /system/overview yaniti (src/main.py: SystemOverviewResponse). */
export interface SystemOverview {
  totals: SystemOverviewTotals
  generated_at: string
  scan_limit: number
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
