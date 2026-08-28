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
export type EscalationTier = 'monitor' | 'notify' | 'alarm' | 'pending_review'
/**
 * 'assessed': risk_score/risk_level guvenilir sekilde hesaplandi (0 dahil gecerli deger).
 * 'unknown': VLM/LLM/ajan karar zincirinde hata olustu; risk_score=null, ASLA dusuk risk
 * olarak yorumlanmamali (bkz. src/agent/langgraph_agent.py: AgentDecision.risk_status).
 */
export type RiskStatus = 'assessed' | 'unknown'

/**
 * A mock action tool the 05 LangGraph Agent actually CALLED (tool_call), not
 * just a text suggestion in `actions` — see src/agent/tools.py:
 * notify_health_team_tool / dispatch_security_tool / trigger_area_lockdown_tool.
 * `args`/`result` shape depends on which tool was called.
 */
export interface TriggeredMockAction {
  /** 'notify_health_team_tool' | 'dispatch_security_tool' | 'trigger_area_lockdown_tool' (bkz. yukarisi). */
  tool: string
  args: Record<string, unknown>
  result: string
}

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
  /** 'rule_engine' | 'agent' | 'unknown' - bkz. src/main.py::build_report. */
  risk_source?: string | null
  risk_explanation?: string | null
  recommended_action: string
  actions: string[]
  /** Ajanın gerçekten çağırdığı mock aksiyon araçları (bkz. TriggeredMockAction). */
  triggered_mock_actions?: TriggeredMockAction[]
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
  | 'rag_security'
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
  /** true when the active VLM is video-based (EVREN) and this stage was intentionally skipped — see BaseVLM.requires_frame_sampling. */
  skipped?: boolean
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

/**
 * Step-by-step VLM progress (src/vlm/evren_vlm.py::VlmProgressCallback via
 * src/main.py::run's _on_vlm_progress). Emitted as one or more `status:
 * "running"` trace events on the SAME "vlm" stage key, BEFORE the final
 * `VlmStageData` "completed"/"failed" event — e.g. while EVREN splits a long
 * video into chunks and sends each one separately. `phase` drives which
 * other fields are present; treat unknown phases as "still working" (the
 * `summary` string is always safe to show regardless).
 */
export interface VlmProgressData {
  progress: {
    phase: 'chunking' | 'chunk_start' | 'chunk_done' | 'chunk_failed' | string
    total_chunks?: number
    chunk_index?: number
    range_label?: string | null
    elapsed_sec?: number
    video_mb?: number
    error?: string
  }
}

/** The "vlm" stage's trace `data` is either a progress tick or the final result — narrow on `'progress' in data`. */
export type VlmStageEventData = VlmStageData | VlmProgressData

/** decision stage — NOTE: raw_response is intentionally absent server-side. */
export interface Decision {
  risk_score: number | null
  risk_level: string
  risk_status: RiskStatus | string
  summary: string
  recommended_action: string
  actions: string[]
  events: unknown[]
  /** Ajanın bu çalışmada gerçekten çağırdığı mock aksiyon araçları (bkz. TriggeredMockAction). */
  triggered_mock_actions?: TriggeredMockAction[]
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


/**
 * rag_security stage: src/observability/trace_serializer.py serialize_rag_security.
 *
 * `embedding_score` (E5/FAISS semantic similarity), `relevance_score`
 * (deterministic weighted-hybrid: semantic+lexical+keyword+metadata+phrase)
 * and `cross_encoder_score` (local Cross-Encoder re-ranking signal, only
 * present when the query-level `cross_encoder_status` is `'used'`) are
 * DELIBERATELY separate fields — never merge them, never label any of them
 * "risk"/"confidence"/"probability". The old `rerank_score` field name from
 * the removed Gemini/Groq LLM-as-judge reranker no longer exists backend-side.
 */
export interface RagResultTelemetry {
  rank: number | null
  final_rank: number | null
  chunk_id: string | null
  document_id: string | null
  document_title: string | null
  article_number: string | null
  source_url: string | null
  embedding_score: number
  relevance_score: number | null
  /** Five components `relevance_score` is computed from (bkz. deterministic_reranker.py). `null` = relevance scoring was disabled/not computed for this candidate — never fabricated. */
  semantic_score: number | null
  lexical_score: number | null
  keyword_score: number | null
  metadata_score: number | null
  phrase_score: number | null
  /** Local Cross-Encoder relevance signal — a ranking signal only, not a risk/confidence value. `null` when the query's cross_encoder_status is not 'used'. */
  cross_encoder_score: number | null
  relevance_status: 'accepted' | 'rejected' | null
  relevance_reason: string | null
  selected: boolean
  text: string
}

export interface RelevanceWeights {
  semantic: number
  lexical: number
  keyword: number
  metadata: number
  phrase: number
}

/**
 * RAG retrieval telemetrisi for tek bir semantik sorgu. `null` = bu turda hic
 * RAG sorgusu yapilmadi (VLM keyword uretmedi) — "0 sonuc" ile KARISTIRILMAZ.
 */
export interface RagTelemetry {
  query_length: number
  candidate_count: number
  final_count: number
  zero_result: boolean
  retrieval_status: 'relevance_scored' | 'embedding_only' | 'insufficient_evidence' | 'empty_index' | string
  threshold: number | null
  embedding_latency_ms: number
  rerank_latency_ms: number | null
  total_latency_ms: number
  avg_embedding_score: number | null
  avg_relevance_score: number | null
  /** Weights `deterministic_reranker.score_candidate()` actually used for this query (read from config, not hardcoded). `null` if unavailable. */
  relevance_weights: RelevanceWeights | null
  /** 'used' | 'unavailable' | 'disabled' — whether the local Cross-Encoder ran for this query. See RagResultTelemetry docstring. */
  cross_encoder_status: 'used' | 'unavailable' | 'disabled' | string
  results: RagResultTelemetry[]
}

/** Tek bir Prompt Injection Guard kontrolunun GUVENLI (ham metin icermeyen) telemetrisi. */
export interface SecurityGuardCheck {
  source: 'user_prompt' | 'vlm_description' | 'vlm_event_description' | string
  is_injection: boolean
  confidence: number
  action: 'allow' | 'quarantine'
  /** Gemini'nin kisa gerekce metni (bilgi amacli — karar mekanizmasi DEGIL, karari `action`/`confidence` belirler). */
  reason: string | null
  guard_failed: boolean
  latency_ms: number
}

export interface RagSecurityStageData {
  rag: RagTelemetry | null
  security: SecurityGuardCheck[]
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
  /** Ask the VIDEO directly (EVREN prefix-cache follow-up); falls back to text silently if unavailable. */
  use_video?: boolean
}

/** A grounded source in the answer (src/assistant: Source). Only real fields. */
export interface AskSource {
  type: 'analysis' | 'regulation' | 'document' | 'video' | string
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

/** GET /ask/suggestions response (src/main.py: AskSuggestionsResponse) — report-specific follow-up chips. */
export interface AskSuggestionsResponse {
  suggestions: string[]
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
  /**
   * RAG + Prompt Injection Guard telemetrisi (persisted trace_events'in
   * "rag_security" asamasindan toplanir). Sayim alanlari her zaman gercek
   * (veri yoksa 0); ortalama alanlari veri yoksa `null` ("N/A") doner —
   * ASLA uydurulmuş bir sayi degildir.
   */
  total_events_detected: number
  critical_risk_analyses: number
  rag_query_count: number
  rag_zero_result_count: number
  avg_embedding_latency_ms: number | null
  avg_rerank_latency_ms: number | null
  avg_total_rag_latency_ms: number | null
  guard_checks: number
  guard_allowed: number
  guard_quarantined: number
  guard_failures: number
  guard_fail_closed_blocks: number
  avg_guard_latency_ms: number | null
  avg_guard_confidence: number | null
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
