/**
 * Maps a REAL SAFIR analysis (from the Pinia analysis store, same store the
 * low-budget Workspace uses) into the VLM Direct dashboard's VlmEvent[]
 * shape. Both analysis modes ultimately run the SAME EVREN-backed pipeline
 * (src/main.py::SafirPipeline.stage_vlm sends the video directly to EVREN,
 * config-driven — see configs/config.yaml `vlm.active_model`/`llm.active_model`,
 * not hardcoded to any provider) — this mode just presents the result
 * differently (video timeline + event table instead of the pipeline-stage
 * workspace).
 *
 * Prefers the VLM stage's structured, per-event data (real risk_score/type/
 * confidence per clustered event) when the trace is available (live SSE or a
 * persisted trace); falls back to the final report's flat timeline (only
 * timestamp+description, no per-event type/score) when it isn't — e.g. a
 * degraded run, or an older persisted analysis without a stored trace.
 */
import type { SafirReport, TimelineEntry, TraceEvent, VlmStageEventData } from '~/types/api'
import type { VlmEvent, VlmRiskLevel } from '~/types/vlm'

function riskLevelFromScore(score: number | null | undefined): VlmRiskLevel {
  if (score == null) return 'mid'
  if (score >= 76) return 'crit'
  if (score >= 51) return 'high'
  if (score >= 26) return 'mid'
  return 'low'
}

function toConfidencePct(confidence: number | null | undefined): number {
  if (confidence == null) return 0
  return Math.round(confidence <= 1 ? confidence * 100 : confidence)
}

export function mapVlmDirectEvents(
  vlmStage: TraceEvent<VlmStageEventData> | undefined,
  report: SafirReport | null,
): VlmEvent[] {
  const data = vlmStage?.data
  const structured = data && !('progress' in data) ? data.structured_events : undefined
  if (structured?.length) {
    return structured
      .map((e, i) => ({
        id: e.event_id || `vlm-evt-${i}`,
        timestamp: e.start_time ?? 0,
        // `event_name` is the model's own free-form label and is ALWAYS
        // present (bkz. types/api.ts VlmEvent) — `canonical_event_type` is
        // only a fallback for the rare case a caller strips it.
        type: e.event_name || e.canonical_event_type || 'Tespit Edilen Olay',
        description: e.description || '—',
        riskLevel: riskLevelFromScore(e.risk_score),
        confidence: toConfidencePct(e.confidence),
      }))
      .sort((a, b) => a.timestamp - b.timestamp)
  }

  if (report?.timeline?.length) {
    const overallTone = riskLevelFromScore(report.risk_status === 'unknown' ? null : report.risk_score)
    return report.timeline.map((entry: TimelineEntry, i) => ({
      id: `timeline-evt-${i}`,
      timestamp: entry.timestamp,
      type: report.detected_event_types?.[0] || 'Gözlem',
      description: entry.description,
      riskLevel: overallTone,
      confidence: 0,
    }))
  }

  return []
}
