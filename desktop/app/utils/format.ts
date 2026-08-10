/** Small presentation helpers shared across workspace components. */
import type { TraceStage } from '~/types/api'

/** Seconds -> MM:SS (matches the backend's timestamp_str convention). */
export function mmss(seconds: number): string {
  const total = Math.max(0, Math.round(seconds))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/** Risk level -> tailwind text/ring/bg tone key. */
export function riskTone(level?: string | null): 'low' | 'mid' | 'high' | 'crit' {
  switch ((level ?? '').toLowerCase()) {
    case 'kritik':
    case 'critical':
      return 'crit'
    case 'yuksek':
    case 'yüksek':
    case 'high':
      return 'high'
    case 'orta':
    case 'medium':
      return 'mid'
    default:
      return 'low'
  }
}

export const RISK_TEXT: Record<string, string> = {
  low: 'text-risk-low',
  mid: 'text-risk-mid',
  high: 'text-risk-high',
  crit: 'text-risk-crit',
}
export const RISK_BG: Record<string, string> = {
  low: 'bg-risk-low',
  mid: 'bg-risk-mid',
  high: 'bg-risk-high',
  crit: 'bg-risk-crit',
}

/** Canonical stage order + display labels (mirror trace_serializer). */
export const STAGE_META: { stage: TraceStage; label: string; blurb: string }[] = [
  { stage: 'sampler', label: 'Frame Sampling', blurb: 'CPU adaptive sampler' },
  { stage: 'vlm', label: 'Multimodal Analysis', blurb: 'Vision-language model' },
  { stage: 'events', label: 'Event Analysis', blurb: 'Detected / temporal / rules' },
  { stage: 'agent_context', label: 'Context & RAG', blurb: 'Agent context assembly' },
  { stage: 'decision', label: 'Agent Decision', blurb: 'Risk & recommended action' },
  { stage: 'escalation', label: 'Risk Escalation', blurb: 'Auto dispatch policy' },
  { stage: 'report', label: 'Final Report', blurb: 'Structured output' },
]

export function stageLabel(stage: TraceStage): string {
  return STAGE_META.find((s) => s.stage === stage)?.label ?? stage
}
