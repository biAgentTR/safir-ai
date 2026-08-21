/** Small presentation helpers shared across workspace components. */
import type { TraceStage } from '~/types/api'

/** Seconds -> MM:SS (matches the backend's timestamp_str convention). */
export function mmss(seconds: number): string {
  const total = Math.max(0, Math.round(seconds))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/**
 * Risk level -> tailwind text/ring/bg tone key.
 *
 * NOTE: this only maps `risk_level` strings. Callers with a `risk_status`
 * field (SafirReport/HistoryListItem/Decision) MUST check
 * `risk_status === 'unknown'` THEMSELVES before calling this — an unrecognized
 * level string here falls back to 'low', which would silently misrender a
 * failed/undetermined analysis as low risk (the exact P0 bug this step fixes).
 */
export function riskTone(level?: string | null): 'low' | 'mid' | 'high' | 'crit' | 'unknown' {
  switch ((level ?? '').toLowerCase()) {
    case 'unknown':
    case 'belirsiz':
      return 'unknown'
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
  unknown: 'text-slate-400',
}
export const RISK_BG: Record<string, string> = {
  low: 'bg-risk-low',
  mid: 'bg-risk-mid',
  high: 'bg-risk-high',
  crit: 'bg-risk-crit',
  unknown: 'bg-slate-500',
}

/** Canonical stage order + display labels (mirror trace_serializer). */
export const STAGE_META: { stage: TraceStage; label: string; blurb: string }[] = [
  { stage: 'sampler', label: 'Kare Örnekleme', blurb: 'CPU uyarlanabilir örnekleyici' },
  { stage: 'vlm', label: 'Çok Modlu Analiz', blurb: 'Görsel-dil modeli' },
  { stage: 'events', label: 'Olay Analizi', blurb: 'Tespit / zamansal / kural' },
  { stage: 'agent_context', label: 'Bağlam ve RAG', blurb: 'Ajan bağlamı oluşturma' },
  { stage: 'decision', label: 'Ajan Kararı', blurb: 'Risk ve önerilen aksiyon' },
  { stage: 'escalation', label: 'Risk Yükseltme', blurb: 'Otomatik tetikleme politikası' },
  { stage: 'report', label: 'Nihai Rapor', blurb: 'Yapılandırılmış çıktı' },
]

export function stageLabel(stage: TraceStage): string {
  return STAGE_META.find((s) => s.stage === stage)?.label ?? stage
}

/**
 * Input -> Output summary per stage, derived ONLY from the real trace payload.
 * When a stage's event exists, counts are filled from it; otherwise the generic
 * role words are shown. No fabricated data.
 */
export function stageFlow(
  stage: TraceStage,
  data: Record<string, unknown> | undefined,
): { in: string; out: string } {
  const n = (v: unknown) => (Array.isArray(v) ? v.length : typeof v === 'number' ? v : 0)
  const d = (data ?? {}) as Record<string, any>
  switch (stage) {
    case 'sampler':
      return {
        in: data ? `${d.stats?.total_frames_scanned ?? 0} kare` : 'ham kareler',
        out: data ? `${n(d.evidence_frames)} kanıt karesi` : 'kanıt kareleri',
      }
    case 'vlm':
      return {
        in: data ? `${d.frames_sent || d.frame_count || 0} kare + prompt` : 'temsili kareler + prompt',
        out: data ? `gözlem + ${n(d.structured_events)} olay` : 'gözlem + yapılandırılmış olaylar',
      }
    case 'events':
      return {
        in: data ? `${n(d.detected_events)} tespit` : 'tespit edilen olaylar',
        out: data ? `${n(d.temporal_events)} zamansal · ${n(d.rule_matches)} kural` : 'zamansal olaylar / kural eşleşmeleri',
      }
    case 'agent_context':
      return {
        in: 'olaylar + mevzuat',
        out: data ? `${d.length ?? 0} karakter bağlam` : 'ajan bağlamı',
      }
    case 'decision': {
      const riskOut = data ? (d.risk_status === 'unknown' || d.risk_score == null ? 'risk belirsiz' : `risk ${d.risk_score}/100`) : 'risk'
      return {
        in: 'ajan bağlamı',
        out: data ? `${riskOut} · ${n(d.actions)} aksiyon` : 'risk / özet / aksiyonlar',
      }
    }
    case 'escalation':
      return {
        in: 'karar / risk',
        out: data ? `${d.tier}${d.auto_dispatched ? ' · oto-tetik' : ''}` : 'kademe / tetikleme',
      }
    case 'report': {
      const riskOut = data ? (d.risk_status === 'unknown' || d.risk_score == null ? 'risk belirsiz' : `risk ${d.risk_score}`) : 'risk'
      return {
        in: 'pipeline çıktıları',
        out: data ? `${riskOut} · nihai rapor` : 'nihai rapor',
      }
    }
    default:
      return { in: '—', out: '—' }
  }
}
