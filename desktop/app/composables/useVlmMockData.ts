/**
 * Shared summary/count helpers for the VLM Direct Analysis dashboard's real
 * event data (see pages/vlm-direct/index.vue + useVlmDirectEvents.ts, which
 * maps the actual backend analysis into VlmEvent[]). No mock or placeholder
 * data lives here — these are pure functions over whatever real events exist.
 */
import type { VlmEvent, VlmAnalysisSummary, VlmRiskLevel } from '~/types/vlm'

const RISK_WEIGHT: Record<VlmRiskLevel, number> = { low: 15, mid: 40, high: 70, crit: 95 }

export function summarize(events: VlmEvent[]): VlmAnalysisSummary {
  if (!events.length) {
    return { overallRiskScore: 0, totalEvents: 0, criticalEvents: 0, averageConfidence: 0 }
  }
  const worst = Math.max(...events.map((e) => RISK_WEIGHT[e.riskLevel]))
  return {
    overallRiskScore: worst,
    totalEvents: events.length,
    criticalEvents: events.filter((e) => e.riskLevel === 'crit').length,
    averageConfidence: Math.round(events.reduce((s, e) => s + e.confidence, 0) / events.length),
  }
}

export function riskLevelCounts(events: VlmEvent[]): Record<VlmRiskLevel, number> {
  const counts: Record<VlmRiskLevel, number> = { low: 0, mid: 0, high: 0, crit: 0 }
  for (const e of events) counts[e.riskLevel]++
  return counts
}

export function eventTypeCounts(events: VlmEvent[]): { type: string; count: number }[] {
  const map = new Map<string, number>()
  for (const e of events) map.set(e.type, (map.get(e.type) ?? 0) + 1)
  return [...map.entries()].map(([type, count]) => ({ type, count })).sort((a, b) => b.count - a.count)
}
