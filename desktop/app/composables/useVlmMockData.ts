/**
 * Mock data for the VLM Direct Analysis dashboard.
 *
 * There is no live VLM-direct backend yet (unlike the low-budget mode, whose
 * dashboard is wired to the real FastAPI backend via useSafirApi). This
 * generates a plausible, deterministic result set scaled to whatever video
 * duration the operator actually uploads, purely so the UI/UX can be built
 * and reviewed now. Every component below reads events/summary through this
 * composable's return shape — swapping this for a real API composable later
 * (e.g. useVlmApi) requires no changes to the components themselves.
 */
import type { VlmEvent, VlmAnalysisSummary, VlmRiskLevel } from '~/types/vlm'

const EVENT_TEMPLATES: { type: string; description: string; riskLevel: VlmRiskLevel }[] = [
  { type: 'PPE İhlali', description: 'Baret takmayan personel tespit edildi.', riskLevel: 'mid' },
  { type: 'Yasak Bölge İhlali', description: 'Personel, forklift operasyon alanına izinsiz girdi.', riskLevel: 'high' },
  { type: 'Düşme Riski', description: 'Yükseklikte çalışan personelde emniyet kemeri gözlemlenmedi.', riskLevel: 'crit' },
  { type: 'Araç-Personel Yakınlığı', description: 'Forklift ile personel arasındaki mesafe güvenlik sınırının altına düştü.', riskLevel: 'high' },
  { type: 'Yangın/Duman Şüphesi', description: 'Kamera görüntüsünde duman benzeri bir görüntü tespit edildi.', riskLevel: 'crit' },
  { type: 'Uygun Olmayan Kaldırma', description: 'Ağır yükün hatalı teknikle kaldırıldığı gözlemlendi.', riskLevel: 'mid' },
  { type: 'Ekipman Hatası', description: 'Konveyör bandında anormal titreşim/duruş tespit edildi.', riskLevel: 'low' },
  { type: 'Toplanma Alanı İhlali', description: 'Acil toplanma alanına malzeme bırakıldığı görüldü.', riskLevel: 'low' },
]

const FRACTIONS = [0.06, 0.14, 0.27, 0.38, 0.5, 0.63, 0.74, 0.86, 0.94]

function seededConfidence(i: number): number {
  // Deterministic 70-96 spread, no Math.random so repeated renders are stable.
  return 70 + ((i * 37) % 27)
}

export function generateMockEvents(durationSeconds: number): VlmEvent[] {
  if (!durationSeconds || durationSeconds <= 0) return []
  return FRACTIONS.map((frac, i) => {
    const tpl = EVENT_TEMPLATES[i % EVENT_TEMPLATES.length]
    return {
      id: `mock-evt-${i}`,
      timestamp: Math.min(durationSeconds - 0.5, Math.max(0, frac * durationSeconds)),
      type: tpl.type,
      description: tpl.description,
      riskLevel: tpl.riskLevel,
      confidence: seededConfidence(i),
    }
  }).sort((a, b) => a.timestamp - b.timestamp)
}

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
