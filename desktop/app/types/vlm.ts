/**
 * Types for the VLM Direct Analysis mode. `VlmEvent` here is a UI-normalized
 * shape — see composables/useVlmDirectEvents.ts for how it's built from the
 * REAL backend analysis (src/main.py::stage_vlm's structured VLM events, or
 * the final report's timeline as a fallback), both served through the same
 * EVREN-backed /analyze/jobs pipeline the low-budget mode uses. The mock
 * generator (useVlmMockData.ts) is now only a placeholder shown before the
 * operator starts a real analysis.
 */

export type VlmRiskLevel = 'low' | 'mid' | 'high' | 'crit'

export interface VlmEvent {
  id: string
  /** Seconds from video start. */
  timestamp: number
  type: string
  description: string
  riskLevel: VlmRiskLevel
  /** VLM confidence, 0-100. */
  confidence: number
}

export interface VlmAnalysisSummary {
  overallRiskScore: number
  totalEvents: number
  criticalEvents: number
  averageConfidence: number
}
