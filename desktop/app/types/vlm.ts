/** Types for the VLM Direct Analysis mode (video -> VLM, mock data for now). */

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
