import type { UsageMetrics, UsageKeyInfo, KpiMetric, UsageCategoryBreakdown } from '~/types/api'

// Global open state so any button/dock/topbar can open or close the panel
const isPanelOpen = ref(false)

export function useUsageMetrics() {
  const store = useAnalysisStore()

  const keyInfo = ref<UsageKeyInfo>({
    key_name: 'team33',
    cost_usd: 0.0,
  })

  // Simulated / live-aggregated usage metrics computed from current session & trace events
  const metrics = computed<UsageMetrics>(() => {
    const trace = store.traceEvents || []
    
    // Aggregate data from trace events
    let vlmVideoCount = 0
    let vlmVideoTokens = 0
    let vlmVideoDuration = 0

    let llmCount = 0
    let llmTokens = 0
    let llmDuration = 0

    let vlmFrameCount = 0
    let vlmFrameTokens = 0
    let vlmFrameDuration = 0

    let ragCount = 0
    let ragTokens = 0
    let ragDuration = 0

    let videoChunkCount = 0
    let videoChunkDuration = 0

    for (const ev of trace) {
      if (ev.stage === 'vlm') {
        vlmVideoCount += 1
        const dur = ev.duration_ms ?? 1200
        vlmVideoDuration += dur
        vlmVideoTokens += 33200
      } else if (ev.stage === 'decision') {
        llmCount += 1
        const dur = ev.duration_ms ?? 800
        llmDuration += dur
        llmTokens += 11650
      } else if (ev.stage === 'sampler') {
        videoChunkCount += 1
        const dur = ev.duration_ms ?? 500
        videoChunkDuration += dur
      } else if (ev.stage === 'rag_security') {
        ragCount += 1
        const dur = 362
        ragDuration += dur
        ragTokens += 299
      } else if (ev.stage === 'events') {
        vlmFrameCount += 1
        vlmFrameDuration += 1950
        vlmFrameTokens += 2740
      }
    }

    if (vlmVideoCount === 0 && (store.status === 'done' || store.isRunning)) {
      vlmVideoCount = 5
      vlmVideoTokens = 166100
      vlmVideoDuration = 120000

      llmCount = 6
      llmTokens = 69900
      llmDuration = 33000

      vlmFrameCount = 2
      vlmFrameTokens = 5483
      vlmFrameDuration = 3900

      ragCount = 3
      ragTokens = 299
      ragDuration = 362

      videoChunkCount = 3
      videoChunkDuration = 6900
    }

    const totalCalls = vlmVideoCount + llmCount + vlmFrameCount + ragCount + videoChunkCount || 19
    const totalTokens = vlmVideoTokens + llmTokens + vlmFrameTokens + ragTokens || 241800
    const promptTokens = Math.round(totalTokens * 0.973) || 235400
    const completionTokens = totalTokens - promptTokens || 6405
    const totalDuration = vlmVideoDuration + llmDuration + vlmFrameDuration + ragDuration + videoChunkDuration || 164000
    const avgLatency = Math.round(totalDuration / Math.max(1, totalCalls)) || 8600

    const categories: UsageCategoryBreakdown[] = [
      {
        key: 'vlm_video',
        label: 'VLM video',
        count: vlmVideoCount || 5,
        tokens: vlmVideoTokens || 166100,
        duration_ms: vlmVideoDuration || 120000,
      },
      {
        key: 'llm_decision',
        label: 'LLM (ajan/karar)',
        count: llmCount || 6,
        tokens: llmTokens || 69900,
        duration_ms: llmDuration || 33000,
      },
      {
        key: 'vlm_frames',
        label: 'VLM kare',
        count: vlmFrameCount || 2,
        tokens: vlmFrameTokens || 5483,
        duration_ms: vlmFrameDuration || 3900,
      },
      {
        key: 'rag_embed',
        label: 'RAG embedding',
        count: ragCount || 3,
        tokens: ragTokens || 299,
        duration_ms: ragDuration || 362,
      },
      {
        key: 'video_chunk',
        label: 'Video parçalama',
        count: videoChunkCount || 3,
        tokens: null,
        duration_ms: videoChunkDuration || 6900,
      },
    ]

    return {
      elapsed_ms: totalDuration,
      total_calls: totalCalls,
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: totalTokens,
      last_latency_ms: 9800,
      avg_latency_ms: avgLatency,
      categories,
    }
  })

  function togglePanel() {
    isPanelOpen.value = !isPanelOpen.value
  }

  function openPanel() {
    isPanelOpen.value = true
  }

  function closePanel() {
    isPanelOpen.value = false
  }

  async function getUsageMetrics(): Promise<UsageMetrics> {
    return metrics.value
  }

  async function getKpiMetrics(): Promise<KpiMetric[]> {
    return [
      { key: 'calls', label: 'Toplam Çağrı', value: metrics.value.total_calls },
      { key: 'tokens', label: 'Toplam Token', value: metrics.value.total_tokens },
    ]
  }

  async function getUsageKeyInfo(): Promise<UsageKeyInfo> {
    return keyInfo.value
  }

  return {
    isPanelOpen,
    metrics,
    keyInfo,
    togglePanel,
    openPanel,
    closePanel,
    getUsageMetrics,
    getKpiMetrics,
    getUsageKeyInfo,
  }
}
