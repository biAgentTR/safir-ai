/**
 * SSE controller for GET /ask/stream (SAFIR Asistan streaming chat).
 *
 * Mirrors useAnalysisStream.ts's EventSource conventions exactly (default
 * `data:` messages, terminal `event: end`, `event: error`) but is decoupled
 * from the Pinia store — assistant.vue keeps its own local, page-scoped chat
 * state (see that page's comments for why).
 */
import { onScopeDispose } from 'vue'
import type { AskStreamMeta } from '~/types/api'

export type AskStreamConnection = 'idle' | 'connecting' | 'open' | 'closed' | 'error'

export interface AskStreamCallbacks {
  onMeta?: (meta: AskStreamMeta) => void
  onDelta?: (delta: string) => void
  onEnd?: () => void
  onError?: (detail: string) => void
}

export function useAskStream() {
  const { askStreamUrl } = useSafirApi()

  let es: EventSource | null = null

  function stop() {
    if (es) {
      es.close()
      es = null
    }
  }

  function start(
    question: string,
    jobId: string | null,
    conversationId: string | null,
    cb: AskStreamCallbacks,
    useVideo = false,
  ) {
    stop()
    es = new EventSource(askStreamUrl(question, jobId, conversationId, useVideo))

    // First default (unnamed) event is the meta payload (sources/job_id/context_used);
    // every subsequent one carries { delta }. Both arrive as plain `data:` lines.
    let gotMeta = false
    es.onmessage = (msg: MessageEvent) => {
      if (!msg.data) return
      let parsed: Record<string, unknown>
      try {
        parsed = JSON.parse(msg.data)
      } catch {
        return // ignore malformed frames
      }
      if (!gotMeta) {
        gotMeta = true
        cb.onMeta?.(parsed as unknown as AskStreamMeta)
        return
      }
      if (typeof parsed.delta === 'string' && parsed.delta) {
        cb.onDelta?.(parsed.delta)
      }
    }

    es.addEventListener('end', () => {
      cb.onEnd?.()
      stop()
    })

    // 'error' surfaces both server `event: error` (has .data) and transport
    // errors (no .data, e.g. connection dropped) — only the former carries a message.
    es.addEventListener('error', (msg) => {
      const data = (msg as MessageEvent).data
      let detail = 'SAFİR şu anda cevap oluşturamadı.'
      if (data) {
        try {
          detail = (JSON.parse(data) as { detail?: string }).detail ?? detail
        } catch {
          /* keep default */
        }
      }
      cb.onError?.(detail)
      stop()
    })
  }

  onScopeDispose(stop)

  return { start, stop }
}
