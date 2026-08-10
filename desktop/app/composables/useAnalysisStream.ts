/**
 * SSE controller for GET /analyze/jobs/{job_id}/stream (built in ADIM 2).
 *
 * Opens a browser EventSource, parses each `data:` line into a TraceEvent, and
 * writes results into the central Pinia store (spec #16). Terminal `event: end`
 * (job done/error) and `event: error` (unknown job) are handled, then the socket
 * is closed so EventSource does not auto-reconnect.
 */
import { onScopeDispose } from 'vue'
import type { JobStatus, TraceEvent } from '~/types/api'

export type StreamConnection =
  | 'idle'
  | 'connecting'
  | 'open'
  | 'closed'
  | 'error'

export function useAnalysisStream() {
  const { streamJobUrl } = useSafirApi()
  const store = useAnalysisStore()

  let es: EventSource | null = null

  function stop() {
    if (es) {
      es.close()
      es = null
    }
    if (store.connection === 'open' || store.connection === 'connecting') {
      store.setConnection('closed')
    }
  }

  function start(jobId: string) {
    stop()
    store.setConnection('connecting')
    es = new EventSource(streamJobUrl(jobId))

    es.onopen = () => store.setConnection('open')

    // Default (unnamed) events carry serialized TraceEvent JSON.
    es.onmessage = (msg: MessageEvent) => {
      if (!msg.data) return
      try {
        store.pushTraceEvent(JSON.parse(msg.data) as TraceEvent)
      } catch {
        // ignore malformed frames
      }
    }

    // Server terminal event: `event: end` -> { status }.
    es.addEventListener('end', (msg) => {
      let status: JobStatus = 'done'
      try {
        status = (JSON.parse((msg as MessageEvent).data) as { status: JobStatus }).status
      } catch {
        /* keep default */
      }
      store.setEndStatus(status)
      stop()
    })

    // 'error' surfaces both server `event: error` (has .data) and transport
    // errors (no .data). Only the former is terminal.
    es.addEventListener('error', (msg) => {
      const data = (msg as MessageEvent).data
      if (data) {
        try {
          store.setStreamError((JSON.parse(data) as { detail?: string }).detail ?? 'stream error')
        } catch {
          store.setStreamError('stream error')
        }
        store.setConnection('error')
        stop()
      } else if (store.connection !== 'closed') {
        store.setConnection('error')
      }
    })
  }

  onScopeDispose(stop)

  return { start, stop }
}
