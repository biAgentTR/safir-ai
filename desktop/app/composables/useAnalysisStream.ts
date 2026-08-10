/**
 * SSE client for GET /analyze/jobs/{job_id}/stream (built in ADIM 2).
 *
 * Connects with the browser EventSource API, parses each `data:` line into a
 * TraceEvent, and exposes the growing TraceEvent[] plus connection state. The
 * server terminates the stream with a named `event: end` (job done/error) or
 * `event: error` (unknown job); both are handled and the socket is closed so
 * EventSource does not auto-reconnect.
 *
 * This is infrastructure only — no advanced pipeline UI here (that is ADIM 4).
 */
import { ref, computed, onScopeDispose } from 'vue'
import type { JobStatus, TraceEvent } from '~/types/api'

export type StreamConnection =
  | 'idle'
  | 'connecting'
  | 'open'
  | 'closed'
  | 'error'

export function useAnalysisStream() {
  const { streamJobUrl } = useSafirApi()

  const events = ref<TraceEvent[]>([])
  const connection = ref<StreamConnection>('idle')
  const endStatus = ref<JobStatus | null>(null)
  const errorDetail = ref<string | null>(null)

  let es: EventSource | null = null

  const stagesSeen = computed(() => events.value.map((e) => e.stage))
  const completedCount = computed(
    () => events.value.filter((e) => e.status === 'completed').length,
  )
  const latestEvent = computed(() =>
    events.value.length ? events.value[events.value.length - 1] : null,
  )

  function stop() {
    if (es) {
      es.close()
      es = null
    }
    if (connection.value === 'open' || connection.value === 'connecting') {
      connection.value = 'closed'
    }
  }

  function reset() {
    stop()
    events.value = []
    endStatus.value = null
    errorDetail.value = null
    connection.value = 'idle'
  }

  function start(jobId: string) {
    reset()
    connection.value = 'connecting'
    es = new EventSource(streamJobUrl(jobId))

    es.onopen = () => {
      connection.value = 'open'
    }

    // Default (unnamed) events carry serialized TraceEvent JSON.
    es.onmessage = (msg: MessageEvent) => {
      if (!msg.data) return
      try {
        const parsed = JSON.parse(msg.data) as TraceEvent
        events.value = [...events.value, parsed]
      } catch {
        // Ignore malformed frames rather than breaking the stream.
      }
    }

    // Server-sent terminal event: `event: end` -> { status }.
    es.addEventListener('end', (msg) => {
      try {
        const payload = JSON.parse((msg as MessageEvent).data) as {
          status: JobStatus
        }
        endStatus.value = payload.status
      } catch {
        endStatus.value = 'done'
      }
      stop()
    })

    // Two things surface on 'error':
    //  1. server-sent `event: error` (unknown job) -> has .data with { detail }
    //  2. transport/connection error -> no .data
    es.addEventListener('error', (msg) => {
      const data = (msg as MessageEvent).data
      if (data) {
        try {
          errorDetail.value = (JSON.parse(data) as { detail?: string }).detail ?? 'stream error'
        } catch {
          errorDetail.value = 'stream error'
        }
        connection.value = 'error'
        stop()
      } else if (connection.value !== 'closed') {
        // Transport hiccup; EventSource will retry unless we are done.
        connection.value = 'error'
      }
    })
  }

  onScopeDispose(stop)

  return {
    events,
    connection,
    endStatus,
    errorDetail,
    stagesSeen,
    completedCount,
    latestEvent,
    start,
    stop,
    reset,
  }
}
