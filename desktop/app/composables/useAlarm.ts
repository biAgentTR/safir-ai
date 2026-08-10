/**
 * Critical-risk audible alarm. Mirrors the Streamlit behaviour: a short beep is
 * played ONCE per analysis when the risk crosses CRITICAL_ALARM_THRESHOLD (70).
 * The "played for this job" guard lives in the store (`alarmPlayedFor`), so the
 * same job never re-triggers the sound (equivalent to `alarm_played_for`).
 *
 * The tone is synthesized with the Web Audio API (no bundled asset, no base64).
 */
export function useAlarm() {
  const store = useAnalysisStore()

  function beep() {
    try {
      const Ctx =
        (window as unknown as { AudioContext?: typeof AudioContext; webkitAudioContext?: typeof AudioContext })
          .AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
      if (!Ctx) return
      const ctx = new Ctx()
      const now = ctx.currentTime
      // two short pulses ~ a klaxon-ish alert
      for (let i = 0; i < 2; i++) {
        const osc = ctx.createOscillator()
        const gain = ctx.createGain()
        osc.type = 'square'
        osc.frequency.value = 880
        const t0 = now + i * 0.28
        gain.gain.setValueAtTime(0.0001, t0)
        gain.gain.exponentialRampToValueAtTime(0.25, t0 + 0.02)
        gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.22)
        osc.connect(gain).connect(ctx.destination)
        osc.start(t0)
        osc.stop(t0 + 0.24)
      }
      setTimeout(() => ctx.close().catch(() => {}), 900)
    } catch {
      // Autoplay may be blocked until the user interacts; fail silently.
    }
  }

  /**
   * Play the alarm once for `signature` (job id + risk score). Safe to call on
   * every report update — it self-dedupes via the store.
   */
  function maybeAlarm(signature: string) {
    if (!store.isCritical) return
    if (store.alarmPlayedFor === signature) return
    store.markAlarmPlayed(signature)
    beep()
  }

  return { beep, maybeAlarm }
}
