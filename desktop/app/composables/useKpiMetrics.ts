/**
 * SAFİR KPI tanımları (şartname: "Ölçümleme ve KPI Tanımlama").
 *
 * Şartnamede istenen beş ölçüt burada AÇIK FORMÜLLERLE tanımlanır ve her biri
 * SADECE gerçek analiz verisinden (Pinia store'daki trace olayları + nihai
 * rapor) hesaplanır. Veri yoksa uydurma sayı üretilmez — `value: null` döner ve
 * panelde "—" gösterilir.
 *
 *   1. Olay Tespit Doğruluğu      — tespit edilen olayların ortalama güveni
 *   2. Özet Kalitesi              — rapor özet alanlarının doluluk oranı
 *   3. Aksiyon Önerisi Doğruluğu  — aksiyon/eskalasyon tutarlılık kontrolleri
 *   4. Kritik Olay Yakalama Oranı — kritik/yüksek kuralla eşleşen olay türü oranı
 *   5. İşlem Süresi               — uçtan uca analiz süresi
 */
import type { DetectedEvent, EventsStageData, RuleMatch, SamplerStats, TemporalEvent } from '~/types/api'
import { durationMs } from '~/utils/format'

export type KpiTone = 'good' | 'warn' | 'bad' | 'neutral' | 'muted'

export interface KpiDefinition {
  key: string
  label: string
  /** Ölçütün nasıl hesaplandığı — panelde açıkça gösterilir (şartname gereği). */
  formula: string
  /** Yüzde KPI'lar için 0-100; süre KPI'ı için saniye. null = ölçülemedi. */
  value: number | null
  display: string
  /** Değerin hangi ham sayılardan geldiği (ör. "12 tespit üzerinden"). */
  detail: string
  tone: KpiTone
}

const CRITICAL_SEVERITIES = ['kritik', 'critical', 'yuksek', 'yüksek', 'high']

function pct(part: number, whole: number): number | null {
  if (!whole) return null
  return Math.max(0, Math.min(100, (part / whole) * 100))
}

function pctTone(v: number | null): KpiTone {
  if (v == null) return 'muted'
  if (v >= 85) return 'good'
  if (v >= 60) return 'warn'
  return 'bad'
}

export function useKpiMetrics() {
  const store = useAnalysisStore()

  const eventsData = computed(() => store.eventForStage('events')?.data as EventsStageData | undefined)

  const samplerStats = computed<SamplerStats | undefined>(() => {
    const fromReport = store.report?.sampler_stats
    if (fromReport) return fromReport
    const s = store.eventForStage('sampler')?.data as { skipped?: boolean; stats?: SamplerStats } | undefined
    if (!s || s.skipped) return undefined
    return s.stats
  })

  /** 1) Olay Tespit Doğruluğu — detected + temporal olayların ortalama confidence'ı. */
  const detectionAccuracy = computed<KpiDefinition>(() => {
    const detected: DetectedEvent[] = eventsData.value?.detected_events ?? []
    const temporal: TemporalEvent[] = eventsData.value?.temporal_events ?? []
    const confidences = [...detected, ...temporal]
      .map((e) => e.confidence)
      .filter((c): c is number => typeof c === 'number' && !Number.isNaN(c))
    const avg = confidences.length
      ? (confidences.reduce((a, b) => a + b, 0) / confidences.length) * 100
      : null
    return {
      key: 'detection_accuracy',
      label: 'Olay Tespit Doğruluğu',
      formula: 'Tespit edilen olayların ortalama güven skoru (confidence)',
      value: avg,
      display: avg == null ? '—' : `%${avg.toFixed(1)}`,
      detail: confidences.length ? `${confidences.length} tespit üzerinden` : 'Tespit verisi yok',
      tone: pctTone(avg),
    }
  })

  /** 2) Özet Kalitesi — raporun özet/kanıt alanlarının doluluk oranı. */
  const summaryQuality = computed<KpiDefinition>(() => {
    const formula =
      'Dolu rapor özet alanı ÷ 7 (özet, doğal dil özeti, gerekçe, zaman çizelgesi, kanıt karesi, olay türü, mevzuat)'
    const r = store.report
    if (!r) {
      return {
        key: 'summary_quality',
        label: 'Özet Kalitesi',
        formula,
        value: null,
        display: '—',
        detail: 'Rapor bekleniyor',
        tone: 'muted',
      }
    }
    const checks = [
      !!r.summary?.trim(),
      (r.natural_language_summary?.trim().length ?? 0) >= 80,
      !!r.risk_explanation?.trim(),
      (r.timeline?.length ?? 0) > 0,
      (r.evidence_frames?.length ?? 0) > 0,
      (r.detected_event_types?.length ?? 0) > 0,
      (r.relevant_regulations?.length ?? 0) > 0,
    ]
    const passed = checks.filter(Boolean).length
    const v = pct(passed, checks.length)
    return {
      key: 'summary_quality',
      label: 'Özet Kalitesi',
      formula,
      value: v,
      display: v == null ? '—' : `%${v.toFixed(1)}`,
      detail: `${passed}/${checks.length} özet alanı dolu`,
      tone: pctTone(v),
    }
  })

  /** 3) Aksiyon Önerisi Doğruluğu — aksiyon/eskalasyon tutarlılık kontrolleri. */
  const actionAccuracy = computed<KpiDefinition>(() => {
    const formula =
      'Geçen tutarlılık kontrolü ÷ 5 (öneri metni, aksiyon listesi, eskalasyon kademesi, risk-kademe uyumu, tetiklenen aksiyon)'
    const r = store.report
    if (!r) {
      return {
        key: 'action_accuracy',
        label: 'Aksiyon Önerisi Doğruluğu',
        formula,
        value: null,
        display: '—',
        detail: 'Rapor bekleniyor',
        tone: 'muted',
      }
    }
    const tier = (r.escalation_tier ?? '') as string
    const score = r.risk_score
    const isCritical = score != null && score >= 70
    const checks = [
      // öneri metni üretildi mi
      !!r.recommended_action?.trim(),
      // somut aksiyon listesi var mı
      (r.actions?.length ?? 0) > 0,
      // eskalasyon kademesi atandı mı
      !!tier,
      // kademe risk seviyesiyle tutarlı mı (kritik risk 'monitor' ile kapatılmamalı)
      isCritical ? ['alarm', 'notify'].includes(tier) : !!tier,
      // kritik riskte gerçekten bir aksiyon tetiklendi mi
      isCritical ? (r.triggered_mock_actions?.length ?? 0) > 0 || r.auto_dispatched : true,
    ]
    const passed = checks.filter(Boolean).length
    const v = pct(passed, checks.length)
    return {
      key: 'action_accuracy',
      label: 'Aksiyon Önerisi Doğruluğu',
      formula,
      value: v,
      display: v == null ? '—' : `%${v.toFixed(1)}`,
      detail: `${passed}/${checks.length} tutarlılık kontrolü geçti`,
      tone: pctTone(v),
    }
  })

  /** 4) Kritik Olay Yakalama Oranı — kritik/yüksek kuralla eşleşen olay türü / tüm olay türü. */
  const criticalCaptureRate = computed<KpiDefinition>(() => {
    const matches: RuleMatch[] = eventsData.value?.rule_matches ?? []
    const detectedTypes = new Set<string>(
      [
        ...(store.report?.detected_event_types ?? []),
        ...(eventsData.value?.detected_events ?? []).map((e) => e.event_type),
      ].filter(Boolean),
    )
    const criticalTypes = new Set(
      matches
        .filter((m) => CRITICAL_SEVERITIES.includes((m.severity ?? '').toLocaleLowerCase('tr-TR')))
        .map((m) => m.event_type)
        .filter(Boolean),
    )
    const rate = pct(criticalTypes.size, detectedTypes.size)
    return {
      key: 'critical_capture',
      label: 'Kritik Olay Yakalama Oranı',
      formula: 'Kritik/yüksek şiddetli kural eşleşmesi olan olay türü ÷ tespit edilen olay türü',
      value: rate,
      display: rate == null ? '—' : `%${rate.toFixed(1)}`,
      detail: detectedTypes.size
        ? `${criticalTypes.size}/${detectedTypes.size} olay türü kritik kuralla eşleşti`
        : 'Olay türü tespit edilmedi',
      tone: rate == null ? 'muted' : 'neutral',
    }
  })

  /** 5) İşlem Süresi — sampler ölçümü, yoksa trace olaylarının toplam süresi. */
  const processingTime = computed<KpiDefinition>(() => {
    const formula = 'Uçtan uca analiz süresi (sampler ölçümü; yoksa boru hattı adım sürelerinin toplamı)'
    const elapsedSec = samplerStats.value?.elapsed_sec
    let seconds: number | null = typeof elapsedSec === 'number' && elapsedSec > 0 ? elapsedSec : null
    let source = 'sampler ölçümü'
    if (seconds == null) {
      const totalMs = (store.traceEvents ?? []).reduce((acc, ev) => acc + (ev.duration_ms ?? 0), 0)
      if (totalMs > 0) {
        seconds = totalMs / 1000
        source = 'boru hattı adım süreleri'
      }
    }
    return {
      key: 'processing_time',
      label: 'İşlem Süresi',
      formula,
      value: seconds,
      display: seconds == null ? '—' : durationMs(seconds * 1000),
      detail: seconds == null ? 'Süre ölçümü yok' : source,
      tone: 'neutral',
    }
  })

  const kpis = computed<KpiDefinition[]>(() => [
    detectionAccuracy.value,
    summaryQuality.value,
    actionAccuracy.value,
    criticalCaptureRate.value,
    processingTime.value,
  ])

  /** Ölçülebilen KPI var mı — panelin "veri bekleniyor" durumunu belirler. */
  const hasData = computed(() => kpis.value.some((k) => k.value != null))

  return {
    kpis,
    hasData,
  }
}
