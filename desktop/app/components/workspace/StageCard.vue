<script setup lang="ts">
// Detail panel for the currently selected pipeline stage. Renders the REAL
// serialized trace payload per stage. Never shows raw_response / secrets /
// system prompt (those are not in the payload by design).
import type {
  ContextStageData,
  Decision,
  Escalation,
  EventsStageData,
  ReportStageData,
  SamplerStageData,
  TraceStage,
  VlmStageData,
} from '~/types/api'

const store = useAnalysisStore()
const { getFrameUrl } = useSafirApi()
const emit = defineEmits<{ (e: 'open-frame', id: string): void }>()

const stage = computed<TraceStage | null>(() => store.selectedStage)
const ev = computed(() => (stage.value ? store.eventForStage(stage.value) : undefined))
const label = computed(() => (stage.value ? stageLabel(stage.value) : '—'))

const statusTone: Record<string, string> = {
  completed: 'text-risk-low bg-risk-low/10 border-risk-low/30',
  running: 'text-accent bg-accent/10 border-accent/30',
  failed: 'text-risk-crit bg-risk-crit/10 border-risk-crit/30',
  pending: 'text-slate-500 bg-surface-2 border-edge',
}

function frameUrl(id: string): string {
  return store.jobId ? getFrameUrl(store.jobId, id) : ''
}

// typed views of the payload (data is Record<string, unknown> -> narrow per stage)
const view = <T,>(s: TraceStage): T | null => (stage.value === s ? (ev.value?.data as unknown as T) : null)
const sampler = computed(() => view<SamplerStageData>('sampler'))
const vlm = computed(() => view<VlmStageData>('vlm'))
const events = computed(() => view<EventsStageData>('events'))
const context = computed(() => view<ContextStageData>('agent_context'))
const decision = computed(() => view<Decision>('decision'))
const escalation = computed(() => view<Escalation>('escalation'))
const report = computed(() => view<ReportStageData>('report'))

// Fix 3: the frames actually sent to the VLM are the sampler's representative
// frames (they live in the sampler stage payload, referenced by thumbnail_url).
const vlmInputFrames = computed(() => {
  if (stage.value !== 'vlm') return []
  const s = store.eventForStage('sampler')?.data as unknown as SamplerStageData | undefined
  return s?.event_groups?.flatMap((g) => g.representative_frames) ?? []
})

// Fix 4: real regulations retrieved by the agent (report.relevant_regulations).
// No fabricated similarity score / chunk id — the backend does not provide them.
const regulations = computed(() => store.report?.relevant_regulations ?? [])
</script>

<template>
  <div class="card p-5 min-h-[22rem]">
    <!-- header -->
    <div class="flex items-center gap-3 mb-4">
      <h3 class="text-base font-semibold text-slate-100">{{ label }}</h3>
      <span
        v-if="ev"
        class="text-[11px] px-2 py-0.5 rounded border font-medium"
        :class="statusTone[ev.status] ?? statusTone.pending"
      >{{ ev.status }}</span>
      <span
        v-if="ev?.duration_ms != null"
        class="ml-auto text-xs font-mono text-slate-500"
      >{{ ev.duration_ms }} ms</span>
    </div>

    <!-- not yet reached -->
    <div v-if="!ev" class="text-sm text-slate-500 py-10 text-center">
      Bu aşama henüz çalışmadı.
    </div>

    <!-- failed (e.g. VLM degraded) -->
    <div v-else-if="ev.status === 'failed'" class="space-y-3">
      <div class="rounded-md border border-risk-crit/30 bg-risk-crit/10 p-4 text-sm text-slate-200">
        <div class="font-semibold text-risk-crit mb-1">
          {{ label }} Tamamlanamadı
        </div>
        <p v-if="ev.error" class="text-slate-200 font-mono text-xs mt-2 bg-surface-1/60 p-2.5 rounded border border-risk-crit/30 whitespace-pre-wrap">
          {{ ev.error }}
        </p>
        <p v-else class="text-slate-400 text-xs mt-1">
          Analiz düşürülmüş (degraded) modda sürdürüldü.
        </p>
      </div>
    </div>

    <template v-else>
      <!-- ============ SAMPLER ============ -->
      <div v-if="sampler" class="space-y-5">
        <div class="grid grid-cols-3 sm:grid-cols-6 gap-3">
          <MetricCell label="Scanned" :value="sampler.stats.total_frames_scanned ?? 0" />
          <MetricCell label="Evaluated" :value="sampler.stats.sampled_frames_evaluated ?? 0" />
          <MetricCell label="Evidence" :value="sampler.stats.evidence_frame_count ?? 0" />
          <MetricCell label="Eliminated" :value="sampler.stats.eliminated_frame_count ?? 0" />
          <MetricCell label="GPU savings" :value="`%${sampler.stats.gpu_savings_ratio_pct ?? 0}`" />
          <MetricCell label="Elapsed" :value="`${sampler.stats.elapsed_sec ?? 0}s`" />
        </div>

        <div v-if="sampler.evidence_frames.length">
          <div class="field-label">Evidence frames</div>
          <div class="flex flex-wrap gap-3">
            <button
              v-for="f in sampler.evidence_frames"
              :key="f.frame_id"
              type="button"
              class="w-36 border border-edge rounded-md overflow-hidden bg-surface-2 text-left hover:ring-1 hover:ring-accent"
              @click="emit('open-frame', f.frame_id)"
            >
              <img :src="frameUrl(f.frame_id)" :alt="f.frame_id" class="w-full h-24 object-cover" loading="lazy" />
              <div class="px-2 py-1 text-[11px] font-mono text-slate-400">
                {{ f.timestamp_str }} · Δ{{ f.change_score }}
                <span v-if="f.is_fallback" class="text-risk-mid"> · fallback</span>
              </div>
            </button>
          </div>
        </div>

        <div v-if="sampler.event_groups.length" class="text-xs text-slate-500">
          {{ sampler.event_groups.length }} olay grubu · temsili kareler VLM'e gönderildi.
        </div>
      </div>

      <!-- ============ VLM ============ -->
      <div v-else-if="vlm" class="space-y-5">
        <div class="grid grid-cols-3 gap-3">
          <MetricCell label="Model" :value="vlm.model_name" />
          <MetricCell label="Frames" :value="vlm.frames_sent || vlm.frame_count" />
          <MetricCell label="Latency" :value="`${vlm.latency_ms} ms`" />
        </div>
        <div>
          <div class="field-label">Input · user prompt</div>
          <p class="text-sm text-slate-300 bg-surface-2 rounded-md p-3 border border-edge">{{ vlm.user_prompt || '—' }}</p>
        </div>
        <div v-if="vlmInputFrames.length">
          <div class="field-label">Input · VLM'e gönderilen kareler ({{ vlmInputFrames.length }})</div>
          <div class="flex flex-wrap gap-3">
            <button
              v-for="rf in vlmInputFrames"
              :key="rf.frame_id"
              type="button"
              class="w-28 border border-edge rounded-md overflow-hidden bg-surface-2 text-left hover:ring-1 hover:ring-accent"
              @click="emit('open-frame', rf.frame_id)"
            >
              <img :src="frameUrl(rf.frame_id)" :alt="rf.frame_id" class="w-full h-20 object-cover" loading="lazy" />
              <div class="px-2 py-1 text-[10px] font-mono text-slate-400">{{ rf.label }} · {{ rf.timestamp_str }}</div>
            </button>
          </div>
        </div>
        <div>
          <div class="field-label">Model output · description</div>
          <p class="text-sm text-slate-200 leading-relaxed whitespace-pre-line">{{ vlm.description || '—' }}</p>
        </div>
        <div v-if="vlm.structured_events?.length">
          <div class="field-label">Structured events ({{ vlm.structured_events.length }})</div>
          <pre class="text-[11px] font-mono text-slate-400 bg-surface-2 border border-edge rounded-md p-3 max-h-56 overflow-auto">{{ JSON.stringify(vlm.structured_events, null, 2) }}</pre>
        </div>
      </div>

      <!-- ============ EVENTS ============ -->
      <div v-else-if="events" class="space-y-5">
        <div v-if="events.detected_events.length">
          <div class="field-label">Detected events</div>
          <ul class="space-y-2">
            <li
              v-for="(d, i) in events.detected_events"
              :key="i"
              class="flex items-center gap-3 bg-surface-2 border border-edge rounded-md px-3 py-2"
            >
              <span class="font-mono text-xs text-slate-400 w-12">{{ mmss(d.timestamp) }}</span>
              <span class="text-sm text-slate-100">{{ d.event_type }}</span>
              <span class="ml-auto text-xs text-slate-400">confidence {{ d.confidence }}</span>
            </li>
          </ul>
        </div>
        <div v-else class="text-sm text-slate-500">Bu analizde ayrık olay tespit edilmedi.</div>

        <div v-if="events.rule_matches.length">
          <div class="field-label">İSG kural eşleşmeleri</div>
          <ul class="flex flex-wrap gap-2">
            <li
              v-for="(r, i) in events.rule_matches"
              :key="i"
              class="text-xs px-2 py-1 rounded border border-risk-high/40 bg-risk-high/10 text-slate-200"
            >
              <span class="font-mono">{{ r.rule_id }}</span>
              <span class="text-slate-400"> · {{ r.severity }}</span>
            </li>
          </ul>
        </div>

        <div v-if="events.temporal_events.length" class="text-xs text-slate-500">
          {{ events.temporal_events.length }} zamansal (sürekli) olay birleştirildi.
        </div>
      </div>

      <!-- ============ CONTEXT & RAG ============ -->
      <div v-else-if="context" class="space-y-4">
        <p class="text-sm text-slate-400">Ajan bağlamı hazırlandı ({{ context.length }} karakter): tespit edilen olaylar + FAISS'ten getirilen İSG mevzuatı birleştirildi.</p>

        <!-- Fix 4: real retrieved regulations (report.relevant_regulations) -->
        <div>
          <div class="field-label">Getirilen İSG mevzuatı (RAG / FAISS)</div>
          <ul v-if="regulations.length" class="space-y-1 text-sm text-slate-200">
            <li v-for="(reg, i) in regulations" :key="i" class="bg-surface-2 border border-edge rounded-md px-3 py-2">{{ reg }}</li>
          </ul>
          <p v-else class="text-sm text-slate-500">
            {{ store.report ? 'Bu analiz için ilgili mevzuat maddesi bulunamadı.' : 'Mevzuat sonuçları rapor tamamlanınca listelenir.' }}
          </p>
        </div>

        <details>
          <summary class="cursor-pointer text-xs text-slate-400">Teknik: agent context prompt_block</summary>
          <pre class="mt-2 text-[11px] font-mono text-slate-500 bg-surface-2 border border-edge rounded-md p-3 max-h-72 overflow-auto whitespace-pre-wrap">{{ context.prompt_block }}</pre>
        </details>
      </div>

      <!-- ============ DECISION ============ -->
      <div v-else-if="decision" class="space-y-5">
        <div class="flex items-end gap-6">
          <div>
            <div class="field-label">Risk</div>
            <template v-if="decision.risk_status === 'unknown' || decision.risk_score == null">
              <div class="text-3xl font-bold text-slate-400">Belirsiz</div>
              <div class="text-sm uppercase tracking-wide text-slate-400">manuel inceleme gerekli</div>
            </template>
            <template v-else>
              <div class="text-3xl font-bold" :class="RISK_TEXT[riskTone(decision.risk_level)]">
                {{ decision.risk_score }}<span class="text-lg text-slate-500"> / 100</span>
              </div>
              <div class="text-sm uppercase tracking-wide" :class="RISK_TEXT[riskTone(decision.risk_level)]">{{ decision.risk_level }}</div>
            </template>
          </div>
        </div>
        <div>
          <div class="field-label">Neden? (özet)</div>
          <p class="text-sm text-slate-200 leading-relaxed">{{ decision.summary || '—' }}</p>
        </div>
        <div>
          <div class="field-label">Önerilen aksiyon</div>
          <p class="text-sm text-slate-100">{{ decision.recommended_action || '—' }}</p>
        </div>
        <div v-if="decision.actions?.length">
          <div class="field-label">Aksiyonlar</div>
          <ol class="list-decimal list-inside space-y-1 text-sm text-slate-200">
            <li v-for="(a, i) in decision.actions" :key="i">{{ a }}</li>
          </ol>
        </div>
      </div>

      <!-- ============ ESCALATION ============ -->
      <div v-else-if="escalation" class="space-y-4">
        <div class="grid grid-cols-3 gap-3">
          <MetricCell label="Tier" :value="escalation.tier" />
          <MetricCell label="Auto-dispatched" :value="escalation.auto_dispatched ? 'evet' : 'hayır'" />
          <MetricCell label="Alert id" :value="escalation.alert_id ?? '—'" mono />
        </div>
        <div>
          <div class="field-label">Gerekçe</div>
          <p class="text-sm text-slate-200">{{ escalation.reason }}</p>
        </div>
      </div>

      <!-- ============ REPORT ============ -->
      <div v-else-if="report" class="space-y-4">
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCell
            label="Risk"
            :value="report.risk_status === 'unknown' || report.risk_score == null ? 'Belirsiz' : `${report.risk_score} (${report.risk_level})`"
          />
          <MetricCell label="Tier" :value="report.escalation_tier ?? '—'" />
          <MetricCell label="VLM" :value="report.vlm_model ?? '—'" mono />
          <MetricCell label="LLM" :value="report.llm_model ?? '—'" mono />
        </div>
        <p class="text-sm text-slate-400">
          Yapılandırılmış nihai rapor üretildi. Kullanıcı-dostu tam görünüm ve dışa aktarma için Rapor sekmesini kullanın.
        </p>
      </div>
    </template>
  </div>
</template>
