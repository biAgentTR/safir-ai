<script setup lang="ts">
// Detail panel for the currently selected pipeline stage. Renders the REAL
// serialized trace payload per stage. Never shows raw_response / secrets /
// system prompt (those are not in the payload by design).
import type {
  ContextStageData,
  Decision,
  Escalation,
  EventsStageData,
  RagSecurityStageData,
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
const ragSecurity = computed(() => view<RagSecurityStageData>('rag_security'))
const decision = computed(() => view<Decision>('decision'))
const escalation = computed(() => view<Escalation>('escalation'))
const report = computed(() => view<ReportStageData>('report'))

// The sampler no longer clusters: EVERY evidence frame it produced is sent
// to the VLM (they live in the sampler stage payload, referenced by
// thumbnail_url). Event clustering happens inside the VLM stage instead.
const vlmInputFrames = computed(() => {
  if (stage.value !== 'vlm') return []
  const s = store.eventForStage('sampler')?.data as unknown as SamplerStageData | undefined
  return s?.evidence_frames ?? []
})

// Fix 4: real regulations retrieved by the agent (report.relevant_regulations).
// No fabricated similarity score / chunk id — the backend does not provide them.
const regulations = computed(() => store.report?.relevant_regulations ?? [])

const GUARD_SOURCE_LABEL: Record<string, string> = {
  user_prompt: 'Kullanıcı İstemi',
  vlm_description: 'VLM Açıklaması',
  vlm_event_description: 'Geçmiş Olay (VLM)',
}
function guardSourceLabel(s: string): string {
  return GUARD_SOURCE_LABEL[s] ?? s
}
function pct(v: number | null): string {
  return v == null ? 'N/A' : `%${Math.round(v * 100)}`
}
function ms(v: number | null): string {
  return v == null ? 'N/A' : `${Math.round(v)} ms`
}
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
        {{ label }} tamamlanamadı. Analiz düşürülmüş (degraded) modda sürdürüldü.
      </div>
      <details v-if="ev.error" class="text-xs">
        <summary class="cursor-pointer text-slate-400">Teknik detay</summary>
        <p class="mt-2 text-slate-500">{{ ev.error }}</p>
      </details>
    </div>

    <template v-else>
      <!-- ============ SAMPLER ============ -->
      <div v-if="sampler" class="space-y-5">
        <div class="grid grid-cols-3 sm:grid-cols-6 gap-3">
          <MetricCell label="Taranan" :value="sampler.stats.total_frames_scanned ?? 0" />
          <MetricCell label="Değerlendirilen" :value="sampler.stats.sampled_frames_evaluated ?? 0" />
          <MetricCell label="Kanıt" :value="sampler.stats.evidence_frame_count ?? 0" />
          <MetricCell label="Elenen" :value="sampler.stats.eliminated_frame_count ?? 0" />
          <MetricCell label="GPU Tasarrufu" :value="`%${sampler.stats.gpu_savings_ratio_pct ?? 0}`" />
          <MetricCell label="Geçen Süre" :value="`${sampler.stats.elapsed_sec ?? 0}s`" />
        </div>

        <div v-if="sampler.evidence_frames.length">
          <div class="field-label">Kanıt kareleri</div>
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
                <span v-if="f.is_fallback" class="text-risk-mid"> · yedek</span>
              </div>
            </button>
          </div>
        </div>

      </div>

      <!-- ============ VLM ============ -->
      <div v-else-if="vlm" class="space-y-5">
        <div class="grid grid-cols-3 gap-3">
          <MetricCell label="Model" :value="vlm.model_name" />
          <MetricCell label="Kareler" :value="vlm.frames_sent || vlm.frame_count" />
          <MetricCell label="Gecikme" :value="`${vlm.latency_ms} ms`" />
        </div>
        <div>
          <div class="field-label">Girdi · kullanıcı istemi</div>
          <p class="text-sm text-slate-300 bg-surface-2 rounded-md p-3 border border-edge">{{ vlm.user_prompt || '—' }}</p>
        </div>
        <div v-if="vlmInputFrames.length">
          <div class="field-label">Input · VLM'e gönderilen kareler ({{ vlmInputFrames.length }})</div>
          <div class="flex flex-wrap gap-3">
            <button
              v-for="ef in vlmInputFrames"
              :key="ef.frame_id"
              type="button"
              class="w-28 border border-edge rounded-md overflow-hidden bg-surface-2 text-left hover:ring-1 hover:ring-accent"
              @click="emit('open-frame', ef.frame_id)"
            >
              <img :src="frameUrl(ef.frame_id)" :alt="ef.frame_id" class="w-full h-20 object-cover" loading="lazy" />
              <div class="px-2 py-1 text-[10px] font-mono text-slate-400">{{ ef.timestamp_str }} · {{ ef.change_score.toFixed(3) }}</div>
            </button>
          </div>
        </div>
        <div>
          <div class="field-label">Model çıktısı · açıklama</div>
          <p class="text-sm text-slate-200 leading-relaxed whitespace-pre-line">{{ vlm.description || '—' }}</p>
        </div>
        <div v-if="vlm.structured_events?.length">
          <div class="field-label">Yapılandırılmış olaylar ({{ vlm.structured_events.length }})</div>
          <pre class="text-[11px] font-mono text-slate-400 bg-surface-2 border border-edge rounded-md p-3 max-h-56 overflow-auto">{{ JSON.stringify(vlm.structured_events, null, 2) }}</pre>
        </div>
      </div>

      <!-- ============ EVENTS ============ -->
      <div v-else-if="events" class="space-y-5">
        <div v-if="events.detected_events.length">
          <div class="field-label">Tespit edilen olaylar</div>
          <ul class="space-y-2">
            <li
              v-for="(d, i) in events.detected_events"
              :key="i"
              class="flex flex-col gap-1 bg-surface-2 border border-edge rounded-md px-3 py-2"
            >
              <div class="flex items-center gap-3">
                <span class="font-mono text-xs text-slate-400 w-12">{{ mmss(d.timestamp) }}</span>
                <span class="text-sm text-slate-100">{{
                  d.event_type === 'siniflandirilamadi' ? d.event_type : (d.event_name || d.event_type)
                }}</span>
                <span class="ml-auto text-xs text-slate-400">güven {{ d.confidence }}</span>
              </div>
              <div
                v-if="d.event_type !== 'siniflandirilamadi' && d.matched_keywords.length"
                class="text-xs text-slate-400 pl-[3.75rem]"
              >
                Risk: {{ d.matched_keywords.join(', ') }}
              </div>
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

      <!-- ============ RAG & SECURITY TELEMETRY ============ -->
      <div v-else-if="ragSecurity" class="space-y-6">
        <div class="rounded-md border border-edge bg-surface-2/60 px-3 py-2 text-xs text-slate-400">
          RAG semantik olarak <span class="text-slate-200">ilgili olabilecek kaynakları</span> getirir; risk skoru/seviyesi
          ve mevzuat eşleşmesi kararı bundan <span class="text-slate-200">bağımsız</span>, deterministik RuleEngine
          tarafından belirlenir (bkz. "Bağlam ve RAG" sekmesindeki "Getirilen İSG mevzuatı").
        </div>

        <!-- RAG -->
        <div>
          <div class="field-label">Semantik RAG Retrieval</div>
          <div v-if="!ragSecurity.rag" class="text-sm text-slate-500">
            Bu turda semantik RAG sorgusu yapılmadı (VLM'den anahtar kelime üretilmedi).
          </div>
          <template v-else>
            <div class="grid grid-cols-3 sm:grid-cols-6 gap-3 mb-3">
              <MetricCell label="Aday" :value="ragSecurity.rag.candidate_count" />
              <MetricCell label="Final" :value="ragSecurity.rag.final_count" />
              <MetricCell label="Durum" :value="ragSecurity.rag.retrieval_status" />
              <MetricCell label="Embedding Gecikme" :value="ms(ragSecurity.rag.embedding_latency_ms)" />
              <MetricCell label="Rerank Gecikme" :value="ms(ragSecurity.rag.rerank_latency_ms)" />
              <MetricCell label="Eşik" :value="ragSecurity.rag.threshold != null ? ragSecurity.rag.threshold : 'N/A'" />
            </div>

            <!-- Deterministic relevance formülü — koddan okunan gerçek ağırlıklar (uydurulmadı). -->
            <div v-if="ragSecurity.rag.relevance_weights" class="rounded-md border border-edge bg-surface-2/60 px-3 py-2 mb-3 text-xs text-slate-400">
              <span class="text-slate-300 font-medium">Deterministic Relevance Formülü</span> (çalışan koddan okunur):
              Semantic × {{ ragSecurity.rag.relevance_weights.semantic }} + Lexical × {{ ragSecurity.rag.relevance_weights.lexical }}
              + Keyword × {{ ragSecurity.rag.relevance_weights.keyword }} + Metadata × {{ ragSecurity.rag.relevance_weights.metadata }}
              + Phrase × {{ ragSecurity.rag.relevance_weights.phrase }}
            </div>
            <div class="text-xs text-slate-500 mb-3">
              Cross-Encoder durumu:
              <span v-if="ragSecurity.rag.cross_encoder_status === 'used'" class="text-risk-low">çalıştı — aşağıdaki Cross-Encoder Skoru gerçek model çıktısıdır.</span>
              <span v-else-if="ragSecurity.rag.cross_encoder_status === 'unavailable'" class="text-risk-mid">kullanılamadı (model ağırlığı yüklenemedi) — sıralama Deterministic Relevance'a düştü, harici bir API'ye düşülmedi.</span>
              <span v-else-if="ragSecurity.rag.cross_encoder_status === 'disabled'" class="text-slate-500">devre dışı.</span>
              <span v-else class="text-slate-600">bu sorgu için bilinmiyor.</span>
            </div>

            <div v-if="ragSecurity.rag.zero_result" class="text-sm text-slate-500 mb-2">
              Eşik üzerinde hiçbir sonuç bulunamadı (0 sonuç geçerli bir sonuçtur).
            </div>
            <div v-if="ragSecurity.rag.results.length" class="space-y-2">
              <details
                v-for="(r, i) in ragSecurity.rag.results"
                :key="i"
                class="rounded-md border border-edge bg-surface-2/40"
                :class="r.selected ? '' : 'opacity-60'"
              >
                <summary class="cursor-pointer px-3 py-2 grid grid-cols-6 gap-2 text-xs items-center">
                  <span class="col-span-2 text-slate-200 truncate">{{ r.document_title ?? r.document_id ?? '—' }} <span class="text-slate-500">· {{ r.article_number ?? '—' }}</span></span>
                  <span class="font-mono text-slate-300">Emb {{ r.embedding_score.toFixed(3) }}</span>
                  <span class="font-mono text-slate-300">Rel {{ r.relevance_score != null ? r.relevance_score.toFixed(3) : '—' }}</span>
                  <span class="font-mono text-slate-300">
                    CE {{ r.cross_encoder_score != null ? r.cross_encoder_score.toFixed(3) : (ragSecurity.rag.cross_encoder_status === 'unavailable' ? 'kullanılamadı' : '—') }}
                  </span>
                  <span :class="r.selected ? 'text-risk-low' : 'text-slate-600'">
                    {{ r.selected ? (r.cross_encoder_score != null ? 'ACCEPTED → CROSS-ENCODER' : 'ACCEPTED') : 'REJECTED BY DETERMINISTIC GATE' }}
                  </span>
                </summary>
                <div class="px-3 pb-3 pt-1 space-y-2 border-t border-edge/60">
                  <div v-if="r.semantic_score != null && ragSecurity.rag.relevance_weights" class="text-[11px] font-mono text-slate-400 space-y-0.5">
                    <div>Semantic&nbsp;&nbsp;{{ r.semantic_score.toFixed(3) }} × {{ ragSecurity.rag.relevance_weights.semantic }} = {{ (r.semantic_score * ragSecurity.rag.relevance_weights.semantic).toFixed(3) }}</div>
                    <div>Lexical&nbsp;&nbsp;&nbsp;{{ r.lexical_score?.toFixed(3) }} × {{ ragSecurity.rag.relevance_weights.lexical }} = {{ ((r.lexical_score ?? 0) * ragSecurity.rag.relevance_weights.lexical).toFixed(3) }}</div>
                    <div>Keyword&nbsp;&nbsp;{{ r.keyword_score?.toFixed(3) }} × {{ ragSecurity.rag.relevance_weights.keyword }} = {{ ((r.keyword_score ?? 0) * ragSecurity.rag.relevance_weights.keyword).toFixed(3) }}</div>
                    <div>Metadata&nbsp;{{ r.metadata_score?.toFixed(3) }} × {{ ragSecurity.rag.relevance_weights.metadata }} = {{ ((r.metadata_score ?? 0) * ragSecurity.rag.relevance_weights.metadata).toFixed(3) }}</div>
                    <div>Phrase&nbsp;&nbsp;&nbsp;&nbsp;{{ r.phrase_score?.toFixed(3) }} × {{ ragSecurity.rag.relevance_weights.phrase }} = {{ ((r.phrase_score ?? 0) * ragSecurity.rag.relevance_weights.phrase).toFixed(3) }}</div>
                  </div>
                  <p v-else class="text-[11px] text-slate-600">Bu kaynak için bileşen skorları taşınmadı (relevance skorlama devre dışıydı).</p>
                  <div v-if="r.text" class="text-xs text-slate-300 bg-surface-2 border border-edge rounded-md p-2 max-h-40 overflow-auto whitespace-pre-wrap">{{ r.text }}</div>
                </div>
              </details>
            </div>
          </template>
        </div>

        <!-- SECURITY -->
        <div>
          <div class="field-label">Prompt Injection Guard</div>
          <div v-if="!ragSecurity.security.length" class="text-sm text-slate-500">
            Bu turda guard kontrolü yapılmadı (guard devre dışı veya kontrol edilecek serbest metin yok).
          </div>
          <template v-else>
            <div class="overflow-x-auto">
              <table class="w-full text-xs">
                <thead>
                  <tr class="text-left text-[10px] uppercase tracking-wide text-slate-500 border-b border-edge">
                    <th class="py-1.5 pr-3">Kaynak</th>
                    <th class="py-1.5 pr-3">Karar</th>
                    <th class="py-1.5 pr-3">Güven</th>
                    <th class="py-1.5 pr-3">Sebep</th>
                    <th class="py-1.5 pr-3">Gecikme</th>
                    <th class="py-1.5 pr-3">Guard Hatası</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(g, i) in ragSecurity.security" :key="i" class="border-b border-edge/60">
                    <td class="py-1.5 pr-3 text-slate-200">{{ guardSourceLabel(g.source) }}</td>
                    <td class="py-1.5 pr-3" :class="g.action === 'quarantine' ? 'text-risk-high' : 'text-risk-low'">
                      {{ g.action === 'quarantine' ? 'Quarantine' : 'Allow' }}
                    </td>
                    <td class="py-1.5 pr-3 font-mono text-slate-300">{{ pct(g.confidence) }}</td>
                    <td class="py-1.5 pr-3 text-slate-400 max-w-[220px] truncate" :title="g.reason ?? ''">{{ g.reason ?? '—' }}</td>
                    <td class="py-1.5 pr-3 font-mono text-slate-400">{{ ms(g.latency_ms) }}</td>
                    <td class="py-1.5 pr-3" :class="g.guard_failed ? 'text-risk-high' : 'text-slate-600'">{{ g.guard_failed ? 'evet' : 'hayır' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </template>
        </div>
      </div>

      <!-- ============ DECISION (Agent'in TASLAK önerisi — resmi risk_score DEĞİL) ============ -->
      <div v-else-if="decision" class="space-y-5">
        <div class="rounded-md border border-accent/30 bg-accent/10 px-3 py-2 text-xs text-slate-300">
          🤖 Bu, Agent'ın (LLM) ürettiği <span class="font-medium">taslak öneridir</span> (backend'de
          <code class="text-slate-200">llm_proposed_score</code> olarak saklanır) — SAFİR'in
          <span class="font-medium">resmi, deterministik risk kararı DEĞİLDİR</span>. Nihai
          <code class="text-slate-200">risk_score</code>/<code class="text-slate-200">risk_level</code>,
          bu değerden BAĞIMSIZ olarak "Nihai Rapor" aşamasında deterministik risk motoru tarafından
          hesaplanır (bkz. Rapor sekmesindeki resmi Risk Skoru).
        </div>
        <div class="flex items-end gap-6">
          <div>
            <div class="field-label">Model Önerisi (taslak, resmi değil)</div>
            <template v-if="decision.risk_status === 'unknown' || decision.risk_score == null">
              <div class="text-3xl font-bold text-slate-400">Belirsiz</div>
              <div class="text-sm uppercase tracking-wide text-slate-400">MANUEL İNCELEME GEREKLİ</div>
            </template>
            <template v-else>
              <div class="text-3xl font-bold text-slate-300">
                {{ decision.risk_score }}<span class="text-lg text-slate-500"> / 100</span>
              </div>
              <div class="text-sm uppercase tracking-wide text-slate-400">{{ trUpper(decision.risk_level) }} (Agent önerisi)</div>
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
          <MetricCell label="Kademe" :value="escalation.tier" />
          <MetricCell label="Otomatik Yönlendirildi" :value="escalation.auto_dispatched ? 'evet' : 'hayır'" />
          <MetricCell label="Uyarı Kimliği" :value="escalation.alert_id ?? '—'" mono />
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
          <MetricCell label="Kademe" :value="report.escalation_tier ?? '—'" />
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
