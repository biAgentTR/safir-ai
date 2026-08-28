<template>
  <div class="max-w-[1440px] mx-auto p-4 sm:p-6 lg:p-8">
    <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
      <div>
        <h1 class="text-2xl font-semibold text-white tracking-tight">Raporlar</h1>
        <p class="text-gray-400 text-sm mt-1">Tamamlanan analizlerin PDF raporlarını görüntüleyin ve bilgisayarınıza kaydedin.</p>
      </div>
      <div class="flex items-center gap-3">
        <button 
          @click="history.refresh"
          :disabled="history.pagination.isLoadingMore && history.items.value?.length > 0"
          class="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border border-gray-700 bg-gray-800/50 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 disabled:opacity-50"
          aria-label="Listeyi yenile"
        >
          <svg :class="{'animate-spin': history.pagination.isLoadingMore && history.items.value?.length > 0}" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
          <span>Yenile</span>
        </button>
      </div>
    </div>

    <!-- Error State -->
    <div v-if="history.state.value === 'failed'" class="bg-red-500/10 border border-red-500/20 rounded-xl p-6 text-center">
      <div class="inline-flex p-3 rounded-full bg-red-500/20 mb-4 text-red-400">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      </div>
      <h3 class="text-lg font-medium text-white mb-2">Rapor geçmişi yüklenemedi</h3>
      <p class="text-red-400/80 mb-6 max-w-md mx-auto">{{ history.errorMessage.value }}</p>
      <button @click="history.refresh" class="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium transition-colors">Tekrar dene</button>
    </div>

    <!-- Service Unavailable -->
    <div v-else-if="history.state.value === 'service-unavailable'" class="bg-gray-800/50 border border-gray-700 rounded-xl p-8 text-center max-w-2xl mx-auto mt-12">
      <div class="inline-flex p-4 rounded-full bg-gray-700/50 mb-4 text-gray-400">
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      </div>
      <h3 class="text-xl font-semibold text-white mb-2">Rapor servisine ulaşılamıyor</h3>
      <p class="text-gray-400 mb-8">Backend servisini kontrol edip tekrar deneyin.</p>
      <div class="flex items-center justify-center gap-4">
        <button @click="history.refresh" class="px-6 py-2.5 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors">Tekrar dene</button>
        <NuxtLink to="/" class="px-6 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium transition-colors">Yeni analiz</NuxtLink>
      </div>
    </div>

    <!-- Empty State -->
    <div v-else-if="history.state.value === 'empty'" class="bg-gray-800/20 border border-dashed border-gray-700 rounded-xl p-12 text-center max-w-3xl mx-auto mt-8">
      <div class="inline-flex p-5 rounded-full bg-gray-800/50 mb-4 text-gray-500">
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/></svg>
      </div>
      <h3 class="text-xl font-semibold text-white mb-2">Henüz rapor bulunmuyor</h3>
      <p class="text-gray-400 mb-8 max-w-md mx-auto">Tamamlanan analizlerden oluşturulan PDF raporları burada görüntülenecek.</p>
      <NuxtLink to="/" class="inline-flex items-center gap-2 px-6 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium transition-colors">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M12 5v14"/></svg>
        Yeni analiz başlat
      </NuxtLink>
    </div>

    <!-- Loading Skeleton -->
    <div v-else-if="history.state.value === 'loading'" class="space-y-4" aria-busy="true">
      <div class="hidden md:grid grid-cols-12 gap-4 px-6 py-3 border-b border-white/10 bg-gray-900/50 rounded-t-xl mb-2">
        <div class="col-span-3 text-xs font-semibold text-gray-500 uppercase">Video</div>
        <div class="col-span-2 text-xs font-semibold text-gray-500 uppercase">Risk</div>
        <div class="col-span-3 text-xs font-semibold text-gray-500 uppercase">Özet</div>
        <div class="col-span-2 text-xs font-semibold text-gray-500 uppercase">Durum</div>
        <div class="col-span-2 text-xs font-semibold text-gray-500 uppercase text-right">Aksiyon</div>
      </div>
      <div v-for="i in 5" :key="i" class="bg-gray-800/30 rounded-xl p-4 md:px-6 md:py-4 flex flex-col md:grid md:grid-cols-12 md:gap-4 md:items-center animate-pulse border border-gray-800">
        <div class="col-span-3 h-5 bg-gray-700/50 rounded w-3/4 mb-4 md:mb-0"></div>
        <div class="col-span-2 h-5 bg-gray-700/50 rounded w-16 mb-4 md:mb-0"></div>
        <div class="col-span-3 h-4 bg-gray-700/30 rounded w-full mb-4 md:mb-0"></div>
        <div class="col-span-2 h-5 bg-gray-700/50 rounded w-24 mb-4 md:mb-0"></div>
        <div class="col-span-2 h-8 bg-gray-700/30 rounded w-24 md:ml-auto"></div>
      </div>
    </div>

    <!-- Data List (Semantic Table for Desktop, Cards for Mobile) -->
    <div v-else class="flex flex-col">
      <!-- Desktop Semantic Table -->
      <div class="hidden md:block overflow-hidden rounded-xl border border-white/5 bg-[#1E2532]/30">
        <table class="w-full text-left border-collapse">
          <thead class="bg-[#141A25]">
            <tr>
              <th scope="col" class="px-6 py-3 border-b border-white/5 text-xs font-semibold text-gray-400 uppercase tracking-wider w-[25%]">Video</th>
              <th scope="col" class="px-6 py-3 border-b border-white/5 text-xs font-semibold text-gray-400 uppercase tracking-wider w-[15%]">Risk</th>
              <th scope="col" class="px-6 py-3 border-b border-white/5 text-xs font-semibold text-gray-400 uppercase tracking-wider w-[25%]">Özet</th>
              <th scope="col" class="px-6 py-3 border-b border-white/5 text-xs font-semibold text-gray-400 uppercase tracking-wider w-[20%]">Rapor Durumu</th>
              <th scope="col" class="px-6 py-3 border-b border-white/5 text-xs font-semibold text-gray-400 uppercase tracking-wider text-right w-[15%]">Aksiyon</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-white/5">
            <tr v-for="item in history.items.value" :key="item.job_id" class="hover:bg-white/[0.02] transition-colors group">
              <!-- 1. Video -->
              <td class="px-6 py-4 align-top">
                <div class="flex flex-col min-w-0">
                  <span class="text-sm font-medium text-white truncate">{{ getBasename(item.video_source) }}</span>
                  <span class="text-xs text-gray-500 mt-1">{{ formatDate(item.created_at) }}</span>
                </div>
              </td>
              <!-- 2. Risk -->
              <td class="px-6 py-4 align-top">
                <HistoryRiskBadge :level="item.risk_level" :score="item.risk_score" />
              </td>
              <!-- 3. Summary -->
              <td class="px-6 py-4 align-top">
                <p class="text-sm text-gray-300 line-clamp-2 leading-snug" v-if="item.summary">{{ item.summary }}</p>
                <p class="text-sm text-gray-500 italic" v-else>Bu analiz için henüz özet bulunmuyor.</p>
              </td>
              <!-- 4. Rapor Durumu -->
              <td class="px-6 py-4 align-top">
                <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border" :class="getReportStatusClasses(item.status)">
                  {{ getReportStatusText(item.status) }}
                </span>
              </td>
              <!-- 5. Aksiyon -->
              <td class="px-6 py-4 align-top text-right">
                <div class="flex flex-col items-end gap-2">
                  <ReportDownloadButton 
                    v-if="item.status === 'done'" 
                    :job-id="item.job_id" 
                    :video-name="getBasename(item.video_source)" 
                  />
                  <NuxtLink 
                    :to="`/analizler/${encodeURIComponent(item.job_id)}`"
                    class="text-xs text-cyan-500 hover:text-cyan-400 font-medium opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity flex items-center gap-1"
                  >
                    Detayı görüntüle
                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
                  </NuxtLink>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Mobile Cards -->
      <div class="flex flex-col gap-3 md:hidden">
        <div 
          v-for="item in history.items.value" 
          :key="'mob-' + item.job_id"
          class="bg-[#1E2532] border border-white/5 rounded-xl p-4 flex flex-col relative"
        >
          <!-- Video -->
          <div class="mb-3">
            <span class="text-sm font-medium text-white truncate block">{{ getBasename(item.video_source) }}</span>
            <span class="text-xs text-gray-500 mt-1 block">{{ formatDate(item.created_at) }}</span>
          </div>
          <!-- Durum ve Risk -->
          <div class="flex items-start justify-between gap-4 mb-4">
            <span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-medium border" :class="getReportStatusClasses(item.status)">
              {{ getReportStatusText(item.status) }}
            </span>
            <HistoryRiskBadge :level="item.risk_level" :score="item.risk_score" />
          </div>
          <!-- Özet -->
          <div class="mb-4">
            <p class="text-sm text-gray-300 line-clamp-2" v-if="item.summary">{{ item.summary }}</p>
            <p class="text-sm text-gray-500 italic" v-else>Özet bulunmuyor.</p>
          </div>
          <!-- Aksiyon -->
          <div class="flex items-center justify-between mt-auto pt-4 border-t border-white/5">
            <NuxtLink 
              :to="`/analizler/${encodeURIComponent(item.job_id)}`"
              class="text-xs text-cyan-500 font-medium py-2"
            >
              Analiz detayı
            </NuxtLink>
            <ReportDownloadButton 
              v-if="item.status === 'done'" 
              :job-id="item.job_id" 
              :video-name="getBasename(item.video_source)" 
            />
          </div>
        </div>
      </div>

      <!-- Load More -->
      <LoadMoreButton 
        :has-more="history.pagination.hasMore" 
        :loading="history.pagination.isLoadingMore" 
        :total-loaded="history.items.value.length"
        @load="history.loadMore" 
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useAnalysisHistory } from '~/composables/useAnalysisHistory'
import HistoryRiskBadge from '~/components/operator/history/HistoryRiskBadge.vue'
import LoadMoreButton from '~/components/operator/history/LoadMoreButton.vue'
import ReportDownloadButton from '~/components/operator/report/ReportDownloadButton.vue'

definePageMeta({ layout: "operator" })

const history = useAnalysisHistory()

onMounted(() => {
  history.fetchList()
})

function getBasename(path: string | null): string {
  if (!path) return 'Bilinmeyen video'
  const parts = path.split(/[/\\]/)
  return parts.pop() || 'Bilinmeyen video'
}

function formatDate(isoString: string): string {
  try {
    const d = new Date(isoString)
    if (isNaN(d.getTime())) return 'Tarih bilgisi yok'
    
    return new Intl.DateTimeFormat('tr-TR', {
      day: 'numeric',
      month: 'long',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(d)
  } catch {
    return 'Tarih bilgisi yok'
  }
}

function getReportStatusText(status: string): string {
  switch (status) {
    case 'done': return 'Rapor oluşturulabilir'
    case 'queued': return 'Analiz sırada'
    case 'running': return 'Analiz devam ediyor'
    case 'error': return 'Rapor kullanılamıyor'
    default: return 'Bilinmiyor'
  }
}

function getReportStatusClasses(status: string): string[] {
  switch (status) {
    case 'done': return ['text-emerald-400', 'bg-emerald-500/10', 'border-emerald-500/20']
    case 'queued': 
    case 'running': return ['text-cyan-400', 'bg-cyan-500/10', 'border-cyan-500/20']
    case 'error': return ['text-red-400', 'bg-red-500/10', 'border-red-500/20']
    default: return ['text-gray-400', 'bg-gray-500/10', 'border-gray-500/20']
  }
}
</script>