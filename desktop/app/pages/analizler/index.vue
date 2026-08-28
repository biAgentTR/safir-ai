<template>
  <div class="max-w-[1440px] mx-auto p-4 sm:p-6 lg:p-8">
    <HistoryPageHeader :is-refreshing="history.pagination.isLoadingMore && history.items.value?.length > 0" @refresh="history.refresh" />

    <!-- Error State -->
    <div v-if="history.state.value === 'failed'" class="bg-red-500/10 border border-red-500/20 rounded-xl p-6 text-center">
      <div class="inline-flex p-3 rounded-full bg-red-500/20 mb-4 text-red-400">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      </div>
      <h3 class="text-lg font-medium text-white mb-2">Analiz geçmişi yüklenemedi</h3>
      <p class="text-red-400/80 mb-6 max-w-md mx-auto">{{ history.errorMessage.value }}</p>
      <button @click="history.refresh" class="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium transition-colors">Tekrar dene</button>
    </div>

    <!-- Service Unavailable -->
    <div v-else-if="history.state.value === 'service-unavailable'" class="bg-gray-800/50 border border-gray-700 rounded-xl p-8 text-center max-w-2xl mx-auto mt-12">
      <div class="inline-flex p-4 rounded-full bg-gray-700/50 mb-4 text-gray-400">
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      </div>
      <h3 class="text-xl font-semibold text-white mb-2">Analiz servisine ulaşılamıyor</h3>
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
      <h3 class="text-xl font-semibold text-white mb-2">Henüz analiz bulunmuyor</h3>
      <p class="text-gray-400 mb-8 max-w-md mx-auto">Başlattığınız video analizleri burada görüntülenecek.</p>
      <NuxtLink to="/" class="inline-flex items-center gap-2 px-6 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium transition-colors">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M12 5v14"/></svg>
        Yeni analiz başlat
      </NuxtLink>
    </div>

    <!-- Loading Skeleton -->
    <div v-else-if="history.state.value === 'loading'" class="space-y-4" aria-busy="true">
      <div class="hidden md:grid grid-cols-12 gap-4 px-6 py-3 border-b border-white/10 bg-gray-900/50 rounded-t-xl mb-2">
        <div class="col-span-3 text-xs font-semibold text-gray-500 uppercase">Video</div>
        <div class="col-span-2 text-xs font-semibold text-gray-500 uppercase">Durum</div>
        <div class="col-span-2 text-xs font-semibold text-gray-500 uppercase">Risk</div>
        <div class="col-span-3 text-xs font-semibold text-gray-500 uppercase">Özet</div>
        <div class="col-span-2 text-xs font-semibold text-gray-500 uppercase text-right">Tarih</div>
      </div>
      <div v-for="i in 5" :key="i" class="bg-gray-800/30 rounded-xl p-4 md:px-6 md:py-4 flex flex-col md:grid md:grid-cols-12 md:gap-4 md:items-center animate-pulse border border-gray-800">
        <div class="col-span-3 h-5 bg-gray-700/50 rounded w-3/4 mb-4 md:mb-0"></div>
        <div class="col-span-2 h-6 bg-gray-700/50 rounded-full w-24 mb-4 md:mb-0"></div>
        <div class="col-span-2 h-5 bg-gray-700/50 rounded w-16 mb-4 md:mb-0"></div>
        <div class="col-span-3 h-4 bg-gray-700/30 rounded w-full mb-4 md:mb-0"></div>
        <div class="col-span-2 h-4 bg-gray-700/30 rounded w-20 md:ml-auto"></div>
      </div>
    </div>

    <!-- Data List -->
    <div v-else class="flex flex-col">
      <!-- Desktop Table Header -->
      <div class="hidden md:block overflow-hidden rounded-xl border border-white/5 bg-[#1E2532]/30 mb-6">
        <table class="w-full text-left border-collapse">
          <thead class="bg-[#141A25]">
            <tr>
              <th scope="col" class="px-6 py-3 border-b border-white/5 text-xs font-semibold text-gray-400 uppercase tracking-wider w-[25%]">Video</th>
              <th scope="col" class="px-6 py-3 border-b border-white/5 text-xs font-semibold text-gray-400 uppercase tracking-wider w-[15%]">Durum</th>
              <th scope="col" class="px-6 py-3 border-b border-white/5 text-xs font-semibold text-gray-400 uppercase tracking-wider w-[15%]">Risk</th>
              <th scope="col" class="px-6 py-3 border-b border-white/5 text-xs font-semibold text-gray-400 uppercase tracking-wider w-[30%]">Özet</th>
              <th scope="col" class="px-6 py-3 border-b border-white/5 text-xs font-semibold text-gray-400 uppercase tracking-wider text-right w-[15%]">Tarih / İşlem</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-white/5">
            <tr v-for="item in history.items.value" :key="item.job_id" class="hover:bg-white/[0.02] transition-colors group">
              <!-- 1. Video -->
              <td class="px-6 py-4 align-top">
                <div class="flex flex-col min-w-0">
                  <span class="text-sm font-medium text-white truncate">{{ getBasename(item.video_source) }}</span>
                  <span class="text-xs text-gray-500 font-mono mt-1 truncate">{{ item.job_id.substring(0, 8) }}&hellip;</span>
                </div>
              </td>
              <!-- 2. Status -->
              <td class="px-6 py-4 align-top">
                <HistoryStatusBadge :status="item.status" />
              </td>
              <!-- 3. Risk -->
              <td class="px-6 py-4 align-top">
                <HistoryRiskBadge :level="item.risk_level" :score="item.risk_score" />
              </td>
              <!-- 4. Summary -->
              <td class="px-6 py-4 align-top">
                <p class="text-sm text-gray-300 line-clamp-2 leading-snug" v-if="item.summary">{{ item.summary }}</p>
                <p class="text-sm text-gray-500 italic" v-else>Bu analiz için henüz özet bulunmuyor.</p>
              </td>
              <!-- 5. Date & Action -->
              <td class="px-6 py-4 align-top text-right">
                <div class="flex flex-col items-end gap-2">
                  <span class="text-xs text-gray-400 whitespace-nowrap">{{ formatDate(item.created_at) }}</span>
                  <NuxtLink 
                    :to="`/analizler/${encodeURIComponent(item.job_id)}`"
                    class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-gray-700 bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 opacity-0 group-hover:opacity-100 focus:opacity-100"
                  >
                    {{ getActionText(item.status) }}
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
                  </NuxtLink>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Mobile Cards -->
      <div class="flex flex-col gap-3 md:hidden mb-6">
        <div 
          v-for="item in history.items.value" 
          :key="`mob-${item.job_id}`"
          class="bg-[#1E2532] border border-white/5 rounded-xl p-4 flex flex-col relative group"
        >
          <!-- Mobile action link wrapper -->
          <NuxtLink :to="`/analizler/${encodeURIComponent(item.job_id)}`" class="absolute inset-0 z-0 rounded-xl focus:outline-none focus:ring-2 focus:ring-cyan-500/50"></NuxtLink>

          <!-- 1. Video Name & ID -->
          <div class="flex flex-col min-w-0 mb-3 z-10 pointer-events-none">
            <span class="text-sm font-medium text-white truncate">{{ getBasename(item.video_source) }}</span>
            <span class="text-xs text-gray-500 font-mono mt-0.5 truncate">{{ item.job_id.substring(0, 8) }}&hellip;</span>
          </div>

          <!-- 2. Status & Risk -->
          <div class="flex items-start justify-between gap-4 mb-4 z-10 pointer-events-none">
            <HistoryStatusBadge :status="item.status" />
            <HistoryRiskBadge :level="item.risk_level" :score="item.risk_score" />
          </div>

          <!-- 4. Summary -->
          <div class="mb-4 z-10 pointer-events-none">
            <p class="text-sm text-gray-300 line-clamp-2 leading-snug" v-if="item.summary">{{ item.summary }}</p>
            <p class="text-sm text-gray-500 italic" v-else>Bu analiz için henüz özet bulunmuyor.</p>
          </div>

          <!-- 5. Date -->
          <div class="flex justify-between items-end mt-auto pt-4 border-t border-white/5 z-10 pointer-events-none">
            <span class="text-xs text-gray-400">{{ formatDate(item.created_at) }}</span>
            <span class="text-xs text-cyan-500 font-medium">{{ getActionText(item.status) }}</span>
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
import HistoryPageHeader from '~/components/operator/history/HistoryPageHeader.vue'
import HistoryStatusBadge from '~/components/operator/history/HistoryStatusBadge.vue'
import HistoryRiskBadge from '~/components/operator/history/HistoryRiskBadge.vue'
import LoadMoreButton from '~/components/operator/history/LoadMoreButton.vue'

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

function getActionText(status: string): string {
  switch (status) {
    case 'done': return 'Sonucu görüntüle'
    case 'queued': 
    case 'running': return 'Durumu görüntüle'
    case 'error': return 'Detayı görüntüle'
    default: return 'Görüntüle'
  }
}
</script>