import { ref, reactive, onBeforeUnmount } from 'vue'
import { useSafirApi } from './useSafirApi'
import type { HistoryListItem } from '../types/api'

export type HistoryPageState =
  | 'idle'
  | 'loading'
  | 'ready'
  | 'empty'
  | 'service-unavailable'
  | 'failed'

export interface HistoryPaginationState {
  limit: number
  offset: number
  hasMore: boolean
  isLoadingMore: boolean
}

const HISTORY_PAGE_SIZE = 20

export function useAnalysisHistory() {
  const api = useSafirApi()
  
  const state = ref<HistoryPageState>('idle')
  const errorMessage = ref<string>('')
  
  const items = ref<HistoryListItem[]>([])
  
  const pagination = reactive<HistoryPaginationState>({
    limit: HISTORY_PAGE_SIZE,
    offset: 0,
    hasMore: false,
    isLoadingMore: false
  })
  
  let abortController: AbortController | null = null

  function cancelCurrentRequest() {
    if (abortController) {
      abortController.abort()
      abortController = null
    }
  }

  async function fetchList(isLoadMore = false) {
    if (!isLoadMore) {
      cancelCurrentRequest()
      // Only set loading state if not load more, but keep existing items if we have them
      // We don't wipe items out on simple refresh to avoid layout jump
      if (items.value.length === 0) {
        state.value = 'loading'
      }
      pagination.offset = 0
    } else {
      pagination.isLoadingMore = true
    }

    abortController = new AbortController()

    try {
      const response = await api.getHistory(
        pagination.limit, 
        pagination.offset, 
        abortController.signal
      )
      
      if (!Array.isArray(response)) {
        throw new Error('Invalid response format: expected an array')
      }

      if (isLoadMore) {
        // Append unique items
        const currentIds = new Set(items.value.map(i => i.job_id))
        const newItems = response.filter(i => !currentIds.has(i.job_id))
        items.value.push(...newItems)
      } else {
        items.value = response
      }

      // If we got exactly the limit, there might be more. Otherwise, we reached the end.
      pagination.hasMore = response.length >= pagination.limit
      
      if (items.value.length === 0) {
        state.value = 'empty'
      } else {
        state.value = 'ready'
      }

    } catch (err: any) {
      if (err.name === 'AbortError' || err.message?.includes('aborted')) {
        // User-initiated cancel, do not show error
        return
      }
      
      // If we are doing a background refresh and it fails, don't wipe the list if we have data
      if (items.value.length === 0 || !isLoadMore) {
        if (err.response?.status === 404 || err.response?.status >= 500 || err.message?.includes('fetch')) {
          state.value = 'service-unavailable'
        } else {
          state.value = 'failed'
          errorMessage.value = err.message || 'Analiz geçmişi yüklenemedi.'
        }
      } else if (isLoadMore) {
        console.error('Load more failed', err)
      }
    } finally {
      pagination.isLoadingMore = false
    }
  }

  async function loadMore() {
    if (pagination.isLoadingMore || !pagination.hasMore) return
    
    // Set offset to current unique items length
    pagination.offset = items.value.length
    await fetchList(true)
  }

  async function refresh() {
    if (pagination.isLoadingMore) return
    await fetchList(false)
  }

  onBeforeUnmount(() => {
    cancelCurrentRequest()
  })

  return {
    state,
    items,
    errorMessage,
    pagination,
    fetchList,
    loadMore,
    refresh
  }
}