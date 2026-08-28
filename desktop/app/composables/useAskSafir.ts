import { ref, type Ref } from 'vue'
import { useSafirApi } from './useSafirApi'
import type { AskSource } from '~/types/api'

export type AskState =
  | 'idle'
  | 'creating-conversation'
  | 'connecting'
  | 'streaming'
  | 'saving'
  | 'completed'
  | 'cancelled'
  | 'failed'

export interface AskUiMessage {
  localId: string
  persistedId?: number
  role: 'user' | 'assistant'
  content: string
  createdAt: string
  status: 'temporary' | 'streaming' | 'saving' | 'persisted' | 'failed'
  sources?: AskSource[]
  contextUsed?: string[]
  usedVideo?: boolean
}

export function useAskSafir(jobId: Ref<string | null>, conversationId: Ref<string | null>) {
  const api = useSafirApi()
  const state = ref<AskState>('idle')
  const messages = ref<AskUiMessage[]>([])
  const errorMsg = ref<string | null>(null)
  
  let abortController: AbortController | null = null
  let eventSource: EventSource | null = null
  
  function generateLocalId() {
    return Math.random().toString(36).substring(2, 9)
  }

  function cancelStream() {
    if (abortController) {
      abortController.abort()
      abortController = null
    }
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
    if (state.value === 'connecting' || state.value === 'streaming') {
      state.value = 'cancelled'
      messages.value = messages.value.filter(m => m.status !== 'temporary' && m.status !== 'streaming')
    }
  }
  
  async function submitQuestion(question: string, useVideo: boolean = false) {
    if (!question.trim()) return
    const q = question.trim()
    errorMsg.value = null

    cancelStream()
    
    state.value = 'creating-conversation'
    if (!conversationId.value) {
      try {
        const title = q.substring(0, 40)
        const conv = await api.createConversation({ title, job_id: jobId.value })
        conversationId.value = conv.conversation_id
      } catch (err) {
        state.value = 'failed'
        errorMsg.value = 'Yeni sohbet başlatılamadı.'
        return
      }
    }
    
    const currentConvId = conversationId.value
    if (!currentConvId) {
      state.value = 'failed'
      errorMsg.value = 'Geçerli bir sohbet bulunamadı.'
      return
    }

    const userMsg: AskUiMessage = {
      localId: generateLocalId(),
      role: 'user',
      content: q,
      createdAt: new Date().toISOString(),
      status: 'temporary'
    }
    const asstMsg: AskUiMessage = {
      localId: generateLocalId(),
      role: 'assistant',
      content: '',
      createdAt: new Date().toISOString(),
      status: 'streaming',
      sources: [],
      contextUsed: [],
      usedVideo: useVideo
    }
    
    messages.value.push(userMsg, asstMsg)

    const MAX_SSE_URL_LENGTH = 1800
    const streamUrl = api.askStreamUrl(q, jobId.value, currentConvId, useVideo)
    
    if (streamUrl.length > MAX_SSE_URL_LENGTH) {
      await doFallbackPost(q, useVideo, userMsg, asstMsg, currentConvId)
    } else {
      doEventSource(streamUrl, userMsg, asstMsg, currentConvId)
    }
  }

  function doEventSource(url: string, userMsg: AskUiMessage, asstMsg: AskUiMessage, currentConvId: string) {
    state.value = 'connecting'
    eventSource = new EventSource(url)

    let isTerminated = false
    
    const handleCompletion = async () => {
      if (isTerminated) return
      isTerminated = true
      
      if (eventSource) {
        eventSource.close()
        eventSource = null
      }

      state.value = 'saving'
      userMsg.status = 'saving'
      asstMsg.status = 'saving'
      await persistMessages(userMsg, asstMsg, currentConvId)
    }

    eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.sources || payload.context_used) {
          if (payload.sources) asstMsg.sources = payload.sources
          if (payload.context_used) asstMsg.contextUsed = payload.context_used
        } else if (payload.delta) {
          state.value = 'streaming'
          asstMsg.content += payload.delta
        }
      } catch (e) {
        // ignore JSON parse errors
      }
    }

    eventSource.addEventListener('end', (event: MessageEvent) => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.status === 'done') {
          handleCompletion()
        }
      } catch(e) {}
    })

    eventSource.addEventListener('error', (event: MessageEvent) => {
      if (isTerminated) return
      isTerminated = true
      
      if (eventSource) {
        eventSource.close()
        eventSource = null
      }
      
      let detailMsg = 'Yanıt bağlantısı kesildi. Mesaj geçmişe kaydedilmedi.'
      if (event.data) {
        try {
          const payload = JSON.parse(event.data)
          if (payload.detail) detailMsg = payload.detail
        } catch(e) {}
      }
      
      state.value = 'failed'
      errorMsg.value = detailMsg
      asstMsg.status = 'failed'
      userMsg.status = 'failed'
    })
    
    eventSource.onerror = () => {
      if (isTerminated) return
      isTerminated = true
      if (eventSource) {
        eventSource.close()
        eventSource = null
      }
      state.value = 'failed'
      errorMsg.value = 'Yanıt bağlantısı kesildi. Mesaj geçmişe kaydedilmedi.'
      asstMsg.status = 'failed'
      userMsg.status = 'failed'
    }
  }

  async function doFallbackPost(question: string, useVideo: boolean, userMsg: AskUiMessage, asstMsg: AskUiMessage, currentConvId: string) {
    state.value = 'connecting'
    abortController = new AbortController()
    
    try {
      const response = await api.ask(question, jobId.value, useVideo) // requires custom signal passing if added to API
      state.value = 'saving'
      userMsg.status = 'saving'
      
      asstMsg.content = response.answer
      asstMsg.sources = response.sources
      asstMsg.contextUsed = response.context_used
      asstMsg.status = 'saving'
      
      await persistMessages(userMsg, asstMsg, currentConvId)
    } catch (err: any) {
      if (err.name === 'AbortError') {
        state.value = 'cancelled'
        messages.value = messages.value.filter(m => m.status !== 'temporary' && m.status !== 'streaming')
      } else {
        state.value = 'failed'
        errorMsg.value = err.data?.detail || 'SAFİR şu anda yanıt oluşturamıyor. Lütfen tekrar deneyin.'
        asstMsg.status = 'failed'
        userMsg.status = 'failed'
      }
    } finally {
      abortController = null
    }
  }

  async function persistMessages(userMsg: AskUiMessage, asstMsg: AskUiMessage, currentConvId: string) {
    try {
      const userRes = await api.addConversationMessage(currentConvId, 'user', userMsg.content)
      userMsg.persistedId = userRes.id
      userMsg.status = 'persisted'
      userMsg.createdAt = userRes.created_at
      
      const asstRes = await api.addConversationMessage(currentConvId, 'assistant', asstMsg.content)
      asstMsg.persistedId = asstRes.id
      asstMsg.status = 'persisted'
      asstMsg.createdAt = asstRes.created_at
      
      state.value = 'completed'
    } catch (e) {
      state.value = 'failed'
      errorMsg.value = 'Yanıt oluşturuldu ancak sohbet geçmişine tam olarak kaydedilemedi.'
      if (userMsg.status !== 'persisted') userMsg.status = 'failed'
      if (asstMsg.status !== 'persisted') asstMsg.status = 'failed'
    }
  }
  
  async function loadConversationHistory() {
    if (!conversationId.value) return
    try {
      const data = await api.getConversation(conversationId.value)
      if (data.job_id && jobId.value && data.job_id !== jobId.value) {
        errorMsg.value = 'Bu sohbet farklı bir analiz kaydına bağlı.'
        conversationId.value = null
        return
      }
      messages.value = data.messages.map(m => ({
        localId: String(m.id),
        persistedId: m.id,
        role: m.role,
        content: m.content,
        createdAt: m.created_at,
        status: 'persisted'
      }))
    } catch (e) {
      errorMsg.value = 'Sohbet geçmişi yüklenemedi.'
      conversationId.value = null
    }
  }
  
  function reset() {
    cancelStream()
    messages.value = []
    errorMsg.value = null
    conversationId.value = null
    state.value = 'idle'
  }
  
  return {
    state,
    messages,
    errorMsg,
    submitQuestion,
    cancelStream,
    loadConversationHistory,
    reset
  }
}
