<template>
  <div>
    <!-- Toggle Button to Open Drawer -->
    <button 
      @click="isOpen = true"
      class="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-cyan-400 bg-cyan-500/10 hover:bg-cyan-500/20 rounded-md border border-cyan-500/20 transition-colors"
      :disabled="!jobId"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
      Ask SAFİR
    </button>

    <!-- Native Dialog for Drawer -->
    <Transition name="fade">
      <div v-if="isOpen" class="relative z-50">
        <!-- Backdrop -->
        <div class="fixed inset-0 bg-black/60 backdrop-blur-sm" @click="closeDrawer" />

        <div class="fixed inset-0 overflow-hidden pointer-events-none">
          <div class="absolute inset-0 overflow-hidden">
            <div class="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10">
              <!-- Drawer Panel -->
              <Transition name="slide">
                <div v-if="isOpen" class="pointer-events-auto w-screen max-w-md h-full bg-zinc-900 border-l border-white/10 shadow-xl flex flex-col">
                  
                  <!-- Header -->
                  <div class="flex items-center justify-between px-4 py-4 border-b border-white/10 bg-zinc-900/50">
                    <h2 class="text-lg font-medium text-white flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-cyan-400"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                      Ask SAFİR
                    </h2>
                    <div class="flex items-center gap-2">
                      <button @click="startNewConversation" class="p-2 text-zinc-400 hover:text-white hover:bg-white/10 rounded-md transition-colors" title="Yeni Sohbet">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="M5 12h14"/></svg>
                      </button>
                      <button @click="closeDrawer" class="p-2 text-zinc-400 hover:text-white hover:bg-white/10 rounded-md transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
                      </button>
                    </div>
                  </div>

                  <!-- Messages Area -->
                  <div class="flex-1 overflow-y-auto p-4 space-y-6" ref="messagesContainer">
                    
                    <!-- Empty State / Suggestions -->
                    <div v-if="messages.length === 0" class="h-full flex flex-col justify-center items-center text-center space-y-6 py-10">
                      <div class="w-16 h-16 rounded-full bg-cyan-500/10 flex items-center justify-center border border-cyan-500/20">
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-cyan-400"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>
                      </div>
                      <div>
                        <h3 class="text-lg font-medium text-white mb-2">Nasıl yardımcı olabilirim?</h3>
                        <p class="text-sm text-zinc-400">Bu analiz raporu hakkında sorular sorabilir, riskler ve bulgularla ilgili ek detay isteyebilirsiniz.</p>
                      </div>
                      
                      <div v-if="suggestions.length > 0" class="w-full space-y-2 mt-4 text-left">
                        <p class="text-xs text-zinc-500 font-medium uppercase tracking-wider pl-1">Öneriler</p>
                        <button 
                          v-for="(sug, idx) in suggestions" 
                          :key="idx"
                          @click="useSuggestion(sug)"
                          class="w-full text-left p-3 text-sm text-zinc-300 bg-white/5 hover:bg-white/10 rounded-md border border-white/5 hover:border-white/20 transition-colors"
                        >
                          {{ sug }}
                        </button>
                      </div>
                    </div>

                      <!-- Chat Bubbles -->
                      <template v-else>
                        <div 
                          v-for="msg in messages" 
                          :key="msg.localId"
                          class="flex flex-col max-w-[90%]"
                          :class="msg.role === 'user' ? 'self-end items-end ml-auto' : 'self-start items-start'"
                        >
                          <div 
                            class="px-4 py-2.5 rounded-2xl whitespace-pre-wrap text-sm"
                            :class="msg.role === 'user' ? 'bg-cyan-600 text-white rounded-tr-sm' : 'bg-zinc-800 text-zinc-200 rounded-tl-sm border border-white/5'"
                          >{{ msg.content }}<span v-if="msg.status === 'streaming'" class="inline-block w-1.5 h-4 ml-1 bg-cyan-400 animate-pulse align-middle"></span></div>
                          
                          <!-- Sources (Assistant only) -->
                          <div v-if="msg.role === 'assistant' && msg.sources && msg.sources.length > 0" class="mt-2 w-full">
                            <details class="text-xs text-zinc-400 bg-zinc-800/50 rounded-md border border-white/5">
                              <summary class="px-3 py-2 cursor-pointer hover:text-zinc-300 font-medium select-none">
                                Kullanılan kaynaklar ({{ msg.sources.length }})
                              </summary>
                              <div class="px-3 pb-3 pt-1 space-y-2 border-t border-white/5 mt-1">
                                <div v-for="(src, idx) in msg.sources" :key="idx" class="p-2 rounded bg-black/20 border border-white/5">
                                  <div class="flex items-center justify-between mb-1">
                                    <span class="font-medium text-cyan-400">{{ src.label || src.type }}</span>
                                    <span v-if="src.score != null && src.score >= 0 && src.score <= 1" class="text-zinc-500">%{{ Math.round(src.score * 100) }} Semantik eşleşme</span>
                                    <span v-else-if="src.score != null" class="text-zinc-500">{{ src.score }}</span>
                                  </div>
                                  <p v-if="src.text" class="text-zinc-400 whitespace-pre-wrap line-clamp-3 hover:line-clamp-none transition-all">{{ src.text }}</p>
                                </div>
                              </div>
                            </details>
                          </div>

                          <!-- Status / Time -->
                          <div class="text-[10px] text-zinc-500 mt-1 flex items-center gap-1" :class="msg.role === 'user' ? 'justify-end' : 'justify-start'">
                            <span v-if="msg.status === 'failed'" class="text-red-400">Kaydedilemedi</span>
                            <span v-else-if="msg.status === 'saving'">Kaydediliyor...</span>
                            <span v-else-if="msg.status === 'persisted' && msg.createdAt">{{ new Date(msg.createdAt).toLocaleTimeString('tr-TR', {hour: '2-digit', minute:'2-digit'}) }}</span>
                            <span v-if="msg.usedVideo" class="flex items-center gap-1 text-cyan-500 ml-2" title="Orijinal video üzerinden (VLM) tarandı"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg> Video modu</span>
                          </div>
                        </div>
                      </template>
                      
                      <!-- Global Error Message -->
                      <div v-if="errorMsg" class="p-3 rounded-md bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                        {{ errorMsg }}
                      </div>
                    </div>

                    <!-- Input Area -->
                    <div class="p-4 bg-zinc-900 border-t border-white/10 shrink-0">
                      <!-- Active Actions -->
                      <div v-if="state === 'connecting' || state === 'streaming'" class="flex justify-center mb-4">
                        <button @click="cancelStream" class="flex items-center gap-2 px-4 py-2 text-sm font-medium text-zinc-300 bg-zinc-800 hover:bg-zinc-700 rounded-full border border-white/10 transition-colors shadow-lg">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/></svg>
                          Yanıtı durdur
                        </button>
                      </div>

                      <div class="flex items-center justify-between mb-2 px-1">
                        <label class="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer group">
                          <div class="relative flex items-center">
                            <input type="checkbox" v-model="useVideo" class="sr-only peer" :disabled="state === 'connecting' || state === 'streaming' || state === 'saving'">
                            <div class="w-8 h-4 bg-zinc-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-cyan-500 peer-disabled:opacity-50"></div>
                          </div>
                          <span class="group-hover:text-zinc-300 transition-colors select-none" title="Soruyu analiz raporuna ek olarak orijinal video üzerinden EVREN modeline yönlendirir.">Videoyu da incele</span>
                        </label>
                      </div>

                      <div class="relative">
                        <textarea
                          v-model="questionInput"
                          @keydown.enter.prevent="handleEnter"
                          placeholder="Analiz hakkında soru sorun..."
                          class="w-full bg-zinc-950 border border-white/10 rounded-xl py-3 pl-4 pr-12 text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 resize-none overflow-hidden transition-all"
                          rows="1"
                          :disabled="state === 'connecting' || state === 'streaming' || state === 'saving'"
                          ref="textareaRef"
                          @input="adjustTextareaHeight"
                          style="min-height: 44px; max-height: 120px;"
                        ></textarea>
                        <button
                          @click="submit"
                          :disabled="!questionInput.trim() || state === 'connecting' || state === 'streaming' || state === 'saving'"
                          class="absolute right-2 bottom-2 p-1.5 rounded-lg text-white bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 disabled:bg-zinc-700 transition-colors"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>
                        </button>
                      </div>
                      <p class="text-[10px] text-center text-zinc-500 mt-2">SAFİR halüsinasyon yapabilir. Önemli kararlarda videoyu kendiniz inceleyin.</p>
                    </div>

                </div>
              </Transition>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick, computed } from 'vue'
import { useAskSafir } from '~/composables/useAskSafir'
import { useSafirApi } from '~/composables/useSafirApi'

const props = defineProps<{
  jobId: string | null
}>()

const isOpen = ref(false)
const questionInput = ref('')
const useVideo = ref(false)
const textareaRef = ref<HTMLTextAreaElement | null>(null)
const messagesContainer = ref<HTMLElement | null>(null)
const suggestions = ref<string[]>([])

const api = useSafirApi()
const jobIdRef = computed(() => props.jobId)
// We will store the active conversation in local ref for the session
const activeConversationId = ref<string | null>(null)

const {
  state,
  messages,
  errorMsg,
  submitQuestion,
  cancelStream,
  loadConversationHistory,
  reset
} = useAskSafir(jobIdRef, activeConversationId)

// Load suggestions
const loadSuggestions = async () => {
  if (!props.jobId) return
  try {
    suggestions.value = await api.getAskSuggestions(props.jobId)
  } catch (e) {
    suggestions.value = []
  }
}

// Watch drawer open state
watch(isOpen, async (newVal) => {
  if (newVal) {
    // When opened
    if (messages.value.length === 0 && props.jobId) {
      await loadSuggestions()
    }
    nextTick(() => {
      textareaRef.value?.focus()
    })
  } else {
    // When closed, don't abort unless desired, but user instruction implies escape should not silently abort if active
    // "Karmaşık confirmation eklemek istemiyorsan Escape aktif stream sırasında paneli kapatmamalı."
    // Actually we will allow closing panel, but let the stream run in background. We just don't abort on close.
  }
})

// Auto-scroll when messages change
watch(() => messages.value, () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}, { deep: true })

watch(() => state.value, (newVal) => {
  if (newVal === 'cancelled' || newVal === 'failed') {
    // Bring back the question to input if it was just failed/cancelled
    if (messages.value.length > 0) {
       // if we removed the temporary messages, the user has to retype? 
       // In useAskSafir, we already handle filtering out the temporary messages on cancel. 
       // We can recover the last question here if needed, but keeping it simple.
    }
  }
})

const adjustTextareaHeight = () => {
  const el = textareaRef.value
  if (!el) return
  el.style.height = '44px' // min-height
  el.style.height = Math.min(el.scrollHeight, 120) + 'px'
}

const handleEnter = (e: KeyboardEvent) => {
  if (e.shiftKey) {
    return // allow newline
  }
  submit()
}

const submit = async () => {
  const q = questionInput.value.trim()
  if (!q || state.value === 'connecting' || state.value === 'streaming' || state.value === 'saving') return
  
  questionInput.value = ''
  if (textareaRef.value) textareaRef.value.style.height = '44px'
  
  await submitQuestion(q, useVideo.value)
}

const useSuggestion = async (sug: string) => {
  questionInput.value = sug
  await submit()
}

const startNewConversation = () => {
  if (state.value === 'connecting' || state.value === 'streaming') {
    cancelStream()
  }
  reset()
  if (props.jobId) {
    loadSuggestions()
  }
  nextTick(() => {
    textareaRef.value?.focus()
  })
}

const closeDrawer = () => {
  // Only close if not streaming, or let it run in background
  if (state.value === 'connecting' || state.value === 'streaming') {
    // Block closing while streaming to prevent accidental silent aborting
    return
  }
  isOpen.value = false
}
function open() { isOpen.value = true }
function close() { isOpen.value = false }
defineExpose({ open, close })
</script>

<style scoped>
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
.slide-enter-active, .slide-leave-active { transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
.slide-enter-from, .slide-leave-to { transform: translateX(100%); }
</style>
