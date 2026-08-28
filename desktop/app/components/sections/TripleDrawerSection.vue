<script setup lang="ts">
// 3-Panel Sliding Drawer Carousel Deck:
// [Panel 0: Geçmiş (Left)] <-> [Panel 1: SAFİR Asistan (Center)] <-> [Panel 2: Raporlar (Right)]
import HistorySection from './HistorySection.vue'
import AssistantSection from './AssistantSection.vue'
import ReportsSection from './ReportsSection.vue'
import type { DrawerSlideId } from '~/composables/useDrawerDeck'

const { activeSlide, setSlide, isTransitioning, setTransitioning } = useDrawerDeck()
const { goToSection } = useSectionNav()

const SLIDES: { id: DrawerSlideId; label: string; icon: string }[] = [
  { id: 'gecmis', label: 'Analiz Geçmişi', icon: '≡' },
  { id: 'asistan', label: 'SAFİR Asistan', icon: '◆' },
  { id: 'raporlar', label: 'Raporlar', icon: '▦' },
]

const SLIDE_ORDER: DrawerSlideId[] = ['gecmis', 'asistan', 'raporlar']

const activeIndex = computed(() => {
  const idx = SLIDES.findIndex((s) => s.id === activeSlide.value)
  return idx !== -1 ? idx : 1 // default to 1 (SAFİR Asistan)
})

let transitionTimer: ReturnType<typeof setTimeout> | null = null

function selectSlide(id: DrawerSlideId) {
  if (isTransitioning.value && activeSlide.value !== id) return
  if (activeSlide.value === id) return

  setTransitioning(true)
  setSlide(id)
  goToSection(id)

  if (transitionTimer) clearTimeout(transitionTimer)
  transitionTimer = setTimeout(() => {
    setTransitioning(false)
  }, 520)
}

function onTransitionEnd() {
  setTransitioning(false)
}

// ---- Horizontal scroll (Wheel / Trackpad / Shift+Wheel) gesture navigation ----
let wheelCooldown = false

function onWheel(e: WheelEvent) {
  // Check if horizontal scroll intent exists (deltaX or Shift + Wheel)
  const isHorizontal = Math.abs(e.deltaX) > Math.abs(e.deltaY) + 8 || e.shiftKey
  const delta = e.shiftKey ? e.deltaY : e.deltaX

  if (!isHorizontal || Math.abs(delta) < 18) return
  if (wheelCooldown || isTransitioning.value) return

  const currentIdx = activeIndex.value

  if (delta > 18) {
    // Scrolling right -> move to next slide
    if (currentIdx < SLIDE_ORDER.length - 1) {
      wheelCooldown = true
      selectSlide(SLIDE_ORDER[currentIdx + 1]!)
      setTimeout(() => { wheelCooldown = false }, 550)
    }
  } else if (delta < -18) {
    // Scrolling left -> move to previous slide
    if (currentIdx > 0) {
      wheelCooldown = true
      selectSlide(SLIDE_ORDER[currentIdx - 1]!)
      setTimeout(() => { wheelCooldown = false }, 550)
    }
  }
}

// ---- Touch / Swipe gesture navigation ----
let touchStartX = 0
let touchStartY = 0

function onTouchStart(e: TouchEvent) {
  if (e.touches[0]) {
    touchStartX = e.touches[0].clientX
    touchStartY = e.touches[0].clientY
  }
}

function onTouchEnd(e: TouchEvent) {
  if (isTransitioning.value || !e.changedTouches[0]) return
  const diffX = e.changedTouches[0].clientX - touchStartX
  const diffY = e.changedTouches[0].clientY - touchStartY

  if (Math.abs(diffX) > Math.abs(diffY) && Math.abs(diffX) > 40) {
    const currentIdx = activeIndex.value
    if (diffX < -40 && currentIdx < SLIDE_ORDER.length - 1) {
      selectSlide(SLIDE_ORDER[currentIdx + 1]!)
    } else if (diffX > 40 && currentIdx > 0) {
      selectSlide(SLIDE_ORDER[currentIdx - 1]!)
    }
  }
}
</script>

<template>
  <section
    id="triple-drawer-deck"
    class="relative max-w-[1540px] mx-auto px-4 sm:px-10 py-6 overflow-hidden select-none scroll-mt-20"
    @wheel.passive="onWheel"
    @touchstart="onTouchStart"
    @touchend="onTouchEnd"
  >
    <!-- Anchor targets for scroll-spy & URL hashes -->
    <div id="gecmis" class="sr-only" />
    <div id="asistan" class="sr-only" />
    <div id="raporlar" class="sr-only" />

    <!-- Top Mode Indicator & Tab Switcher for the 3'lü Çekmece -->
    <div class="flex items-center justify-center mb-5">
      <div class="inline-flex p-1 rounded-xl bg-surface-1 border border-edge shadow-lg">
        <button
          v-for="s in SLIDES"
          :key="s.id"
          type="button"
          class="relative flex items-center gap-2 px-4 py-2 rounded-lg text-xs sm:text-sm font-medium transition-all duration-200"
          :class="[
            activeSlide === s.id
              ? 'bg-accent text-white shadow-[0_0_15px_rgba(20,184,166,0.35)] font-semibold'
              : 'text-slate-400 hover:text-slate-100 hover:bg-surface-2',
            isTransitioning ? 'cursor-wait' : 'cursor-pointer'
          ]"
          :disabled="isTransitioning && activeSlide !== s.id"
          @click="selectSlide(s.id)"
        >
          <span v-if="activeSlide === s.id" class="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
          <span class="text-xs">{{ s.icon }}</span>
          <span>{{ s.label }}</span>
        </button>
      </div>
    </div>

    <!-- 3-Card Carousel Track with edge fade masks, scroll support and exact centering -->
    <div class="relative w-full flex items-center justify-center min-h-[660px] overflow-hidden rounded-2xl">
      <!-- Floating Navigation Arrow (Left) -->
      <button
        v-if="activeIndex > 0"
        type="button"
        class="absolute left-3 top-1/2 -translate-y-1/2 z-40 w-11 h-11 rounded-full bg-surface-1/90 backdrop-blur-md border border-edge/80 hover:border-accent text-slate-300 hover:text-white shadow-2xl hover:shadow-[0_0_20px_rgba(20,184,166,0.35)] flex items-center justify-center transition-all duration-200 hover:scale-110 cursor-pointer"
        :disabled="isTransitioning"
        aria-label="Önceki Bölüm"
        @click="selectSlide(SLIDE_ORDER[activeIndex - 1]!)"
      >
        ❮
      </button>

      <!-- Floating Navigation Arrow (Right) -->
      <button
        v-if="activeIndex < SLIDE_ORDER.length - 1"
        type="button"
        class="absolute right-3 top-1/2 -translate-y-1/2 z-40 w-11 h-11 rounded-full bg-surface-1/90 backdrop-blur-md border border-edge/80 hover:border-accent text-slate-300 hover:text-white shadow-2xl hover:shadow-[0_0_20px_rgba(20,184,166,0.35)] flex items-center justify-center transition-all duration-200 hover:scale-110 cursor-pointer"
        :disabled="isTransitioning"
        aria-label="Sonraki Bölüm"
        @click="selectSlide(SLIDE_ORDER[activeIndex + 1]!)"
      >
        ❯
      </button>

      <!-- Left & Right Soft Edge Gradient Vignette Masks -->
      <div class="pointer-events-none absolute left-0 top-0 bottom-0 w-12 sm:w-16 bg-gradient-to-r from-surface-0 via-surface-0/60 to-transparent z-30" />
      <div class="pointer-events-none absolute right-0 top-0 bottom-0 w-12 sm:w-16 bg-gradient-to-l from-surface-0 via-surface-0/60 to-transparent z-30" />

      <div
        class="flex items-stretch gap-8 transition-transform duration-500 ease-out py-2"
        :style="{
          transform: activeIndex === 0
            ? 'translateX(calc(min(1040px, 84vw) + 2rem))'
            : activeIndex === 2
            ? 'translateX(calc(-1 * (min(1040px, 84vw) + 2rem)))'
            : 'translateX(0px)',
          transitionTimingFunction: 'cubic-bezier(0.16, 1, 0.3, 1)'
        }"
        @transitionend="onTransitionEnd"
      >
        <!-- CARD 0: Geçmiş (Left Card) -->
        <div
          class="w-[min(1040px,84vw)] shrink-0 rounded-2xl border transition-all duration-500 bg-surface-1/95 backdrop-blur-md p-6 flex flex-col justify-between relative overflow-hidden"
          :class="
            activeIndex === 0
              ? 'scale-100 opacity-100 z-20 border-accent/60 shadow-[0_20px_50px_-12px_rgba(0,0,0,0.6),0_0_25px_-5px_rgba(20,184,166,0.22)] ring-1 ring-accent/40 cursor-default'
              : 'scale-[0.93] opacity-40 hover:opacity-75 z-10 border-edge cursor-pointer hover:border-edge-strong shadow-lg'
          "
          @click="activeIndex !== 0 && selectSlide('gecmis')"
        >
          <div v-if="activeIndex === 0" class="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-transparent via-accent/70 to-transparent" />
          <div :class="activeIndex !== 0 ? 'pointer-events-none' : ''">
            <HistorySection />
          </div>
        </div>

        <!-- CARD 1: SAFİR Asistan (Center Card) -->
        <div
          class="w-[min(1040px,84vw)] shrink-0 rounded-2xl border transition-all duration-500 bg-surface-1/95 backdrop-blur-md p-6 flex flex-col justify-between relative overflow-hidden"
          :class="
            activeIndex === 1
              ? 'scale-100 opacity-100 z-20 border-accent/60 shadow-[0_20px_50px_-12px_rgba(0,0,0,0.6),0_0_25px_-5px_rgba(20,184,166,0.22)] ring-1 ring-accent/40 cursor-default'
              : 'scale-[0.93] opacity-40 hover:opacity-75 z-10 border-edge cursor-pointer hover:border-edge-strong shadow-lg'
          "
          @click="activeIndex !== 1 && selectSlide('asistan')"
        >
          <div v-if="activeIndex === 1" class="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-transparent via-accent/70 to-transparent" />
          <div :class="activeIndex !== 1 ? 'pointer-events-none' : ''">
            <AssistantSection />
          </div>
        </div>

        <!-- CARD 2: Raporlar (Right Card) -->
        <div
          class="w-[min(1040px,84vw)] shrink-0 rounded-2xl border transition-all duration-500 bg-surface-1/95 backdrop-blur-md p-6 flex flex-col justify-between relative overflow-hidden"
          :class="
            activeIndex === 2
              ? 'scale-100 opacity-100 z-20 border-accent/60 shadow-[0_20px_50px_-12px_rgba(0,0,0,0.6),0_0_25px_-5px_rgba(20,184,166,0.22)] ring-1 ring-accent/40 cursor-default'
              : 'scale-[0.93] opacity-40 hover:opacity-75 z-10 border-edge cursor-pointer hover:border-edge-strong shadow-lg'
          "
          @click="activeIndex !== 2 && selectSlide('raporlar')"
        >
          <div v-if="activeIndex === 2" class="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-transparent via-accent/70 to-transparent" />
          <div :class="activeIndex !== 2 ? 'pointer-events-none' : ''">
            <ReportsSection />
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
