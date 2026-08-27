<script setup lang="ts">
// High-performance ambient backdrop with giant glowing glass bubbles & 3D parallax scroll reactivity.
// Listens to scroll on #app-scroll-region to smoothly glide background bubbles across depth layers.
// Pure GPU transforms (translate3d, scale) — 60/120 FPS buttery smooth motion.

const scrollY = ref(0)
let ticking = false

function onScroll(e: Event) {
  const target = e.target as HTMLElement | Window
  const y = target instanceof HTMLElement ? target.scrollTop : (window.scrollY || 0)
  if (!ticking) {
    window.requestAnimationFrame(() => {
      scrollY.value = y
      ticking = false
    })
    ticking = true
  }
}

onMounted(() => {
  const scrollEl = document.getElementById('app-scroll-region')
  if (scrollEl) {
    scrollEl.addEventListener('scroll', onScroll, { passive: true })
  }
  window.addEventListener('scroll', onScroll, { passive: true })
})

onBeforeUnmount(() => {
  const scrollEl = document.getElementById('app-scroll-region')
  if (scrollEl) {
    scrollEl.removeEventListener('scroll', onScroll)
  }
  window.removeEventListener('scroll', onScroll)
})
</script>

<template>
  <div class="pointer-events-none fixed inset-0 z-0 overflow-hidden select-none" aria-hidden="true">
    <!-- Base Canvas Background -->
    <div class="absolute inset-0 bg-surface-0 transition-colors duration-500" />

    <!-- Unified Global Seamless Grid Texture (Whisper-faint) -->
    <div
      class="absolute inset-0 opacity-[0.009] dark:opacity-[0.012] [background-image:linear-gradient(currentColor_1px,transparent_1px),linear-gradient(90deg,currentColor_1px,transparent_1px)] [background-size:56px_56px] text-slate-800 dark:text-slate-100"
    />

    <!-- Ambient Fluid Glow Orbs (Ultra-soft, faint atmospheric breathing in the far depth) -->
    <div
      class="absolute -top-[15%] -left-[10%] w-[70vw] h-[70vh] rounded-full bg-accent/[0.035] dark:bg-accent/[0.02] blur-[140px] animate-orb-1"
      :style="{ transform: `translate3d(0, ${-scrollY * 0.08}px, 0)` }"
    />
    <div
      class="absolute -bottom-[20%] -right-[10%] w-[65vw] h-[65vh] rounded-full bg-cyan-500/[0.03] dark:bg-cyan-500/[0.015] blur-[150px] animate-orb-2"
      :style="{ transform: `translate3d(0, ${-scrollY * 0.12}px, 0)` }"
    />
    <div
      class="absolute top-[35%] left-[25%] w-[55vw] h-[55vh] rounded-full bg-teal-400/[0.02] dark:bg-indigo-500/[0.015] blur-[160px] animate-orb-3"
      :style="{ transform: `translate3d(0, ${-scrollY * 0.1}px, 0)` }"
    />

    <!-- ========================================================================= -->
    <!-- 🫧 PARALLAX LAYER 1: DEEP BACKGROUND GIANT BUBBLES (Slow Parallax) 🫧 -->
    <!-- ========================================================================= -->
    <div
      class="absolute inset-0 transition-transform duration-75 ease-out will-change-transform"
      :style="{ transform: `translate3d(0, ${-scrollY * 0.18}px, 0)` }"
    >
      <!-- Giant Hero Bubble 1: Upper Left Deep Orb -->
      <div class="absolute top-[6%] left-[3%] animate-bubble-1">
        <div class="bubble-orb w-48 h-48 sm:w-64 sm:h-64 lg:w-72 lg:h-72" />
      </div>

      <!-- Giant Hero Bubble 2: Upper Right Deep Orb -->
      <div class="absolute top-[12%] right-[4%] animate-bubble-3">
        <div class="bubble-orb bubble-orb-cyan w-52 h-52 sm:w-72 sm:h-72 lg:w-80 lg:h-80" />
      </div>

      <!-- Large Bubble 3: Lower Left Depth Sphere -->
      <div class="absolute top-[68%] left-[6%] animate-bubble-8">
        <div class="bubble-orb w-40 h-40 sm:w-56 sm:h-56" />
      </div>

      <!-- Giant Bubble 4: Lower Right Atmosphere Sphere -->
      <div class="absolute top-[72%] right-[6%] animate-bubble-10">
        <div class="bubble-orb bubble-orb-cyan w-48 h-48 sm:w-64 sm:h-64 lg:w-72 lg:h-72" />
      </div>
    </div>

    <!-- ========================================================================= -->
    <!-- 🫧 PARALLAX LAYER 2: MID-GROUND FLOATING BUBBLES (Medium Parallax) 🫧 -->
    <!-- ========================================================================= -->
    <div
      class="absolute inset-0 transition-transform duration-75 ease-out will-change-transform"
      :style="{ transform: `translate3d(0, ${-scrollY * 0.38}px, 0)` }"
    >
      <!-- Mid Bubble 5: Top Center-Left Floating Bubble -->
      <div class="absolute top-[18%] left-[28%] animate-bubble-2">
        <div class="bubble-orb w-28 h-28 sm:w-36 sm:h-36" />
      </div>

      <!-- Mid Bubble 6: Mid-Left Floating Glass Sphere -->
      <div class="absolute top-[36%] left-[10%] animate-bubble-4">
        <div class="bubble-orb bubble-orb-cyan w-24 h-24 sm:w-32 sm:h-32" />
      </div>

      <!-- Mid Bubble 7: Center Wandering Luminous Orb -->
      <div class="absolute top-[45%] left-[38%] animate-bubble-5">
        <div class="bubble-orb w-32 h-32 sm:w-44 sm:h-44" />
      </div>

      <!-- Mid Bubble 8: Mid-Right Floating Glass Bubble -->
      <div class="absolute top-[40%] right-[14%] animate-bubble-6">
        <div class="bubble-orb bubble-orb-cyan w-36 h-36 sm:w-48 sm:h-48" />
      </div>

      <!-- Mid Bubble 9: Center-Right Floating Orb -->
      <div class="absolute top-[56%] right-[28%] animate-bubble-7">
        <div class="bubble-orb w-20 h-20 sm:w-28 sm:h-28" />
      </div>

      <!-- Mid Bubble 10: Bottom Center Floating Bubble -->
      <div class="absolute top-[80%] left-[42%] animate-bubble-9">
        <div class="bubble-orb bubble-orb-cyan w-28 h-28 sm:w-36 sm:h-36" />
      </div>
    </div>

    <!-- ========================================================================= -->
    <!-- 🫧 PARALLAX LAYER 3: FOREGROUND ACCENT BUBBLES (Fast Parallax) 🫧 -->
    <!-- ========================================================================= -->
    <div
      class="absolute inset-0 transition-transform duration-75 ease-out will-change-transform"
      :style="{ transform: `translate3d(0, ${-scrollY * 0.65}px, 0)` }"
    >
      <!-- Accent Bubble 11: Top Right Sparkle Orb -->
      <div class="absolute top-[24%] right-[22%] animate-bubble-11">
        <div class="bubble-orb w-16 h-16 sm:w-20 sm:h-20" />
      </div>

      <!-- Accent Bubble 12: Bottom Left Gliding Bubble -->
      <div class="absolute top-[86%] left-[20%] animate-bubble-12">
        <div class="bubble-orb bubble-orb-cyan w-18 h-18 sm:w-22 sm:h-22" />
      </div>
    </div>

    <!-- ========================================================================= -->
    <!-- ✨ DRIFTING GEOMETRIC PARTICLES WITH PARALLAX ("Kayan Elementler") ✨ -->
    <!-- ========================================================================= -->
    <div
      class="absolute inset-0 transition-transform duration-75 ease-out will-change-transform"
      :style="{ transform: `translate3d(0, ${-scrollY * 0.28}px, 0)` }"
    >
      <!-- Element 1: Soft Floating Diamond -->
      <div class="absolute top-[20%] left-[16%] animate-float-slow-1">
        <div class="w-4 h-4 rounded-sm bg-accent/[0.06] border border-accent/[0.12] rotate-45 backdrop-blur-sm" />
      </div>

      <!-- Element 2: Delicate Floating Ring -->
      <div class="absolute top-[28%] right-[16%] animate-float-slow-2">
        <div class="w-16 h-16 rounded-full border border-accent/[0.08] border-dashed" />
      </div>

      <!-- Element 3: Ambient Micro Glyph (Cross) -->
      <div class="absolute top-[64%] left-[8%] animate-float-slow-3">
        <div class="text-accent/[0.12] font-mono text-lg font-light select-none">＋</div>
      </div>

      <!-- Element 4: Glowing Soft Particle -->
      <div class="absolute top-[76%] right-[18%] animate-float-slow-4">
        <div class="w-3.5 h-3.5 rounded-full bg-cyan-400/[0.08] blur-[1px]" />
      </div>

      <!-- Element 5: Floating Geometric Dot -->
      <div class="absolute top-[50%] left-[45%] animate-float-slow-5">
        <div class="w-3 h-3 rounded-full bg-accent/[0.10]" />
      </div>

      <!-- Element 6: Soft Outer Ambient Ring -->
      <div class="absolute top-[14%] right-[36%] animate-float-slow-6">
        <div class="w-20 h-20 rounded-full border border-slate-400/[0.07] dark:border-slate-500/[0.07]" />
      </div>

      <!-- Element 7: Subtle Drifting Dot -->
      <div class="absolute top-[82%] left-[34%] animate-float-slow-7">
        <div class="w-4 h-4 rounded-sm border border-accent/[0.08] rotate-12" />
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Glass Bubble Base Styles (Faint, Soft Ethereal Glowing Sphere) */
.bubble-orb {
  position: relative;
  border-radius: 9999px;
  background: radial-gradient(
    circle at 32% 32%,
    rgba(20, 184, 166, 0.055) 0%,
    rgba(20, 184, 166, 0.02) 45%,
    transparent 85%
  );
  border: 1px solid rgba(20, 184, 166, 0.065);
  box-shadow:
    0 0 20px 1px rgba(20, 184, 166, 0.035),
    inset 0 0 10px rgba(255, 255, 255, 0.04);
}

.bubble-orb-cyan {
  background: radial-gradient(
    circle at 32% 32%,
    rgba(6, 182, 212, 0.055) 0%,
    rgba(6, 182, 212, 0.02) 45%,
    transparent 85%
  );
  border: 1px solid rgba(6, 182, 212, 0.065);
  box-shadow:
    0 0 20px 1px rgba(6, 182, 212, 0.035),
    inset 0 0 10px rgba(255, 255, 255, 0.04);
}

/* Light Theme Adaptations (Ultra-Light, Clean & Translucent) */
:global([data-theme='light']) .bubble-orb {
  background: radial-gradient(
    circle at 32% 32%,
    rgba(13, 148, 136, 0.04) 0%,
    rgba(13, 148, 136, 0.015) 45%,
    transparent 85%
  );
  border: 1px solid rgba(13, 148, 136, 0.055);
  box-shadow:
    0 0 16px 1px rgba(13, 148, 136, 0.025),
    inset 0 0 8px rgba(255, 255, 255, 0.12);
}

:global([data-theme='light']) .bubble-orb-cyan {
  background: radial-gradient(
    circle at 32% 32%,
    rgba(8, 145, 178, 0.04) 0%,
    rgba(8, 145, 178, 0.015) 45%,
    transparent 85%
  );
  border: 1px solid rgba(8, 145, 178, 0.055);
  box-shadow:
    0 0 16px 1px rgba(8, 145, 178, 0.025),
    inset 0 0 8px rgba(255, 255, 255, 0.12);
}

/* Fluid Glow Orbs Animations */
@keyframes orb-1 {
  0% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(8vw, 6vh, 0) scale(1.08); }
  100% { transform: translate3d(-4vw, 4vh, 0) scale(0.95); }
}
@keyframes orb-2 {
  0% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(-7vw, -8vh, 0) scale(1.1); }
  100% { transform: translate3d(5vw, -4vh, 0) scale(0.92); }
}
@keyframes orb-3 {
  0% { transform: translate3d(0, 0, 0) scale(0.95); opacity: 0.5; }
  50% { transform: translate3d(4vw, -5vh, 0) scale(1.12); opacity: 0.8; }
  100% { transform: translate3d(-6vw, 3vh, 0) scale(1); opacity: 0.5; }
}

.animate-orb-1 { animation: orb-1 34s ease-in-out infinite alternate; }
.animate-orb-2 { animation: orb-2 40s ease-in-out infinite alternate; }
.animate-orb-3 { animation: orb-3 28s ease-in-out infinite alternate; }

/* Wandering Glowing Bubbles Animations (Visible, organic 3D gliding) */
@keyframes bubble-drift-1 {
  0% { transform: translate3d(0, 0, 0) scale(1); }
  33% { transform: translate3d(55px, -65px, 0) scale(1.06); }
  66% { transform: translate3d(-35px, -45px, 0) scale(0.94); }
  100% { transform: translate3d(25px, 50px, 0) scale(1.03); }
}
@keyframes bubble-drift-2 {
  0% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(-45px, 55px, 0) scale(1.12); }
  100% { transform: translate3d(35px, -35px, 0) scale(0.92); }
}
@keyframes bubble-drift-3 {
  0% { transform: translate3d(0, 0, 0) scale(0.98); }
  50% { transform: translate3d(-60px, -55px, 0) scale(1.08); }
  100% { transform: translate3d(45px, 55px, 0) scale(0.94); }
}
@keyframes bubble-drift-4 {
  0% { transform: translate3d(0, 0, 0); }
  50% { transform: translate3d(45px, -40px, 0) scale(1.2); }
  100% { transform: translate3d(-25px, 35px, 0) scale(0.88); }
}
@keyframes bubble-drift-5 {
  0% { transform: translate3d(0, 0, 0) scale(1); }
  33% { transform: translate3d(-48px, 38px, 0) scale(1.06); }
  66% { transform: translate3d(42px, -52px, 0) scale(0.95); }
  100% { transform: translate3d(-18px, -24px, 0) scale(1.04); }
}
@keyframes bubble-drift-6 {
  0% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(50px, 60px, 0) scale(1.08); }
  100% { transform: translate3d(-40px, -40px, 0) scale(0.93); }
}
@keyframes bubble-drift-7 {
  0% { transform: translate3d(0, 0, 0) scale(0.95); }
  50% { transform: translate3d(40px, -45px, 0) scale(1.15); }
  100% { transform: translate3d(-30px, 30px, 0) scale(1); }
}
@keyframes bubble-drift-8 {
  0% { transform: translate3d(0, 0, 0); }
  50% { transform: translate3d(-38px, -50px, 0) scale(1.15); }
  100% { transform: translate3d(32px, 42px, 0) scale(0.9); }
}
@keyframes bubble-drift-9 {
  0% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(-52px, -58px, 0) scale(1.08); }
  100% { transform: translate3d(45px, 35px, 0) scale(0.93); }
}
@keyframes bubble-drift-10 {
  0% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(35px, -35px, 0) scale(1.2); }
  100% { transform: translate3d(-30px, 25px, 0) scale(0.88); }
}
@keyframes bubble-drift-11 {
  0% { transform: translate3d(0, 0, 0) scale(0.9); }
  50% { transform: translate3d(-30px, 40px, 0) scale(1.25); }
  100% { transform: translate3d(35px, -25px, 0) scale(0.88); }
}
@keyframes bubble-drift-12 {
  0% { transform: translate3d(0, 0, 0); }
  50% { transform: translate3d(32px, -35px, 0) scale(1.18); }
  100% { transform: translate3d(-24px, 30px, 0) scale(0.92); }
}

.animate-bubble-1 { animation: bubble-drift-1 22s ease-in-out infinite alternate; }
.animate-bubble-2 { animation: bubble-drift-2 18s ease-in-out infinite alternate; }
.animate-bubble-3 { animation: bubble-drift-3 26s ease-in-out infinite alternate; }
.animate-bubble-4 { animation: bubble-drift-4 16s ease-in-out infinite alternate; }
.animate-bubble-5 { animation: bubble-drift-5 23s ease-in-out infinite alternate; }
.animate-bubble-6 { animation: bubble-drift-6 25s ease-in-out infinite alternate; }
.animate-bubble-7 { animation: bubble-drift-7 19s ease-in-out infinite alternate; }
.animate-bubble-8 { animation: bubble-drift-8 20s ease-in-out infinite alternate; }
.animate-bubble-9 { animation: bubble-drift-9 28s ease-in-out infinite alternate; }
.animate-bubble-10 { animation: bubble-drift-10 24s ease-in-out infinite alternate; }
.animate-bubble-11 { animation: bubble-drift-11 15s ease-in-out infinite alternate; }
.animate-bubble-12 { animation: bubble-drift-12 17s ease-in-out infinite alternate; }

/* Slowly Floating & Drifting Geometric Elements */
@keyframes float-slow-1 {
  0% { transform: translate3d(0, 0, 0) rotate(0deg); }
  50% { transform: translate3d(24px, -35px, 0) rotate(45deg); }
  100% { transform: translate3d(-18px, 20px, 0) rotate(90deg); }
}
@keyframes float-slow-2 {
  0% { transform: translate3d(0, 0, 0) rotate(0deg); }
  50% { transform: translate3d(-30px, 25px, 0) rotate(180deg); }
  100% { transform: translate3d(20px, -20px, 0) rotate(360deg); }
}
@keyframes float-slow-3 {
  0% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(15px, 30px, 0) scale(1.15); }
  100% { transform: translate3d(-20px, -15px, 0) scale(0.9); }
}
@keyframes float-slow-4 {
  0% { transform: translate3d(0, 0, 0); opacity: 0.3; }
  50% { transform: translate3d(-25px, -30px, 0); opacity: 0.7; }
  100% { transform: translate3d(18px, 22px, 0); opacity: 0.3; }
}
@keyframes float-slow-5 {
  0% { transform: translate3d(0, 0, 0); }
  50% { transform: translate3d(20px, 15px, 0); }
  100% { transform: translate3d(-15px, -20px, 0); }
}
@keyframes float-slow-6 {
  0% { transform: translate3d(0, 0, 0) rotate(0deg); }
  50% { transform: translate3d(-20px, 28px, 0) rotate(90deg); }
  100% { transform: translate3d(25px, -15px, 0) rotate(180deg); }
}
@keyframes float-slow-7 {
  0% { transform: translate3d(0, 0, 0) rotate(12deg); }
  50% { transform: translate3d(30px, -25px, 0) rotate(60deg); }
  100% { transform: translate3d(-10px, 20px, 0) rotate(12deg); }
}

.animate-float-slow-1 { animation: float-slow-1 26s ease-in-out infinite alternate; }
.animate-float-slow-2 { animation: float-slow-2 32s ease-in-out infinite alternate; }
.animate-float-slow-3 { animation: float-slow-3 24s ease-in-out infinite alternate; }
.animate-float-slow-4 { animation: float-slow-4 22s ease-in-out infinite alternate; }
.animate-float-slow-5 { animation: float-slow-5 29s ease-in-out infinite alternate; }
.animate-float-slow-6 { animation: float-slow-6 36s ease-in-out infinite alternate; }
.animate-float-slow-7 { animation: float-slow-7 27s ease-in-out infinite alternate; }

@media (prefers-reduced-motion: reduce) {
  .animate-orb-1, .animate-orb-2, .animate-orb-3,
  .animate-bubble-1, .animate-bubble-2, .animate-bubble-3, .animate-bubble-4, .animate-bubble-5,
  .animate-bubble-6, .animate-bubble-7, .animate-bubble-8, .animate-bubble-9, .animate-bubble-10,
  .animate-bubble-11, .animate-bubble-12,
  .animate-float-slow-1, .animate-float-slow-2, .animate-float-slow-3,
  .animate-float-slow-4, .animate-float-slow-5, .animate-float-slow-6, .animate-float-slow-7 {
    animation: none;
  }
}
</style>
