<script setup lang="ts">
// Purely decorative, low-cost ambient backdrop for the app shell. No Canvas/
// WebGL/Three.js, no Vue reactivity in the render loop — just a couple of
// fixed layers animated with `transform`/`opacity` via CSS, so the compositor
// does the work and idle CPU/GPU stays near zero. Respects
// prefers-reduced-motion (freezes the animation, keeps the static gradient).
</script>

<template>
  <div class="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden="true">
    <div class="absolute inset-0 bg-surface-0" />
    <div class="absolute -top-1/4 -left-1/4 h-[60vh] w-[60vh] rounded-full bg-accent/[0.06] blur-3xl animate-safir-drift-1" />
    <div class="absolute -bottom-1/3 -right-1/4 h-[55vh] w-[55vh] rounded-full bg-accent/[0.05] blur-3xl animate-safir-drift-2" />
    <div class="absolute inset-0 opacity-[0.03] [background-image:linear-gradient(rgb(var(--c-slate-100))_1px,transparent_1px),linear-gradient(90deg,rgb(var(--c-slate-100))_1px,transparent_1px)] [background-size:48px_48px]" />
  </div>
</template>

<style scoped>
@keyframes safir-drift-1 {
  0%,
  100% {
    transform: translate(0, 0);
  }
  50% {
    transform: translate(6%, 4%);
  }
}
@keyframes safir-drift-2 {
  0%,
  100% {
    transform: translate(0, 0);
  }
  50% {
    transform: translate(-5%, -6%);
  }
}
.animate-safir-drift-1 {
  animation: safir-drift-1 32s ease-in-out infinite;
}
.animate-safir-drift-2 {
  animation: safir-drift-2 38s ease-in-out infinite;
}
@media (prefers-reduced-motion: reduce) {
  .animate-safir-drift-1,
  .animate-safir-drift-2 {
    animation: none;
  }
}
</style>
