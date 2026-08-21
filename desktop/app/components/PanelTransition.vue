<script setup lang="ts">
// Reusable panel-switch welcome overlay. Mounted once in layouts/default.vue
// so every sidebar panel shares the same component and timing — see
// usePanelTransition.ts for the trigger/content logic and AppSidebar.vue for
// where it's fired. Sits above the routed page content (which has already
// mounted underneath by the time this shows) inside the layout's <main>, so
// the sidebar and topbar stay visible and there is no blank/white flash.
const { active, welcome } = usePanelTransition()
</script>

<template>
  <Transition name="panel-fade">
    <div
      v-if="active && welcome"
      class="absolute inset-0 z-40 flex flex-col items-center justify-center gap-4 bg-surface-0"
      role="status"
      aria-live="polite"
    >
      <img src="~/assets/images/logo.png" alt="SAFİR" class="w-14 h-14 object-contain" />
      <div class="text-center px-6">
        <h2 class="text-lg font-semibold text-slate-100">{{ welcome.title }}</h2>
        <p class="mt-1.5 text-sm text-slate-500 max-w-sm">{{ welcome.subtitle }}</p>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.panel-fade-enter-active {
  transition: opacity 0.18s ease;
}
.panel-fade-leave-active {
  transition: opacity 0.28s ease;
}
.panel-fade-enter-from,
.panel-fade-leave-to {
  opacity: 0;
}
</style>
