<script setup lang="ts">
// Desktop application shell: top bar + horizontal scroll-synced tab bar +
// scrollable main. The tab bar (AppTabNav) and the section anchors it
// targets live together — see pages/index.vue and components/sections/*.
</script>

<template>
  <div class="h-full flex flex-col bg-surface-0">
    <BackgroundScene />
    <AppTopbar />
    <AppTabNav />
    <main id="app-scroll-region" class="relative flex-1 min-h-0 overflow-y-auto">
      <slot />
    </main>
  </div>
</template>
