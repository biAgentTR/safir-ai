<script setup lang="ts">
// Desktop application shell: top bar + scrollable main region.
const analysisStore = useAnalysisStore()
</script>

<template>
  <div class="h-full flex flex-col relative bg-surface-0 overflow-hidden">
    <BackgroundScene />
    <AppTopbar />
    <main id="app-scroll-region" class="relative z-10 flex-1 min-h-0 overflow-y-auto pt-14">
      <slot />
    </main>
    <VlmChatPanel :job-id="analysisStore.jobId" />
  </div>
</template>
