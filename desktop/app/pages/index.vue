<script setup lang="ts">
// Hub page: mounts every panel as a scroll-anchored section (see
// components/AppTabNav.vue for the tab bar that drives/reflects scroll
// position, and components/sections/* for each panel's own logic).
const { mode } = useAnalysisMode()
</script>

<template>
  <div>
    <!-- welcome overlay: shown once until the first backend health probe resolves -->
    <AppSplash />

    <!-- fixed landing section — always the first tab, regardless of mode -->
    <HomeSection />

    <!-- VLM Direct mode's own dashboard is its own section, right after Ana Sayfa when active -->
    <VlmDirectSection v-if="mode === 'vlm_direct'" />

    <NewAnalysisSection />
    <HistorySection />
    <ReportsSection />
    <AssistantSection />
    <SystemSection />

    <!-- lets a short last section still scroll flush to the top of the viewport
         (browsers otherwise clamp at the end of the document) so the tab bar's
         scroll-spy lands on it correctly — see AppTabNav.vue -->
    <div class="h-[60vh]" aria-hidden="true" />
  </div>
</template>
