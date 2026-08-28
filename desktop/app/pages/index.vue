<script setup lang="ts">
// Hub page: mounts every panel as a scroll-anchored section (see
// components/AppTabNav.vue for the tab bar that drives/reflects scroll
// position, and components/sections/* for each panel's own logic).
const { mode, isModeSwitching } = useAnalysisMode()
const analysisStore = useAnalysisStore()
</script>

<template>
  <div>
    <!-- welcome overlay: shown once until the first backend health probe resolves -->
    <AppSplash />

    <!-- fixed landing section — always the first tab, regardless of mode -->
    <HomeSection />

    <!-- Mode-specific pipeline dashboard with smooth crossfade and skeleton loader -->
    <Transition name="mode-crossfade" mode="out-in">
      <div v-if="isModeSwitching" :key="'skeleton-' + mode" class="w-full">
        <ModeSkeletonLoader :mode="mode" />
      </div>
      <div v-else-if="mode === 'vlm_direct'" key="vlm_direct" class="w-full">
        <VlmDirectSection />
      </div>
      <div v-else key="low_budget" class="w-full">
        <NewAnalysisSection />
      </div>
    </Transition>

    <TripleDrawerSection />
    <SystemSection />

    <!-- lets a short last section still scroll flush to the top of the viewport
         (browsers otherwise clamp at the end of the document) so the tab bar's
         scroll-spy lands on it correctly — see AppTabNav.vue -->
    <div class="h-[60vh]" aria-hidden="true" />
  </div>
</template>
