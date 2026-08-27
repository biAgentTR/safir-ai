<script setup lang="ts">
// Realistic sweeping light skeleton loader for Lite <-> Direct mode switching
defineProps<{
  mode: 'low_budget' | 'vlm_direct'
}>()
</script>

<template>
  <div class="w-full max-w-4xl mx-auto px-4 py-8 select-none">
    <!-- VLM Direct Mode Skeleton -->
    <div v-if="mode === 'vlm_direct'" class="space-y-6">
      <!-- Hero / Eyebrow / Title placeholder -->
      <div class="flex flex-col items-center gap-3 max-w-lg mx-auto text-center">
        <div class="h-4 w-44 rounded-full skeleton-shimmer-accent" />
        <div class="h-8 w-72 sm:w-96 rounded-xl skeleton-shimmer" />
        <div class="h-4 w-64 sm:w-80 rounded-md skeleton-shimmer" />
      </div>

      <!-- Composer Card placeholder -->
      <div class="rounded-2xl border border-edge/80 bg-surface-1/90 p-5 space-y-4 shadow-xl">
        <div class="h-24 w-full rounded-xl skeleton-shimmer" />
        <div class="flex items-center justify-between pt-2">
          <div class="flex gap-2.5">
            <div class="h-9 w-28 rounded-lg skeleton-shimmer" />
            <div class="h-9 w-24 rounded-lg skeleton-shimmer" />
          </div>
          <div class="h-10 w-36 rounded-lg skeleton-shimmer-accent" />
        </div>
      </div>

      <!-- Suggestion chips placeholder -->
      <div class="flex flex-wrap gap-2.5 justify-center pt-2">
        <div v-for="i in 4" :key="i" class="h-7 w-36 rounded-full skeleton-shimmer" />
      </div>
    </div>

    <!-- Low Budget (Lite) Pipeline Mode Skeleton -->
    <div v-else class="space-y-6 max-w-3xl mx-auto">
      <!-- Section Header placeholder -->
      <div class="space-y-2">
        <div class="h-7 w-40 rounded-lg skeleton-shimmer" />
        <div class="h-4 w-80 rounded-md skeleton-shimmer" />
      </div>

      <!-- Form Card placeholder -->
      <div class="rounded-2xl border border-edge/80 bg-surface-1/90 p-6 space-y-5 shadow-xl">
        <!-- Source tabs -->
        <div class="inline-flex p-1 bg-surface-2 rounded-lg border border-edge gap-1">
          <div class="h-9 w-32 rounded-md skeleton-shimmer-accent" />
          <div class="h-9 w-32 rounded-md skeleton-shimmer" />
        </div>

        <!-- File input -->
        <div class="space-y-2">
          <div class="h-3.5 w-28 rounded skeleton-shimmer" />
          <div class="flex gap-2">
            <div class="h-11 flex-1 rounded-lg skeleton-shimmer" />
            <div class="h-11 w-20 rounded-lg skeleton-shimmer" />
          </div>
        </div>

        <!-- Prompt input -->
        <div class="space-y-2">
          <div class="h-3.5 w-36 rounded skeleton-shimmer" />
          <div class="h-20 w-full rounded-lg skeleton-shimmer" />
        </div>

        <!-- Sliders -->
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div class="space-y-2">
            <div class="h-3.5 w-24 rounded skeleton-shimmer" />
            <div class="h-10 w-full rounded-lg skeleton-shimmer" />
          </div>
          <div class="space-y-2">
            <div class="h-3.5 w-32 rounded skeleton-shimmer" />
            <div class="h-10 w-full rounded-lg skeleton-shimmer" />
          </div>
        </div>

        <!-- Submit Button -->
        <div class="pt-2">
          <div class="h-11 w-44 rounded-lg skeleton-shimmer-accent" />
        </div>
      </div>
    </div>
  </div>
</template>
