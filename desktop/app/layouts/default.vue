<script setup lang="ts">
// Desktop application shell: fixed navigation rail + top bar + scrollable main.
</script>

<template>
  <div class="h-full flex bg-surface-0">
    <AppSidebar />
    <div class="flex-1 min-w-0 flex flex-col">
      <AppTopbar />
      <main class="flex-1 min-h-0 overflow-y-auto">
        <slot />
      </main>
    </div>
  </div>
</template>
