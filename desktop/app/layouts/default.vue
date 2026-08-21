<script setup lang="ts">
// Desktop application shell: fixed navigation rail + top bar + scrollable main.
// PanelTransition overlays just the main area (sidebar/topbar stay visible)
// briefly when the operator switches sidebar panels — see AppSidebar.vue.
</script>

<template>
  <div class="h-full flex bg-surface-0">
    <AppSidebar />
    <div class="flex-1 min-w-0 flex flex-col">
      <AppTopbar />
      <main class="relative flex-1 min-h-0 overflow-y-auto">
        <slot />
        <PanelTransition />
      </main>
    </div>
  </div>
</template>
