<script setup lang="ts">
// Full-bleed layout with no sidebar/topbar — used for the admin login gate,
// which should read as a standalone screen (matches the reference design),
// not a page nested inside the operator navigation shell.
</script>

<template>
  <div class="h-full overflow-y-auto bg-surface-0">
    <slot />
  </div>
</template>
