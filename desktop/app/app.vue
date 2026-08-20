<script setup lang="ts">
// First paint's theme is set synchronously by the blocking inline script in
// nuxt.config.ts (app.head.script) — this just brings useTheme's reactive
// state in sync and starts listening for OS theme changes.
const { init } = useTheme()
onMounted(init)
</script>

<template>
  <NuxtLayout>
    <NuxtPage />
  </NuxtLayout>
</template>
