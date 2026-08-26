<script setup lang="ts">
// First paint's theme is set synchronously by the blocking inline script in
// nuxt.config.ts (app.head.script) — this just brings useTheme's reactive
// state in sync and starts listening for OS theme changes.
const { init } = useTheme()
onMounted(init)

// One-time analysis mode picker (ModeSelectGate) replaces the whole app
// shell until the operator has chosen a mode; after that it never shows
// again (see useAnalysisMode.ts / hasChosen, persisted in localStorage).
const { hasChosen } = useAnalysisMode()
</script>

<template>
  <ModeSelectGate v-if="!hasChosen" />
  <NuxtLayout v-else>
    <NuxtPage />
  </NuxtLayout>
</template>
