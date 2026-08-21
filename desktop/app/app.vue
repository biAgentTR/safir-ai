<script setup lang="ts">
const { initTheme } = useTheme()

onMounted(() => {
  initTheme()
})
</script>

<template>
  <NuxtLayout>
    <NuxtPage />
  </NuxtLayout>
</template>
