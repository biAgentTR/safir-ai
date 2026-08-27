<script setup lang="ts">
// Moved into the single-scroll hub (pages/index.vue, #raporlar section) — see
// pages/new-analysis.vue for why this stub exists. /reports/[jobId] (the
// report detail drill-down) is untouched and still its own route.
const router = useRouter()
onMounted(() => router.replace('/#raporlar'))
</script>

<template>
  <div />
</template>
