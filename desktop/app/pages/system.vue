<script setup lang="ts">
// Moved into the single-scroll hub (pages/index.vue, #sistem section) — see
// pages/new-analysis.vue for why this stub exists. The admin-only gate that
// used to live in middleware/admin-auth.ts now lives inline in
// components/sections/SystemSection.vue (it self-guards as a section).
const router = useRouter()
onMounted(() => router.replace('/#sistem'))
</script>

<template>
  <div />
</template>
