<script setup lang="ts">
// This content now lives inline in the single-scroll hub (pages/index.vue,
// #yeni-analiz section) behind the horizontal tab bar. This stub exists only
// so old links/bookmarks to /new-analysis still land in the right place.
const router = useRouter()
onMounted(() => router.replace('/#yeni-analiz'))
</script>

<template>
  <div />
</template>
