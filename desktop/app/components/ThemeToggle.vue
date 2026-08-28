<script setup lang="ts">
// Compact, accessible icon-only theme control for the header. One button cycles
// system -> light -> dark -> system; the icon reflects the active preference.
const { preference, resolved, cycle } = useTheme()

const label = computed(() => {
  if (preference.value === 'system') {
    return `Sistem (${resolved.value === 'light' ? 'açık' : 'koyu'})`
  }
  return preference.value === 'light' ? 'Açık' : 'Koyu'
})
</script>

<template>
  <button
    type="button"
    class="w-8 h-8 rounded-md border border-edge bg-surface-2 flex items-center justify-center text-slate-300 hover:bg-surface-3 hover:text-slate-100 transition-colors shrink-0"
    :aria-label="`Tema: ${label}. Değiştirmek için tıklayın.`"
    :title="`Tema: ${label}`"
    @click="cycle"
  >
    <svg
      v-if="preference === 'light'"
      class="w-4 h-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
    </svg>
    <svg
      v-else-if="preference === 'dark'"
      class="w-4 h-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
    </svg>
    <svg
      v-else
      class="w-4 h-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M12 3v18a9 9 0 0 0 0-18z" fill="currentColor" />
    </svg>
  </button>
</template>
