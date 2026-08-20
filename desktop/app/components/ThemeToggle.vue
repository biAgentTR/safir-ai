<script setup lang="ts">
// Compact, accessible theme control for the header. One button cycles
// system -> light -> dark -> system; the icon + label always state the
// CURRENT theme (not the action), so it reads correctly for screen readers
// and at a glance on a control-room monitor.
const { preference, resolved, cycle } = useTheme()

const meta = computed(() => {
  if (preference.value === 'system') {
    return { icon: '◐', label: `Sistem (${resolved.value === 'light' ? 'açık' : 'koyu'})` }
  }
  return preference.value === 'light' ? { icon: '☀', label: 'Açık' } : { icon: '☾', label: 'Koyu' }
})
</script>

<template>
  <button
    type="button"
    class="inline-flex items-center gap-1.5 rounded-md border border-edge bg-surface-2 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-surface-3 transition-colors"
    :aria-label="`Tema: ${meta.label}. Değiştirmek için tıklayın.`"
    :title="`Tema: ${meta.label}`"
    @click="cycle"
  >
    <span aria-hidden="true">{{ meta.icon }}</span>
    <span class="hidden sm:inline">{{ meta.label }}</span>
  </button>
</template>
