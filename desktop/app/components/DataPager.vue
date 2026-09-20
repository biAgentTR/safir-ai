<script setup lang="ts">
/**
 * Yeniden kullanilabilir sayfalama denetimi (alt bilgi seridi).
 *
 * Gorsel dil, `sections/HistorySection.vue` icindeki mevcut sayfalama
 * serisinden BIREBIR devralinmistir (ayni buton/aralik/renk siniflari) -
 * boylece Sistem Verileri'ne eklenen sayfalama, uygulamaya sonradan
 * yamanmis gibi DEGIL, yerlesik bir parcasi gibi gorunur. Tek fark, bu
 * bilesenin duruma sahip OLMAMASI: sayfa numarasini `v-model:page` ile
 * cagirandan alir, boylece ayni bilesen bir ekranda birden fazla liste
 * icin (analizler / konusmalar / iz kaydi / kanitlar) kullanilabilir.
 *
 * Toplam kayit `pageSize`i asmiyorsa bilesen HICBIR SEY cizmez - kisa
 * listelerde gereksiz bir arayuz gurultusu olusmaz.
 */
const props = withDefaults(
  defineProps<{
    /** 1 tabanli gecerli sayfa. */
    page: number
    /** Sayfalanan listedeki TOPLAM kayit sayisi. */
    total: number
    /** Sayfa basina kayit. */
    pageSize?: number
    /** Alt bilgide gosterilen kayit adi (ör. "kayıt", "olay"). */
    noun?: string
  }>(),
  { pageSize: 25, noun: 'kayıt' },
)

const emit = defineEmits<{ 'update:page': [value: number] }>()

const totalPages = computed(() => Math.max(1, Math.ceil(props.total / props.pageSize)))

/** Bu sayfada gosterilen araligin ilk sirasi (1 tabanli, kullaniciya donuk). */
const rangeStart = computed(() => (props.total === 0 ? 0 : (props.page - 1) * props.pageSize + 1))
/** Bu sayfada gosterilen araligin son sirasi (son sayfada toplamla sinirlanir). */
const rangeEnd = computed(() => Math.min(props.page * props.pageSize, props.total))

function go(next: number) {
  // Sinirlarin disina cikilmasi ENGELLENIR: cagiran taraf bos bir dilim
  // (slice) alip "veri yok" saniyor gibi yanlis bir duruma DUSMEZ.
  const clamped = Math.min(Math.max(1, next), totalPages.value)
  if (clamped !== props.page) emit('update:page', clamped)
}
</script>

<template>
  <div
    v-if="totalPages > 1"
    class="pt-4 mt-4 flex items-center justify-between border-t border-edge text-xs text-slate-500"
  >
    <span>
      Toplam {{ total }} {{ noun }} ·
      <span class="font-mono text-slate-400">{{ rangeStart }}–{{ rangeEnd }}</span> arası gösteriliyor
    </span>
    <div class="flex items-center gap-2">
      <button
        type="button"
        class="px-2.5 py-1 rounded bg-surface-2 hover:bg-surface-3 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        :disabled="page <= 1"
        aria-label="Önceki sayfa"
        @click.stop="go(page - 1)"
      >
        ❮ Önceki
      </button>
      <span class="font-mono text-slate-400" aria-live="polite">{{ page }} / {{ totalPages }}</span>
      <button
        type="button"
        class="px-2.5 py-1 rounded bg-surface-2 hover:bg-surface-3 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        :disabled="page >= totalPages"
        aria-label="Sonraki sayfa"
        @click.stop="go(page + 1)"
      >
        Sonraki ❯
      </button>
    </div>
  </div>
</template>
