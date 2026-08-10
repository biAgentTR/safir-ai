<script setup lang="ts">
// Overview / landing. Entry point into a New Analysis; History/Reports are not
// implemented in this step.
const { state } = useBackendHealth()
</script>

<template>
  <div class="max-w-5xl mx-auto px-6 py-8">
    <section class="card p-6 mb-6">
      <div class="flex items-start justify-between gap-6">
        <div>
          <h2 class="text-xl font-semibold text-slate-100">Saha Analiz Merkezi</h2>
          <p class="mt-2 text-sm text-slate-400 max-w-2xl leading-relaxed">
            Bir video kaynağı veya kamera akışı seçin; SAFİR uyarlanabilir kare
            örnekleme, çok-modlu analiz ve İSG karar zincirini çalıştırıp canlı
            olarak buraya aktarsın. Boru hattının her adımı gerçek zamanlı (SSE)
            izlenebilir.
          </p>
        </div>
        <div class="text-right text-xs text-slate-500 shrink-0">
          <div class="uppercase tracking-wide">Backend</div>
          <div
            class="mt-1 font-medium"
            :class="state === 'online' ? 'text-risk-low' : state === 'offline' ? 'text-risk-crit' : 'text-slate-400'"
          >
            {{ state }}
          </div>
        </div>
      </div>

      <div class="mt-6 flex gap-3">
        <NuxtLink to="/new-analysis" class="btn-primary">
          <span>＋</span> Yeni Analiz Başlat
        </NuxtLink>
        <button class="btn-ghost cursor-not-allowed opacity-60" disabled title="Bu adımda etkin değil">
          Geçmiş (yakında)
        </button>
      </div>
    </section>

    <section class="grid grid-cols-3 gap-4">
      <div class="card p-5">
        <div class="text-xs uppercase tracking-wide text-slate-500">Boru Hattı</div>
        <div class="mt-2 text-2xl font-semibold text-slate-100">7 Aşama</div>
        <p class="mt-1 text-xs text-slate-500">Sampling → VLM → Karar → Rapor</p>
      </div>
      <div class="card p-5">
        <div class="text-xs uppercase tracking-wide text-slate-500">Gözlemlenebilirlik</div>
        <div class="mt-2 text-2xl font-semibold text-slate-100">Canlı SSE</div>
        <p class="mt-1 text-xs text-slate-500">Gerçek trace olayları</p>
      </div>
      <div class="card p-5">
        <div class="text-xs uppercase tracking-wide text-slate-500">Arayüz</div>
        <div class="mt-2 text-2xl font-semibold text-slate-100">Tauri 2</div>
        <p class="mt-1 text-xs text-slate-500">Streamlit ile paralel</p>
      </div>
    </section>
  </div>
</template>
