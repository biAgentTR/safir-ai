import sys
content = '''<template>
  <div class=\"min-h-screen relative overflow-x-hidden pb-12\">
    <div class=\"fixed inset-0 z-0 pointer-events-none opacity-40\">
      <BackgroundScene />
    </div>
    
    <div class=\"relative z-10 w-full max-w-[1320px] mx-auto px-6 lg:px-10 pt-8\">
      
      <!-- Loading State -->
      <div v-if=\"state === 'loading'\" class=\"grid grid-cols-1 lg:grid-cols-3 gap-6\" aria-busy=\"true\">
        <div class=\"lg:col-span-2 space-y-6\">
          <div class=\"aspect-video bg-[var(--color-surface)] rounded-2xl border border-[var(--color-border)] animate-pulse\"></div>
          <div class=\"h-32 bg-[var(--color-surface)] rounded-[18px] border border-[var(--color-border)] animate-pulse\"></div>
        </div>
        <div class=\"h-full min-h-[400px] bg-[var(--color-surface)] rounded-[18px] border border-[var(--color-border)] animate-pulse\"></div>
      </div>

      <!-- Processing State -->
      <div v-else-if=\"state === 'processing'\" class=\"flex flex-col items-center justify-center min-h-[500px] bg-[var(--color-surface)] border border-[var(--color-border)] rounded-[18px] p-8 text-center\">
        <div class=\"w-16 h-16 rounded-2xl bg-[var(--color-primary)]/10 flex items-center justify-center mb-6\">
          <svg class=\"animate-spin text-[var(--color-primary)] w-8 h-8\" xmlns=\"http://www.w3.org/2000/svg\" fill=\"none\" viewBox=\"0 0 24 24\"><circle class=\"opacity-25\" cx=\"12\" cy=\"12\" r=\"10\" stroke=\"currentColor\" stroke-width=\"4\"></circle><path class=\"opacity-75\" fill=\"currentColor\" d=\"M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z\"></path></svg>
        </div>
        <h2 class=\"text-xl font-bold text-white mb-2\">
          {{ jobStatus === "queued" ? "Analiz Sýrada" : "Analiz Devam Ediyor" }}
        </h2>
        <p class=\"text-[var(--color-text-secondary)] mb-8 max-w-md\">
          {{ jobStatus === "queued" ? "Videonuz analiz edilmek üzere kuyrukta bekliyor." : totalSteps ? "Analiz adýmý " + step + " / " + totalSteps : "Videonuz þu anda iþleniyor. Lütfen bekleyin." }}
        </p>
        <NuxtLink to=\"/\" class=\"px-6 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-colors font-medium\">
          Ana Ekrana Dön
        </NuxtLink>
      </div>

      <!-- Not Found -->
      <div v-else-if=\"state === 'not-found'\" class=\"flex flex-col items-center justify-center min-h-[500px] bg-[var(--color-surface)] border border-[var(--color-border)] rounded-[18px] p-8 text-center\">
        <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"48\" height=\"48\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" class=\"mb-4 text-slate-500\"><circle cx=\"12\" cy=\"12\" r=\"10\"/><line x1=\"12\" y1=\"8\" x2=\"12\" y2=\"12\"/><line x1=\"12\" y1=\"16\" x2=\"12.01\" y2=\"16\"/></svg>
        <h2 class=\"text-xl font-bold text-white mb-2\">Analiz Bulunamadý</h2>
        <p class=\"text-[var(--color-text-secondary)] mb-8 max-w-md\">Bu analiz kaldýrýlmýþ, süresi dolmuþ veya geçersiz bir baðlantý kullanýlmýþ olabilir.</p>
        <NuxtLink to=\"/\" class=\"px-6 py-2.5 bg-[var(--color-primary)] hover:bg-cyan-400 text-[#05090c] rounded-xl transition-colors font-medium\">
          Ana Ekrana Dön
        </NuxtLink>
      </div>

      <!-- Service Unavailable -->
      <div v-else-if=\"state === 'service-unavailable'\" class=\"flex flex-col items-center justify-center min-h-[500px] bg-[var(--color-surface)] border border-[var(--color-border)] rounded-[18px] p-8 text-center\">
        <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"48\" height=\"48\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" class=\"mb-4 text-rose-500 opacity-80\"><path d=\"M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z\"/><line x1=\"12\" y1=\"9\" x2=\"12\" y2=\"13\"/><line x1=\"12\" y1=\"17\" x2=\"12.01\" y2=\"17\"/></svg>
        <h2 class=\"text-xl font-bold text-white mb-2\">Analiz Servisine Ulaþýlamýyor</h2>
        <p class=\"text-[var(--color-text-secondary)] mb-8 max-w-md\">Sonuçlar þu anda yüklenemiyor. Backend servisini kontrol edip tekrar deneyin.</p>
        <div class=\"flex gap-4\">
          <button @click=\"reload()\" class=\"px-6 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-colors font-medium\">
            Tekrar Dene
          </button>
          <NuxtLink to=\"/\" class=\"px-6 py-2.5 bg-[var(--color-primary)] hover:bg-cyan-400 text-[#05090c] rounded-xl transition-colors font-medium\">
            Ana Ekrana Dön
          </NuxtLink>
        </div>
      </div>

      <!-- Failed -->
      <div v-else-if=\"state === 'failed'\" class=\"flex flex-col items-center justify-center min-h-[500px] bg-[#0f0709] border border-[#ff7f91]/30 rounded-[18px] p-8 text-center\">
        <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"48\" height=\"48\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" class=\"mb-4 text-[#ff7f91]\"><circle cx=\"12\" cy=\"12\" r=\"10\"/><line x1=\"15\" y1=\"9\" x2=\"9\" y2=\"15\"/><line x1=\"9\" y1=\"9\" x2=\"15\" y2=\"15\"/></svg>
        <h2 class=\"text-xl font-bold text-white mb-2\">Analiz Tamamlanamadý</h2>
        <p class=\"text-[#ff7f91]/80 mb-8 max-w-md\">{{ errorMsg || "Ýþlem sýrasýnda beklenmeyen bir hata oluþtu.\" }}</p>
        <NuxtLink to=\"/\" class=\"px-6 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-colors font-medium\">
          Ana Ekrana Dön
        </NuxtLink>
      </div>

      <!-- Completed / Empty -->
      <div v-else-if=\"(state === 'completed' || state === 'empty') && report\" class=\"flex flex-col\">
        
        <!-- TOP HEADER -->
        <div class=\"flex flex-col md:flex-row md:items-start justify-between mb-8 gap-4\">
          <div>
            <div class=\"flex items-center gap-1.5 text-[10px] font-bold tracking-widest uppercase mb-3\" :class=\"state === 'completed' ? 'text-[var(--color-success)]' : 'text-slate-400'\">
              <svg v-if=\"state === 'completed'\" xmlns=\"http://www.w3.org/2000/svg\" width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"3\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M20 6 9 17l-5-5\"/></svg>
              <span v-if=\"state === 'completed'\">ANALÝZ TAMAMLANDI</span>
              <span v-else>ANALÝZ BOÞ</span>
              <span v-if=\"report.generated_at\" class=\"text-[var(--color-text-muted)]\"> · {{ formatDateUpper(report.generated_at) }}</span>
            </div>
            <h1 class=\"text-4xl font-medium text-[var(--color-text)] tracking-tight\">
              Olay deðerlendirmesi
            </h1>
          </div>
        </div>

        <!-- MAIN RESULT GRID -->
        <div class=\"grid grid-cols-1 lg:grid-cols-[minmax(0,2fr)_minmax(320px,0.95fr)] gap-4 mb-4\">
          <!-- LEFT: VIDEO -->
          <div class=\"flex flex-col min-w-0\">
            <div class=\"w-full bg-[#05090c] rounded-[16px] border border-[var(--color-border)] overflow-hidden\" style=\"height: clamp(300px, 40vh, 380px);\">
              <ReportVideoViewer :videoPath=\"report.video_source\" />
            </div>
            <!-- BOTTOM METRICS -->
            <div class=\"grid grid-cols-3 gap-px bg-[var(--color-border)] border border-[var(--color-border)] mt-4 rounded-[12px] overflow-hidden shrink-0\">
              <div class=\"bg-[var(--color-surface)] p-4 flex flex-col justify-center\">
                <div class=\"text-[9px] font-bold text-[var(--color-text-muted)] uppercase tracking-wider mb-1 flex items-center gap-1.5\">
                  <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M12 2v20\"/><path d=\"M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6\"/></svg>
                  Risk skoru
                </div>
                <div class=\"text-lg font-bold text-[#ff7f91]\">
                  <span v-if=\"report.risk_score != null\">{{ report.risk_score }} <span class=\"text-sm font-medium text-[var(--color-text-secondary)]\">/ 100</span></span>
                  <span v-else class=\"text-sm text-[var(--color-text-muted)]\">Deðerlendirilmedi</span>
                </div>
              </div>
              <div class=\"bg-[var(--color-surface)] p-4 flex flex-col justify-center\">
                <div class=\"text-[9px] font-bold text-[var(--color-text-muted)] uppercase tracking-wider mb-1 flex items-center gap-1.5\">
                  <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M22 12h-4l-3 9L9 3l-3 9H2\"/></svg>
                  Risk Seviyesi
                </div>
                <div class=\"text-lg font-bold text-[var(--color-text)] truncate\">
                  {{ formatRiskLevel(report.risk_level) }}
                </div>
              </div>
              <div class=\"bg-[var(--color-surface)] p-4 flex flex-col justify-center\">
                <div class=\"text-[9px] font-bold text-[var(--color-text-muted)] uppercase tracking-wider mb-1 flex items-center gap-1.5\">
                  <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><polygon points=\"5 3 19 12 5 21 5 3\"/></svg>
                  Dosya Adý
                </div>
                <div class=\"text-[13px] font-medium text-[var(--color-text)] truncate\" :title=\"getBasename(report.video_source)\">
                  {{ getBasename(report.video_source) }}
                </div>
              </div>
            </div>
          </div>

          <!-- RIGHT: RISK CARD -->
          <div class=\"h-full min-w-0\">
            <ReportRiskCard
              :score=\"report.risk_score\"
              :level=\"report.risk_level\"
              :summary=\"report.summary\"
              :recommendedAction=\"report.recommended_action\"
              :actions=\"report.actions\"
              @ask-safir=\"openAskSafir\"
            />
          </div>
        </div>

        <!-- BOTTOM STRIP -->
        <div v-if=\"state === 'completed'\" class=\"flex flex-col sm:flex-row sm:items-center justify-between bg-[var(--color-surface)] border border-[var(--color-border)] rounded-[16px] p-5 gap-4\">
          <div class=\"flex items-center gap-4\">
            <div class=\"w-10 h-10 rounded-lg bg-[var(--color-primary)]/10 text-[var(--color-primary)] flex items-center justify-center shrink-0 border border-[var(--color-primary)]/20\">
              <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z\"/><polyline points=\"14 2 14 8 20 8\"/><line x1=\"16\" y1=\"13\" x2=\"8\" y2=\"13\"/><line x1=\"16\" y1=\"17\" x2=\"8\" y2=\"17\"/><polyline points=\"10 9 9 9 8 9\"/></svg>
            </div>
            <div>
              <div class=\"text-[13px] font-bold text-[var(--color-text)]\">Analiz raporu hazýr</div>
              <div class=\"text-[11px] text-[var(--color-text-secondary)]\">Detaylý analiz ve operatör deðerlendirmelerini PDF raporunda inceleyin.</div>
            </div>
          </div>
          <div class=\"flex items-center shrink-0\">
            <ReportDownloadButton :job-id=\"jobId\" :video-name=\"report ? getBasename(report.video_source) : ''\" />
          </div>
        </div>

      </div>
    </div>
    
    <!-- Hidden Ask Safir instance -->
    <div class=\"hidden\">
      <AskSafirDrawer ref=\"askSafirRef\" :job-id=\"jobId\" />
    </div>
  </div>
</template>
'''
with open('new_analizler.vue', 'a', encoding='utf-8') as f:
  f.write(content)
