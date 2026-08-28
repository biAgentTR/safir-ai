<template>
  <div class="h-screen w-screen flex bg-[#0b0e14] text-slate-200 overflow-hidden font-sans">
    <aside class="w-64 shrink-0 border-r border-white/5 bg-[#0b0e14]/90 flex flex-col z-10 relative">
      <div class="h-16 flex items-center px-6 border-b border-white/5">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded bg-cyan-500/10 flex items-center justify-center border border-cyan-500/20 text-cyan-400">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/></svg>
          </div>
          <div>
            <div class="font-bold text-lg tracking-wider text-white">SAFİR</div>
            <div class="text-[9px] text-slate-500 tracking-widest uppercase">Yönetim Konsolu</div>
          </div>
        </div>
      </div>
      
      <div class="p-4 flex-1 overflow-y-auto space-y-6">
        <div>
          <div class="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2 px-2">Genel</div>
          <nav class="space-y-1">
            <NuxtLink to="/admin" class="flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors hover:bg-white/5" active-class="bg-cyan-500/10 text-cyan-400 font-medium" exact><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/></svg>Genel Bakış</NuxtLink>
            <NuxtLink to="/admin/islemler" class="flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors hover:bg-white/5" active-class="bg-cyan-500/10 text-cyan-400 font-medium"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>İşlem Geçmişi</NuxtLink>
          </nav>
        </div>
        
        <div>
          <div class="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2 px-2">Sistem</div>
          <nav class="space-y-1">
            <NuxtLink to="/admin/modeller" class="flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors hover:bg-white/5" active-class="bg-cyan-500/10 text-cyan-400 font-medium"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/></svg>Model Ayarları</NuxtLink>
            <NuxtLink to="/admin/servisler" class="flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors hover:bg-white/5" active-class="bg-cyan-500/10 text-cyan-400 font-medium"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>Servis Sağlığı</NuxtLink>
            <NuxtLink to="/admin/loglar" class="flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors hover:bg-white/5" active-class="bg-cyan-500/10 text-cyan-400 font-medium"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>Sistem Kayıtları</NuxtLink>
          </nav>
        </div>
      </div>
      
      <div class="p-4 border-t border-white/5 mt-auto">
        <NuxtLink to="/" class="flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors hover:bg-white/5 text-slate-400"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>Operatör Ekranına Dön</NuxtLink>
      </div>
    </aside>
    
    <main class="flex-1 overflow-auto relative">
      <div class="absolute inset-0 bg-[url('~/assets/images/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))] opacity-5 pointer-events-none"></div>
      <slot />
    </main>
  </div>
</template>
