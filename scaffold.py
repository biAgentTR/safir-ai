import os

app_dir = r'c:\ins-son\desktop\app'
pages_dir = os.path.join(app_dir, 'pages')
layouts_dir = os.path.join(app_dir, 'layouts')

# Layouts
operator_layout = '''<template>
  <div class="h-screen w-screen flex flex-col bg-[#0b0e14] text-slate-200 overflow-hidden font-sans">
    <header class="h-16 shrink-0 flex items-center justify-between px-6 border-b border-white/5 bg-[#0b0e14]/80 backdrop-blur-md">
      <div class="flex items-center gap-8">
        <NuxtLink to="/" class="flex items-center gap-3">
          <div class="w-8 h-8 rounded bg-cyan-500/10 flex items-center justify-center border border-cyan-500/20 text-cyan-400">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/></svg>
          </div>
          <div>
            <div class="font-bold text-lg tracking-wider text-white">SAFİR</div>
            <div class="text-[10px] text-cyan-400/80 tracking-widest uppercase">Vision Intelligence</div>
          </div>
        </NuxtLink>
        
        <nav class="hidden md:flex items-center gap-1">
          <NuxtLink to="/" class="px-4 py-2 text-sm font-medium rounded-md transition-colors hover:text-white hover:bg-white/5" active-class="text-cyan-400 bg-cyan-400/10">Operasyon</NuxtLink>
          <NuxtLink to="/analizler" class="px-4 py-2 text-sm font-medium rounded-md transition-colors hover:text-white hover:bg-white/5" active-class="text-cyan-400 bg-cyan-400/10">Analizler</NuxtLink>
          <NuxtLink to="/raporlar" class="px-4 py-2 text-sm font-medium rounded-md transition-colors hover:text-white hover:bg-white/5" active-class="text-cyan-400 bg-cyan-400/10">Raporlar</NuxtLink>
        </nav>
      </div>
      
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
          <div class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
          Sistem Aktif
        </div>
        <NuxtLink to="/admin/login" class="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md border border-white/10 hover:border-white/20 transition-colors">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
          Yönetici Girişi
        </NuxtLink>
        <div class="w-8 h-8 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center text-sm font-bold text-white">OP</div>
      </div>
    </header>
    
    <main class="flex-1 overflow-auto relative">
      <!-- Grid Background -->
      <div class="absolute inset-0 bg-[url('~/assets/images/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))] opacity-5 pointer-events-none"></div>
      
      <!-- Ambient light effect -->
      <div class="absolute top-0 left-1/4 w-1/2 h-[500px] bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none"></div>
      
      <slot />
    </main>
  </div>
</template>
<script setup>
</script>
<style>
body { background-color: #0b0e14; color: #e2e8f0; }
</style>
'''

admin_layout = '''<template>
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
'''

with open(os.path.join(layouts_dir, 'operator.vue'), 'w', encoding='utf-8') as f:
    f.write(operator_layout)
    
with open(os.path.join(layouts_dir, 'admin.vue'), 'w', encoding='utf-8') as f:
    f.write(admin_layout)

os.makedirs(os.path.join(pages_dir, 'analizler'), exist_ok=True)
os.makedirs(os.path.join(pages_dir, 'raporlar'), exist_ok=True)
os.makedirs(os.path.join(pages_dir, 'admin'), exist_ok=True)

dummy_pages = {
    'index.vue': '<template><div class="max-w-6xl mx-auto p-8 relative z-10"><h1>Operatör Analiz Ekranı</h1><p class="text-slate-400 mt-2">Video yükleme paneli buraya gelecek.</p></div></template><script setup>definePageMeta({ layout: "operator" })</script>',
    'analizler/index.vue': '<template><div class="max-w-6xl mx-auto p-8 relative z-10"><h1>Önceki Analizler</h1></div></template><script setup>definePageMeta({ layout: "operator" })</script>',
    'analizler/[id].vue': '<template><div class="max-w-6xl mx-auto p-8 relative z-10"><h1>Analiz Sonucu: {{ $route.params.id }}</h1></div></template><script setup>definePageMeta({ layout: "operator" })</script>',
    'raporlar/index.vue': '<template><div class="max-w-6xl mx-auto p-8 relative z-10"><h1>Oluşturulan Raporlar</h1></div></template><script setup>definePageMeta({ layout: "operator" })</script>',
    'admin/index.vue': '<template><div class="p-8 relative z-10"><h1>Yönetici Genel Bakış</h1></div></template><script setup>definePageMeta({ layout: "admin" })</script>',
    'admin/islemler.vue': '<template><div class="p-8 relative z-10"><h1>Video İşleme Geçmişi</h1></div></template><script setup>definePageMeta({ layout: "admin" })</script>',
    'admin/modeller.vue': '<template><div class="p-8 relative z-10"><h1>Model ve Ajan Ayarları</h1></div></template><script setup>definePageMeta({ layout: "admin" })</script>',
    'admin/servisler.vue': '<template><div class="p-8 relative z-10"><h1>Servis Sağlığı ve Sistem Durumu</h1></div></template><script setup>definePageMeta({ layout: "admin" })</script>',
    'admin/loglar.vue': '<template><div class="p-8 relative z-10"><h1>Sistem Kayıtları</h1></div></template><script setup>definePageMeta({ layout: "admin" })</script>'
}

for path, content in dummy_pages.items():
    file_path = os.path.join(pages_dir, path)
    if os.path.exists(file_path) and 'login.vue' in path:
        continue # Don't overwrite login if not needed, wait actually I didn't include login in dummy pages.
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Success")
