<script setup lang="ts">
// Visual skeleton only — there is no admin auth backend yet anywhere in
// SAFİR (no endpoint, no token, no session). This page must NOT simulate a
// working login: no fake token, no "success" state, no credential storage.
const username = ref('')
const password = ref('')
</script>

<template>
  <div class="max-w-sm mx-auto px-6 py-16">
    <div class="card p-6">
      <h2 class="text-lg font-bold tracking-tight text-slate-100">Yönetici Girişi</h2>
      <p class="mt-1 text-sm text-slate-500">Saha operasyon ekibi dışındaki yönetici işlemleri için.</p>

      <form class="mt-5 space-y-4" @submit.prevent>
        <div>
          <label class="field-label" for="admin-username">Kullanıcı Adı</label>
          <input id="admin-username" v-model="username" class="field-input" autocomplete="username" disabled />
        </div>
        <div>
          <label class="field-label" for="admin-password">Parola</label>
          <input id="admin-password" v-model="password" type="password" class="field-input" autocomplete="current-password" disabled />
        </div>
        <button type="submit" class="btn-primary w-full" disabled>Giriş Yap</button>
      </form>

      <div class="mt-5 rounded-md border border-info/30 bg-info/10 px-3 py-2.5 text-xs text-slate-300">
        Kimlik doğrulama sonraki aşamada backend servisine bağlanacaktır.
      </div>
    </div>
  </div>
</template>
