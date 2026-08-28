<script setup lang="ts">
// Demo yönetici girişi — bkz. stores/auth.ts ve utils/demoCredentials.ts.
// Gerçek bir kimlik doğrulama servisi yoktur; bu ekran ileride JWT/oturum
// tabanlı bir sisteme kolayca bağlanabilecek şekilde ayrı bir katmanda tutulur.
definePageMeta({ layout: 'blank' })

const auth = useAuthStore()
const router = useRouter()
const route = useRoute()

const username = ref('')
const password = ref('')
const rememberMe = ref(false)
const submitted = ref(false)

const usernameInput = ref<HTMLInputElement | null>(null)
onMounted(() => usernameInput.value?.focus())

async function onSubmit() {
  submitted.value = true
  const ok = await auth.login(username.value, password.value, rememberMe.value)
  if (ok) {
    const next = typeof route.query.next === 'string' ? route.query.next : '/#sistem'
    router.push(next)
  }
}
</script>

<template>
  <div class="min-h-full flex items-center justify-center px-4 py-10 relative overflow-hidden bg-surface-0">
    <div class="ambient-rings" aria-hidden="true">
      <span class="ambient-ring ambient-ring-a" />
      <span class="ambient-ring ambient-ring-b" />
    </div>
    <div class="grid-texture" aria-hidden="true" />

    <div class="relative w-full max-w-sm">
      <NuxtLink to="/" class="mb-6 flex items-center gap-2.5 justify-center text-slate-300 hover:text-slate-100 transition-colors">
        <img src="~/assets/images/logo.png" alt="" class="w-6 h-6 object-contain" />
        <span class="text-sm font-bold tracking-[0.24em]">SAFİR</span>
      </NuxtLink>

      <div class="glass-panel rounded-lg p-7">
        <div class="flex items-center gap-2 text-accent mb-1">
          <span aria-hidden="true">🔒</span>
          <span class="eyebrow !text-accent">Yönetim Konsolu</span>
        </div>
        <h1 class="text-xl font-bold text-slate-100 mb-1">Yetkili erişimi</h1>
        <p class="text-sm text-slate-500 mb-6">İşleme hattı, modeller ve sistem kayıtlarını yönetmek için oturum açın.</p>

        <form class="space-y-4" novalidate @submit.prevent="onSubmit">
          <div>
            <label for="admin-username" class="field-label">Kullanıcı adı</label>
            <input
              id="admin-username"
              ref="usernameInput"
              v-model="username"
              type="text"
              autocomplete="username"
              class="field-input"
              :aria-invalid="submitted && !!auth.error"
              required
            />
          </div>
          <div>
            <label for="admin-password" class="field-label">Parola</label>
            <input
              id="admin-password"
              v-model="password"
              type="password"
              autocomplete="current-password"
              class="field-input"
              :aria-invalid="submitted && !!auth.error"
              required
            />
          </div>

          <div class="flex items-center justify-between">
            <label class="flex items-center gap-2 text-sm text-slate-400 select-none">
              <input v-model="rememberMe" type="checkbox" class="rounded border-edge bg-surface-2 accent-accent" />
              Beni hatırla
            </label>
          </div>

          <p v-if="submitted && auth.error" role="alert" class="text-sm text-risk-crit flex items-center gap-1.5">
            <span aria-hidden="true">⚠</span>{{ auth.error }}
          </p>

          <button type="submit" class="btn-primary w-full" :disabled="auth.pending">
            {{ auth.pending ? 'Kontrol ediliyor…' : 'Konsola giriş yap' }}
            <span aria-hidden="true">→</span>
          </button>
        </form>

        <p class="mt-5 text-[11px] text-slate-600 flex items-center gap-1.5">
          <span aria-hidden="true">🛡</span>
          Bu demo oturumu yalnızca bu cihazda tutulur; gerçek bir kimlik doğrulama sunucusuna bağlı değildir.
        </p>
      </div>
    </div>
  </div>
</template>
