/**
 * Demo admin session store. Gates the Sistem Verileri (/system) route behind
 * a local, fake login (see utils/demoCredentials.ts) — there is no real
 * backend auth endpoint to call yet.
 *
 * Persistence: sessionStorage by default (cleared when the window closes),
 * or localStorage when the operator checks "Beni hatırla" — still not a
 * secure token, just a plain flag readable by any script in this origin.
 * Fine for a desktop demo shell; NOT how a real deployment should persist a
 * session. Swapping to real auth later means: verifyDemoCredentials ->
 * a real login API call, and this plain flag -> a signed/expiring token.
 */
import { defineStore } from 'pinia'
import { verifyDemoCredentials } from '~/utils/demoCredentials'

const STORAGE_KEY = 'safir-admin-session'

interface AuthState {
  isAuthenticated: boolean
  username: string | null
  error: string | null
  pending: boolean
}

function readPersisted(): { username: string } | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY) ?? localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as { username: string }) : null
  } catch {
    return null
  }
}

export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({
    isAuthenticated: false,
    username: null,
    error: null,
    pending: false,
  }),
  actions: {
    /** Rehydrate from storage on app start (called once from a plugin/layout). */
    init() {
      const persisted = readPersisted()
      if (persisted) {
        this.isAuthenticated = true
        this.username = persisted.username
      }
    },
    async login(username: string, password: string, rememberMe: boolean): Promise<boolean> {
      this.pending = true
      this.error = null
      try {
        const ok = await verifyDemoCredentials(username, password)
        if (!ok) {
          this.error = 'Kullanıcı adı veya parola hatalı.'
          return false
        }
        this.isAuthenticated = true
        this.username = username.trim()
        if (typeof window !== 'undefined') {
          const payload = JSON.stringify({ username: this.username })
          // demo-only: plain, unsigned flag — see file header
          ;(rememberMe ? localStorage : sessionStorage).setItem(STORAGE_KEY, payload)
        }
        return true
      } finally {
        this.pending = false
      }
    },
    logout() {
      this.isAuthenticated = false
      this.username = null
      if (typeof window !== 'undefined') {
        sessionStorage.removeItem(STORAGE_KEY)
        localStorage.removeItem(STORAGE_KEY)
      }
    },
  },
})
