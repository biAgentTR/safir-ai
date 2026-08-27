// Guards Sistem Verileri (/system) behind the demo admin login. Reads
// sessionStorage/localStorage directly (not just Pinia state) because a hard
// page refresh on /system, before the auth store has rehydrated, would
// otherwise briefly think the operator is logged out.
export default defineNuxtRouteMiddleware((to) => {
  if (import.meta.server) return
  const auth = useAuthStore()
  if (!auth.isAuthenticated) auth.init()
  if (!auth.isAuthenticated) {
    return navigateTo({ path: '/admin/login', query: { next: to.fullPath } })
  }
})
