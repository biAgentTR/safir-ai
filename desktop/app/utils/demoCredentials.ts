/**
 * Demo-only admin credentials for the local login gate in front of Sistem
 * Verileri (/system). NOT a real auth backend — there is none. Centralized
 * here (not scattered across components) so swapping this for a real
 * JWT/session call later means replacing verifyDemoCredentials' body only;
 * every call site (stores/auth.ts) already awaits it as if it were async.
 */
const DEMO_USERNAME = 'admin@safir.ai'
const DEMO_PASSWORD = 'safir2026'

export async function verifyDemoCredentials(username: string, password: string): Promise<boolean> {
  return username.trim().toLowerCase() === DEMO_USERNAME && password === DEMO_PASSWORD
}
