/**
 * Arka ucta O AN aktif olan model adlarini saglar (GET /health -> `models`).
 *
 * NEDEN VAR: operator metinleri saglayici adini KODA GOMUYORDU
 * ("Video EVREN'e gonderiliyor…"). Saglayici degistiginde (EVREN -> Gemini)
 * bu cumleler sessizce YANLIS hale geldi ve kullaniciya artik var olmayan bir
 * servisin adi gosterildi. Arayuz artik adi backend'den okur: config neyi
 * aktif ederse ekranda o yazar.
 *
 * Durum `useState` ile PAYLASILIR - birden fazla bilesen bunu cagirsa bile
 * `/health` yalnizca BIR KEZ istenir (`useBackendHealth`in 5 sn'lik yoklamasi
 * bundan bagimsizdir ve tekrarlanmaz).
 */
export interface ActiveModels {
  vlm?: string | null
  vlm_frames?: string | null
  llm?: string | null
  embedding?: string | null
  guard?: string | null
}

/**
 * Model adi bilinmiyorsa kullanilacak NOTR metin.
 *
 * Kullaniciyi BOSLUKLA karsilastirmamak icin gereklidir: backend'e henuz
 * ulasilamadiysa "Video  modeline gonderiliyor…" gibi yarim bir cumle yerine
 * "Video gorsel-dil modeline gonderiliyor…" okunur - dogru, genel ve eksiksiz.
 */
const VLM_FALLBACK = 'görsel-dil modeli'

export function useActiveModels() {
  const models = useState<ActiveModels>('safir-active-models', () => ({}))
  const loaded = useState<boolean>('safir-active-models-loaded', () => false)
  const { health } = useSafirApi()

  async function load(force = false) {
    if (loaded.value && !force) return
    try {
      const res = await health()
      models.value = res.models ?? {}
      loaded.value = true
    } catch {
      // Arka uca ulasilamadi: `models` bos kalir ve tum etiketler notr
      // fallback'e duser. Saglik gostergesi zaten ayri uyari veriyor
      // (`useBackendHealth`), burada ikinci bir hata mesaji URETILMEZ.
    }
  }

  onMounted(() => {
    void load()
  })

  /** Video-dogrudan analizde kullanilan VLM'in gosterim adi. */
  const vlmLabel = computed(() => models.value.vlm?.trim() || VLM_FALLBACK)
  /** Kare-tabanli (Dusuk Butceli) modda kullanilan VLM'in gosterim adi. */
  const framesLabel = computed(() => models.value.vlm_frames?.trim() || VLM_FALLBACK)

  return { models, vlmLabel, framesLabel, load }
}
