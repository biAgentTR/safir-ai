# SAFİR — Kurulum ve Çalıştırma Rehberi (Google Gemini)

> **Bu belge 2026-09-18'de gerçek, aktif mimariye göre yeniden yazılmıştır.**
> Önceki sürüm yerel vLLM + RTX 5090 + Qwen kurulumunu, ondan önceki sürüm ise
> TEKNOFEST EVREN servisini anlatıyordu. **İkisi de artık geçerli değildir:**
> EVREN takıma kapatıldı, yerel GPU servislemesi ise bu projede kullanılmıyor.
> Şu an VLM, LLM/ajan, embedding ve güvenlik katmanlarının **tamamı Google
> Gemini** üzerinden çalışır. Vektör veritabanı (Qdrant) ise **yerel/gömülü**
> çalışır — ayrı bir sunucu kurmanız gerekmez.

---

## 0. Ön koşullar

| Gereksinim | Sürüm / Not |
|---|---|
| Python | 3.10 – 3.12 (geliştirme 3.12.10 ile yapıldı) |
| GPU | **Gerekmez.** Model servislemesi Gemini'de; yerel katman (frame sampler) CPU-only çalışır. |
| Gemini API anahtarı | https://aistudio.google.com/apikey — VLM, LLM/ajan ve embedding için |
| Groq API anahtarı | https://console.groq.com/keys — prompt-injection guard için |
| Node.js | Yalnızca masaüstü arayüzü (Nuxt) için — 18+ |
| ffmpeg | **Opsiyonel.** Yoksa video parçalama OpenCV ile (daha yavaş) yapılır; sistem yine çalışır. |

---

## 1. Depoyu hazırla

```bash
git clone <repo-url> safir-ai
cd safir-ai/safir-ai
```

## 2. Sanal ortam + bağımlılıklar

Gemini profili `requirements-gemini.txt` dosyasındadır. Bu dosya
`requirements.txt` ile aynı çekirdeği içerir ancak **`vllm`** (Linux/GPU
paketi) ve **`sentence-transformers`/`torch`** (yalnızca opsiyonel yerel
cross-encoder için; production akışında çağrılmaz) paketlerini içermez —
kurulum bu sayede dakikalar yerine saniyeler sürer.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-gemini.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-gemini.txt
```

## 3. API anahtarlarını tanımla

Sistem **iki sağlayıcı** kullanır — model/anlama katmanı **Gemini**, güvenlik
katmanı **Groq**. Bu bilinçli bir izolasyondur: bir sağlayıcıda kota/kesinti
olursa diğeri ayakta kalır. Anahtarlar **hiçbir zaman** repoya yazılmaz.

**A) Terminalde (oturumluk):**
```powershell
$env:GEMINI_API_KEY = "AIza...."
$env:GROQ_API_KEY   = "gsk_...."
```
```bash
export GEMINI_API_KEY="AIza...."
export GROQ_API_KEY="gsk_...."
```

**B) `.env` dosyasıyla (kalıcı):** `.env.example` dosyasını `.env` olarak
kopyalayıp iki satırı da doldurun. `.env` `.gitignore` kapsamındadır.

| Anahtar | Hangi katmanlar |
|---|---|
| `GEMINI_API_KEY` | VLM (video), VLM (kare), LLM/ajan, karar sentezi, embedding |
| `GROQ_API_KEY` | Prompt-injection guard |

> **Guard `fail_closed: true` ile çalışır:** `GROQ_API_KEY` tanımsızsa her
> analizde VLM açıklaması ve kullanıcı istemi **quarantine** damgası alır
> (içerik gizlenmez, ama görünür şekilde işaretlenir). Geçici olarak kapatmak
> için `configs/config.yaml → guard.enabled: false`.

> **Embedding neden Groq'ta değil?** Groq'un model kataloğunda **hiçbir
> embedding modeli yoktur** (yalnızca metin üretimi, Whisper/STT, TTS ve
> prompt-guard sınıflandırıcıları). `/v1/embeddings` ucu desteklenmez, bu
> yüzden RAG embedding katmanı Gemini'de kalır.

## 4. Mevzuat bilgi tabanını indeksle (RAG — tek seferlik)

748 parçalık Türkçe İSG mevzuatı külliyatı repoda hazır gelir
(`data/knowledge_base/chunks/`), ancak vektörlerinin bir kez üretilmesi
gerekir. Qdrant **yerel/gömülü** çalışır ve `data/qdrant/` altında bir klasör
oluşturur — kurulacak bir Qdrant sunucusu **yoktur**.

```bash
python -m src.rag.build_knowledge_index
```

Bu adım Gemini embedding API'sine gerçek istek atar. Atlanırsa sistem
çalışmaya devam eder, yalnızca raporlardaki **mevzuat atıf bölümü boş kalır**.

## 5. Backend'i başlat

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Doğrulama:
```bash
curl http://localhost:8000/health
```
Beklenen: `{"status":"ok","system":"SAFIR"}`

Açılışta bir **model ısınma raporu** loglanır (VLM / ajan / karar modeli /
embedding). Bu rapor yalnızca gözlemlenebilirlik içindir — bir katman hata
verse bile uygulama açılmaya devam eder. `Please pass a valid API key`
görüyorsanız 3. adımdaki anahtar tanımlı değildir.

> **Port 8000 önemlidir:** masaüstü arayüzü `/api` isteklerini
> `http://localhost:8000` adresine proxy'ler (`desktop/nuxt.config.ts`).
> Farklı bir port kullanırsanız arayüz backend'i bulamaz.

## 6. Masaüstü arayüzünü başlat

```bash
cd ../desktop
npm install
npm run dev
```
Arayüz `http://localhost:3000` adresinde açılır ve backend'e aynı-köken
(`/api` proxy) üzerinden bağlanır — CORS ayarı gerekmez.

## 7. Uçtan uca doğrulama

1. Bir `.mp4` dosyasını `safir-ai/data/` altına koyun.
2. Arayüzden **VLM Direct** modunu seçip analizi başlatın.
3. Canlı iz (SSE) panelinde sırayla şunları görmelisiniz: video parçalama →
   VLM çağrısı → olay tespiti → kural motoru → RAG → ajan → risk skoru.

Arayüzsüz, yalnızca API ile:
```bash
curl -X POST http://localhost:8000/analyze/jobs -H "Content-Type: application/json" -d "{\"video_source\":\"ornek.mp4\",\"analysis_mode\":\"vlm_direct\"}"
```

---

## Mimari özeti (hangi katman nereye gidiyor)

| Katman | Sağlayıcı | Uç nokta |
|---|---|---|
| VLM — video (VLM Direct) | Gemini `gemini-2.5-flash` | native `/v1beta/models/...:generateContent` |
| VLM — kare (Düşük Bütçeli) | Gemini `gemini-2.5-flash` | OpenAI-uyumlu `/v1beta/openai` |
| LLM / LangGraph ajanı | Gemini `gemini-2.5-flash` | OpenAI-uyumlu `/v1beta/openai` |
| Nihai karar sentezi | Gemini `gemini-2.5-pro` | OpenAI-uyumlu `/v1beta/openai` |
| Embedding (RAG) | Gemini `gemini-embedding-001` (1536 boyut) | OpenAI-uyumlu `/v1beta/openai` |
| Vektör deposu | **Qdrant — yerel/gömülü** | `data/qdrant/` (sunucu yok) |
| Prompt injection guard | **Groq** `openai/gpt-oss-20b` | OpenAI-uyumlu `https://api.groq.com/openai/v1` |
| Frame sampler | **Yerel, CPU-only** | — |
| Kural motoru + risk modeli | **Yerel, deterministik** | — |

Video neden native uca gidiyor? Gemini'nin OpenAI-uyumlu katmanı **görüntü**
kabul eder ama **video kabul etmez**; bu yüzden video-doğrudan yol
`src/vlm/gemini_vlm.py::GeminiVLM` içinde native `:generateContent` uç
noktasını kullanır. 12 MB'den büyük videolar otomatik olarak Files API'ye
yüklenir.

---

## Sorun giderme

| Belirti | Neden / Çözüm |
|---|---|
| `Please pass a valid API key` | `GEMINI_API_KEY` tanımlı değil veya geçersiz (bkz. adım 3). |
| Her raporda "quarantine" uyarısı | `GROQ_API_KEY` tanımlı değil — guard `fail_closed` davranıyor (bkz. adım 3). |
| Groq `400 ... logprobs` | Groq `logprobs`/`logit_bias`/`top_logprobs` kabul etmez. Guard bunları göndermez; bu hatayı görüyorsanız guard dışı bir katman yanlışlıkla Groq'a yönlendirilmiştir. |
| Raporda mevzuat bölümü boş | 4. adım (indeksleme) çalıştırılmamış. |
| Arayüz "Arka uca ulaşılamıyor" diyor | Backend 8000 portunda değil (bkz. adım 5 notu). |
| `ffmpeg/ffprobe PATH'te bulunamadi` | Bilgi amaçlı; parçalama OpenCV'ye düşer, sistem çalışır. |
| Ajan `risk_status=unknown` veriyor | Model boş içerik döndürmüş olabilir — `config.yaml` içindeki `reasoning_effort: "none"` / `thinkingConfig.thinkingBudget: 0` ayarlarının durduğundan emin olun. |
| Karar sentezi boş dönüyor | `gemini-2.5-pro` düşünme modu kapatılamaz; `llm.decision_model` değerini `"gemini"` yapmak güvenli geri adımdır. |

## Süreçleri durdurmak

Backend ve arayüz, çalıştıkları terminalde `Ctrl+C` ile durdurulur.

## EVREN'e geri dönmek gerekirse

Sağlayıcı kodu silinmedi (`src/vlm/evren_vlm.py`, `EvrenEmbeddingProvider`,
`EvrenPromptInjectionGuard`). `configs/config.yaml` içindeki `provider` /
`active_model` değerlerini `evren` ailesine çevirip `.env.example` içinde
yorum satırına alınmış `EVREN_*` değişkenlerini doldurmak yeterlidir.
