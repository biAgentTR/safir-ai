# SAFIR — Saha Analiz ve Farkındalık İçin Yapay Zekâ Destekli Karar Sistemi

Video tabanlı İSG (iş sağlığı ve güvenliği) analizi ve karar destek sistemi.
Akış: **Video → Adaptive Frame Sampler (CPU) → VLM (görsel anlama) → Olay
Analizi + Hibrit Bellek/RAG → LangGraph Ajanı (muhakeme) → Otomatik Eskalasyon →
Yapılandırılmış Rapor (JSON) + Operatör Paneli**.

> **Şartname dokümantasyonu:** mimari diyagramı, kullanılan agentic framework/LLM'ler,
> senaryolar ve mock fonksiyonlar, ölçümleme sonuçları ve ölçekleme ihtiyaçları için
> bkz. [`../DOKUMANTASYON.md`](../DOKUMANTASYON.md).

## Kurulum (GPU GEREKMEZ)

Aktif mimaride model servislemesi **Google Gemini** üzerindedir; NVIDIA GPU,
CUDA toolkit, vLLM veya Docker **gerekmez**. Yerel çalışan tek ağır bileşen
CPU-only frame sampler'dır.

```bash
cd safir-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-gemini.txt      # Gemini profili (vllm/torch İÇERMEZ)
pip install -r requirements-dashboard.txt   # opsiyonel: operatör paneli (Streamlit)
```

## Çalıştırma

```bash
# 1) API anahtarı (tek anahtar; VLM + LLM + embedding + guard hepsi bunu kullanır)
export GEMINI_API_KEY=AIza...        # Windows PowerShell: $env:GEMINI_API_KEY="AIza..."

# 2) Mevzuat bilgi tabanını indeksle (tek seferlik; Qdrant yerel/gömülü çalışır)
python -m src.rag.build_knowledge_index

# 3) Backend (FastAPI)
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Operatör paneli (Streamlit, opsiyonel)
streamlit run src/ui/dashboard.py
```

Masaüstü arayüzü `/api` isteklerini `http://localhost:8000` adresine proxy'ler
(`desktop/nuxt.config.ts`), bu yüzden **port 8000 korunmalıdır**. Ayrıntılı
kurulum, mimari tablosu ve sorun giderme için bkz.
[`KURULUM.md`](KURULUM.md).

## Model backend'leri

Sistem üç backend'i tek soyutlama üzerinden destekler (`configs/config.yaml`):

| Backend | Ne zaman | Nasıl |
|---|---|---|
| **Gemini** | **Aktif** — VLM, LLM/ajan, embedding | `vlm.active_model: gemini`, `llm.active_model: gemini` |
| **Groq** | **Aktif** — prompt-injection guard | `guard.provider: groq` |
| **vLLM (yerel)** | Yerel GPU ile çalıştırmak istenirse | `vlm.active_model: qwen`, `llm.active_model: qwen3` |
| **Mock** | GPU/ağ'sız, offline geliştirme | `app.use_mock_vlm: true`, `app.use_mock_llm: true` |
| **EVREN (TEKNOFEST servisi)** | ~~Aktif~~ — **servis takıma kapatıldı** | kod duruyor, config artık seçmiyor |

### Aktif kurulum: Gemini (model) + Groq (güvenlik)

Model/anlama katmanı Gemini'de, güvenlik katmanı **ayrı bir sağlayıcıda**
(Groq) çalışır — bir sağlayıcıda kota/kesinti olursa güvenlik katmanı da
birlikte düşmesin diye bilinçli bir izolasyon.

```bash
export GEMINI_API_KEY=AIza...   # VLM (video+kare), LLM/ajan, karar sentezi, embedding
export GROQ_API_KEY=gsk_...     # prompt-injection guard
```

> Guard `fail_closed: true` çalışır: `GROQ_API_KEY` yoksa içerik quarantine
> damgası alır (gizlenmez). Kapatmak için `guard.enabled: false`.

> **Embedding neden Groq'ta değil?** Groq'un kataloğunda hiçbir embedding
> modeli yok (yalnızca metin üretimi, Whisper, TTS, prompt-guard
> sınıflandırıcıları) — `/v1/embeddings` desteklenmiyor.

Video-doğrudan yol, Gemini'nin **native** `:generateContent` ucunu kullanır —
çünkü Gemini'nin OpenAI-uyumlu katmanı görüntü kabul eder ama **video kabul
etmez** (`src/vlm/gemini_vlm.py::GeminiVLM`). 12 MB üstü videolar otomatik
olarak Files API'ye yüklenir. Kare-tabanlı yol, LLM, embedding ve guard ise
OpenAI-uyumlu uç üzerinden gider, dolayısıyla mevcut istemci kodu değişmeden
çalışır.

Vektör deposu artık **yerel/gömülü Qdrant**'tır (`data/qdrant/`) — ayrı bir
Qdrant sunucusu ve ek ortam değişkeni gerekmez.

Yerel vLLM sınıfları (`qwen`/`gemma`) ve EVREN sınıfları hiç değişmeden kalır;
sağlayıcı seçimi `VLLMEndpointConfig.provider` alanı üzerinden yapılır.

## Otomatik Eskalasyon (Human-on-the-Loop)

Önceki tasarımda operatör, saha alarmını tetiklemek için "onayla" butonuna
basmak zorundaydı (bloke edici Human-in-the-Loop kapısı). Mentör geri bildirimi
doğrultusunda bu kapı kaldırıldı: sistem, risk skoruna göre aksiyon kademesini
**kendisi** belirler (`src/decision/escalation.py`):

- **monitor** (düşük): yalnızca kaydet, izlemeye devam.
- **notify** (orta): kaydet + bildirim.
- **alarm** (yüksek/kritik): saha alarmı **otomatik** tetiklenir (operatör onayı
  beklenmez).

Operatör, alarmı engellemek için değil; sonradan **denetlemek/geri almak** için
devrededir (`POST /alerts/{alert_id}/acknowledge`). Eşikler
`configs/config.yaml → escalation` altında ayarlanır.

## Uçtan uca test

```bash
# Tamamen offline, deterministik (mock VLM+LLM) — ağ/GPU/anahtar gerekmez:
python scripts/e2e_smoke.py --mock

# Gerçek E2E (Gemini backend'i ayarlıysa):
export GEMINI_API_KEY=...
python scripts/e2e_smoke.py --video data/ornek.mp4

# Birim + entegrasyon testleri:
pytest -q
```

## Adım adım Jupyter walkthrough

Pipeline'ı aşama aşama görmek için (sampler → temsili kareler → VLM → olay
tespiti → RAG → ajan → otomatik eskalasyon → nihai rapor):

```bash
pip install notebook ipykernel
jupyter notebook notebooks/SAFIR_walkthrough.ipynb
# veya VS Code'da .ipynb dosyasını açıp hücreleri sırayla çalıştır
```

Defterin ilk hücresindeki iki anahtar:
- `USE_MOCK`: `False` → `config.yaml`'daki backend (Gemini; `GEMINI_API_KEY` gerekir).
  `True` → tamamen offline (sabit örnek çıktı, GPU/anahtar gerekmez).
- `USE_FAKE_RAG`: `True` → ağır embedding modelini (bge-m3) indirmez (demo için hızlı).

Notebook, `scripts/build_notebook.py` ile üretilir (yeniden üretmek için:
`python scripts/build_notebook.py`).

## KPI / Benchmark (ölçümleme)

Şartname kendi metriklerinizi tanımlamanızı ister. `scripts/benchmark.py`,
etiketli klipler üzerinde tüm pipeline'ı koşup KPI'ları raporlar: olay tespiti
precision/recall/F1 (kategori + makro/mikro), kritik olay yakalama oranı ve
klip başı gecikme.

```bash
# Hazır sentetik demo (offline, mock) — harness'i ve metrikleri gösterir:
python scripts/benchmark.py --synthetic --mock

# Gerçek etiketli klipler (Gemini/vLLM backend ayarlıysa):
python scripts/benchmark.py --manifest benchmarks/manifest.json --out benchmarks/result.json
```

Manifest biçimi:
```json
[
  {"video": "data/clip1.mp4", "expected_events": ["arac_yaya_yakinligi"], "critical": true},
  {"video": "data/clip2.mp4", "expected_events": ["kkd_ihlali"]}
]
```

## Dayanıklılık (hata işleme)

- VLM çağrıları geçici ağ hatalarında üstel geri-çekilmeli yeniden denenir; kalıcı
  hatada iş çökmez, **degraded** (hata notlu) rapor üretilir ve operatör manuel
  incelemeye yönlendirilir.
- Ajan muhakemesi hata verirse risk uydurulmaz; güvenli bir degraded karar döner.
- Küçük modellerin bozuk JSON'u: ajan tarafında JSON-modu yeniden-denemesi
  (`agent.guided_json`), VLM tarafında toleranslı EVENTS_JSON ayrıştırma +
  anahtar-kelime yedeği ile kurtarılır.

> **Bilinen sorun:** `tests/test_sampler.py` içindeki iki test
> (`test_motion_produces_real_evidence_frames`,
> `test_fallback_frame_used_when_no_threshold_crossed`) bu değişikliklerden
> **önce de** başarısızdı: testler `process_video` çıktısındaki her karede
> `saved_path` bekliyor; oysa tasarım gereği yalnızca **zirve** kareler diske
> yazılır. Sampler'ın GPU-tasarrufu davranışını bozmamak için bu davranış
> korundu; testlerin mi yoksa kaydetme politikasının mı güncelleneceği ayrı bir
> karar olarak bırakıldı.

## Proje yapısı (özet)

```
src/
  sampler/        CPU Adaptive Frame Sampler (kanıt karesi + olay kümeleme)
  vlm/            VLM/LLM istemcileri (Qwen/Gemma/Gemini + mock), factory
  prompts/        Merkezi istemler (VLM gözlem + ajan muhakeme/JSON)
  event_analysis/ Olay tespiti, zamansal muhakeme, kural motoru
  memory/         SQLite olay belleği + Embedding/FAISS RAG
  agent/          LangGraph muhakeme ajanı (JSON karar çıktısı)
  decision/       Otomatik eskalasyon (Human-on-the-Loop)
  schemas/        SafirReport (şartname-uyumlu JSON)
  ui/             Operatör paneli (OOP bileşenler: api_client, theme,
                  report_export, components/, app)
  main.py         FastAPI servisi + SafirPipeline orkestratörü
```
