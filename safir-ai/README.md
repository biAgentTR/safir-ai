# SAFIR — Saha Analiz ve Farkındalık İçin Yapay Zekâ Destekli Karar Sistemi

Video tabanlı İSG (iş sağlığı ve güvenliği) analizi ve karar destek sistemi.
Akış: **Video → Adaptive Frame Sampler (CPU) → VLM (görsel anlama) → Olay
Analizi + Hibrit Bellek/RAG → LangGraph Ajanı (muhakeme) → Otomatik Eskalasyon →
Yapılandırılmış Rapor (JSON) + Operatör Paneli**.

## Kurulum

```bash
cd safir-ai
pip install -r requirements.txt            # çekirdek servis (backend)
pip install -r requirements-dashboard.txt  # operatör paneli (Streamlit)
```

## Çalıştırma

```bash
# Backend (FastAPI)
python -m src.main            # veya: uvicorn src.main:app --host 0.0.0.0 --port 8000

# Operatör paneli (Streamlit)
streamlit run src/ui/dashboard.py
```

## Model backend'leri

Sistem üç backend'i tek soyutlama üzerinden destekler (`configs/config.yaml`):

| Backend | Ne zaman | Nasıl |
|---|---|---|
| **vLLM (yerel)** | Yarışma teslimi (varsayılan) | `vlm.active_model: qwen`, `llm.active_model: qwen3` |
| **Mock** | GPU'suz, offline geliştirme | `app.use_mock_vlm: true`, `app.use_mock_llm: true` |
| **Gemini** | VRAM yetersizken **test** | aşağıya bakın |

### Gemini test backend'i (geçici — yalnızca geliştirme/test)

> ⚠️ **Uyarı:** Harici API, şartnamenin "tamamen yerel/offline çalışma"
> gereksinimini ihlal eder. Gemini yalnızca yerel VRAM (8 GB) iki 3B modeli
> makul hızda koşamadığında **test** amacıyla kullanılır. **Yarışma teslimi
> için** `active_model` değerleri yerel modele (`qwen`/`qwen3`) geri
> alınmalıdır.

Aynı pipeline'ı Gemini'nin OpenAI-uyumlu ucu üzerinden test etmek için **iki
satır** yeterlidir:

```yaml
# configs/config.yaml
vlm:  { active_model: "gemini" }
llm:  { active_model: "gemini" }
```

ve API anahtarı (bkz. `.env.example`):

```bash
export GEMINI_API_KEY=...   # https://aistudio.google.com/apikey
```

Yerel vLLM sınıfları hiç değişmeden kalır; Gemini `src/vlm/gemini_vlm.py`
adaptörü ve `VLLMEndpointConfig.provider` alanı üzerinden eklenir.

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
