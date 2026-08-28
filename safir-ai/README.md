# SAFIR — Saha Analiz ve Farkındalık İçin Yapay Zekâ Destekli Karar Sistemi

Video tabanlı İSG (iş sağlığı ve güvenliği) analizi ve karar destek sistemi.
Akış: **Video → Adaptive Frame Sampler (CPU) → VLM (görsel anlama) → Olay
Analizi + Hibrit Bellek/RAG → LangGraph Ajanı (muhakeme) → Otomatik Eskalasyon →
Yapılandırılmış Rapor (JSON) + Operatör Paneli**.

> **Şartname dokümantasyonu:** mimari diyagramı, kullanılan agentic framework/LLM'ler,
> senaryolar ve mock fonksiyonlar, ölçümleme sonuçları ve ölçekleme ihtiyaçları için
> bkz. [`../DOKUMANTASYON.md`](../DOKUMANTASYON.md).

## Kurulum (native — Docker GEREKMEZ)

Sistemde bir NVIDIA GPU + sürücü (Blackwell/RTX 5090 için ≥570) varsa yeterlidir;
ayrı bir CUDA toolkit veya Docker kurulumuna gerek yoktur (`vllm` paketi kendi
uyumlu torch/CUDA wheel'lerini pip ile getirir).

```bash
cd safir-ai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # backend + yerel vLLM istemcisi/servisi (tek kurulum)
pip install -r requirements-dashboard.txt  # opsiyonel: operatör paneli (Streamlit)
```

## Çalıştırma

Modelleri servis eden vLLM süreçlerini (VLM + LLM), ardından API'yi başlatın —
üçü de aynı `venv` içindeki `vllm`/`python` komutlarıyla, Docker'sız:

```bash
# 1) VLM sunucusu (arka planda)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --trust-remote-code \
  --quantization fp8 --dtype bfloat16 --gpu-memory-utilization 0.50 \
  --max-model-len 8192 --limit-mm-per-prompt image=12 --max-num-seqs 2 &

# 2) LLM sunucusu (arka planda)
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8003 --trust-remote-code \
  --dtype bfloat16 --gpu-memory-utilization 0.30 --max-model-len 4096 --max-num-seqs 4 &

# 3) Backend (FastAPI)
python -m src.main            # veya: uvicorn src.main:app --host 0.0.0.0 --port 8000

# Operatör paneli (Streamlit, opsiyonel)
streamlit run src/ui/dashboard.py
```

`configs/config.yaml` içindeki `vlm.models.qwen` / `llm.models.qwen3` altındaki
`vllm_host`/`vllm_port` değerleri (varsayılan `127.0.0.1:8001` / `127.0.0.1:8003`)
yukarıdaki `--port` değerleriyle eşleşmelidir. Ayrıntılı sıfırdan-kurulum adımları
(sürücü kurulumu, `nohup` ile arka planda çalıştırma, sorun giderme) için bkz.
[`KURULUM.md`](KURULUM.md).

## Model backend'leri

Sistem üç backend'i tek soyutlama üzerinden destekler (`configs/config.yaml`):

| Backend | Ne zaman | Nasıl |
|---|---|---|
| **EVREN (TEKNOFEST servisi)** | Aktif (varsayılan) | `vlm.active_model: evren`, `llm.active_model: evren` |
| **vLLM (yerel)** | Yerel GPU ile çalıştırmak istenirse | `vlm.active_model: qwen`, `llm.active_model: qwen3` |
| **Mock** | GPU/ağ'sız, offline geliştirme | `app.use_mock_vlm: true`, `app.use_mock_llm: true` |

### EVREN backend'i (aktif)

Video **doğrudan** EVREN'in video-analiz ucuna (`model: "vlm"`) gönderilir;
yerel GPU/vLLM gerekmez. Gerekli tek şey API anahtarı (bkz. `.env.example`):

```bash
export EVREN_API_KEY=sk-evren-teamNN-XXXXXXXX
```

Yerel vLLM sınıfları (`qwen`/`gemma`) hiç değişmeden kalır; EVREN
`src/vlm/evren_vlm.py` adaptörü ve `VLLMEndpointConfig.provider` alanı
üzerinden eklenir.

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
