# SAFİR AI — Otonom Saha Güvenliği & İSG Video İnceleme Sistemi

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Framework: LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange.svg)](https://www.langchain.com/langgraph)
[![Status: TEKNOFEST Ready](https://img.shields.io/badge/TEKNOFEST-2026_Ready-purple.svg)]()

**SAFİR AI**, kritik tesislerde, fabrika sahalarında ve askeri/sivil operasyon alanlarında **"Operasyonel Risk"** şemsiyesi altında iki temel kategoriyi kapsayan **7 katmanlı, agentic (ajan tabanlı), provider-agnostic** yapay zeka sistemidir:
- 🏭 **SAFETY (İş Sağlığı ve Güvenliği)**: Kaza, yaralanma, KKD eksikliği, yangın/duman, ekipman arızası tespiti.
- 🛡️ **SECURITY (Tesis ve Çevre Güvenliği)**: İzinsiz alan girişi, tel örgü/nizamiye sızması, terk edilmiş şüpheli çanta/paket, İHA/drone hava ihlali.

> **Savunma Sanayi & Saha Operasyonları Notu:** Sistem hem endüstriyel tesisler (İSG/Safety) hem de savunma sanayi kritik tesisleri (Nizamiye/Perimeter/Security) için çift katmanlı bütünleşik bir operasyonel risk çözümü sunmaktadır.

---

## 🏛️ 1. Sistem Mimarisi ve Katmanlar

Sistem, ham video akışını ağır tespit modellerine (YOLO/ByteTrack) sokup GPU'yu yormak yerine; tamamen CPU üzerinde çalışan **OpenCV Adaptive Frame Sampler** ile süzerek **%80-%98 oranında GPU tasarrufu** sağlar. VLM ve LLM muhakeme katmanları **LangGraph** durum makinesi ile yönetilir.

Detaylı şemalar ve bileşen açıklamaları için: 📖 **[Sistem Mimari Dokümanı (`docs/ARCHITECTURE.md`)](docs/ARCHITECTURE.md)**

```text
[RTSP Canlı Kamera] 
       │
       ▼
[01. CPU Frame Sampler] ── (Δ ≥ 0.001 Filtresi ile %85+ GPU Tasarrufu)
       │
       ▼
[02. Peak Frame Exporter] ── (Zaman Dilimi Kümeleme & Zirve Kare Seçimi)
       │
       ▼
[03. Provider-Agnostic VLM] ── (Gemini API [Dev] / Yerel vLLM + Qwen2-VL [Prod])
       │
       ▼
[04. Hibrit BM25 + FAISS RAG] ── (0.5 Vektör + 0.5 Kelime, İSG & Tesis Yönergeleri)
       │
       ▼
[05. LangGraph Decision Agent] ── (Birincil LLM Muhakemesi + İkincil Guardrail)
       │
       ▼
[06. Escalation & Otonom Alarm] ── (Risk ≥ 51 -> POST /alerts/trigger)
       │
       ▼
[07. Komuta Merkezi & Audit UI] ── (Human-on-the-Loop Operatör Denetimi)
```

---

## ⚡ 2. Kurulum ve Çalıştırma Rehberi

### **Gereksinimler**
- Python 3.10+ (Windows / Linux / macOS)
- Virtualenv (`.venv`)
- OpenCV (`opencv-python`)
- PyTorch & FAISS (`faiss-cpu`)

### **1. Depoyu Klonlayın ve Sanal Ortamı Kurun**
```bash
git clone https://github.com/user/p3-project.git
cd p3-project/safir-ai

# Sanal ortam oluşturun ve aktif edin
python -m venv .venv
# Windows için:
.\.venv\Scripts\activate
# Linux/macOS için:
source .venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### **2. Ortam Değişkenlerini Yapılandırın**
`.env.example` dosyasını `.env` olarak kopyalayın:
```bash
cp .env.example .env
```
`.env` dosyasındaki ayarları düzenleyin:
```ini
VLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...your_key_here
FAISS_WEIGHT=0.5
BM25_WEIGHT=0.5
ALERT_THRESHOLD=51
```

### **3. Sunucuyu Başlatın (FastAPI Backend)**
```bash
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
Swagger API Dokümantasyonu: `http://localhost:8000/docs`

---

## 🧪 3. Test ve Benchmarking Betikleri

### **Çevrimdışı (Offline) Birim ve Senaryo Testlerini Çalıştırma**
Tüm ajan muhakemesi, eskalasyon ve VLM provider testleri harici API/GPU gerektirmeden çevrimdışı koşturulabilir:
```bash
python -m pytest
```

### **Tüm Sistemi Otomatik Benchmark Etme**
```bash
python -m evaluation.run_benchmark
```
Bu betik şunları koşturur:
1. CPU Adaptive Frame Sampler $\Delta$ eşik duyarlılık taraması.
2. Hibrit RAG Ağırlık duyarlılık ölçümü (FAISS/BM25).
3. Ground Truth veri seti başarım ölçümü (Precision, Recall, F1, RTF).

---

## 📊 4. Deneysel Ölçümleme ve Analiz Raporları

Sistemin başarım ve verimlilik iddiaları deneysel olarak kanıtlanmış ve dokümante edilmiştir:

1. 📄 **[Değerlendirme Metrikleri Raporu (`docs/METRICS.md`)](docs/METRICS.md)**
   - Precision (%100.0), Recall (%100.0), F1-Score (%100.0), Real-Time Factor (590x RTF).
2. 🎬 **[CPU Frame Sampler & GPU Tasarruf Analizi (`docs/FRAME_SAMPLER_ANALYSIS.md`)](docs/FRAME_SAMPLER_ANALYSIS.md)**
   - $\Delta = 0.001$ eşik değerinde %80-%98 GPU yükü tasarrufu kanıtı ve Matplotlib grafikleri.
3. 🧠 **[FAISS + BM25 Hibrit RAG Duyarlılık Raporu (`docs/RAG_EVALUATION.md`)](docs/RAG_EVALUATION.md)**
   - 0.5 FAISS + 0.5 BM25 dengeli ağırlık konfigürasyonunda %100 Top-1 doğruluğu ve 1.0000 MRR kanıtı.

---

## ⚠️ 5. Karşılaşılan Zorluklar ve Çözümler (Engineering Challenges)

> **Geliştirme Mimari Notu:**
> *"Yerel GPU kaynağı kısıtı nedeniyle geliştirme aşamasında VLM katmanı provider-agnostic tasarlandı; mevcut geliştirme Gemini API ile, teslim edilecek üretim sistemi yerel vLLM + Qwen2-VL ile çalışacaktır."*

- **Gereksiz VLM Çıkarım Maliyeti**: Durağan kamera görüntülerinin sürekli VLM'e gönderilmesi maliyeti artırıyordu. CPU tabanlı `AdaptiveFrameSampler` ile çözüldü (%85+ kare elendi).
- **Yanlış Alarm (False Positive) Riski**: Kural tabanlı regex süzgeçleri ana karar mekanizması yapıldığında esneklik yitiriliyordu. Sistem **LLM-Primary Decision Engine + Secondary Guardrail Filter** mimarisine geçirildi.

---

## 📜 6. Lisans

Bu proje **[Apache License 2.0](LICENSE)** altında lisanslanmıştır. TEKNOFEST 2026 Yarışması şartnamelerine tam uyumludur.
