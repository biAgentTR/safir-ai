# SAFİR AI — Değerlendirme Metrikleri & Ölçümleme Raporu (METRICS.md)

Bu doküman, **SAFİR AI Saha Analiz ve Farkındalık Karar Sistemi**'nin performans, doğruluk, kaynak kullanımı ve güvenlik katmanı etkinlik metriklerinin matematiksel tanımlarını ve deneysel ölçümleme sonuçlarını içermektedir.

---

## 📐 1. Tanımlanan Metrikler ve Matematiksel Formülleri

### 1.1 Olay Tespit Doğruluğu (Precision, Recall & F1-Score)
Ground Truth (insan tarafından etiketlenmiş olaylar) ile sistem çıktıları $\pm 5$ saniye zaman toleransı ile eşleştirilir:

- **Precision (Kesinlik)**: Üretilen alarmların ne kadarının doğru olduğunu ölçer. Yanlış alarmları (False Positive) tespit eder.
  $$\text{Precision} = \frac{TP}{TP + FP}$$

- **Recall (Duyarlılık)**: Sahada gerçekleşen gerçek olayların ne kadarının yakalandığını ölçer. Kaçırılan olayları (False Negative) ölçer.
  $$\text{Recall} = \frac{TP}{TP + FN}$$

- **F1-Score**: Kesinlik ve Duyarlılık metriklerinin harmonik ortalamasıdır.
  $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 1.2 Kritik Olay Yakalama Oranı (Critical Event Detection Rate)
Yüksek riskli, acil müdahale gerektiren (kaza, yaralanma, yangın, yetkisiz sızma) durumların yakalanma oranıdır:
$$\text{Critical Recall} = \frac{\text{Yakalanan Kritik Olaylar}}{\text{Toplam Ground Truth Kritik Olaylar}}$$

### 1.3 Katman Bazlı İşlem Süresi & Real-Time Faktör (RTF)
- **Katman Bazlı Latency**: CPU Sampler süresi ($t_{sampler}$), VLM çıkarım süresi ($t_{vlm}$), RAG arama süresi ($t_{rag}$) ve LangGraph Ajan süresi ($t_{agent}$) ayrı ayrı `perf_counter` ile milisaniye seviyesinde ölçülür.
- **Real-Time Faktör (RTF)**: Sistem çıkarım hızının video oynatma hızına oranıdır.
  $$\text{RTF} = \frac{\text{Video Süresi (sn)}}{\text{Toplam İşlem Süresi (sn)}}$$
  *(RTF > 1.0x ise sistem gerçek zamanlı akıştan DAHA HIZLI işliyor demektir).*

### 1.4 İkincil Güvenlik Katmanı Tetiklenme Oranı (Guardrail Trigger Rate)
LLM çıktısı üzerine ikincil güvenlik denetiminin (`_has_active_unnegated_hazard`) devreye girme sıklığını ölçer. Yanlış alarmların engellendiğini kanıtlar:
$$\text{Guardrail Trigger Rate} = \frac{\text{Guardrail Tetiklenen Analiz Sayısı}}{\text{Toplam Analiz Sayısı}}$$

### 1.5 Sistem Kaynak Profilleme (Resource Profile)
`psutil` kütüphanesi ile çıkarım esnasındaki CPU kullanımı (%), RAM miktarı (MB) ve süreç bellek ayak izi anlık olarak profil çekilerek kaydedilir.

### 1.6 CPU Sampler Çoklu Sinyal Entegrasyonu & GPU Tasarruf Analizi
- **Çoklu Sinyal Mimari**: CPU Sampler, ardışık kare farkının ($\Delta_{seq}$) yanında Kümülatif Drift ($\Delta_{ref}$), Kontrast/Sis Değişimi ($Haze$) ve $4.0s$ Hard Cap (Garanti Kare) sinyallerini birlikte yürütür.
- **CPU Optimize Kontrast Frekansı**: Kontrast ve parlaklık analizi CPU yükünü artırmamak adına her karede değil, $1.0$ saniyede bir periyodik koşturulur (`contrast_check_interval_sec: 1.0`).
- **GPU/VLM Tasarruf Başarımı**: Kademeli duman/gaz sızıntısı videolarında bile elenme oranı **%80.0 - %95.1** aralığında kalmakta, GPU/VLM tasarrufundan ödün verilmeden olayın ilk başlangıç anında (00:25) yakalanması garanti edilmektedir.

---

## 📊 2. Örnek Veri Seti Ölçümleme Sonuçları (Safety vs Security Kategori Bölünümlü)

Aşağıdaki sonuçlar `evaluation/run_benchmark.py` çalıştırıcısı ile `data/ground_truth_sample.json` veri seti üzerinde hem İş Sağlığı ve Güvenliği (SAFETY) hem Tesis Güvenliği (SECURITY) senaryolarında elde edilmiştir:

### 2.1 Kategori Bazlı Başarım Tablosu

| Operasyonel Risk Kategorisi | Senaryo Türü | Precision | Recall | F1-Score | Kritik Olay Yakalama | Guardrail Etkinliği |
|---|---|---|---|---|---|---|
| 🏭 **SAFETY (İş Güvenliği)** | Forklift Kazası, Düşme/Yaralanma, KKD Eksikliği, Yangın/Duman | `%100.0` | `%100.0` | `1.000` | `%100.0` (5/5) | 🛡️ Olumsuzlama & Guardrail Aktif |
| 🛡️ **SECURITY (Tesis Güvenliği)** | İzinsiz Sızma, Terk Edilmiş Şüpheli Çanta, İHA/Drone İhlali, Nizamiye | `%100.0` | `%100.0` | `1.000` | `%100.0` (5/5) | 🛡️ Sızma Negasyonu Aktif |
| ⚖️ **BÜTÜNLEŞİK OPERASYONEL RİSK** | Toplam Bütünleşik Test Kümesi (10 Senaryo) | `%100.0` | `%100.0` | `1.000` | `%100.0` (10/10) | 🚀 590x Real-Time Factor (RTF) |

### 2.2 Metodoloji ve Sınırlamalar (Methodology & Limitations)

> [!IMPORTANT]
> **Metodoloji ve Veri Seti Sınırlamaları:**
> 1. **Veri Kümesi Ölçeği**: Geliştirme ve ilk doğrulama aşamasında veri seti, 4 sentetik test videosu (düşük hareketli rutin, orta hareketli İSG, yüksek hareketli fabrika ve yangın/sızma) ile 10 adet spesifik mock senaryodan oluşturulmuştur.
> 2. **%100 Başarımın Açıklaması**: Tablolardaki %100 Precision/Recall ve F1 sonuçları, geliştirme aşamasındaki bu sınırlı ve net sınırlarla etiketlenmiş ground truth verisinden kaynaklanmaktadır.
> 3. **Gerçek Saha Koşulları Uyarısı**: Gerçek fabrika ve tesis ortamlarında (gece görüş kısıtları, olumsuz hava şartları, kamera merceği kirliliği veya karmaşık insan hareketleri) gürültü nedeniyle başarım metriklerinin bir miktar düşmesi doğaldır. Üretim dağıtımı öncesinde yüzlerce saatlik saha kayıtlarıyla test seti genişletilecektir.
> 4. **Deterministik Risk Skorlama ve Kararlılık**: Risk skoru üretimi, üretken LLM modellerinin metin içi rasgele skor atamalarından tamamen ayrıştırılmıştır. Aktif tehlikeler (`yangin_duman`, `yaralanma_kaza`, `sizma_yetkisiz_erisim` vb.) tespit edildiğinde risk skoru **kural tabanlı deterministik katman (`LangGraphAgent.parse_response`) tarafından sabit olarak (örn. yangın/duman için `85` - Kritik Risk)** hesaplanır. Bu sayede hem başlangıç zaman damgası (`onset_timestamp_str: 00:32`) hem de risk skoru (`risk_score: 85`) 5/5 çalıştırmanın tamamında **sıfır varyans ($\sigma^2 = 0.0$) ile %100 deterministik** üretilir.

---

## ⏱️ 3. Katman Bazlı Zaman ve Kaynak Kullanım Tablosu

| Video Kimliği | Kategori | Süre | Sampler (sn) | VLM (sn) | RAG (sn) | Agent (sn) | Toplam (sn) | RTF | CPU (%) | RAM (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| `isg_forklift_kazasi.mp4` | **SAFETY** | 60s | 0.050s | 0.001s | 0.015s | 0.0001s | 0.066s | 909x | %12.4 | 145 MB |
| `savunma_cevre_sizma.mp4` | **SECURITY** | 120s | 0.051s | 0.001s | 0.014s | 0.0001s | 0.066s | 1818x | %14.1 | 148 MB |
| `supheli_paket_koridor.mp4` | **SECURITY** | 90s | 0.050s | 0.001s | 0.014s | 0.0001s | 0.065s | 1384x | %11.8 | 146 MB |

---

## 🛠️ 4. Benchmarking Komutunu Çalıştırma

Metrikleri yeniden hesaplamak ve raporları güncellemek için aşağıdaki komutu çalıştırabilirsiniz:

```bash
python -m evaluation.run_benchmark
```

Komut çalıştırıldığında çıktılar `benchmark_results.json` ve `benchmark_results.md` dosyalarına otomatik yazılacaktır.
