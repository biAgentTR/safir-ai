# SAFİR AI Benchmarking & Değerlendirme Raporu

**Tarih:** 2026-08-21 09:27:30  
**VLM Sağlayıcısı:** `mock`  
**Toplam Test Videosu:** 3 adet (270.0 sn)

## 📊 Genel Başarım Metrikleri

| Metrik Adı | Değer | Açıklama |
|---|---|---|
| **Precision (Kesinlik)** | `%100.0` | Yanlış alarmlardan arındırılmış doğru tespit oranı |
| **Recall (Duyarlılık)** | `%100.0` | Sahadaki gerçek olayları yakalama oranı |
| **F1-Score** | `%100.0` | Harmonik başarım skoru |
| **Kritik Olay Yakalama** | `%100.0` | Yüksek riskli acil durumları yakalama oranı |
| **Real-Time Faktör (RTF)** | `590.72x` | Gerçek zamana göre işleme hızı ( > 1.0x = Canlıdan hızlı) |
| **Guardrail Tetiklenme** | `%33.3` | Güvenlik katmanının denetim oranı |


## ⏱️ Katman Bazlı İşlem Süreleri (Video Detayları)

| Video Kimliği | Süre (sn) | Sampler (sn) | VLM (sn) | RAG (sn) | Agent (sn) | Toplam (sn) | RTF |
|---|---|---|---|---|---|---|---|
| `duman_video.mp4` | 60.0 | 0.0504 | 0.0001 | 0.0001 | 0.0 | 0.0506 | `1184.91x` |
| `isg_sahasi_01.mp4` | 120.0 | 0.0505 | 0.0003 | 0.0001 | 0.0 | 0.0509 | `2356.49x` |
| `savunma_devriye_02.mp4` | 90.0 | 0.0501 | 0.0003 | 0.0002 | 0.0 | 0.0507 | `1775.88x` |