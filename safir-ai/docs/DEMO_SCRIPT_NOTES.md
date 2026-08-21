# SAFİR AI — TEKNOFEST Sunumu Demo Video Kurgu Planı (DEMO_SCRIPT_NOTES.md)

Bu doküman, yarışma jürisine sunulacak demo videosunun akışını, zaman çizelgesini (timeline) ve videoda mutlaka gösterilmesi gereken **SAFETY (İş Güvenliği)** ve **SECURITY (Tesis/Perimeter Güvenliği)** senaryolarının kurgu notlarını içerir.

---

## 🎬 1. Genel Demo Video Kurgu Akışı (Toplam Süre: ~3-4 Dakika)

```text
[00:00 - 00:30] Giriş & Sistem Mimarısı Özet Sunumu (7 Katman, Provider-Agnostic, LangGraph)
       │
       ▼
[00:30 - 01:45] SENARYO 1: SAFETY (İş Güvenliği & Forklift Kazası)
       │
       ▼
[01:45 - 03:00] SENARYO 2: SECURITY (Tesis Çevre Güvenliği & İzinsiz Sızma)
       │
       ▼
[03:00 - 03:30] Hibrit RAG Mevzuat Sorgulama & Human-in-the-Loop Operatör Onayı
```

---

## 🏭 2. Senaryo 1: SAFETY (İş Sağlığı ve Güvenliği)

- **Video İçeriği**: Fabrika içi forklift trafiğinde bir yaya personeline çarpma ve personelin hareketsiz kalması (Yarışma şartname örneğine sadık).
- **Zaman Çizelgesi İşaretçileri**:
  - `00:05` — Personel baret ve yelekle yürüyor (`safe_timestamps`).
  - `00:18` — Forklift yaya geçidine yaklaşıyor, anlık görüş engeli.
  - `00:22` — **OLAY ANI**: Forklift personelle temas ediyor; personel yerde kalıyor (`incident_timestamps`).
- **Sistem Ekranında Vurgulanacak Detaylar**:
  - **Kategori Rozeti**: `[CATEGORY: SAFETY]` (Yeşil/Mavi Sanayi Simgesi)
  - **Risk Skoru**: `85 / 100` (Kritik Risk)
  - **İlgili Mevzuat (RAG)**: `ISG Yönetmeliği Madde 24` & `Operasyonel Kural OK-07` (Forklift-Yaya Ayrımı).
  - **Aksiyon Önerisi**: `"Acil Sağlık Ekibini Yönlendirin", "Alanı Güvenlik Altına Alın"`.
  - **Otonom Alarm**: `POST /alerts/trigger` ile `SAHA ALARMI OTOMATIK TETIKLENDI [SAFETY]` günlüğü.

---

## 🛡️ 3. Senaryo 2: SECURITY (Tesis ve Çevre Güvenliği)

- **Video İçeriği**: Kritik savunma sanayi tesisi dış tel örgü hattından gece/gündüz izinsiz bir şahsın tırmanarak sızma girişimi.
- **Zaman Çizelgesi İşaretçileri**:
  - `01:48` — Dış çevre koruma hattı kamera görüntüsü (`safe_timestamps`).
  - `02:05` — **OLAY ANI**: Şüpheli şahıs tel örgüye yanaşıyor ve fiziki tırmanma başlatıyor (`incident_timestamps`).
  - `02:15` — Şahıs nizamiye arka koridoruna giriş yapıyor.
- **Sistem Ekranında Vurgulanacak Detaylar**:
  - **Kategori Rozeti**: `[CATEGORY: SECURITY]` (Kırmızı Kalkan Simgesi)
  - **Risk Skoru**: `85 / 100` (Kritik Risk)
  - **İlgili Yönerge (RAG)**: `Erişim Kontrol Prosedürü EK-01` & `Çevre Koruma İhlali SEC-02`.
  - **Aksiyon Önerisi**: `"Nizamye ve Güvenlik Devriyesini Yönlendirin", "Perimeter Kapılarını Kilitleyin"`.
  - **Otonom Alarm**: `POST /alerts/trigger` ile `SAHA ALARMI OTOMATIK TETIKLENDI [SECURITY]` günlüğü.

---

## 🧠 4. Gösterilecek Ek Özellikler & Jüri Notları

1. **CPU Adaptive Sampler GPU Tasarruf Göstergesi**:
   - Ekranın alt köşesinde `GPU Tasarrufu: %88.4` ve `İşleme Hızı: 590x RTF` göstergeleri açıkça vurgulanmalı.
2. **Negasyon Guardrail Kanıtı**:
   - `"Herhangi bir yetkisiz giriş veya kaza görülmedi"` cümlesinin sistemi yanlış alarma geçirmediği, `Risk Skoru: 0` olarak doğru süzüldüğü ekranda gösterilmeli.
3. **Human-on-the-Loop Denetimi**:
   - Operatörün tetiklenen alarmları durdurmak zorunda olmadığı, ancak otonom alarmlara sonradan `[ONAYLA / ACKNOWLEDGE]` notu düşebildiği arayüz üzerinden sunulmalı.
