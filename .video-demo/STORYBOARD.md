# SAFİR — 60 sn Demo Videosu / Storyboard

**Format:** 1920×1080, 30 fps, sessiz (müzik/seslendirme yok — telifsiz kaynak ve API anahtarı
mevcut olmadığı için uydurma ses eklenmedi), yakılmış Türkçe altyazı.

**Kaynak ilkesi:** Her kare gerçek üründen veya gerçek saha kaydından. Hiçbir UI mockup'ı,
hiçbir uydurma skor/benchmark yok. Ekrandaki bütün sayılar `job 16986632-1de6-4d1b-8b2f-3302857a947c`
analizinin gerçek çıktısıdır (bucket11.mp4, VLM Direct modu, EVREN).

## Doğrulanmış veri (videoda görünen her iddia)

| İddia | Kaynak |
|---|---|
| Risk 76 (kritik) = deterministik 62 + ajan (LLM) 90 ortalaması | `report.risk_score` / `deterministic_score` / `llm_proposed_score` |
| 2 kritik olay: `çöp_kutusunda_yangin` 00:18, `sabit_alev_boru_hattinda` 01:00, güven %95 | `report.events`, Riskli Olaylar tablosu |
| VLM metni: "00:18'de kişi koltuğundan kalkarken… 00:32'de duman alevlenir" | `report.natural_language_summary` |
| Gerekçe: YG-03 kuralı, `safir_evidence_weighted_v2`, feature'lar | `report.risk_explanation` |
| Operatör cevabı: "yangın söndürme ekiplerini çağırmalı…" | Canlı `/ask/stream` yanıtı (36 sn, 2 çağrı) |
| Mevzuat kaynakları (7 kaynak) | Ask yanıtının gerçek RAG kaynakları |
| İşleme süresi 1 dk 43 sn | Pipeline trace `duration_ms` toplamı |
| KPI: Özet Kalitesi %89,6 (155/173) · Kritik Olay Yakalama %77,9 (95/122) | `GET /metrics/kpi` |

**Kasıtlı olarak GÖSTERİLMEYEN:** bounding box / nesne-ID overlay'i. SAFİR nesne dedektörü
değil, video-dil modeli tabanlıdır; kutu çizmek ürünün yapmadığı bir şeyi göstermek olurdu.
Algı katmanı, ürünün gerçekten ürettiği şeyle anlatılır: zaman çizelgesindeki riskli-an
işaretleri ve zaman damgalı olay tespiti.

## Timeline

| # | Zaman | Süre | Görüntü | Altyazı |
|---|---|---|---|---|
| 1 | 0:00–0:05 | 5 sn | `09_home.png` hero, yavaş zoom-in | "Saha kamerası görüntüsünden risk kararına — saniyeler içinde." |
| 2 | 0:05–0:09 | 4 sn | `saha_yangin.mp4` gerçek saha kaydı, tam ekran | "Gerçek saha kaydı. Kimse izlemiyor olabilir." |
| 3 | 0:09–0:13 | 4 sn | `02_dashboard_clean.png` video paneli + zaman şeridi zoom | "Video doğrudan görsel-dil modeline (EVREN) gider." |
| 4 | 0:13–0:18 | 5 sn | `saha_forklift.mp4` + şerit işaretleri | "Riskli saniyeler zaman çizelgesinde işaretlenir." |
| 5 | 0:18–0:25 | 7 sn | `03_events.png` Riskli Olaylar tablosu zoom | "Nesne değil, olay: tür, zaman damgası, güven." |
| 6 | 0:25–0:32 | 7 sn | `06_risk_reasoning.png` VLM çıktısı zoom | "Olayın başlangıcı, gelişimi ve sonucu okunur." |
| 7 | 0:32–0:39 | 7 sn | `06_risk_reasoning.png` risk kırılımı zoom | "Risk skoru iki bağımsız kaynaktan: kural motoru + ajan." |
| 8 | 0:39–0:45 | 6 sn | `06_risk_reasoning.png` gerekçe metni zoom | "Her skorun gerekçesi ve mevzuat dayanağı görünür." |
| 9 | 0:45–0:50 | 5 sn | `04_ask_answer.png` cevap zoom | "Operatöre uygulanabilir aksiyon." |
| 10 | 0:50–0:54 | 4 sn | `08_kpi.png` KPI paneli zoom | "Ölçümleme paneli: her metrik gerçek veriden." |
| 11 | 0:54–1:00 | 6 sn | `09_home.png` zoom-out, temiz son kare | "SAFİR — Görüntüyü izlemeyin. Ne olduğunu anlayın." |

**Toplam: 60,0 sn**

## Uygulama notları

- Geçişler: 12 kare (0,4 sn) xfade — hızlı ve temiz.
- Hareket: her sahnede yumuşak zoom (Ken Burns); statik kare yok.
- Altyazı: Segoe UI Bold, alt %12'de, koyu şerit üzerinde, 2 satırı geçmez.
- Renk dili: SAFİR'in kendi paleti (koyu yüzey + turkuaz vurgu) korunur; ek grafik katmanı yok.
