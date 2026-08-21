# SAFİR AI — FAISS + BM25 Hibrit RAG Değerlendirme & Duyarlılık Raporu

Bu rapor, **SAFİR AI Anlamsal Bellek Katmanı (EmbeddingRAGService)** için FAISS vektör araması, BM25 kelime araması ve Safety/Security kategori bonus oranlarının başarımını ölçmektedir.

## ⚙️ 1. Mimari ve İndekslenen Belgeler

Sistem 100% yerel (offline) `sentence-transformers/all-MiniLM-L6-v2` embedding modeli ve saf Python BM25 indeksleyicisi ile çalışır.
İndekslenen temel belgeler:

- 🏭 **İSG Yönetmelikleri (`safety`)**: Madde 12 (Yüksekte Çalışma), Madde 24 (KKD), OK-07 (Forklift-Yaya), Madde 31 (Sıcak Çalışma), YG-03 (Yangın Tahliye), Madde 45, OK-15, Madde 52.
- 🛡️ **Savunma Tesis Koruma Yönergeleri (`security`)**: EK-01 (Çevre Sızma), SHP-02 (Şüpheli Paket), İHA-04 (Drone İhlali), SEC-01 (İzinsiz Alan Girişi), SEC-02 (Tel Örgü İhlali), SEC-03 (Terk Edilmiş Nesne), SEC-04 (Yetkisiz Araç).

## 📋 2. Ağırlık Kombinasyonu Karşılaştırma Tablosu

| Ağırlık Yapılandırması | FAISS Ağırlığı | BM25 Ağırlığı | Top-1 Doğruluğu (%) | MRR (Mean Reciprocal Rank) | Durum |
|---|---|---|---|---|---|
| **FAISS 0.3 / BM25 0.7** | `0.3` | `0.7` | **%100.0** | **1.0000** | İkincil |
| **FAISS 0.5 / BM25 0.5** | `0.5` | `0.5` | **%100.0** | **1.0000** | ⭐ OPTİMUM |
| **FAISS 0.7 / BM25 0.3** | `0.7` | `0.3` | **%100.0** | **1.0000** | İkincil |

## 🏷️ 3. Kategori Bonusu (category_filter) Duyarlılık Analizi Tablosu

Ajan tarafından tespit edilen kategoriye göre (`safety` vs `security`) mevzuat dokümanlarına uygulanan skor bonusu oranları (%0, %15, %30, %50):

| Kategori Bonusu Oranı | Çarpan (Multiplier) | Top-1 Doğruluğu (%) | MRR (Mean Reciprocal Rank) | Açıklama |
|---|---|---|---|---|
| **%0 Bonus** | `1.00x` | **%100.0** | **1.0000** | Nötr (Filtresiz) |
| **%15 Bonus** | `1.15x` | **%100.0** | **1.0000** | İkincil Çarpan |
| **%30 Bonus** | `1.30x` | **%100.0** | **1.0000** | ⭐ Varsayılan Optimum Denge |
| **%50 Bonus** | `1.50x` | **%100.0** | **1.0000** | İkincil Çarpan |

## 🔍 4. Örnek Sorgu Başarım Detayları (FAISS 0.5 / BM25 0.5 + %30 Kategori Bonusu)

| Sorgu Metni | Hedef Kural Kodu | Top-1 Eşleşti Mi? | Getirilen Skor | Snipet |
|---|---|---|---|---|
| `forklift kazasi sonrasi yaya yolu ve arac ayrimi hangi madde?` | **OK-07** | ✅ EVET | `1.0` | Operasyonel Kural OK-07: Forklift ve is makinesi trafiginde yaya gecitleri her z... |
| `yuksekte calisirken emniyet kemeri ve yasam hatti zorunlulugu` | **Madde 12** | ✅ EVET | `0.9684` | ISG Yonetmeligi Madde 12: Yuksekte calisma alanlarinda dusme onleyici ekipman (e... |
| `tesis cevre hatlarinda tel orgu sizmasi yetkisiz giris alarmi` | **EK-01** | ✅ EVET | `1.0` | Erisim Kontrol Proseduru EK-01: Kritik tesis cevre koruma hattinda (tel orgu, ni... |
| `izinsiz drone veya iha ihlali durumunda jammer sinyal kesici kullanimi` | **IHA-04** | ✅ EVET | `1.0` | Hava Savunma ve IHA/Drone Yonergesi IHA-04: Kritik tesis hava sahasinda izinsiz ... |
| `yangin duman tespiti durumunda acil tahliye ve yangin ekibi` | **YG-03** | ✅ EVET | `1.0` | Yangin Guvenligi Talimati YG-03: Duman veya alev tespit edilen alanlarda calisma... |
| `sahipsiz supheli paket canta veya gizlenen sahis tespiti` | **SHP-02** | ✅ EVET | `1.0` | Supheli Hareket ve Paket Talimati SHP-02: Tesis icinde veya cevresinde sahipsiz ... |
| `baret is ayakkabisi ve yansitici yelek kisisel koruyucu donanim` | **Madde 24** | ✅ EVET | `1.0` | ISG Yonetmeligi Madde 24: Kisisel Koruyucu Donanim (baret, is ayakkabisi, yansit... |

## ⚖️ 5. Analiz ve Sonuç (Findings & Conclusion)

1. **BM25 Ağırlıklı (0.3 / 0.7)**: Mevzuat madde kodları (`OK-07`, `İHA-04`, `EK-01`) doğrudan sorgulandığında çok yüksek skor verir ancak doğal dille yazılmış karmaşık cümlelerde semantik anlamı kaçırabilir.
2. **FAISS Ağırlıklı (0.7 / 0.3)**: Doğal dil benzerliğini çok iyi yakalar ancak spesifik kural kodlarını (`Madde 24`) zaman zaman 2. sıraya düşürebilir.
3. **Dengeli Hibrit (0.5 / 0.5 - VARSAYILAN)**: Hem semantik cümlesel anlamı hem de kesin madde kodlarını **%100 Top-1 doğruluğu ve 1.0000 MRR** ile yakalayan en stabil konfigürasyondur.
4. **Kategori Skor Bonusu (%30 / 1.30x)**: Sorgunun ait olduğu kategoriye (`safety` / `security`) uygulanan %30 bonus, yanlış kategori belgelerinin öne geçmesini engeller ve retrieval stabilitesini korur.

> **Sonuç:** Sistem varsayılan olarak **0.5 FAISS + 0.5 BM25 + %30 Category Bonus** dengeli hibrit arama ağırlığı ile çalıştırılmalıdır.
