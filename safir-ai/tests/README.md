# SAFİR AI — Çevrimdışı (Offline) Birim ve Entegrasyon Testleri

Bu dizin, **SAFİR AI Karar Ajanı, Escalation Mekanizması, VLM Sağlayıcıları ve RAG Katmanı** için %100 çevrimdışı (GPU veya harici API gerektirmeyen) pytest testlerini barındırır.

---

## 🧪 Test Dosyaları Düzeni

| Test Dosyası | Açıklama |
|---|---|
| `test_agent_scenarios.py` | 4 farklı senaryo (Yüksek risk, Düşük risk, Negasyon/Guardrail, Orta risk) için ajanın uçtan uca JSON çıktısı ve risk skoru testleri. |
| `test_escalation_integration.py` | Risk skoru $\ge 51$ olduğunda otonom saha alarmının (`POST /alerts/trigger`) tetiklendiğini, altında tetiklenmediğini doğrulayan entegrasyon testi (HTTP mock'lu). |
| `test_vlm_provider.py` | Provider-Agnostic VLM mimarisini (`VLMProvider`, `GeminiProvider`, `VLLMQwenProvider`, `MockVLMProvider`) test eder. |
| `mock_scenarios.py` | Testlerde kullanılan sabit senaryo verileri. |

---

## 🚀 Testleri Çalıştırma Komutları

Tüm testleri sanal ortam içinden çalıştırmak için:

```bash
cd safir-ai
.\.venv\Scripts\python.exe -m pytest
```

Belirli bir test dosyasını çalıştırmak için:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_agent_scenarios.py
.\.venv\Scripts\python.exe -m pytest tests/test_escalation_integration.py
```
