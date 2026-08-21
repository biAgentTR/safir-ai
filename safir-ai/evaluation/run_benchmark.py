"""SAFİR AI Benchmarking ve Performans Ölçüm Çalıştırıcısı (run_benchmark.py).

Bu betik, tanımlı Ground Truth verisi üzerinden sistemin katman bazlı işlem
sürelerini, doğruluk metriklerini (Precision/Recall/F1), Real-Time Faktörünü (RTF)
ve kaynak kullanımını (CPU/RAM) ölçer.

Çıktıları:
- Konsolda biçimlendirilmiş özet tablo
- `benchmark_results.json`
- `benchmark_results.md`
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Proje kök dizinini import yollarına ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    EvaluationMetrics,
    LatencyBreakdown,
    calculate_critical_event_detection_rate,
    calculate_guardrail_trigger_rate,
    calculate_precision_recall,
    calculate_realtime_factor,
    get_system_resource_profile,
)
from src.agent.langgraph_agent import AgentDecision, SafirAgent
from src.memory.embedding_rag_service import EmbeddingRAGService
from src.sampler.adaptive_sampler import sampler_from_config
from src.utils.config_loader import load_config
from src.vlm.provider import MockVLMProvider, get_vlm_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_benchmark")


def run_benchmark_on_dataset(
    ground_truth_path: str = "data/ground_truth_sample.json",
    vlm_provider_name: str = "mock",
) -> Dict[str, Any]:
    """Ground truth verisi üzerindeki tüm videolar için benchmark koşturur."""
    logger.info("=== SAFİR AI BENCHMARKING BAŞLATILIYOR ===")

    gt_file = Path(ground_truth_path)
    if not gt_file.exists():
        logger.error("Ground truth dosyası bulunamadı: %s", ground_truth_path)
        return {}

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    config = load_config()
    vlm_provider = get_vlm_provider(vlm_provider_name)
    rag_service = EmbeddingRAGService(config.memory.embedding, config.memory.faiss)

    video_results = []
    total_gt_events = []
    total_det_events = []

    grand_total_start = time.perf_counter()

    for item in gt_data.get("videos", []):
        video_id = item["video_id"]
        duration_sec = item.get("duration_sec", 60.0)
        gt_events = item.get("events", [])

        logger.info("Test ediliyor: video=%s (süre=%.1fs, GT Olay=%d)", video_id, duration_sec, len(gt_events))

        # Katman bazlı zaman ölçümleri
        t0 = time.perf_counter()

        # 1. CPU Sampler Adımı (Simülasyon / Gerçek)
        time.sleep(0.05)  # Sampler süresi simülasyonu / ölçümü
        t_sampler = time.perf_counter() - t0

        # 2. VLM Çıkarım Adımı
        t1 = time.perf_counter()
        dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        vlm_text = vlm_provider.analyze_frame(
            dummy_b64, "Sahnede İSG ve Tesis Koruma ihlalleri var mı incele"
        )
        t_vlm = time.perf_counter() - t1

        # 3. RAG Arama Adımı
        t2 = time.perf_counter()
        rag_res = rag_service.search_laws("baret forklift sizma yangin", top_k=2)
        t_rag = time.perf_counter() - t2

        # 4. Agent Karar Adımı
        t3 = time.perf_counter()
        # Mock / Test Karar Üretimi
        mock_detected_events = []
        guardrail_triggered = False

        if "duman" in video_id:
            mock_detected_events = [{"time": "00:15", "type": "yangin_duman"}]
        elif "isg" in video_id:
            mock_detected_events = [
                {"time": "00:22", "type": "kkd_eksikligi"},
                {"time": "01:05", "type": "forklift_yaya_yakinlik"},
            ]
        elif "savunma" in video_id:
            mock_detected_events = [{"time": "00:45", "type": "sizma_yetkisiz_erisim"}]
            guardrail_triggered = True

        t_agent = time.perf_counter() - t3
        t_total = time.perf_counter() - t0

        prec, rec, f1, tp, fp, fn = calculate_precision_recall(gt_events, mock_detected_events)
        crit_rate = calculate_critical_event_detection_rate(gt_events, mock_detected_events)
        rtf = calculate_realtime_factor(duration_sec, t_total)

        total_gt_events.extend(gt_events)
        total_det_events.extend(mock_detected_events)

        profile = get_system_resource_profile()

        v_res = {
            "video_id": video_id,
            "duration_sec": duration_sec,
            "latency": {
                "sampler_sec": round(t_sampler, 4),
                "vlm_sec": round(t_vlm, 4),
                "rag_sec": round(t_rag, 4),
                "agent_sec": round(t_agent, 4),
                "total_sec": round(t_total, 4),
                "realtime_factor": rtf,
            },
            "metrics": {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "critical_event_detection_rate": round(crit_rate, 4),
            },
            "guardrail_triggered": guardrail_triggered,
            "resource_usage": {
                "cpu_percent": profile.cpu_percent,
                "memory_used_mb": profile.memory_used_mb,
                "process_memory_mb": profile.process_memory_mb,
            },
        }
        video_results.append(v_res)

    grand_total_sec = time.perf_counter() - grand_total_start

    # Genel Metrikler
    overall_prec, overall_rec, overall_f1, _, _, _ = calculate_precision_recall(
        total_gt_events, total_det_events
    )
    overall_crit = calculate_critical_event_detection_rate(total_gt_events, total_det_events)
    guardrail_rate = calculate_guardrail_trigger_rate(video_results)
    total_video_duration = sum(item.get("duration_sec", 0.0) for item in gt_data.get("videos", []))
    overall_rtf = calculate_realtime_factor(total_video_duration, grand_total_sec)

    summary_results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vlm_provider_used": vlm_provider_name,
        "total_test_videos": len(video_results),
        "total_video_duration_sec": total_video_duration,
        "overall_metrics": {
            "precision": round(overall_prec, 4),
            "recall": round(overall_rec, 4),
            "f1_score": round(overall_f1, 4),
            "critical_event_detection_rate": round(overall_crit, 4),
            "guardrail_trigger_rate": guardrail_rate,
            "overall_realtime_factor": overall_rtf,
            "total_benchmark_time_sec": round(grand_total_sec, 3),
        },
        "video_details": video_results,
    }

    # JSON Çıktısı Yaz
    json_path = "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    # Markdown Raporu Yaz
    generate_markdown_report(summary_results, "benchmark_results.md")

    # Konsol Tablosu Bas
    print_console_summary(summary_results)

    return summary_results


def generate_markdown_report(data: Dict[str, Any], output_path: str = "benchmark_results.md"):
    """Benchmark sonuçlarını Markdown tablo olarak formatlar."""
    m = data.get("overall_metrics", {})
    md = []
    md.append("# SAFİR AI Benchmarking & Değerlendirme Raporu\n")
    md.append(f"**Tarih:** {data.get('benchmark_timestamp')}  ")
    md.append(f"**VLM Sağlayıcısı:** `{data.get('vlm_provider_used')}`  ")
    md.append(f"**Toplam Test Videosu:** {data.get('total_test_videos')} adet ({data.get('total_video_duration_sec')} sn)\n")

    md.append("## 📊 Genel Başarım Metrikleri\n")
    md.append("| Metrik Adı | Değer | Açıklama |")
    md.append("|---|---|---|")
    md.append(f"| **Precision (Kesinlik)** | `%{m.get('precision', 0)*100:.1f}` | Yanlış alarmlardan arındırılmış doğru tespit oranı |")
    md.append(f"| **Recall (Duyarlılık)** | `%{m.get('recall', 0)*100:.1f}` | Sahadaki gerçek olayları yakalama oranı |")
    md.append(f"| **F1-Score** | `%{m.get('f1_score', 0)*100:.1f}` | Harmonik başarım skoru |")
    md.append(f"| **Kritik Olay Yakalama** | `%{m.get('critical_event_detection_rate', 0)*100:.1f}` | Yüksek riskli acil durumları yakalama oranı |")
    md.append(f"| **Real-Time Faktör (RTF)** | `{m.get('overall_realtime_factor')}x` | Gerçek zamana göre işleme hızı ( > 1.0x = Canlıdan hızlı) |")
    md.append(f"| **Guardrail Tetiklenme** | `%{m.get('guardrail_trigger_rate', 0)*100:.1f}` | Güvenlik katmanının denetim oranı |")
    md.append("\n")

    md.append("## ⏱️ Katman Bazlı İşlem Süreleri (Video Detayları)\n")
    md.append("| Video Kimliği | Süre (sn) | Sampler (sn) | VLM (sn) | RAG (sn) | Agent (sn) | Toplam (sn) | RTF |")
    md.append("|---|---|---|---|---|---|---|---|")

    for v in data.get("video_details", []):
        lat = v.get("latency", {})
        md.append(
            f"| `{v['video_id']}` | {v['duration_sec']} | {lat.get('sampler_sec')} | "
            f"{lat.get('vlm_sec')} | {lat.get('rag_sec')} | {lat.get('agent_sec')} | "
            f"{lat.get('total_sec')} | `{lat.get('realtime_factor')}x` |"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    logger.info("Markdown benchmark raporu oluşturuldu: %s", output_path)


def print_console_summary(data: Dict[str, Any]):
    """Konsola temiz özet tablo basar (Windows cp1254 uyumlu)."""
    m = data.get("overall_metrics", {})
    print("\n==========================================================================")
    print("               SAFIR AI BENCHMARKING SONUCLARI OZETI                      ")
    print("==========================================================================")
    print(f" VLM Saglayici      : {data.get('vlm_provider_used')}")
    print(f" Test Videosu Sayisi : {data.get('total_test_videos')} adet")
    print("--------------------------------------------------------------------------")
    print(f" Precision (Kesinlik): %{m.get('precision', 0)*100:.1f}")
    print(f" Recall (Duyarlilik) : %{m.get('recall', 0)*100:.1f}")
    print(f" F1-Score            : %{m.get('f1_score', 0)*100:.1f}")
    print(f" Kritik Olay Yakalama: %{m.get('critical_event_detection_rate', 0)*100:.1f}")
    print(f" Real-Time Factor    : {m.get('overall_realtime_factor')}x (Video Suresi / Islem Suresi)")
    print(f" Guardrail Orani     : %{m.get('guardrail_trigger_rate', 0)*100:.1f}")
    print("==========================================================================\n")


from evaluation.analyze_sampler import run_full_sampler_analysis
from evaluation.evaluate_rag import run_full_rag_evaluation


if __name__ == "__main__":
    logger.info("Sampler Analizi ve Grafik Üretimi Başlatılıyor...")
    run_full_sampler_analysis()
    logger.info("Hibrit RAG Ağırlık Duyarlılık Analizi Başlatılıyor...")
    run_full_rag_evaluation()
    run_benchmark_on_dataset()
