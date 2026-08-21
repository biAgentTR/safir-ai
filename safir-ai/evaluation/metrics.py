"""SAFİR AI Değerlendirme ve Metrik Ölçüm Modülü (Evaluation & Benchmarking).

Bu modül, yarışma şartnamesine tam uyumlu şekilde sistemin doğruluğunu,
işlem hızını, kaynak kullanımını ve güvenlik katmanı (Guardrail) etkinliğini
ölçmek için tanımlanmış metrikleri içerir.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class ResourceProfile:
    """Sistem kaynak kullanımı profili (CPU, Bellek, Disk IO)."""

    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    process_memory_mb: float = 0.0


@dataclass
class LatencyBreakdown:
    """Katman bazlı işlem süreleri (saniye)."""

    sampler_elapsed_sec: float = 0.0
    vlm_elapsed_sec: float = 0.0
    rag_elapsed_sec: float = 0.0
    agent_elapsed_sec: float = 0.0
    total_pipeline_elapsed_sec: float = 0.0
    realtime_factor: float = 0.0  # (Video Süresi / Toplam İşlem Süresi)


@dataclass
class EvaluationMetrics:
    """Tek bir video veya tüm veri seti için hesaplanan doğruluk metrikleri."""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    critical_detection_rate: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_ground_truth_events: int = 0
    total_detected_events: int = 0
    guardrail_trigger_rate: float = 0.0


def parse_timestamp_to_seconds(ts: Any) -> float:
    """'MM:SS' veya sayısal zaman damgasını saniyeye çevirir."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        parts = ts.strip().split(":")
        if len(parts) == 2:
            try:
                return float(parts[0]) * 60.0 + float(parts[1])
            except ValueError:
                pass
        try:
            return float(ts)
        except ValueError:
            pass
    return 0.0


def calculate_precision_recall(
    ground_truth_events: List[Dict[str, Any]],
    detected_events: List[Dict[str, Any]],
    tolerance_sec: float = 5.0,
) -> Tuple[float, float, float, int, int, int]:
    """Ground truth olayları ile sistem çıktılarını zaman damgalarına göre eşleştirip
    Precision, Recall ve F1-Score hesaplar.

    Args:
        ground_truth_events: Ground truth JSON'daki olay listesi.
        detected_events: Sistem çıktısındaki tespit edilen olay listesi.
        tolerance_sec: Eşleşme kabul edilecek zaman damgası toleransı (varsayılan ±5 sn).

    Returns:
        (precision, recall, f1_score, true_positives, false_positives, false_negatives)
    """
    if not ground_truth_events and not detected_events:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if not ground_truth_events:
        return 0.0, 1.0, 0.0, 0, len(detected_events), 0
    if not detected_events:
        return 1.0, 0.0, 0.0, 0, 0, len(ground_truth_events)

    matched_gt = set()
    true_positives = 0

    for det in detected_events:
        det_sec = parse_timestamp_to_seconds(det.get("timestamp", det.get("time", 0.0)))
        det_type = str(det.get("type", det.get("event", ""))).lower()

        best_gt_idx = None
        min_diff = float("inf")

        for idx, gt in enumerate(ground_truth_events):
            if idx in matched_gt:
                continue
            gt_sec = parse_timestamp_to_seconds(gt.get("timestamp", gt.get("time", 0.0)))
            gt_type = str(gt.get("type", gt.get("event", ""))).lower()

            diff = abs(det_sec - gt_sec)
            # Tip eşleşiyorsa veya genel eşleşme yapılıyorsa
            type_match = not det_type or not gt_type or (det_type in gt_type or gt_type in det_type)

            if diff <= tolerance_sec and type_match and diff < min_diff:
                min_diff = diff
                best_gt_idx = idx

        if best_gt_idx is not None:
            matched_gt.add(best_gt_idx)
            true_positives += 1

    false_positives = len(detected_events) - true_positives
    false_negatives = len(ground_truth_events) - true_positives

    precision = true_positives / len(detected_events) if detected_events else 0.0
    recall = true_positives / len(ground_truth_events) if ground_truth_events else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, true_positives, false_positives, false_negatives


def calculate_critical_event_detection_rate(
    ground_truth_events: List[Dict[str, Any]],
    detected_events: List[Dict[str, Any]],
    tolerance_sec: float = 5.0,
) -> float:
    """Yalnızca kritik/yüksek riskli olayların yakalanma oranını (Critical Event Recall) hesaplar.

    Args:
        ground_truth_events: Ground truth olay listesi (`is_critical: true` olanlar süzülür).
        detected_events: Tespit edilen olaylar.
        tolerance_sec: Zaman toleransı.

    Returns:
        Kritik olay yakalanma oranı (0.0 - 1.0).
    """
    critical_gt = [e for e in ground_truth_events if e.get("is_critical", False) or e.get("severity") in ("yuksek", "kritik")]
    if not critical_gt:
        # Eğer hiç kritik olay tanımlanmamışsa tüm GT olaylarını kritik kabul et
        critical_gt = ground_truth_events

    if not critical_gt:
        return 1.0

    _, recall, _, _, _, _ = calculate_precision_recall(critical_gt, detected_events, tolerance_sec=tolerance_sec)
    return recall


def calculate_realtime_factor(video_duration_sec: float, total_processing_time_sec: float) -> float:
    """Real-Time Faktörünü (RTF) hesaplar: (Video Süresi / İşlem Süresi).
    RTF > 1.0 ise sistem gerçek zamandan DAHA HIZLI işliyor demektir.
    """
    if total_processing_time_sec <= 0:
        return 0.0
    return round(video_duration_sec / total_processing_time_sec, 2)


def calculate_guardrail_trigger_rate(results_list: List[Dict[str, Any]]) -> float:
    """Tüm analizler içindeki ikincil güvenlik katmanının (Guardrail) tetiklenme oranını hesaplar."""
    if not results_list:
        return 0.0
    triggered_count = sum(1 for r in results_list if r.get("guardrail_triggered", False))
    return round(triggered_count / len(results_list), 4)


def get_system_resource_profile() -> ResourceProfile:
    """`psutil` kullanarak anlık sistem CPU ve RAM profilini okur."""
    if psutil is None:
        return ResourceProfile()

    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        sys_mem = psutil.virtual_memory()

        return ResourceProfile(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_used_mb=round(sys_mem.used / (1024 * 1024), 2),
            memory_percent=sys_mem.percent,
            process_memory_mb=round(mem_info.rss / (1024 * 1024), 2),
        )
    except Exception as exc:
        logger.warning("Resource profiling hatası: %s", exc)
        return ResourceProfile()
