"""KPI metrik hesaplayicilari (saf, test edilebilir fonksiyonlar).

Sartname, katilimcilarin kendi metriklerini tanimlamasini ister (olay tespit
dogrulugu, kritik olay yakalama orani, islem suresi vb.). Bu modul, pipeline'i
CALISTIRMADAN — yalnizca (beklenen, tespit edilen) etiket kumeleri ve gecikme
listeleri uzerinden — bu metrikleri deterministik olarak hesaplar; boylece
`scripts/benchmark.py`den bagimsiz olarak birim testi yapilabilir.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set


@dataclass
class TypeMetric:
    """Tek bir olay kategorisi icin tespit sayaclari ve turetilmis metrikler."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class BenchmarkResult:
    """Bir benchmark kosusunun toplulastirilmis KPI sonuclari."""

    n: int
    per_type: Dict[str, TypeMetric] = field(default_factory=dict)
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    critical_total: int = 0
    critical_caught: int = 0
    critical_catch_rate: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_p50: float = 0.0


@dataclass
class ClipRecord:
    """Tek bir klip icin beklenen/tespit edilen etiketler + kritiklik + gecikme."""

    expected: Set[str]
    detected: Set[str]
    expected_critical: bool = False
    detected_critical: bool = False
    latency_ms: float = 0.0


def _micro(per_type: Dict[str, TypeMetric]) -> TypeMetric:
    total = TypeMetric()
    for metric in per_type.values():
        total.tp += metric.tp
        total.fp += metric.fp
        total.fn += metric.fn
    return total


def compute_event_metrics(records: Sequence[ClipRecord]) -> BenchmarkResult:
    """Klip kayitlarindan olay-tespit KPI'larini (precision/recall/F1, kritik yakalama, gecikme) hesaplar.

    Cok-etiketli (multi-label) degerlendirme: her klip icin her kategori,
    beklenen ve/veya tespit edilen kumelerde bulunusuna gore TP/FP/FN sayilir.

    Args:
        records: Her klip icin beklenen/tespit edilen kategori kumeleri,
            kritiklik bayraklari ve gecikme (ms).

    Returns:
        Kategori bazli ve makro/mikro toplulastirilmis metrikleri, kritik olay
        yakalama oranini ve gecikme ozetini iceren `BenchmarkResult`.
    """
    per_type: Dict[str, TypeMetric] = {}

    def _slot(event_type: str) -> TypeMetric:
        return per_type.setdefault(event_type, TypeMetric())

    for record in records:
        for event_type in record.expected | record.detected:
            metric = _slot(event_type)
            in_expected = event_type in record.expected
            in_detected = event_type in record.detected
            if in_expected and in_detected:
                metric.tp += 1
            elif in_detected and not in_expected:
                metric.fp += 1
            elif in_expected and not in_detected:
                metric.fn += 1

    macro_p = statistics.mean([m.precision for m in per_type.values()]) if per_type else 0.0
    macro_r = statistics.mean([m.recall for m in per_type.values()]) if per_type else 0.0
    macro_f1 = statistics.mean([m.f1 for m in per_type.values()]) if per_type else 0.0
    micro = _micro(per_type)

    critical_total = sum(1 for r in records if r.expected_critical)
    critical_caught = sum(1 for r in records if r.expected_critical and r.detected_critical)
    critical_rate = critical_caught / critical_total if critical_total else 0.0

    latencies = [r.latency_ms for r in records if r.latency_ms > 0]

    return BenchmarkResult(
        n=len(records),
        per_type=per_type,
        macro_precision=round(macro_p, 4),
        macro_recall=round(macro_r, 4),
        macro_f1=round(macro_f1, 4),
        micro_precision=round(micro.precision, 4),
        micro_recall=round(micro.recall, 4),
        micro_f1=round(micro.f1, 4),
        critical_total=critical_total,
        critical_caught=critical_caught,
        critical_catch_rate=round(critical_rate, 4),
        latency_ms_mean=round(statistics.mean(latencies), 1) if latencies else 0.0,
        latency_ms_p50=round(statistics.median(latencies), 1) if latencies else 0.0,
    )
