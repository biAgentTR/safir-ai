"""#1 KPI metrik hesaplayicisi (src/eval/metrics.py) icin deterministik birim testleri."""

from __future__ import annotations

from src.eval.metrics import ClipRecord, compute_event_metrics


def test_perfect_detection_gives_full_scores() -> None:
    records = [
        ClipRecord(expected={"kkd_ihlali"}, detected={"kkd_ihlali"}, latency_ms=100.0),
        ClipRecord(expected={"yangin_duman"}, detected={"yangin_duman"}, latency_ms=200.0),
    ]
    result = compute_event_metrics(records)
    assert result.micro_precision == 1.0
    assert result.micro_recall == 1.0
    assert result.macro_f1 == 1.0
    assert result.latency_ms_mean == 150.0
    assert result.latency_ms_p50 == 150.0


def test_counts_tp_fp_fn_per_type() -> None:
    # klip1: forklift dogru; yangin bekleniyor ama forklift tespit (FP forklift? hayir):
    records = [
        ClipRecord(expected={"arac_yaya_yakinligi"}, detected={"arac_yaya_yakinligi"}),  # TP
        ClipRecord(expected={"yangin_duman"}, detected={"arac_yaya_yakinligi"}),  # FN yangin, FP forklift
    ]
    result = compute_event_metrics(records)
    forklift = result.per_type["arac_yaya_yakinligi"]
    fire = result.per_type["yangin_duman"]
    assert (forklift.tp, forklift.fp, forklift.fn) == (1, 1, 0)
    assert (fire.tp, fire.fp, fire.fn) == (0, 0, 1)
    assert forklift.precision == 0.5
    assert forklift.recall == 1.0


def test_critical_catch_rate() -> None:
    records = [
        ClipRecord(expected=set(), detected=set(), expected_critical=True, detected_critical=True),
        ClipRecord(expected=set(), detected=set(), expected_critical=True, detected_critical=False),
        ClipRecord(expected=set(), detected=set(), expected_critical=False, detected_critical=True),
    ]
    result = compute_event_metrics(records)
    assert result.critical_total == 2
    assert result.critical_caught == 1
    assert result.critical_catch_rate == 0.5


def test_empty_records_are_safe() -> None:
    result = compute_event_metrics([])
    assert result.n == 0
    assert result.macro_f1 == 0.0
    assert result.critical_catch_rate == 0.0
