"""SAFIR KPI benchmark harness: olay tespit dogrulugu, kritik yakalama, gecikme.

Etiketli klipler uzerinde tum pipeline'i calistirir ve sartname-uyumlu KPI'lari
raporlar: olay tespiti precision/recall/F1 (kategori bazli + makro/mikro),
kritik olay yakalama orani ve klip basi gecikme.

Kullanim (repo kokunden veya safir-ai/ icinden):
    # Hazir sentetik demo (offline, mock; harness'i ve metrikleri gosterir):
    python scripts/benchmark.py --synthetic --mock

    # Gercek klipler uzerinde (Gemini/vLLM backend'i config'te ayarliysa):
    python scripts/benchmark.py --manifest benchmarks/manifest.json --out benchmarks/result.json

Manifest bicimi (JSON dizisi):
    [
      {"video": "data/clip1.mp4", "expected_events": ["arac_yaya_yakinligi"], "critical": true},
      {"video": "data/clip2.mp4", "expected_events": ["kkd_ihlali"]}
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from src.eval.metrics import BenchmarkResult, ClipRecord, compute_event_metrics  # noqa: E402

_CRITICAL_LEVELS = {"yuksek", "kritik"}


def _make_synthetic_dataset(base_dir: Path) -> List[dict]:
    """Offline demo icin sentetik videolar + beklenen etiketli bir manifest uretir.

    Mock backend, sabit "forklift ... korumasiz alan" tasviri dondurdugu icin
    'arac_yaya_yakinligi' beklendiginde isabet, 'yangin_duman' beklendiginde
    iskalama gozlenir; boylece metrikler onemsiz (hepsi 1.0) olmaz ve harness'in
    dogru hesapladigi gorulur.
    """
    import cv2
    import numpy as np

    def _write(path: Path) -> None:
        frames = []
        for i in range(60):
            frame = np.full((48, 64, 3), 30, dtype=np.uint8)
            if 20 <= i < 30:
                cv2.rectangle(frame, (5, 5), (58, 42), (255, 255, 255), -1)
            frames.append(frame)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
        for frame in frames:
            writer.write(frame)
        writer.release()

    manifest = []
    for name, expected, critical in [
        ("clip_forklift.mp4", ["arac_yaya_yakinligi"], False),
        ("clip_fire.mp4", ["yangin_duman"], True),
    ]:
        path = base_dir / name
        _write(path)
        manifest.append({"video": str(path), "expected_events": expected, "critical": critical})
    return manifest


def _print_report(result: BenchmarkResult) -> None:
    print("\n" + "=" * 68)
    print(f"SAFIR KPI BENCHMARK — {result.n} klip")
    print("=" * 68)
    print(f"{'Kategori':<28}{'P':>8}{'R':>8}{'F1':>8}{'TP/FP/FN':>14}")
    print("-" * 68)
    for event_type, metric in sorted(result.per_type.items()):
        print(
            f"{event_type:<28}{metric.precision:>8.2f}{metric.recall:>8.2f}{metric.f1:>8.2f}"
            f"{f'{metric.tp}/{metric.fp}/{metric.fn}':>14}"
        )
    print("-" * 68)
    print(f"{'MAKRO':<28}{result.macro_precision:>8.2f}{result.macro_recall:>8.2f}{result.macro_f1:>8.2f}")
    print(f"{'MIKRO':<28}{result.micro_precision:>8.2f}{result.micro_recall:>8.2f}{result.micro_f1:>8.2f}")
    print("-" * 68)
    print(
        f"Kritik olay yakalama: {result.critical_caught}/{result.critical_total} "
        f"(%{result.critical_catch_rate * 100:.1f})"
    )
    print(f"Gecikme (klip basi): ort={result.latency_ms_mean:.0f} ms, medyan={result.latency_ms_p50:.0f} ms")
    print("=" * 68)


def main() -> int:
    parser = argparse.ArgumentParser(description="SAFIR KPI benchmark.")
    parser.add_argument("--manifest", type=str, default=None, help="Etiketli klip manifesti (JSON).")
    parser.add_argument("--synthetic", action="store_true", help="Hazir sentetik demo veri seti uret.")
    parser.add_argument("--mock", action="store_true", help="VLM+LLM'i mock'a zorla (offline).")
    parser.add_argument("--real-rag", action="store_true", help="Gercek Embedding+FAISS RAG kullan (model indirir).")
    parser.add_argument("--out", type=str, default=None, help="Sonuc JSON'unun yazilacagi yol.")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.WARNING)

    import src.main as main_module
    from src.utils.config_loader import load_config

    if not args.real_rag:
        from scripts.e2e_smoke import _FakeRagService  # ayni sahte RAG'i yeniden kullan

        main_module.EmbeddingRAGService = lambda *a, **k: _FakeRagService()  # type: ignore[assignment]

    config = load_config()
    if args.mock:
        config = config.model_copy(
            update={"app": config.app.model_copy(update={"use_mock_vlm": True, "use_mock_llm": True})}
        )

    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    if args.synthetic:
        tmp_dir = tempfile.TemporaryDirectory()
        manifest = _make_synthetic_dataset(Path(tmp_dir.name))
    elif args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    else:
        parser.error("--manifest veya --synthetic verilmelidir.")

    print(f"Backend -> VLM: {config.vlm.active_model} | LLM: {config.llm.active_model} | mock={args.mock}")
    pipeline = main_module.SafirPipeline(config)

    records: List[ClipRecord] = []
    try:
        for entry in manifest:
            video = entry["video"]
            expected = set(entry.get("expected_events", []))
            expected_critical = bool(entry.get("critical", False)) or (
                entry.get("expected_risk_level") in _CRITICAL_LEVELS
            )

            started = time.perf_counter()
            report = pipeline.run(video, entry.get("prompt", "Sahnede riskli bir durum var mi degerlendir."))
            latency_ms = (time.perf_counter() - started) * 1000.0

            detected = set(report.detected_event_types)
            print(
                f"  - {Path(video).name}: beklenen={sorted(expected)} "
                f"tespit={sorted(detected)} risk={report.risk_level} ({latency_ms:.0f} ms)"
            )
            records.append(
                ClipRecord(
                    expected=expected,
                    detected=detected,
                    expected_critical=expected_critical,
                    detected_critical=report.risk_level in _CRITICAL_LEVELS,
                    latency_ms=latency_ms,
                )
            )
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    result = compute_event_metrics(records)
    _print_report(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n": result.n,
            "macro": {"precision": result.macro_precision, "recall": result.macro_recall, "f1": result.macro_f1},
            "micro": {"precision": result.micro_precision, "recall": result.micro_recall, "f1": result.micro_f1},
            "critical_catch_rate": result.critical_catch_rate,
            "latency_ms_mean": result.latency_ms_mean,
            "latency_ms_p50": result.latency_ms_p50,
            "per_type": {
                t: {"precision": m.precision, "recall": m.recall, "f1": m.f1, "tp": m.tp, "fp": m.fp, "fn": m.fn}
                for t, m in result.per_type.items()
            },
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSonuc yazildi: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
