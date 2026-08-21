"""Birim/Entegrasyon Testi: Sampler Katmanı Çoklu-Çalıştırma Kararlılık (Multi-Run Stability) Testi.

Bu test, `AdaptiveFrameSampler` ve Önem Ağırlıklı Seyreltme (Importance-Weighted Pinned Frame Decimation)
mantığının aynı video girdisi üzerinde N kez tekrarlı çalıştırıldığında onset zaman damgası varyansı
üretmediğini (`variance == 0.0s`) ve kritik trend karelerinin (`has_gradual_trend=True`) seyreltmede
korunduğunu doğrular.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.sampler.adaptive_sampler import AdaptiveFrameSampler
from src.utils.config_loader import load_config


def test_sampler_importance_weighted_decimation_preserves_trend_frames():
    """Seyreltme sırasında has_gradual_trend=True karelerin korunduğunu doğrular."""
    config = load_config()
    sampler = AdaptiveFrameSampler(
        min_change_threshold=config.sampler.min_change_threshold,
        max_sampling_gap_sec=config.sampler.max_sampling_gap_sec,
        max_evidence_buffer=10,  # Kasıtlı olarak küçük tampon seçerek seyreltmeyi zorla
        enable_contrast_check=True,
        contrast_change_threshold=0.020,
    )

    video_path = "data/uploads/yaz_c_-duman.mp4"
    if not Path(video_path).exists():
        pytest.skip(f"Test videosu bulunamadı: {video_path}")

    evidence_frames = sampler.process_video(video_path, sample_fps=5)

    # Tampon sınırı (10) aşılmamış olmalı
    assert len(evidence_frames) <= 10

    # Trend taşıyan kritik karelerin seyreltmede korunduğunu doğrula
    trend_frames = [ef for ef in evidence_frames if ef.has_gradual_trend]
    assert len(trend_frames) > 0, "Kademeli trend kareleri önem ağırlıklı seyreltmede silinmemeli!"


def test_sampler_multi_run_onset_stability():
    """Aynı gerçek video üzerinde N kez çalıştırmada onset zaman damgası kararlılığını ölçer."""
    config = load_config()
    video_path = "data/uploads/yaz_c_-duman.mp4"
    if not Path(video_path).exists():
        pytest.skip(f"Test videosu bulunamadı: {video_path}")

    onsets = []
    for _ in range(3):
        sampler = AdaptiveFrameSampler(
            min_change_threshold=config.sampler.min_change_threshold,
            max_sampling_gap_sec=config.sampler.max_sampling_gap_sec,
            max_evidence_buffer=50,
            enable_contrast_check=True,
            contrast_change_threshold=0.020,
        )
        evidence = sampler.process_video(video_path, sample_fps=5)
        # Onset timestamp of first evidence frame past 20.0s
        first_smoke_frame = next((ef for ef in evidence if ef.timestamp_sec >= 20.0), None)
        assert first_smoke_frame is not None
        onsets.append(first_smoke_frame.timestamp_sec)

    # 3 çalıştırmadaki onset değerlerinin tamamen aynı olduğunu doğrula (variance == 0)
    assert len(set(onsets)) == 1, f"Multi-run onset zaman damgası kararsız! Elde edilenler: {onsets}"
