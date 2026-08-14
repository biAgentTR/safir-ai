"""`RepresentativeFrameExtractor` (src/sampler/context/representative_frame_extractor.py) icin birim testleri.

Kok neden Hata #4 duzeltmesi: KISA Olay Gruplarinda mevcut sabit ±pencere
(pre/peak/post, en fazla 3 kare) davranisi AYNEN korunurken, UZUN Olay
Gruplarinda (suresi `pre_event_sec + post_event_sec` toplamini asan) zirve
etrafinda sikismak yerine `start_time` -> `end_time` araligina dengeli
yayilan, ust sinirli (`max_representative_frames`) ve tekrarsiz bir kare
seti uretildigini dogrular (bkz. VLM Pipeline Kok Neden Analiz Raporu).

Tamamen sentetik, testin kendi urettigi videolar uzerinde calisir; GPU/aga/
baska bir module (clustering, VLM, Agent) bagimliligi yoktur.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.sampler.context.representative_frame_extractor import RepresentativeFrameExtractor
from src.sampler.schema import EvidenceFrame

_NATIVE_FPS = 25.0


def _write_video(path: Path, frame_count: int) -> None:
    """Sabit renkli, `frame_count` kareli, 25 fps'lik sentetik bir video yazar."""
    frame = np.full((48, 64, 3), 30, dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, _NATIVE_FPS, (64, 48))
    for _ in range(frame_count):
        writer.write(frame)
    writer.release()


def _evidence_frame(timestamp_sec: float) -> EvidenceFrame:
    """Testler icin, gercek bir video karesinden gecmeden sentetik bir zirve `EvidenceFrame` uretir."""
    frame_id = int(round(timestamp_sec * _NATIVE_FPS))
    minutes, seconds = divmod(int(timestamp_sec), 60)
    return EvidenceFrame(
        frame_id=frame_id,
        timestamp_sec=timestamp_sec,
        timestamp_str=f"{minutes:02d}:{seconds:02d}",
        change_score=1.0,
        image_bytes=b"\xff\xd8\xff",
        base64_image="data:image/jpeg;base64,AAAA",
        image_shape=(48, 64, 3),
    )


@pytest.fixture
def long_video(tmp_path: Path) -> str:
    """24 saniyelik (600 kare @ 25fps), uzun Olay Gruplarini test etmeye yetecek sentetik video."""
    path = tmp_path / "long.mp4"
    _write_video(path, frame_count=600)
    return str(path)


# --------------------------- KISA olay: mevcut davranis korunmali ---------------------------


def test_short_cluster_preserves_fixed_window_behavior(long_video: str) -> None:
    """Sure, `pre_event_sec + post_event_sec` esigini asmiyorsa en fazla 3 kare (pre/peak/post) uretilmeli."""
    extractor = RepresentativeFrameExtractor(pre_event_sec=2.0, post_event_sec=2.0)
    peak = _evidence_frame(10.0)

    result = extractor.extract(long_video, peak, start_time=9.0, end_time=11.5)  # sure=2.5s <= 4.0s esigi

    assert len(result) <= 3
    labels = {rf.label for rf in result}
    assert "peak" in labels
    assert labels <= {"pre-event", "peak", "post-event"}


def test_extract_without_cluster_times_falls_back_to_fixed_window(long_video: str) -> None:
    """`start_time`/`end_time` hic verilmezse (eski cagiran kodla geriye-donuk uyumluluk) sabit pencere kullanilmali."""
    extractor = RepresentativeFrameExtractor(pre_event_sec=2.0, post_event_sec=2.0)
    peak = _evidence_frame(10.0)

    result = extractor.extract(long_video, peak)

    assert len(result) <= 3
    assert any(rf.label == "peak" for rf in result)


# --------------------------- UZUN olay: cluster'a yayilma ---------------------------


def test_long_cluster_spreads_frames_across_start_to_end(long_video: str) -> None:
    """Sure esigi asarsa kareler zirve etrafinda sikismamali; start_time->end_time araligina yayilmali."""
    extractor = RepresentativeFrameExtractor(pre_event_sec=2.0, post_event_sec=2.0)
    peak = _evidence_frame(10.0)

    result = extractor.extract(long_video, peak, start_time=0.0, end_time=20.0)  # sure=20s > 4.0s esigi

    assert len(result) > 3, "Uzun olayda eski 3-kareli sabit pencereden fazlasi uretilmeli."
    timestamps = [rf.timestamp_sec for rf in result]
    assert timestamps == sorted(timestamps)
    # Ilk ve son kare, olayin baslangic/bitisine yakin olmali (eski ±2s pencereyle SINIRLI KALMAMALI).
    assert timestamps[0] <= 1.0
    assert timestamps[-1] >= 19.0
    # En az bir kare, eski sabit pencerenin (peak ± pre/post = [8.0, 12.0]) DISINDA olmali;
    # bu, olayin artik yalnizca zirve aninin degil, tamaminin temsil edildiginin kaniti.
    assert any(ts < 8.0 or ts > 12.0 for ts in timestamps)


def test_long_cluster_uses_actual_start_and_end_timestamps(long_video: str) -> None:
    """Uretilen kareler, cluster'in GERCEK start_time/end_time'ina gore konumlanmali (baska bir sabit degil)."""
    extractor = RepresentativeFrameExtractor(pre_event_sec=1.0, post_event_sec=1.0)
    peak = _evidence_frame(15.0)

    result = extractor.extract(long_video, peak, start_time=5.0, end_time=22.0)

    timestamps = [rf.timestamp_sec for rf in result]
    assert min(timestamps) >= 5.0 - (1.0 / _NATIVE_FPS)  # bir kare payi tolerans
    assert max(timestamps) <= 22.0 + (1.0 / _NATIVE_FPS)
    assert min(timestamps) <= 5.5  # baslangica yakin bir kare secilmis olmali
    assert max(timestamps) >= 21.0  # bitise yakin bir kare secilmis olmali


# --------------------------- Zirve karenin korunmasi ---------------------------


def test_peak_frame_is_always_included_in_short_cluster(long_video: str) -> None:
    extractor = RepresentativeFrameExtractor(pre_event_sec=2.0, post_event_sec=2.0)
    peak = _evidence_frame(10.0)

    result = extractor.extract(long_video, peak, start_time=9.0, end_time=11.0)

    peak_entries = [rf for rf in result if rf.label == "peak"]
    assert len(peak_entries) == 1
    assert peak_entries[0].timestamp_sec == peak.timestamp_sec
    assert peak_entries[0].base64_image == peak.base64_image


def test_peak_frame_is_always_included_in_long_cluster(long_video: str) -> None:
    extractor = RepresentativeFrameExtractor(pre_event_sec=2.0, post_event_sec=2.0)
    peak = _evidence_frame(10.0)

    result = extractor.extract(long_video, peak, start_time=0.0, end_time=20.0)

    peak_entries = [rf for rf in result if rf.label == "peak"]
    assert len(peak_entries) == 1
    assert peak_entries[0].timestamp_sec == peak.timestamp_sec
    assert peak_entries[0].base64_image == peak.base64_image


# --------------------------- Ust sinir ---------------------------


def test_max_representative_frames_upper_bound_is_respected(long_video: str) -> None:
    """Cok uzun bir olayda bile kare sayisi `max_representative_frames`i asmamali (GPU/API maliyet kontrolu)."""
    extractor = RepresentativeFrameExtractor(
        pre_event_sec=0.1, post_event_sec=0.1, max_representative_frames=6
    )
    peak = _evidence_frame(10.0)

    result = extractor.extract(long_video, peak, start_time=0.0, end_time=20.0)

    assert len(result) <= 6


def test_default_max_representative_frames_is_five(long_video: str) -> None:
    """Varsayilan ust sinir 5'tir (Gemini payload/gecikme/kota baskisini sinirlamak icin 8'den dusuruldu)."""
    extractor = RepresentativeFrameExtractor(pre_event_sec=0.1, post_event_sec=0.1)
    peak = _evidence_frame(10.0)

    result = extractor.extract(long_video, peak, start_time=0.0, end_time=20.0)

    assert len(result) <= 5


def test_constructor_rejects_max_representative_frames_below_three() -> None:
    with pytest.raises(ValueError):
        RepresentativeFrameExtractor(max_representative_frames=2)


# --------------------------- Tekrarsizlik ---------------------------


def test_no_duplicate_timestamps_when_target_frames_exceed_available_distinct_frames(
    long_video: str,
) -> None:
    """Hedeflenen kare sayisi, o kisa araliktaki mevcut ayirt edici kare sayisindan fazlaysa, ayni kare TEKRAR secilmemeli."""
    extractor = RepresentativeFrameExtractor(
        pre_event_sec=0.02, post_event_sec=0.02, max_representative_frames=6
    )
    peak = _evidence_frame(10.0)

    # sure=0.1s (~2.5 kare @25fps) cok kisa; window_span=0.04s oldugundan UZUN olay
    # dalina duser ve hedef kare sayisi mevcut ayirt edici kare sayisindan fazla olur.
    result = extractor.extract(long_video, peak, start_time=9.95, end_time=10.05)

    timestamps = [rf.timestamp_sec for rf in result]
    assert len(timestamps) == len(set(timestamps)), "Ayni zaman damgasi/kare iki kez secilmemeli."


def test_no_duplicate_timestamps_in_normal_long_cluster(long_video: str) -> None:
    extractor = RepresentativeFrameExtractor(pre_event_sec=2.0, post_event_sec=2.0)
    peak = _evidence_frame(10.0)

    result = extractor.extract(long_video, peak, start_time=0.0, end_time=20.0)

    timestamps = [rf.timestamp_sec for rf in result]
    assert len(timestamps) == len(set(timestamps))
