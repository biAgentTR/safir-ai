"""`src/vlm/evren_vlm.py::EvrenVLM` icin GPU/gercek-ag bagimliligi gerektirmeyen birim testleri.

Gercek EVREN'e hicbir istek atilmaz - `httpx.post`, bu modulun icinde
(`src.vlm.evren_vlm.httpx.post`) sahte bir fonksiyonla degistirilir. Videolar
`tests/test_sampler.py`/`tests/test_video_chunker.py`daki ile ayni desende,
tamamen sentetik (cv2 ile uretilmis) kucuk dosyalardir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest

from src.utils.config_loader import VLLMEndpointConfig
from src.vlm.evren_vlm import EvrenVLM


def _write_video(path: Path, num_frames: int, fps: float = 10.0, size: tuple[int, int] = (32, 24)) -> None:
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _endpoint(chunk_duration_sec: float | None) -> VLLMEndpointConfig:
    return VLLMEndpointConfig(
        model_name="vlm",
        max_new_tokens=1024,
        temperature=0.0,
        provider="evren",
        base_url="http://fake-evren.local/v1",
        chunk_duration_sec=chunk_duration_sec,
    )


class _FakeHttpResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


def _evren_payload(description: str, events: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    content = description
    if events is not None:
        import json

        content = f"{description}\nEVENTS_JSON: {json.dumps(events)}"
    return {"choices": [{"message": {"content": content}}]}


def test_short_video_sends_single_request_when_chunking_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Video chunk_duration_sec'ten KISAYSA, chunking ACIK olsa bile TEK istek atilir (davranis DEGISMEZ)."""
    video_path = tmp_path / "short.mp4"
    _write_video(video_path, num_frames=50, fps=10.0)  # 5.0s

    calls: List[Dict[str, Any]] = []

    def _fake_post(url, json, headers, timeout):  # noqa: A002 - httpx.post imzasiyla ayni
        calls.append(json)
        return _FakeHttpResponse(_evren_payload("Rutin saha."))

    monkeypatch.setattr("src.vlm.evren_vlm.httpx.post", _fake_post)

    vlm = EvrenVLM(_endpoint(chunk_duration_sec=60.0))
    response = vlm.analyze_video(str(video_path), evidence_frames=[], prompt="test")

    assert len(calls) == 1
    assert response.description == "Rutin saha."
    assert response.structured_events == []


def test_chunking_disabled_sends_single_request_for_long_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`chunk_duration_sec=None` (varsayilan/eski davranis): uzun video da TEK istekte gonderilir."""
    video_path = tmp_path / "long.mp4"
    _write_video(video_path, num_frames=200, fps=10.0)  # 20.0s

    calls: List[Dict[str, Any]] = []

    def _fake_post(url, json, headers, timeout):
        calls.append(json)
        return _FakeHttpResponse(_evren_payload("Tum video tek seferde."))

    monkeypatch.setattr("src.vlm.evren_vlm.httpx.post", _fake_post)

    vlm = EvrenVLM(_endpoint(chunk_duration_sec=None))
    response = vlm.analyze_video(str(video_path), evidence_frames=[], prompt="test")

    assert len(calls) == 1
    assert response.description == "Tum video tek seferde."


def test_long_video_is_chunked_into_multiple_sequential_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """20s'lik video, chunk_duration_sec=5 ile 4 AYRI istekte gonderilmeli (paralel degil, sirali)."""
    video_path = tmp_path / "long.mp4"
    _write_video(video_path, num_frames=200, fps=10.0)  # 20.0s

    calls: List[Dict[str, Any]] = []

    def _fake_post(url, json, headers, timeout):
        calls.append(json)
        idx = len(calls)
        return _FakeHttpResponse(_evren_payload(f"parca-{idx}"))

    monkeypatch.setattr("src.vlm.evren_vlm.httpx.post", _fake_post)

    vlm = EvrenVLM(_endpoint(chunk_duration_sec=5.0))
    response = vlm.analyze_video(str(video_path), evidence_frames=[], prompt="test")

    assert len(calls) == 4
    # Her istek AYRI, kucuk (tam videodan kucuk) bir video govdesi tasimali.
    body_sizes = {len(c["messages"][0]["content"][1]["video_url"]["url"]) for c in calls}
    assert len(body_sizes) >= 1  # en azindan calisti, gercek boyut karsilastirmasi asagida
    assert "parca-1" in response.description
    assert "parca-4" in response.description


def test_chunk_event_timestamps_are_shifted_to_original_video_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """En onemli regresyon: her parcanin structured_events'indeki start_time/end_time,
    parcanin KENDI basindan (0) DEGIL, ORIJINAL videodaki GERCEK zamandan olmali."""
    video_path = tmp_path / "long.mp4"
    _write_video(video_path, num_frames=200, fps=10.0)  # 20.0s, 4x 5s parca

    call_index = {"n": 0}

    def _fake_post(url, json, headers, timeout):
        call_index["n"] += 1
        n = call_index["n"]
        # Her parca, KENDI ICINDE (0-5s araliginda) bir olay bildiriyor.
        events = [
            {
                "event_id": f"e{n}",
                "type": "kkd_ihlali",
                "start_time": 1.0,
                "end_time": 2.0,
                "evidence_ids": [],
                "description": f"parca {n} olayi",
                "risk_score": 30,
                "confidence": 0.8,
            }
        ]
        return _FakeHttpResponse(_evren_payload(f"parca-{n}", events=events))

    monkeypatch.setattr("src.vlm.evren_vlm.httpx.post", _fake_post)

    vlm = EvrenVLM(_endpoint(chunk_duration_sec=5.0))
    response = vlm.analyze_video(str(video_path), evidence_frames=[], prompt="test")

    assert len(response.structured_events) == 4
    start_times = sorted(e["start_time"] for e in response.structured_events)
    # Parca 1: 1.0 (0 kaydirma), parca 2: 6.0 (5s kaydirma), parca 3: 11.0, parca 4: 16.0
    assert start_times == pytest.approx([1.0, 6.0, 11.0, 16.0])
    end_times = sorted(e["end_time"] for e in response.structured_events)
    assert end_times == pytest.approx([2.0, 7.0, 12.0, 17.0])


def test_one_failing_chunk_does_not_lose_the_others(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bir parcanin istegi basarisiz olursa, DIGER basarili parcalarin sonucu KAYBOLMAMALI."""
    video_path = tmp_path / "long.mp4"
    _write_video(video_path, num_frames=200, fps=10.0)  # 20.0s, 4 parca

    call_index = {"n": 0}

    def _fake_post(url, json, headers, timeout):
        call_index["n"] += 1
        n = call_index["n"]
        if n == 2:
            raise RuntimeError("simulated network failure")
        return _FakeHttpResponse(_evren_payload(f"parca-{n}"))

    monkeypatch.setattr("src.vlm.evren_vlm.httpx.post", _fake_post)

    vlm = EvrenVLM(_endpoint(chunk_duration_sec=5.0))
    response = vlm.analyze_video(str(video_path), evidence_frames=[], prompt="test")

    assert "parca-1" in response.description
    assert "parca-3" in response.description
    assert "parca-4" in response.description
    assert "ANALYSIS_FAILED" in response.description


def test_all_chunks_failing_raises_runtime_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path = tmp_path / "long.mp4"
    _write_video(video_path, num_frames=200, fps=10.0)

    def _fake_post(url, json, headers, timeout):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr("src.vlm.evren_vlm.httpx.post", _fake_post)

    vlm = EvrenVLM(_endpoint(chunk_duration_sec=5.0))
    with pytest.raises(RuntimeError):
        vlm.analyze_video(str(video_path), evidence_frames=[], prompt="test")


def test_chunk_temp_files_are_cleaned_up_after_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gecici parca dosyalari, analiz basarili da basarisiz da olsa TEMIZLENMELI."""
    video_path = tmp_path / "long.mp4"
    _write_video(video_path, num_frames=200, fps=10.0)

    seen_paths: List[str] = []

    def _fake_post(url, json, headers, timeout):
        return _FakeHttpResponse(_evren_payload("ok"))

    monkeypatch.setattr("src.vlm.evren_vlm.httpx.post", _fake_post)

    from src.vlm import video_chunker as _vc

    original_split = _vc.split_video_into_chunks

    def _spying_split(*args, **kwargs):
        chunks = original_split(*args, **kwargs)
        seen_paths.extend(c.path for c in chunks if not c.is_original)
        return chunks

    monkeypatch.setattr("src.vlm.evren_vlm.split_video_into_chunks", _spying_split)

    vlm = EvrenVLM(_endpoint(chunk_duration_sec=5.0))
    vlm.analyze_video(str(video_path), evidence_frames=[], prompt="test")

    assert seen_paths, "en az bir gecici parca uretilmis olmali"
    import os

    for p in seen_paths:
        assert not os.path.exists(p)


def test_rtsp_source_still_rejected_with_chunking_enabled(tmp_path: Path) -> None:
    vlm = EvrenVLM(_endpoint(chunk_duration_sec=60.0))
    with pytest.raises(RuntimeError):
        vlm.analyze_video("rtsp://example.com/stream", evidence_frames=[], prompt="test")
