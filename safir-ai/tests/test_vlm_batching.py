"""Olay-bazli (event-based) VLM dagitimi icin testler.

`BaseVLM.describe_events_batched` (src/vlm/base_vlm.py) ve `main.py::
_aggregate_vlm_responses` birlikte, tum Olay Gruplarini TEK bir dev payload'a
doldurmak yerine kontrollu batch'ler halinde ayri ayri analiz eder; bir
batch'in basarisiz olmasi digerlerinin sonucunu kaybetmez ve basarisizlik
asla risk=0 gibi yorumlanmaz (acik `status`/`[ANALYSIS_FAILED]`/`[UYARI]`
isaretleri kullanilir).
"""

from __future__ import annotations

from typing import List

import pytest

from src.main import _aggregate_vlm_responses
from src.sampler.schema import EventCluster, EvidenceFrame
from src.vlm.base_vlm import BaseVLM, VLMResponse


def _cluster(event_id: int) -> EventCluster:
    frame = EvidenceFrame(
        frame_id=event_id,
        timestamp_sec=float(event_id),
        timestamp_str=f"00:0{event_id}",
        change_score=0.5,
        image_bytes=b"\xff\xd8\xff",
        base64_image="data:image/jpeg;base64,AA==",
        image_shape=(1, 1, 3),
    )
    return EventCluster(
        event_id=event_id, start_time=float(event_id), end_time=float(event_id), peak_frame=frame, total_candidate_frames=1
    )


class _StubVLM(BaseVLM):
    """`describe_events_batched`in temel siniftan (BaseVLM) miras alinan, gercek
    batching/izolasyon davranisini test eden sahte VLM."""

    def __init__(self, fail_event_ids: set) -> None:
        from src.utils.config_loader import VLLMEndpointConfig

        super().__init__(
            VLLMEndpointConfig(model_name="stub-vlm", max_new_tokens=8, temperature=0.0)
        )
        self.fail_event_ids = fail_event_ids
        self.calls: List[List[int]] = []

    def describe_events(self, clusters: List[EventCluster], prompt: str) -> VLMResponse:
        ids = [c.event_id for c in clusters]
        self.calls.append(ids)
        if any(eid in self.fail_event_ids for eid in ids):
            raise RuntimeError(f"simulated failure for {ids}")
        return VLMResponse(
            description=f"Olay(lar) {ids} icin gozlem.",
            model_name="stub-vlm",
            frame_count=len(clusters),
            latency_ms=1.0,
            structured_events=[{"type": "genel_gozlem", "timestamp": float(ids[0]), "confidence": 0.9}],
        )

    def health_check(self) -> bool:
        return True


# --------------------------- describe_events_batched: izolasyon ---------------------------


def test_batched_dispatch_sends_one_request_per_event_by_default() -> None:
    """`batch_size=1` (varsayilan) ile HER olay kendi ayri VLM istegini almali (tek dev payload YOK)."""
    vlm = _StubVLM(fail_event_ids=set())
    clusters = [_cluster(1), _cluster(2), _cluster(3)]

    responses = vlm.describe_events_batched(clusters, prompt="p")

    assert len(responses) == 3
    assert vlm.calls == [[1], [2], [3]]
    assert all(r.status == "completed" for r in responses)
    assert [r.cluster_event_ids for r in responses] == [[1], [2], [3]]


def test_batched_dispatch_controlled_batch_size_groups_events() -> None:
    vlm = _StubVLM(fail_event_ids=set())
    clusters = [_cluster(1), _cluster(2), _cluster(3), _cluster(4), _cluster(5)]

    responses = vlm.describe_events_batched(clusters, prompt="p", batch_size=2)

    assert vlm.calls == [[1, 2], [3, 4], [5]]
    assert len(responses) == 3


def test_one_failed_event_does_not_lose_other_events_results() -> None:
    """Bir olayin VLM cagrisi basarisiz olsa da diger olaylarin sonucu KAYBEDILMEMELI."""
    vlm = _StubVLM(fail_event_ids={2})
    clusters = [_cluster(1), _cluster(2), _cluster(3)]

    responses = vlm.describe_events_batched(clusters, prompt="p")

    assert len(responses) == 3  # hicbir olay dusurulmedi
    statuses = {r.cluster_event_ids[0]: r.status for r in responses}
    assert statuses == {1: "completed", 2: "failed", 3: "completed"}
    # Basarili olaylarin structured_events'i hala mevcut.
    assert responses[0].structured_events
    assert responses[2].structured_events
    # Basarisiz olan acik bicimde isaretlenmis, sessizce bos birakilmamis.
    assert responses[1].structured_events == []
    assert "ANALYSIS_FAILED" in responses[1].description


def test_failed_batch_never_silently_treated_as_success() -> None:
    vlm = _StubVLM(fail_event_ids={1})
    responses = vlm.describe_events_batched([_cluster(1)], prompt="p")

    assert responses[0].status == "failed"
    assert responses[0].status != "completed"


def test_empty_clusters_returns_empty_batched_response() -> None:
    vlm = _StubVLM(fail_event_ids=set())
    assert vlm.describe_events_batched([], prompt="p") == []


# --------------------------- _aggregate_vlm_responses ---------------------------


def test_aggregate_all_success_merges_structured_events_and_stamps_event_id() -> None:
    vlm = _StubVLM(fail_event_ids=set())
    clusters = [_cluster(1), _cluster(2)]
    responses = vlm.describe_events_batched(clusters, prompt="p")

    aggregated = _aggregate_vlm_responses(responses, clusters, fallback_model_name="stub-vlm")

    assert aggregated.status == "completed"
    assert len(aggregated.structured_events) == 2
    assert {ev["cluster_event_id"] for ev in aggregated.structured_events} == {1, 2}
    assert not aggregated.description.startswith("[HATA]")


def test_aggregate_partial_failure_preserves_successful_events() -> None:
    vlm = _StubVLM(fail_event_ids={2})
    clusters = [_cluster(1), _cluster(2), _cluster(3)]
    responses = vlm.describe_events_batched(clusters, prompt="p")

    aggregated = _aggregate_vlm_responses(responses, clusters, fallback_model_name="stub-vlm")

    assert aggregated.status == "partial_failure"
    # Basarisiz olan risk=0 gibi degil, acik bir UYARI notuyla isaretlenmeli.
    assert "[UYARI]" in aggregated.description
    assert "analysis_failed" in aggregated.description
    # Basarili olan 2 olayin structured_events'i KAYBOLMAMIS.
    assert {ev["cluster_event_id"] for ev in aggregated.structured_events} == {1, 3}
    # Tum-basarisiz degraded formatina (eski [HATA]) DUSMEMIS olmali.
    assert not aggregated.description.startswith("[HATA]")


def test_aggregate_total_failure_falls_back_to_legacy_hata_format() -> None:
    """Gerye-donuk uyumluluk: TUM batch'ler basarisizsa eski `[HATA]` formatina AYNEN dusulmeli."""
    vlm = _StubVLM(fail_event_ids={1, 2})
    clusters = [_cluster(1), _cluster(2)]
    responses = vlm.describe_events_batched(clusters, prompt="p")

    aggregated = _aggregate_vlm_responses(responses, clusters, fallback_model_name="stub-vlm")

    assert aggregated.status == "failed"
    assert aggregated.description.startswith("[HATA]")
    assert aggregated.structured_events == []
