"""Evidence-bazli VLM dagitimi + reconciliation icin testler.

`BaseVLM.analyze_evidence_batched` (src/vlm/base_vlm.py) TUM evidence
karelerini tek bir dev payload'a doldurmak yerine kronolojik, kayipsiz
batch'ler halinde ayri ayri analiz eder; batch siniri bir OLAY SINIRI
DEGILDIR. Birden fazla batch varsa `BaseVLM.reconcile_events` batch-yerel
olaylari TEK bir global olay listesine birlestirir. `src/main.py` katmani
(`_naive_concat_batches`, `_reconcile_unassigned_evidence`) bu ciktiyi
evidence_id gecerliligi/atanmamislik acisindan dogrular; bir batch'in
basarisiz olmasi digerlerinin sonucunu kaybetmez ve basarisizlik asla
risk=0 gibi yorumlanmaz (acik `status`/`[ANALYSIS_FAILED]`/`[UYARI]`
isaretleri kullanilir).
"""

from __future__ import annotations

from typing import List

import pytest

from src.main import _naive_concat_batches, _reconcile_unassigned_evidence
from src.sampler.schema import EvidenceFrame
from src.vlm.base_vlm import BaseVLM, VLMResponse


def _evidence_frame(idx: int) -> EvidenceFrame:
    return EvidenceFrame(
        evidence_id=f"ev{idx}",
        frame_id=idx,
        timestamp_sec=float(idx),
        timestamp_str=f"00:0{idx}",
        change_score=0.5,
        image_bytes=b"\xff\xd8\xff",
        base64_image="data:image/jpeg;base64,AA==",
        image_shape=(1, 1, 3),
    )


class _StubVLM(BaseVLM):
    """`analyze_evidence_batched`in temel siniftan (BaseVLM) miras alinan, gercek
    batching/izolasyon davranisini test eden sahte VLM."""

    def __init__(self, fail_evidence_ids: set) -> None:
        from src.utils.config_loader import VLLMEndpointConfig

        super().__init__(
            VLLMEndpointConfig(model_name="stub-vlm", max_new_tokens=8, temperature=0.0)
        )
        self.fail_evidence_ids = fail_evidence_ids
        self.calls: List[List[str]] = []

    def analyze_evidence(self, evidence_frames: List[EvidenceFrame], prompt: str) -> VLMResponse:
        ids = [ef.evidence_id for ef in evidence_frames]
        self.calls.append(ids)
        if any(eid in self.fail_evidence_ids for eid in ids):
            raise RuntimeError(f"simulated failure for {ids}")
        return VLMResponse(
            description=f"Evidence {ids} icin gozlem.",
            model_name="stub-vlm",
            frame_count=len(evidence_frames),
            latency_ms=1.0,
            structured_events=[
                {
                    "event_id": f"e_{ids[0]}",
                    "type": "genel_gozlem",
                    "start_time": float(evidence_frames[0].timestamp_sec),
                    "end_time": float(evidence_frames[-1].timestamp_sec),
                    "evidence_ids": ids,
                    "description": "gozlem",
                    "risk_score": 20,
                    "confidence": 0.9,
                }
            ],
        )

    def health_check(self) -> bool:
        return True


# --------------------------- analyze_evidence_batched: izolasyon ---------------------------


def test_batched_dispatch_controlled_batch_size_groups_evidence() -> None:
    vlm = _StubVLM(fail_evidence_ids=set())
    frames = [_evidence_frame(i) for i in range(5)]

    responses = vlm.analyze_evidence_batched(frames, prompt="p", batch_size=2)

    assert vlm.calls == [["ev0", "ev1"], ["ev2", "ev3"], ["ev4"]]
    assert len(responses) == 3
    assert all(r.status == "completed" for r in responses)
    assert [r.evidence_ids for r in responses] == [["ev0", "ev1"], ["ev2", "ev3"], ["ev4"]]


def test_one_failed_batch_does_not_lose_other_batches_results() -> None:
    """Bir batch'in VLM cagrisi basarisiz olsa da diger batch'lerin sonucu KAYBEDILMEMELI."""
    vlm = _StubVLM(fail_evidence_ids={"ev1"})
    frames = [_evidence_frame(i) for i in range(3)]

    responses = vlm.analyze_evidence_batched(frames, prompt="p", batch_size=1)

    assert len(responses) == 3  # hicbir evidence dusurulmedi
    statuses = {r.evidence_ids[0]: r.status for r in responses}
    assert statuses == {"ev0": "completed", "ev1": "failed", "ev2": "completed"}
    assert responses[0].structured_events
    assert responses[2].structured_events
    assert responses[1].structured_events == []
    assert "ANALYSIS_FAILED" in responses[1].description


def test_failed_batch_never_silently_treated_as_success() -> None:
    vlm = _StubVLM(fail_evidence_ids={"ev0"})
    responses = vlm.analyze_evidence_batched([_evidence_frame(0)], prompt="p")

    assert responses[0].status == "failed"
    assert responses[0].status != "completed"


def test_empty_evidence_returns_empty_batched_response() -> None:
    vlm = _StubVLM(fail_evidence_ids=set())
    assert vlm.analyze_evidence_batched([], prompt="p") == []


# --------------------------- reconcile_events ---------------------------


def test_reconcile_single_batch_returns_it_directly() -> None:
    vlm = _StubVLM(fail_evidence_ids=set())
    frames = [_evidence_frame(0)]
    responses = vlm.analyze_evidence_batched(frames, prompt="p")

    reconciled = vlm.reconcile_events(responses, prompt="p")

    assert reconciled is responses[0]


def test_reconcile_all_failed_returns_failed_status_preserving_evidence_ids() -> None:
    vlm = _StubVLM(fail_evidence_ids={"ev0", "ev1"})
    frames = [_evidence_frame(0), _evidence_frame(1)]
    responses = vlm.analyze_evidence_batched(frames, prompt="p", batch_size=1)

    reconciled = vlm.reconcile_events(responses, prompt="p")

    assert reconciled.status == "failed"
    assert set(reconciled.evidence_ids) == {"ev0", "ev1"}


# --------------------------- _naive_concat_batches (main.py fallback) ---------------------------


def test_naive_concat_batches_prefixes_event_ids_and_loses_no_evidence() -> None:
    vlm = _StubVLM(fail_evidence_ids=set())
    frames = [_evidence_frame(0), _evidence_frame(1)]
    responses = vlm.analyze_evidence_batched(frames, prompt="p", batch_size=1)

    merged = _naive_concat_batches(responses)

    assert len(merged.structured_events) == 2
    event_ids = {ev["event_id"] for ev in merged.structured_events}
    assert event_ids == {"b0_e_ev0", "b1_e_ev1"}


# --------------------------- _reconcile_unassigned_evidence ---------------------------


def test_reconcile_unassigned_evidence_drops_invented_ids() -> None:
    frames = [_evidence_frame(0), _evidence_frame(1)]
    structured_events = [
        {"event_id": "e1", "evidence_ids": ["ev0", "ev_fabricated"]},
    ]

    cleaned = _reconcile_unassigned_evidence(structured_events, frames, batch_responses=[])

    assert cleaned[0]["evidence_ids"] == ["ev0"]


def test_reconcile_unassigned_evidence_preserves_leftover_evidence() -> None:
    """Hicbir olaya atanmayan evidence, 'unassigned/siniflandirilamadi' kaydinda KORUNUR (silinmez)."""
    frames = [_evidence_frame(0), _evidence_frame(1), _evidence_frame(2)]
    structured_events = [
        {"event_id": "e1", "evidence_ids": ["ev0"]},
    ]

    cleaned = _reconcile_unassigned_evidence(structured_events, frames, batch_responses=[])

    unassigned = next(e for e in cleaned if e["event_id"] == "unassigned")
    assert unassigned["canonical_event_type"] == "siniflandirilamadi"
    assert unassigned["event_name"] == "siniflandirilamadi"
    assert set(unassigned["evidence_ids"]) == {"ev1", "ev2"}


def test_reconcile_unassigned_evidence_prevents_duplicate_assignment() -> None:
    """Ayni evidence_id iki olaya atanmissa, yalnizca ILK atama korunur."""
    frames = [_evidence_frame(0)]
    structured_events = [
        {"event_id": "e1", "evidence_ids": ["ev0"]},
        {"event_id": "e2", "evidence_ids": ["ev0"]},
    ]

    cleaned = _reconcile_unassigned_evidence(structured_events, frames, batch_responses=[])

    assert cleaned[0]["evidence_ids"] == ["ev0"]
    assert cleaned[1]["evidence_ids"] == []


def test_reconcile_unassigned_evidence_merges_into_existing_unassigned_record() -> None:
    """VLM zaten kendi 'unassigned' kaydini uretmisse, leftover'lar bu kayda EKLENIR (yeni kayit acilmaz)."""
    frames = [_evidence_frame(0), _evidence_frame(1)]
    structured_events = [
        {
            "event_id": "unassigned",
            "event_name": "siniflandirilamadi",
            "canonical_event_type": "siniflandirilamadi",
            "evidence_ids": ["ev0"],
        },
    ]

    cleaned = _reconcile_unassigned_evidence(structured_events, frames, batch_responses=[])

    assert len(cleaned) == 1
    assert set(cleaned[0]["evidence_ids"]) == {"ev0", "ev1"}
