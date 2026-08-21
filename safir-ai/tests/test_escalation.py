"""Otomatik eskalasyon (Human-on-the-Loop) katmani icin testler.

Bloke edici operator kapisinin kaldirildigini ve saha alarminin yuksek/kritik
risk skorunda OTOMATIK tetiklendigini; operatorun yalnizca sonradan
onayladigini (acknowledge) dogrular.
"""

from __future__ import annotations

from typing import List

import pytest

from src.decision.escalation import (
    EscalationPolicy,
    EscalationTier,
    FieldAlarmDispatcher,
)
from src.utils.config_loader import EscalationConfig


class _RecordingSink:
    """Dispatch cagrilarini kaydeden sahte alarm arka ucu (otomatik tetigi dogrulamak icin)."""

    def __init__(self) -> None:
        self.dispatched: List[dict] = []

    def dispatch(self, *, risk_score, risk_level, recommended_action, summary, auto, event_category: str = "safety") -> str:
        self.dispatched.append({"risk_score": risk_score, "auto": auto, "event_category": event_category})
        return f"alert-{len(self.dispatched)}"

    def acknowledge(self, alert_id: str, operator_note: str = ""):  # pragma: no cover - bu testte kullanilmaz
        raise NotImplementedError


@pytest.fixture
def config() -> EscalationConfig:
    return EscalationConfig(notify_score=26, auto_alarm_score=51)


def test_low_risk_stays_monitor_no_dispatch(config: EscalationConfig) -> None:
    sink = _RecordingSink()
    policy = EscalationPolicy(config, sink=sink)

    decision = policy.evaluate(risk_score=10, risk_level="dusuk", recommended_action="izle", summary="s")

    assert decision.tier is EscalationTier.MONITOR
    assert decision.auto_dispatched is False
    assert decision.alert_id is None
    assert sink.dispatched == []


def test_medium_risk_notify_no_dispatch(config: EscalationConfig) -> None:
    sink = _RecordingSink()
    policy = EscalationPolicy(config, sink=sink)

    decision = policy.evaluate(risk_score=35, risk_level="orta", recommended_action="bildir", summary="s")

    assert decision.tier is EscalationTier.NOTIFY
    assert decision.auto_dispatched is False
    assert sink.dispatched == []


@pytest.mark.parametrize("score", [51, 75, 100])
def test_high_risk_auto_dispatches_alarm_without_operator_gate(config: EscalationConfig, score: int) -> None:
    """Yuksek/kritik risk: operator onayi BEKLENMEDEN alarm otomatik tetiklenir."""
    sink = _RecordingSink()
    policy = EscalationPolicy(config, sink=sink)

    decision = policy.evaluate(
        risk_score=score, risk_level="yuksek", recommended_action="alarm", summary="s"
    )

    assert decision.tier is EscalationTier.ALARM
    assert decision.auto_dispatched is True
    assert decision.alert_id is not None
    assert len(sink.dispatched) == 1
    assert sink.dispatched[0]["auto"] is True


def test_field_dispatcher_acknowledge_marks_record(config: EscalationConfig) -> None:
    """Operator, otomatik tetiklenen alarmi sonradan onaylayabilir (Human-on-the-Loop)."""
    policy = EscalationPolicy(config, sink=FieldAlarmDispatcher())

    decision = policy.evaluate(risk_score=90, risk_level="kritik", recommended_action="alarm", summary="s")
    assert decision.alert_id is not None

    record = policy.sink.acknowledge(decision.alert_id, "operator kontrol etti")
    assert record.acknowledged is True
    assert record.operator_note == "operator kontrol etti"


def test_acknowledge_unknown_alert_raises(config: EscalationConfig) -> None:
    policy = EscalationPolicy(config, sink=FieldAlarmDispatcher())
    with pytest.raises(KeyError):
        policy.sink.acknowledge("bilinmeyen-id")
