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

    def dispatch(self, *, risk_score, risk_level, recommended_action, summary, auto) -> str:
        self.dispatched.append({"risk_score": risk_score, "auto": auto})
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


def test_unassessed_risk_status_falls_to_pending_review_no_auto_action(config: EscalationConfig) -> None:
    """`risk_status != "assessed"` (Agent muhakemesi basarisiz): otomatik islem/alarm YAPILMAZ,
    PENDING_REVIEW'e dusulur - operatorun ACIK karari beklenir (bkz. escalation.py 2026-08-25 notu)."""
    sink = _RecordingSink()
    policy = EscalationPolicy(config, sink=sink)

    decision = policy.evaluate(
        risk_score=None, risk_level="unknown", recommended_action="incele", summary="s", risk_status="unknown"
    )

    assert decision.tier is EscalationTier.PENDING_REVIEW
    assert decision.auto_dispatched is False
    assert decision.alert_id is None
    assert sink.dispatched == []


@pytest.mark.parametrize("score,risk_level", [(35, "orta"), (90, "kritik")])
def test_notify_or_alarm_without_deterministic_backing_falls_to_pending_review(
    config: EscalationConfig, score: int, risk_level: str
) -> None:
    """Risk skoru NOTIFY/ALARM esigine ulassa BILE, HICBIR RuleEngine kaniti yoksa
    (skor tamamen Agent'in/LLM'in kendi taslak tahmini), otomatik islem/alarm
    YAPILMAZ - PENDING_REVIEW'e dusulur; gercek bir saha alarmi dogrulanmamis,
    tek-kaynakli bir tahminle OTOMATIK tetiklenmez."""
    sink = _RecordingSink()
    policy = EscalationPolicy(config, sink=sink)

    decision = policy.evaluate(
        risk_score=score,
        risk_level=risk_level,
        recommended_action="degerlendir",
        summary="s",
        has_deterministic_backing=False,
    )

    assert decision.tier is EscalationTier.PENDING_REVIEW
    assert decision.auto_dispatched is False
    assert decision.alert_id is None
    assert sink.dispatched == []


def test_low_risk_without_deterministic_backing_still_stays_monitor(config: EscalationConfig) -> None:
    """Dusuk risk icin `has_deterministic_backing=False` KISITLAMASI UYGULANMAZ - operatoru
    dusuk riskte meshgul etmenin faydasi yoktur, MONITOR olarak kalir."""
    sink = _RecordingSink()
    policy = EscalationPolicy(config, sink=sink)

    decision = policy.evaluate(
        risk_score=10,
        risk_level="dusuk",
        recommended_action="izle",
        summary="s",
        has_deterministic_backing=False,
    )

    assert decision.tier is EscalationTier.MONITOR
    assert decision.auto_dispatched is False
    assert sink.dispatched == []


@pytest.mark.parametrize("score", [51, 90])
def test_alarm_with_deterministic_backing_still_auto_dispatches(config: EscalationConfig, score: int) -> None:
    """Deterministik (RuleEngine) kanit VARSA, yuksek/kritik risk HALA operator
    onayi beklenmeden otomatik tetiklenir - PENDING_REVIEW yalnizca kanitsiz/
    belirsiz durumlar icindir, GERCEK dogrulanmis alarmlari GECIKTIRMEZ."""
    sink = _RecordingSink()
    policy = EscalationPolicy(config, sink=sink)

    decision = policy.evaluate(
        risk_score=score,
        risk_level="kritik",
        recommended_action="alarm",
        summary="s",
        has_deterministic_backing=True,
    )

    assert decision.tier is EscalationTier.ALARM
    assert decision.auto_dispatched is True
    assert decision.alert_id is not None


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


class _FakeAlertStore:
    """`AlertStore` Protocol'une uyan, gercek SQLite'a bagli olmayan in-memory test cifti."""

    def __init__(self) -> None:
        self.rows: dict = {}

    def record_alert(self, alert_id, risk_score, risk_level, recommended_action, summary, auto, created_at) -> None:
        self.rows[alert_id] = {
            "alert_id": alert_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommended_action": recommended_action,
            "summary": summary,
            "auto": auto,
            "created_at": created_at,
            "acknowledged": False,
            "operator_note": "",
        }

    def get_alert(self, alert_id):
        return self.rows.get(alert_id)

    def acknowledge_alert(self, alert_id, operator_note: str = "") -> bool:
        if alert_id not in self.rows:
            return False
        self.rows[alert_id]["acknowledged"] = True
        self.rows[alert_id]["operator_note"] = operator_note
        return True


class _BrokenAlertStore:
    """Her cagrida istisna firlatan sahte depo - kalicilik hatasinin kritik yolu KESMEDIGINI dogrulamak icin."""

    def record_alert(self, *args, **kwargs) -> None:
        raise RuntimeError("disk dolu")

    def get_alert(self, alert_id):
        raise RuntimeError("disk dolu")

    def acknowledge_alert(self, alert_id, operator_note: str = "") -> bool:
        raise RuntimeError("disk dolu")


def test_dispatch_persists_alert_to_store(config: EscalationConfig) -> None:
    store = _FakeAlertStore()
    policy = EscalationPolicy(config, sink=FieldAlarmDispatcher(store=store))

    decision = policy.evaluate(risk_score=90, risk_level="kritik", recommended_action="alarm", summary="s")

    assert decision.alert_id in store.rows
    assert store.rows[decision.alert_id]["risk_score"] == 90
    assert store.rows[decision.alert_id]["acknowledged"] is False


def test_acknowledge_persists_to_store(config: EscalationConfig) -> None:
    store = _FakeAlertStore()
    policy = EscalationPolicy(config, sink=FieldAlarmDispatcher(store=store))
    decision = policy.evaluate(risk_score=90, risk_level="kritik", recommended_action="alarm", summary="s")

    policy.sink.acknowledge(decision.alert_id, "operator kontrol etti")

    assert store.rows[decision.alert_id]["acknowledged"] is True
    assert store.rows[decision.alert_id]["operator_note"] == "operator kontrol etti"


def test_acknowledge_after_restart_recovers_alert_from_store(config: EscalationConfig) -> None:
    """Surec yeniden baslarsa (bellek-ici dict BOS), alarm kalici depodan geri yuklenip onaylanabilmeli."""
    store = _FakeAlertStore()
    dispatcher_before_restart = FieldAlarmDispatcher(store=store)
    alert_id = dispatcher_before_restart.dispatch(
        risk_score=90, risk_level="kritik", recommended_action="alarm", summary="s", auto=True
    )

    # "Restart": TAMAMEN YENI bir dispatcher, ayni store, bos bellek-ici dict.
    dispatcher_after_restart = FieldAlarmDispatcher(store=store)
    record = dispatcher_after_restart.acknowledge(alert_id, "restart sonrasi onay")

    assert record.acknowledged is True
    assert record.risk_score == 90
    assert store.rows[alert_id]["acknowledged"] is True


def test_persistence_failure_does_not_block_dispatch_or_acknowledge(config: EscalationConfig) -> None:
    """Kalicilik best-effort'tur: depo hata verse bile alarm tetiklenir/onaylanir (kritik yol KESILMEZ)."""
    dispatcher = FieldAlarmDispatcher(store=_BrokenAlertStore())

    alert_id = dispatcher.dispatch(
        risk_score=90, risk_level="kritik", recommended_action="alarm", summary="s", auto=True
    )
    assert alert_id is not None

    record = dispatcher.acknowledge(alert_id, "onaylandi")
    assert record.acknowledged is True


def test_no_store_behaves_identically_to_before(config: EscalationConfig) -> None:
    """`store=None` (varsayilan) ile davranis eski (bellek-ici SADECE) haliyle BIREBIR AYNI kalmali."""
    dispatcher = FieldAlarmDispatcher()
    alert_id = dispatcher.dispatch(
        risk_score=90, risk_level="kritik", recommended_action="alarm", summary="s", auto=True
    )
    record = dispatcher.acknowledge(alert_id)
    assert record.acknowledged is True

    # Bellek-ici kayit yoksa (yeni bir dispatcher, store yok) KeyError beklenir - eski davranis.
    with pytest.raises(KeyError):
        FieldAlarmDispatcher().acknowledge(alert_id)
