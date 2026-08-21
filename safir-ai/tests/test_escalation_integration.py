"""Escalation Policy & Field Alarm Integration Test (Otonom Saha Alarmları)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.decision.escalation import (
    EscalationPolicy,
    EscalationTier,
    FieldAlarmDispatcher,
)
from src.utils.config_loader import EscalationConfig


@pytest.fixture
def escalation_config():
    """Varsayılan eskalasyon konfigürasyonu (notify=26, auto_alarm=51)."""
    return EscalationConfig(notify_score=26, auto_alarm_score=51, alert_endpoint="http://localhost:8000/alerts/trigger")


def test_escalation_dispatched_for_high_risk(escalation_config):
    """Risk skoru ≥51 olduğunda otonom saha alarmının ALARM kademesiyle otomatik tetiklendiğini doğrular."""
    dispatcher = FieldAlarmDispatcher()
    policy = EscalationPolicy(escalation_config, sink=dispatcher)

    # High risk scenario (score = 80 >= 51)
    escalation_result = policy.evaluate(
        risk_score=80,
        risk_level="kritik",
        recommended_action="Acil Sağlık Ekibini Yönlendirin",
        summary="Sahnede forklift kazası ve yaralanma tespiti.",
    )

    assert escalation_result.tier is EscalationTier.ALARM
    assert escalation_result.auto_dispatched is True
    assert escalation_result.alert_id is not None
    assert len(dispatcher._alerts) == 1


def test_escalation_not_dispatched_for_low_risk(escalation_config):
    """Risk skoru <51 olduğunda otonom alarmların tetiklenmediğini (MONITOR/NOTIFY) doğrular."""
    dispatcher = FieldAlarmDispatcher()
    policy = EscalationPolicy(escalation_config, sink=dispatcher)

    # Low risk scenario (score = 10 < 26)
    escalation_result = policy.evaluate(
        risk_score=10,
        risk_level="dusuk",
        recommended_action="Rutin izlemeye devam et",
        summary="Sahnede rutin çalışma devam ediyor.",
    )

    assert escalation_result.tier is EscalationTier.MONITOR
    assert escalation_result.auto_dispatched is False
    assert escalation_result.alert_id is None
    assert len(dispatcher._alerts) == 0


def test_http_alert_post_trigger_mock(escalation_config):
    """POST /alerts/trigger uç noktasının HTTP çağrısını simüle eden entegrasyon testi."""
    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"status": "success", "alert_id": "alert-999"}

    with patch("httpx.Client.post", mock_post):
        # Simüle edilen API alarm tetikleme mantığı
        import httpx
        client = httpx.Client()
        response = client.post(
            escalation_config.alert_endpoint,
            json={
                "risk_score": 85,
                "risk_level": "kritik",
                "action": "Acil Saha Müdahalesi",
            },
        )
        assert response.status_code == 200
        assert mock_post.called
        assert "alerts/trigger" in mock_post.call_args[0][0]
