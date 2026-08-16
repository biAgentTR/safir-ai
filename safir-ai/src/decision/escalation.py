"""06 - Otomatik Eskalasyon Katmani: risk skorunu OTOMATIK aksiyon kademesine cevirir.

Mentor geri bildirimi: onceki tasarimda operator, saha alarmini tetiklemek icin
"onayla" butonuna basmak zorundaydi (Human-in-the-Loop KAPI = pipeline'i bloke
eden erken mudahale). Bu katman, o bloke edici kapiyi kaldirir: sistem, risk
skoruna gore aksiyon kademesini KENDISI belirler ve yuksek/kritik durumda saha
alarmini OTOMATIK tetikler. Operator, alarmi engellemek icin degil; sonradan
DENETLEMEK/GERI ALMAK icin devrededir (Human-on-the-Loop).

Kademeler:
    MONITOR (dusuk)          -> yalnizca kaydet, izlemeye devam et.
    NOTIFY  (orta)           -> kaydet + yumusak bildirim.
    ALARM   (yuksek/kritik)  -> saha alarmini OTOMATIK tetikle (operator onayi beklemez).
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Protocol

from src.utils.config_loader import EscalationConfig

logger = logging.getLogger(__name__)


class EscalationTier(str, Enum):
    """Otomatik karar motorunun risk skoruna gore sectigi aksiyon kademesi."""

    MONITOR = "monitor"
    NOTIFY = "notify"
    ALARM = "alarm"


@dataclass
class EscalationDecision:
    """Otomatik eskalasyon sonucunun ozeti (rapora ve UI'a tasinir)."""

    tier: EscalationTier
    auto_dispatched: bool = False
    alert_id: Optional[str] = None
    reason: str = ""


@dataclass
class AlertRecord:
    """Tetiklenen bir saha alarminin kaydi (otomatik veya manuel)."""

    alert_id: str
    risk_score: int
    risk_level: str
    recommended_action: str
    summary: str
    auto: bool
    created_at: str
    acknowledged: bool = False
    operator_note: str = ""


class AlarmSink(Protocol):
    """Saha alarmini fiili bir kanala (SMS/anons/SCADA) ileten arka uc sozlesmesi."""

    def dispatch(
        self, *, risk_score: int, risk_level: str, recommended_action: str, summary: str, auto: bool
    ) -> str:
        """Alarmi iletir ve takip icin bir `alert_id` dondurur."""
        ...

    def acknowledge(self, alert_id: str, operator_note: str = "") -> "AlertRecord":
        """Operatorun alarmi denetledigini/geri aldigini isaretler (Human-on-the-Loop)."""
        ...


class FieldAlarmDispatcher:
    """Saha alarmlarini kaydeden/loglayan varsayilan `AlarmSink` (mock saha entegrasyonu).

    Gercek bir dagitimda `dispatch` SMS/anons/SCADA'ya baglanir; bu iskelette
    alarmi loglar, bellek-ici bir kayit defterinde tutar (operatorun sonradan
    onaylamasi/geri almasi icin) ve bir `alert_id` dondurur.
    """

    def __init__(self) -> None:
        self._alerts: Dict[str, AlertRecord] = {}
        self._lock = threading.Lock()

    def dispatch(
        self, *, risk_score: int, risk_level: str, recommended_action: str, summary: str, auto: bool
    ) -> str:
        """Alarmi tetikler, kayit defterine ekler ve `alert_id` dondurur."""
        alert_id = str(uuid.uuid4())
        record = AlertRecord(
            alert_id=alert_id,
            risk_score=risk_score,
            risk_level=risk_level,
            recommended_action=recommended_action,
            summary=summary,
            auto=auto,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._alerts[alert_id] = record
        logger.warning(
            "SAHA ALARMI %s TETIKLENDI: alert_id=%s risk=%d(%s) aksiyon=%s",
            "OTOMATIK" if auto else "MANUEL",
            alert_id,
            risk_score,
            risk_level,
            recommended_action,
        )
        return alert_id

    def acknowledge(self, alert_id: str, operator_note: str = "") -> AlertRecord:
        """Operatorun bir alarmi denetledigini/geri aldigini isaretler (Human-on-the-Loop).

        Raises:
            KeyError: `alert_id` kayit defterinde yoksa.
        """
        with self._lock:
            record = self._alerts.get(alert_id)
            if record is None:
                raise KeyError(f"Alarm bulunamadi: {alert_id}")
            record.acknowledged = True
            record.operator_note = operator_note
        logger.info("Saha alarmi operator tarafindan onaylandi: alert_id=%s not=%s", alert_id, operator_note or "(yok)")
        return record


class EscalationPolicy:
    """Risk skorunu OTOMATIK olarak bir `EscalationTier`e cevirir ve gerekirse alarmi tetikler.

    Bloke edici bir operator kapisi YOKTUR: `evaluate` cagrildiginda, yuksek/
    kritik risk icin alarm dogrudan `AlarmSink.dispatch` ile tetiklenir.
    """

    def __init__(self, config: EscalationConfig, sink: Optional[AlarmSink] = None) -> None:
        """Politikayi eskalasyon esikleri ve bir alarm arka ucuyla kurar.

        Args:
            config: `notify_score`/`auto_alarm_score` esiklerini iceren config.
            sink: Alarm dagitim arka ucu; verilmezse `FieldAlarmDispatcher` kullanilir.
        """
        self._config = config
        self._sink: AlarmSink = sink or FieldAlarmDispatcher()

    @property
    def sink(self) -> AlarmSink:
        """Alarm dagitim arka ucunu dondurur (operator onay uc noktasi icin)."""
        return self._sink

    def classify(self, risk_score: int) -> EscalationTier:
        """Risk skorunu esiklere gore bir kademeye cevirir (alarm tetiklemeden).

        Args:
            risk_score: 0-100 arasi risk skoru.

        Returns:
            `MONITOR` (< notify_score), `NOTIFY` (< auto_alarm_score) veya `ALARM`.
        """
        if risk_score >= self._config.auto_alarm_score:
            return EscalationTier.ALARM
        if risk_score >= self._config.notify_score:
            return EscalationTier.NOTIFY
        return EscalationTier.MONITOR

    def evaluate(
        self,
        *,
        risk_score: Optional[int],
        risk_level: str,
        recommended_action: str,
        summary: str,
        risk_status: str = "assessed",
    ) -> EscalationDecision:
        """Kademeyi belirler ve ALARM kademesinde saha alarmini OTOMATIK tetikler.

        `risk_status != "assessed"` (guvenilir bir karar uretilemedi) durumunda,
        sayisal esik karsilastirmasi HIC yapilmaz: dogrudan `NOTIFY` kademesine
        dusulur (operator incelemesi icin yumusak sinyal) ve alarm OTOMATIK
        tetiklenmez. `unknown -> MONITOR` (sessizce dusuk risk kabul etme) veya
        `unknown -> ALARM` (sayisal olmayan veriyle otomatik alarm) ASLA yapilmaz.

        Args:
            risk_score: 0-100 arasi risk skoru; risk_status="unknown" ise None olabilir.
            risk_level: `dusuk|orta|yuksek|kritik|unknown`.
            recommended_action: Alarma iliştirilecek birincil aksiyon onerisi.
            summary: Alarma iliştirilecek kisa durum ozeti.
            risk_status: `assessed` | `unknown` (bkz. `AgentDecision.risk_status`).

        Returns:
            Secilen kademe, otomatik tetiklenip tetiklenmedigi ve (varsa) `alert_id`.
        """
        if risk_status != "assessed" or risk_score is None:
            return EscalationDecision(
                tier=EscalationTier.NOTIFY,
                auto_dispatched=False,
                alert_id=None,
                reason="Risk durumu belirsiz — analiz guvenilir sekilde tamamlanamadi; manuel inceleme gerekli.",
            )

        tier = self.classify(risk_score)

        if tier is EscalationTier.ALARM:
            alert_id = self._sink.dispatch(
                risk_score=risk_score,
                risk_level=risk_level,
                recommended_action=recommended_action,
                summary=summary,
                auto=True,
            )
            return EscalationDecision(
                tier=tier,
                auto_dispatched=True,
                alert_id=alert_id,
                reason=f"Risk skoru {risk_score} >= otomatik alarm esigi {self._config.auto_alarm_score}.",
            )

        reason = (
            f"Risk skoru {risk_score} >= bildirim esigi {self._config.notify_score}."
            if tier is EscalationTier.NOTIFY
            else f"Risk skoru {risk_score} < bildirim esigi {self._config.notify_score}; izlemeye devam."
        )
        return EscalationDecision(tier=tier, auto_dispatched=False, alert_id=None, reason=reason)
