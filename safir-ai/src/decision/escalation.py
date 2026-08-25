"""06 - Otomatik Eskalasyon Katmani: risk skorunu OTOMATIK aksiyon kademesine cevirir.

Mentor geri bildirimi: onceki tasarimda operator, saha alarmini tetiklemek icin
"onayla" butonuna basmak zorundaydi (Human-in-the-Loop KAPI = pipeline'i bloke
eden erken mudahale). Bu katman, o bloke edici kapiyi kaldirir: sistem, risk
skoruna gore aksiyon kademesini KENDISI belirler ve yuksek/kritik durumda saha
alarmini OTOMATIK tetikler. Operator, alarmi engellemek icin degil; sonradan
DENETLEMEK/GERI ALMAK icin devrededir (Human-on-the-Loop).

Kademeler:
    MONITOR         (dusuk)                          -> yalnizca kaydet, izlemeye devam et.
    NOTIFY          (orta, RuleEngine-dogrulanmis)    -> kaydet + yumusak bildirim.
    ALARM           (yuksek/kritik, RuleEngine-dogrulanmis) -> saha alarmini OTOMATIK tetikle.
    PENDING_REVIEW  (belirsiz VEYA deterministik kanitsiz) -> OTOMATIK ISLEM YAPILMAZ,
                    operatorun ACIK kararini bekler (bkz. asagida).

2026-08-25 (PENDING_REVIEW - dar kapsamli, BLOKE EDEN insan-onayi kapisi):
Yukaridaki "bloke edici kapi kaldirildi" karari HALA gecerlidir - gercek,
RuleEngine-dogrulanmis (deterministik kanitli) yuksek/kritik risk durumlarinda
saha alarmi HALA OTOMATIK/gecikmesiz tetiklenir (bir yangin/gaz kacagi
senaryosunda operator onayini beklemek, yanlis pozitiften daha tehlikelidir).
Ancak IKI DAR senaryoda tam tersi gecerlidir - otomatik islem/alarm YERINE
operatorun ACIK bir karar vermesi (onayla/geri cevir/manuel alarm) beklenir:
1. `risk_status != "assessed"` (Agent muhakemesi basarisiz oldu - sayisal bir
   risk skoru bile YOK).
2. Risk skoru NOTIFY/ALARM esigine ulasti AMA HICBIR deterministik (RuleEngine)
   kanit yok - skor tamamen Agent'in (LLM) kendi taslak tahminine dayaniyor.
Bu iki durumda "otomatik olarak dusuk risk say" (sessiz MONITOR) veya "otomatik
alarm tetikle" (dogrulanmamis veriyle) ASLA yapilmaz - ikisi de yanlis
yonlendirme riski tasir.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Protocol, runtime_checkable

from src.utils.config_loader import EscalationConfig

logger = logging.getLogger(__name__)


class EscalationTier(str, Enum):
    """Otomatik karar motorunun risk skoruna gore sectigi aksiyon kademesi."""

    MONITOR = "monitor"
    NOTIFY = "notify"
    ALARM = "alarm"
    PENDING_REVIEW = "pending_review"
    """Otomatik islem/alarm YAPILMAZ - operatorun ACIK karari (onay/geri cevirme/
    manuel alarm) beklenir. Bkz. modul dokustringi 2026-08-25 notu."""


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


@runtime_checkable
class AlertStore(Protocol):
    """`src.memory.event_store.EventStore`in `alerts` tablosuyla ayni imzaya sahip herhangi bir nesne icin duck-typing sozlesmesi.

    `EventStoreLike` (bkz. `src/event_analysis/event_history.py`) ile ayni
    gerekceyle: `FieldAlarmDispatcher`in gercek SQLite'a bagimli olmadan
    testte calisabilmesi icin.
    """

    def record_alert(
        self,
        alert_id: str,
        risk_score: int,
        risk_level: str,
        recommended_action: str,
        summary: str,
        auto: bool,
        created_at: str,
    ) -> None:
        """Bkz. `EventStore.record_alert`."""
        ...

    def get_alert(self, alert_id: str) -> Optional[Dict[str, object]]:
        """Bkz. `EventStore.get_alert`."""
        ...

    def acknowledge_alert(self, alert_id: str, operator_note: str = "") -> bool:
        """Bkz. `EventStore.acknowledge_alert`."""
        ...


class FieldAlarmDispatcher:
    """Saha alarmlarini kaydeden/loglayan varsayilan `AlarmSink` (mock saha entegrasyonu).

    Gercek bir dagitimda `dispatch` SMS/anons/SCADA'ya baglanir; bu iskelette
    alarmi loglar, bellek-ici bir kayit defterinde tutar (operatorun sonradan
    onaylamasi/geri almasi icin) ve bir `alert_id` dondurur.

    Kalicilik (audit P0 duzeltmesi): `store` verilirse (bkz. `AlertStore`),
    her `dispatch`/`acknowledge` bellek-ici kaydin YANI SIRA kalici olarak da
    (SQLite) yazilir - surec yeniden baslasa bile alarmlar VE operatorun
    onceki onaylari kaybolmaz. `store=None` (varsayilan) ile davranis eskisiyle
    BIREBIR AYNI kalir (yalnizca bellek-ici, mevcut testler etkilenmez).
    Kalicilik best-effort'tur: yazma basarisiz olursa alarmin kendisi (bellek-
    ici kayit + dispatch) yine de tamamlanir - persistence hatasi kritik
    alarm yolunu KESMEZ.
    """

    def __init__(self, store: Optional[AlertStore] = None) -> None:
        self._alerts: Dict[str, AlertRecord] = {}
        self._lock = threading.Lock()
        self._store = store

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
        if self._store is not None:
            try:
                self._store.record_alert(
                    alert_id=alert_id,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    recommended_action=recommended_action,
                    summary=summary,
                    auto=auto,
                    created_at=record.created_at,
                )
            except Exception:  # noqa: BLE001 - kalicilik best-effort'tur, alarmin kendisini KESMEZ
                logger.exception("Alarm kalici depoya yazilamadi (alert_id=%s); bellek-ici kayit korunuyor.", alert_id)
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

        `alert_id` bellek-ici kayit defterinde yoksa (orn. surec bu alarmi
        dispatch ETTIKTEN SONRA yeniden baslatildiysa), `store` verilmisse
        kalici depodan geri yuklenir; boylece restart sonrasi da eski bir
        alarm onaylanabilir.

        Raises:
            KeyError: `alert_id` ne bellek-ici kayitta ne de (varsa) kalici depoda bulunursa.
        """
        with self._lock:
            record = self._alerts.get(alert_id)
            if record is not None:
                record.acknowledged = True
                record.operator_note = operator_note
            elif self._store is not None:
                persisted = self._store.get_alert(alert_id)
                if persisted is None:
                    raise KeyError(f"Alarm bulunamadi: {alert_id}")
                record = AlertRecord(
                    alert_id=persisted["alert_id"],
                    risk_score=persisted["risk_score"],
                    risk_level=persisted["risk_level"],
                    recommended_action=persisted["recommended_action"],
                    summary=persisted["summary"],
                    auto=persisted["auto"],
                    created_at=persisted["created_at"],
                    acknowledged=True,
                    operator_note=operator_note,
                )
                self._alerts[alert_id] = record
            else:
                raise KeyError(f"Alarm bulunamadi: {alert_id}")
        if self._store is not None:
            try:
                self._store.acknowledge_alert(alert_id, operator_note)
            except Exception:  # noqa: BLE001 - kalicilik best-effort'tur, onay islemini KESMEZ
                logger.exception("Alarm onayi kalici depoya yazilamadi (alert_id=%s).", alert_id)
        logger.info("Saha alarmi operator tarafindan onaylandi: alert_id=%s not=%s", alert_id, operator_note or "(yok)")
        return record


class EscalationPolicy:
    """Risk skorunu OTOMATIK olarak bir `EscalationTier`e cevirir ve gerekirse alarmi tetikler.

    GERCEK (RuleEngine-dogrulanmis) yuksek/kritik risk icin bloke edici bir
    operator kapisi YOKTUR: alarm dogrudan `AlarmSink.dispatch` ile OTOMATIK
    tetiklenir. Yalnizca DAR iki senaryoda (`risk_status="unknown"` VEYA
    deterministik kanit yok) bloke eden bir kapi VARDIR - `PENDING_REVIEW`
    (bkz. modul dokustringi 2026-08-25 notu): bu durumda otomatik islem/alarm
    YAPILMAZ, operatorun ACIK karari beklenir.
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
        has_deterministic_backing: bool = True,
    ) -> EscalationDecision:
        """Kademeyi belirler ve ALARM kademesinde (deterministik kanit VARSA) saha alarmini OTOMATIK tetikler.

        `risk_status != "assessed"` (guvenilir bir karar uretilemedi) durumunda,
        sayisal esik karsilastirmasi HIC yapilmaz: dogrudan `PENDING_REVIEW`
        kademesine dusulur - operatorun ACIK karari (onay/geri cevirme/manuel
        alarm) BEKLENIR, hicbir otomatik islem/alarm YAPILMAZ. `unknown ->
        MONITOR` (sessizce dusuk risk kabul etme) veya `unknown -> ALARM`
        (sayisal olmayan veriyle otomatik alarm) ASLA yapilmaz.

        `has_deterministic_backing=False` (RuleEngine hicbir kural eslestirmedi,
        risk_score TAMAMEN Agent'in/LLM'in kendi taslak tahmini) durumunda da
        AYNI mantik gecerlidir: skor NOTIFY/ALARM esigine ulassa BILE otomatik
        islem/alarm YAPILMAZ, `PENDING_REVIEW`e dusulur - dogrulanmamis, tek
        kaynakli (LLM-only) bir tahminle gercek bir saha alarmi/bildirimi
        OTOMATIK tetiklenmez. Dusuk risk (`MONITOR`) icin bu kisitlama
        UYGULANMAZ - dusuk riskte operatoru meshgul etmenin faydasi yoktur.

        Args:
            risk_score: 0-100 arasi risk skoru; risk_status="unknown" ise None olabilir.
            risk_level: `dusuk|orta|yuksek|kritik|unknown`.
            recommended_action: Alarma iliştirilecek birincil aksiyon onerisi.
            summary: Alarma iliştirilecek kisa durum ozeti.
            risk_status: `assessed` | `unknown` (bkz. `AgentDecision.risk_status`).
            has_deterministic_backing: Nihai risk_score'un RuleEngine tarafindan
                dogrulanmis en az bir `RuleMatch`e dayanip dayanmadigi (bkz.
                `RiskProvenance.rule_ids`). `True` (varsayilan) = mevcut/eski
                davranis (geriye-uyumluluk icin).

        Returns:
            Secilen kademe, otomatik tetiklenip tetiklenmedigi ve (varsa) `alert_id`.
        """
        if risk_status != "assessed" or risk_score is None:
            return EscalationDecision(
                tier=EscalationTier.PENDING_REVIEW,
                auto_dispatched=False,
                alert_id=None,
                reason="Risk durumu belirsiz — analiz guvenilir sekilde tamamlanamadi; operator karari gerekli.",
            )

        tier = self.classify(risk_score)

        if tier is not EscalationTier.MONITOR and not has_deterministic_backing:
            return EscalationDecision(
                tier=EscalationTier.PENDING_REVIEW,
                auto_dispatched=False,
                alert_id=None,
                reason=(
                    f"Risk skoru {risk_score} {tier.value} esigine ulasti ama HICBIR deterministik "
                    "(RuleEngine) kanit yok - skor tamamen Agent'in kendi tahminine dayaniyor; "
                    "otomatik islem/alarm YAPILMADI, operator karari gerekli."
                ),
            )

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
