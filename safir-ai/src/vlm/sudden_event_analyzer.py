"""Ani Olay Analiz Orkestrasyonu.

`sudden_event_detector.detect_sudden_events` ile bulunan ONCESI/SONRASI kare
ciftlerini, goruntu-kabul eden bir `BaseVLM` (bu projede: `EvrenFramesVLM`,
model="llm-large", istek basina en fazla 2 goruntu - bkz. EVREN
dokumantasyonu SS 7.5) araciligiyla dogrulatir. Yalnizca GERCEKTEN onemli
(`is_notable_event=true`) bulunan adaylar, ana VLM'in urettigi
`structured_events` ile TAMAMEN AYNI sozluk semasinda (EVENTS_JSON sekli)
dondurulur - boylece cagiran taraf (`src/main.py`) bunlari mevcut olay
listesine DOGRUDAN EKLEYEBILIR; ne rapor semasinda ne de asagi akis
(EventEngine/RuleEngine/rapor olusturma) kodunda HICBIR degisiklik gerekmez.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from src.prompts.sudden_event_prompts import SUDDEN_EVENT_VERIFICATION_PROMPT
from src.vlm.base_vlm import BaseVLM
from src.vlm.sudden_event_detector import detect_sudden_events

logger = logging.getLogger(__name__)


def _parse_verification(raw_text: str) -> Dict[str, Any] | None:
    """Dogrulama modelinin JSON yanitini toleransli sekilde ayristirir.

    Bazi kucuk modeller kod blogu isaretleyicisi (` ```json ... ``` `) ekleyebilir
    veya sonuna fazladan metin iliştirebilir; bu fonksiyon bunlari temizlemeye
    calisir. Gecerli bir JSON nesnesi elde edilemezse `None` doner (aday
    sessizce ATLANIR - rapor cokmez, yalnizca bu tek aday dogrulanamamis sayilir).
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Ani olay dogrulama yaniti gecerli JSON degil, atlaniyor: %r", raw_text[:200])
        return None
    if not isinstance(data, dict):
        return None
    return data


def analyze_sudden_events(
    video_path: str,
    frames_vlm: BaseVLM,
    max_candidates: int = 6,
) -> List[Dict[str, Any]]:
    """Videoyu ani-degisim adaylari icin tarar, her adayi goruntu modeliyle
    dogrulatir ve GERCEKTEN onemli olanlari EVENTS_JSON semasinda dondurur.

    Bir adayin dogrulama cagrisi basarisiz olursa (ag hatasi, gecersiz yanit
    vb.) yalnizca o aday atlanir - diger adaylarin degerlendirmesi ve ana VLM
    akisi ETKILENMEZ.

    Args:
        video_path: Kaynak video dosyasi.
        frames_vlm: Goruntu (image_url) kabul eden bir `BaseVLM` orneği -
            bu projede `SafirPipeline._vlm_frames` (zaten var olan, dusuk-
            butceli mod icin kurulmus istemci; burada BASKA bir prompt'la
            YENIDEN kullanilir, YENI bir istemci/config eklenmez).
        max_candidates: Taramada bulunacak azami aday sayisi (maliyet siniri).

    Returns:
        Yalnizca DOGRULANMIS (is_notable_event=true) ani olaylarin, ana
        VLM'in urettigi `structured_events` ile AYNI sozluk semasinda
        listesi (bos olabilir).
    """
    try:
        candidates = detect_sudden_events(video_path, max_candidates=max_candidates)
    except Exception:  # noqa: BLE001 - tespit asamasi basarisiz olsa da ana VLM akisi ETKILENMEMELI
        logger.exception("Ani olay tespiti basarisiz (video=%s); bu katman atlaniyor.", video_path)
        return []

    if not candidates:
        return []

    confirmed: List[Dict[str, Any]] = []
    for i, candidate in enumerate(candidates):
        try:
            response = frames_vlm.analyze_evidence(
                [candidate.before_frame, candidate.after_frame], SUDDEN_EVENT_VERIFICATION_PROMPT
            )
        except Exception:  # noqa: BLE001 - bir adayin dogrulamasi basarisiz olsa da digerleri denenmeye devam eder
            logger.exception("Ani olay dogrulama cagrisi basarisiz (t=%.2fs)", candidate.timestamp_sec)
            continue

        verification = _parse_verification(response.description)
        if verification is None or not verification.get("is_notable_event"):
            continue

        event_name = str(verification.get("event_name") or "ani_olay").strip() or "ani_olay"
        description = str(verification.get("description") or "").strip()
        try:
            confidence = float(verification.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        confirmed.append(
            {
                "event_id": f"sudden_{i}",
                "event_name": event_name,
                "canonical_event_type": None,
                "start_time": max(0.0, candidate.timestamp_sec - 0.5),
                "end_time": candidate.timestamp_sec + 0.5,
                "evidence_ids": [],
                "description": description or "Ani/kisa sureli bir guvenlik olayi tespit edildi (detay saglanmadi).",
                "keywords": [event_name],
                "risk_score": None,
                "confidence": confidence,
            }
        )
        logger.info(
            "Ani olay dogrulandi: t=%.2fs event_name=%s confidence=%.2f",
            candidate.timestamp_sec,
            event_name,
            confidence,
        )

    return confirmed


__all__ = ["analyze_sudden_events"]
