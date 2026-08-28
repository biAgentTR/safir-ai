"""Ani Olay Tespiti (Sudden Event Detection) - VLM Direct icin ek katman.

VLM Direct modunda (`EvrenVLM.analyze_video`) modele TUM video TEK seferde
gonderilir; modelin kendisi kare-kare fark hesaplamaz, bu yuzden cok kisa/
ani gecen olaylar (bir kapinin aniden kapanmasi, bir kisinin sikismasi,
ani bir dusme vb.) surekli anlatinin icinde kaybolabilir. Bu modul, Adaptive
Frame Sampler'DAN TAMAMEN BAGIMSIZ (o katmana hicbir sekilde dokunulmaz),
hafif bir OpenCV kare-farki taramasi yaparak videoyu ONCEDEN tarar: ortalama
gurultu seviyesinden (robust Z-skor - medyan + k*MAD) ANLAMLI OLCUDE sapan
kare-farki "sicrama"larini bulur ve her sicrama icin ONCESI/SONRASI kare
cifti cikarir.

Bu ciftler saf HAREKET tespitidir - "onemli bir guvenlik olayi" ile "kamera
titremesi" arasinda AYRIM YAPMAZ. O ayrimi `sudden_event_analyzer.py`,
ciftleri goruntu-kabul eden bir modele gonderip GERCEKTEN dogrulatarak yapar.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.sampler.schema import EvidenceFrame

logger = logging.getLogger(__name__)


@dataclass
class SuddenEventCandidate:
    """Videoda tespit edilen, GORSEL OLARAK DOGRULANMASI gereken bir 'ani degisim' adayi."""

    timestamp_sec: float
    """Sicramanin tespit edildigi (ONCESI/SONRASI karelerinin ortasindaki) zaman damgasi."""
    change_score: float
    """Kare-farki skoru (yalnizca tanilama/siralama amaclidir, rapora YANSIMAZ)."""
    before_frame: EvidenceFrame
    """Sicramadan hemen ONCEKI ana ait kare."""
    after_frame: EvidenceFrame
    """Sicramadan hemen SONRAKI ana ait kare."""


def _encode_frame(frame: np.ndarray, jpeg_quality: int) -> Tuple[bytes, str]:
    """Bir OpenCV karesini JPEG/base64'e kodlar (`EvidenceFrame` alanlari icin)."""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("Ani olay karesi JPEG'e kodlanamadi.")
    image_bytes = buffer.tobytes()
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return image_bytes, f"data:image/jpeg;base64,{base64_str}"


def _grab_frame_at(cap: cv2.VideoCapture, timestamp_sec: float) -> Optional[np.ndarray]:
    """Videoda verilen saniyeye atlayip tek bir kare okur (bulamazsa `None`)."""
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def _build_evidence_frame(
    frame: np.ndarray, evidence_id: str, timestamp_sec: float, change_score: float, jpeg_quality: int
) -> EvidenceFrame:
    """Ham bir OpenCV karesinden, goruntu-kabul eden modele gonderilebilir bir `EvidenceFrame` uretir."""
    image_bytes, base64_image = _encode_frame(frame, jpeg_quality)
    minutes, seconds = divmod(int(max(0.0, timestamp_sec)), 60)
    return EvidenceFrame(
        evidence_id=evidence_id,
        frame_id=0,
        timestamp_sec=round(max(0.0, timestamp_sec), 2),
        timestamp_str=f"{minutes:02d}:{seconds:02d}",
        change_score=round(change_score, 4),
        image_bytes=image_bytes,
        base64_image=base64_image,
        image_shape=frame.shape,
        saved_path=None,
        is_fallback=False,
        selection_reason="significant_change",
    )


def detect_sudden_events(
    video_path: str,
    sample_fps: float = 4.0,
    before_offset_sec: float = 0.5,
    after_offset_sec: float = 0.5,
    spike_sensitivity: float = 6.0,
    max_candidates: int = 6,
    min_gap_sec: float = 1.5,
    jpeg_quality: int = 85,
) -> List[SuddenEventCandidate]:
    """Videoyu hafifce tarayip ANI (sicrama tarzi) kare-farki anlarini bulur.

    Iki gecisli calisir: (1) videonun TAMAMI, `sample_fps` hizinda alt-
    ornekleyerek TEK GECISTE taranir (native cozunurlukte tutulmaz - yalnizca
    gri-tonlama fark skoru hesaplanir, hafif/hizli); (2) yalnizca secilen
    sicrama anlari icin video YENIDEN ACILIP tam cozunurlukte ONCESI/SONRASI
    kareleri cikarilir (yalnizca birkac aday oldugu icin ucuz).

    Args:
        video_path: Analiz edilecek video dosyasinin yolu.
        sample_fps: Kare-farki hesaplamasi icin ornekleme hizi (saniyede kac
            kare degerlendirilecegi) - videonun HER karesi degil, bu hizda
            alt-ornekleme yapilir.
        before_offset_sec / after_offset_sec: Tespit edilen anin ONCESI/
            SONRASI icin cikarilacak karelerin zaman farki.
        spike_sensitivity: Robust-Z benzeri esik carpani (medyan + k*MAD);
            buyudukce daha az/daha belirgin sicrama yakalanir.
        max_candidates: Videodan en fazla kac aday cikarilacagi (maliyet
            sinirlamasi - her aday ayri bir goruntu-modeli cagrisi gerektirir).
        min_gap_sec: Iki aday arasindaki asgari zaman farki (ayni ani
            olayin birden fazla kez yakalanmasini onler - non-max suppression).
        jpeg_quality: Cikarilan ONCESI/SONRASI karelerinin JPEG kalitesi.

    Returns:
        Zaman sirali `SuddenEventCandidate` listesi (bos olabilir - ani
        degisim yoksa, video cok kisaysa veya acilamazsa).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Ani olay tespiti icin video acilamadi: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps))) if sample_fps > 0 else 1

    scores: List[Tuple[float, float]] = []  # (timestamp_sec, score)
    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    scores.append((frame_idx / fps, float(np.mean(diff))))
                prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()

    if len(scores) < 6:
        # Video cok kisa/az ornek - guvenilir bir gurultu tabani hesaplanamaz;
        # sessizce bos liste dondurulur (rapor cokmez, sadece ani olay katmani atlanir).
        return []

    values = np.array([s[1] for s in scores])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) or 1e-6
    threshold = median + spike_sensitivity * mad

    spikes = [(ts, score) for ts, score in scores if score > threshold]
    if not spikes:
        return []

    # En yuksek skordan baslayarak, birbirine yakin (min_gap_sec icindeki)
    # sicramalari TEK bir temsilci ile birlestir (non-max suppression).
    spikes.sort(key=lambda x: -x[1])
    selected: List[Tuple[float, float]] = []
    for ts, score in spikes:
        if all(abs(ts - s[0]) >= min_gap_sec for s in selected):
            selected.append((ts, score))
        if len(selected) >= max_candidates:
            break
    selected.sort(key=lambda x: x[0])

    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        return []
    candidates: List[SuddenEventCandidate] = []
    try:
        for i, (ts, score) in enumerate(selected):
            before_img = _grab_frame_at(cap2, ts - before_offset_sec)
            after_img = _grab_frame_at(cap2, ts + after_offset_sec)
            if before_img is None or after_img is None:
                continue
            before_ef = _build_evidence_frame(
                before_img, f"sudden{i}_before", ts - before_offset_sec, score, jpeg_quality
            )
            after_ef = _build_evidence_frame(
                after_img, f"sudden{i}_after", ts + after_offset_sec, score, jpeg_quality
            )
            candidates.append(
                SuddenEventCandidate(
                    timestamp_sec=round(ts, 2),
                    change_score=round(score, 4),
                    before_frame=before_ef,
                    after_frame=after_ef,
                )
            )
    finally:
        cap2.release()

    if candidates:
        logger.info(
            "Ani olay tespiti: %d aday bulundu (video=%s, esik=%.2f)", len(candidates), video_path, threshold
        )
    return candidates


__all__ = ["SuddenEventCandidate", "detect_sudden_events"]
