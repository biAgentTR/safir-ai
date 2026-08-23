"""02 - VLM Oncesi Katman: OpenCV tabanli Adaptive Frame Sampler (CPU ONLY).

Bu modul, ham video akisini agir bir tespit modeli (YOLO/ByteTrack) ve GPU
kullanmadan tarar. Ardisik gri-tonlama kareler arasindaki mutlak fark ile
degisimi olcer, dinamik (medyan tabanli) bir gurultu tabani (noise floor)
ile gereksiz titremeleri eler ve yalnizca anlamli degisim iceren kareleri
"Kanit Karesi" (Evidence Frame) olarak isaretler.

ONEMLI (mimari): Sampler burada DURUR - hicbir olay kumelemesi (event
clustering), zirve/temsili kare secimi veya `event_id`/`cluster_id` uretimi
YAPMAZ. Esik-gecmis TUM Kanit Kareleri, video geneli SIRALI (kronolojik) ve
KAYIPSIZ olarak `process_video()`den doner; hicbir global buffer/kare
limiti, temporal voting/clustering/deduplication veya liste kesme nedeniyle
bir kare burada elenmez. Kumeleme + olay analizi artik VLM katmaninda
yapilir (bkz. `src/vlm/base_vlm.py::BaseVLM.analyze_evidence_batched` +
`reconcile_events`).

ONEMLI (zamansal kapsama / temporal coverage): Esik tabanli secim, uzun bir
sessiz araliktaki (ör. 00:15 -> 01:45) esik-alti kareleri DOGASI GEREGI hic
evidence yapmaz; bu, gercek bir olayin tamamen kacirilmasina yol acabilir.
Bunu onlemek icin `process_video()`, son evidence karesinden bu yana gecen
sure `max_temporal_gap_sec`i asarsa, o pencerede degerlendirilen esik-alti
adaylar arasindan `net_change_score`'u EN YUKSEK olani (rastgele veya sabit
periyodik bir kare DEGIL) `selection_reason="temporal_coverage"` ile
evidence listesine ekler. Bu bir OLAY KUMELEMESI DEGILDIR ve pre/peak/post
gibi bir konumsal rol getirmez - yalnizca "bu kare neden secildi" bilgisini
tasiyan bir metadata alanidir (bkz. `EvidenceFrame.selection_reason`).

ONEMLI (yogunluk-uyarlamali secim / adaptive selection density): Video HER
ZAMAN tek geciste, tek bir sabit `sample_fps` ile okunur - bu, kaynak/analiz
hizidir ve DEGISMEZ. Adaptif olan yalnizca VLM'e GONDERILECEK karelerin
SECIM SIKLIGIDIR. `process_video()`, her ornekte hesaplanan `net_change_score`
uzerinden UC yogunluk seviyesi arasinda O(1) durum (iki cooldown zaman
damgasi + sabit boyutlu bir "supheli-erken" penceresi) ile gecis yapar:

  1. SAKIN: `max_temporal_gap_sec` araligiyla (mevcut coverage davranisi).
  2. ERKEN DEGISIM: ana esigin ALTINDA ama `early_change_min_count` kadar
     ardisik/pencere-ici "supheli-erken" (`early_change_score_ratio *
     min_change_threshold`i gecen) sinyal SURDURULDUGUNDE tetiklenir. Secilen
     kare, ONAYIN GERCEKLESTIGI (min_count'a ulasilan) kare DEGIL, zincirin
     SAKIN durumdayken ILK supheli-erken sinyali verdigi kareDIR (bkz.
     `pending_early_candidate` - `_PendingSelectionCandidate`nin bu amacla
     yeniden kullanilan tek-ornekli hali; buyuyen bir gecmis buffer'i
     DEGILDIR). Boylece bir olayin baslangic karesi (ör. izmaritin atildigi
     an, dumanin ilk gorulme ani), onay `early_change_min_count` kareyi
     bulana kadar geciken 1-2 karelik bir gecikme OLMADAN korunur. Aktifken
     `early_change_selection_interval_sec` ile daha sik secim yapilir.
  3. GUCLU DEGISIM: ana esik gecildiginde (mevcut `threshold_exceeded`
     davranisi DEGISMEDEN korunur - HER esik-gecen kare secilir). Esik
     gecildikten sonra `strong_change_cooldown_sec` boyunca bir hysteresis
     penceresi acilir; bu pencerede skor esigin altina dusse bile
     `significant_change_selection_interval_sec` ile (en sik) secim
     surdurulur - boylece tek bir anlik skor dususunde hemen sakin secime
     donulmez.

Bu mekanizma OLAY KUMELEMESI YAPMAZ, olay baslangic/bitis zamani URETMEZ ve
pre/peak/post gibi bir konumsal rol GETIRMEZ; yalnizca sampler'in HANGI
SIKLIKTA kare sectigini kontrol eder. Esik-alti secimler (coverage/erken/
guclu-hysteresis) icin, son SECILEN evidence karesine gore gorsel olarak
"neredeyse ayni" olan adaylar `dedup_similarity_ratio` ile elenir; bu kontrol
`threshold_exceeded` karelere ASLA uygulanmaz ve olayin ilk ortaya ciktigi
(baseline'dan farkli olan) erken-degisim karesini yanlislikla SILEMEZ (bkz.
`_is_near_duplicate`).
"""

from __future__ import annotations

import base64
import csv
import datetime
import json
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from src.sampler.payload_builder import format_evidence_metadata_label
from src.sampler.schema import EvidenceFrame

_SELECTION_REASON_THRESHOLD = "threshold_exceeded"
_SELECTION_REASON_COVERAGE = "temporal_coverage"
_SELECTION_REASON_EARLY_CHANGE = "early_change"
_SELECTION_REASON_SIGNIFICANT_CHANGE = "significant_change"
_SELECTION_REASON_SINGLE_FRAME_CHANGE = "single_frame_change"
_SELECTION_REASON_FALLBACK = "fallback"

# Tanilama (diagnostic) modu icin RED nedeni sabitleri (bkz. `_DiagnosticRecorder`).
# Bu deger kumesi sampler ALGORITMASINI degistirmez - yalnizca mevcut karar
# degiskenlerinin (esik, onay, aralik, dedup, yogunluk) OKUNABILIR bir
# aciklamasidir. NOT: `below_early_threshold`/`temporal_vote_failed` KASITLI
# OLARAK burada YOKTUR - mevcut mimaride hicbir kod yolu bu iki nedeni TEK ve
# KESIN (nihai) red gerekcesi olarak veremez (bkz. modul rapor notu #3):
# esik-alti HER aday, skoru ne kadar dusuk olursa olsun, genel `pending_best`
# / coverage yarismasina KATILMAYA DEVAM eder (sessiz bolgede "temsili bir
# kare hep kalsin" garantisi geregi) - dolayisiyla nihai red her zaman
# `not_best_coverage_candidate`/`near_duplicate`/`early_not_confirmed`/
# `selection_interval_not_elapsed` gibi GERCEKTEN o kareyi eleyen mekanizmaya
# ait olur. Sahte bir kullanim uretmek yerine bu iki deger BASTAN KALDIRILDI.
_REJECTION_NO_REFERENCE_FRAME = "no_reference_frame"
_REJECTION_EARLY_NOT_CONFIRMED = "early_not_confirmed"
_REJECTION_SELECTION_INTERVAL_NOT_ELAPSED = "selection_interval_not_elapsed"
_REJECTION_NEAR_DUPLICATE = "near_duplicate"
_REJECTION_NOT_BEST_COVERAGE_CANDIDATE = "not_best_coverage_candidate"
# IKI kullanim yeri vardir: (1) `_DiagnosticRecorder.record()` icindeki
# savunmaci "her frame_index icin tek satir" korumasi; (2) `process_video`
# icindeki GERCEK algoritmik guvenlik agi - `_discard_pending_early_if_same_
# frame()` - `pending_best` VE `pending_early_candidate` AYNI frame_id'yi
# tuttugunda (bkz. modul rapor notu #2), digerinin DAHA SONRA bagimsiz
# olarak AYNI kareyi tekrar eklemesini engeller.
_REJECTION_ALREADY_SELECTED_FRAME_INDEX = "already_selected_frame_index"

# CSV cikti icin sabit sutun sirasi (JSONL'de de ayni anahtar kumesi kullanilir,
# sira onemsizdir orada). Goruntu/base64 alani KASITLI OLARAK YOKTUR.
_DIAGNOSTIC_FIELDNAMES: Tuple[str, ...] = (
    "frame_index",
    "timestamp_ms",
    "timestamp_str",
    "net_change_score",
    "raw_change_ratio",
    "adaptive_noise_floor",
    "tile_scores",
    "early_threshold_value",
    "main_threshold_value",
    "early_threshold_exceeded",
    "main_threshold_exceeded",
    "recent_early_decisions",
    "recent_early_true_count",
    "early_change_confirmed",
    "density_state",
    "pending_early_candidate_frame_index",
    "pending_early_candidate_timestamp_ms",
    "temporal_vote_passed",
    "temporal_vote_true_count",
    "temporal_vote_min_count",
    "strong_cooldown_active",
    "strong_cooldown_until_ms",
    "early_cooldown_active",
    "early_cooldown_until_ms",
    "selection_interval_sec",
    "selection_interval_elapsed",
    "gap_since_last_evidence_ms",
    "dedup_checked",
    "dedup_is_duplicate",
    "coverage_window_start_ms",
    "coverage_window_end_ms",
    "selected",
    "selection_reason",
    "rejection_reason",
    "evidence_id",
    "vlm_payload_label",
    "long_baseline_net_score",
    "long_baseline_early_suspicious",
    "long_baseline_slow_net_score",
    "long_baseline_slow_early_suspicious",
)


def _ms(timestamp_sec: float) -> int:
    """Saniyeyi tam milisaniyeye yuvarlar (tanilama zaman damgalari icin)."""
    return int(round(timestamp_sec * 1000))


def _finite_ms(timestamp_sec: float) -> Optional[int]:
    """`_ms`, ancak `-inf`/`inf` (hic tetiklenmemis cooldown) icin `None` doner."""
    if timestamp_sec in (float("-inf"), float("inf")):
        return None
    return _ms(timestamp_sec)


def _native(value: Any) -> Any:
    """Numpy skalerlerini (ör. `np.float64`/`np.bool_`/`np.int64`) yerel
    Python tiplerine cevirir - CSV/JSON yazicilari numpy skalerlerini
    (`np.bool_`/`np.int64` `float`/`int`in alt sinifi DEGILDIR) reddeder."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _readable_timestamp(timestamp_sec: float) -> str:
    """`MM:SS.mmm` formatinda, milisaniye hassasiyeti korunmus okunabilir zaman damgasi."""
    total_ms = _ms(timestamp_sec)
    minutes, rem_ms = divmod(total_ms, 60_000)
    seconds, millis = divmod(rem_ms, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


class _DiagnosticRecorder:
    """Config ile acilip kapanabilen, sampler'in HER ornek karesi icin tam
    karar izini diske yazan tanilama (diagnostic) kaydedicisi.

    YALNIZCA `SamplerConfig.diagnostic_enabled=True` iken ORNEKLENIR;
    KAPALIYKEN `AdaptiveFrameSampler` bu sinifa hic dokunmaz - davranis ve
    performans BIREBIR korunur (bkz. `process_video`).

    Goruntu/base64 veri ASLA tutulmaz/yazilmaz. Satirlar TEK TEK diske
    yazilir (bellekte biriktirilmez); yalnizca KUCUK ozet sayaclari
    (Counter/set - toplam ornek kare sayisiyla degil, YALNIZCA farkli
    selection/rejection reason SAYISIYLA orantili sabit sayida anahtar)
    bellekte tutulur. Bu kaydedici YALNIZ-YAZILIR bir gozlem katmanidir;
    hicbir metodu sampler'in secim KARARLARINI okumaz/etkilemez - bkz.
    gorev kisiti "diagnostic kayitlarinin kendisi sampler secim state'ine
    dahil olmasin".
    """

    def __init__(self, output_path: Path, fmt: str) -> None:
        if fmt not in ("csv", "jsonl"):
            raise ValueError(f"diagnostic_output_format 'csv' veya 'jsonl' olmalidir, verilen: {fmt!r}")
        self._fmt = fmt
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(output_path, "w", newline="", encoding="utf-8")
        self._csv_writer = None
        if fmt == "csv":
            self._csv_writer = csv.DictWriter(self._fh, fieldnames=_DIAGNOSTIC_FIELDNAMES)
            self._csv_writer.writeheader()

        self.total_rows = 0
        self.selection_reason_counts: "Counter[str]" = Counter()
        self.rejection_reason_counts: "Counter[str]" = Counter()
        self._recorded_frame_indices: Set[int] = set()
        self.duplicate_record_attempts = 0

    def record(self, row: Dict[str, Any]) -> None:
        """Tek bir tanilama satirini diske yazar ve ozet sayaclarini gunceller.

        Ayni `frame_index` icin ikinci bir cagri gelirse (bkz.
        `_REJECTION_ALREADY_SELECTED_FRAME_INDEX`), bu kaydedicinin KENDI
        "her frame_index icin tek satir" garantisini korumak amaciyla o
        girisim `selected=False, rejection_reason=
        already_selected_frame_index` olarak GUVENLE loglanir ve
        `duplicate_record_attempts` sayaci artirilir - hicbir istisna
        firlatilmaz, sampler akisini kesintiye UGRATMAZ.

        Args:
            row: `_DIAGNOSTIC_FIELDNAMES` anahtarlarini iceren satir sozlugu.
                `evidence_id`/`vlm_payload_label` cagiran tarafindan
                VERILMEZ - `selected=True` iken burada, `frame_index`/
                `timestamp_str`/`timestamp_ms`/`net_change_score`/
                `selection_reason`den TUTARLI sekilde turetilir (bkz.
                asagisi); boylece TUM secim yollari (esik/coverage/erken/
                guclu/tek-kare) icin uctan uca izlenebilirlik TEK bir
                yerden garanti edilir.
        """
        row = {key: _native(value) for key, value in row.items()}
        frame_index = row["frame_index"]
        if frame_index in self._recorded_frame_indices:
            row = dict(row)
            row["selected"] = False
            row["selection_reason"] = None
            row["rejection_reason"] = _REJECTION_ALREADY_SELECTED_FRAME_INDEX
            self.duplicate_record_attempts += 1
        self._recorded_frame_indices.add(frame_index)
        self.total_rows += 1

        # UCTAN UCA IZLENEBILIRLIK (bkz. gorev kisiti "VLM payload'undaki
        # sira ve etiket, sampler'daki gercek evidence_id/frame_index/
        # timestamp_ms ile eslessin"): bu kare GERCEKTEN evidence olarak
        # secildiyse (`selected=True`), diagnostic satirina - `EvidenceFrame`
        # nesnesine burada erisim OLMASA da - AYNI `evidence_id` formulunu
        # (`f"ev{frame_index}"`, bkz. `_build_evidence_frame`) ve VLM
        # payload'una GERCEKTEN gomulecek etiketi (bkz.
        # `src.sampler.payload_builder.format_evidence_metadata_label` -
        # `VLMPayloadBuilder.build_content_blocks` ile AYNI fonksiyon, iki
        # ayri f-string'in zamanla birbirinden sapmasi riski YOKTUR) ekler.
        # `selected=False` satirlarda ikisi de `None` kalir - sampler'da hic
        # var olmamis bir kareye asla bir evidence_id/etiket UYDURULMAZ.
        if row.get("selected"):
            timestamp_sec = (row["timestamp_ms"] / 1000.0) if row.get("timestamp_ms") is not None else 0.0
            row["evidence_id"] = f"ev{frame_index}"
            row["vlm_payload_label"] = format_evidence_metadata_label(
                evidence_id=row["evidence_id"],
                frame_index=frame_index,
                # ONEMLI: `row["timestamp_str"]` BURADA KASITLI OLARAK
                # KULLANILMAZ - o, tanilama gorunumu icin milisaniye
                # hassasiyetli `MM:SS.mmm` bicimindedir (bkz.
                # `_readable_timestamp`), ama GERCEK `EvidenceFrame.
                # timestamp_str` (bkz. `_build_evidence_frame`) SANIYE
                # hassasiyetli `MM:SS` bicimindedir. Etiketin VLM'e
                # GERCEKTEN giden metinle BIREBIR ayni olmasi icin ayni
                # `MM:SS` formulu burada da UYGULANIR.
                timestamp_str=f"{int(timestamp_sec) // 60:02d}:{int(timestamp_sec) % 60:02d}",
                timestamp_sec=timestamp_sec,
                change_score=row.get("net_change_score") or 0.0,
                selection_reason=row.get("selection_reason"),
            )
        else:
            row["evidence_id"] = None
            row["vlm_payload_label"] = None

        if row.get("selected"):
            self.selection_reason_counts[row.get("selection_reason") or "unknown"] += 1
        else:
            self.rejection_reason_counts[row.get("rejection_reason") or "unknown"] += 1

        if self._fmt == "csv":
            self._csv_writer.writerow(row)
        else:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._fh.close()

    def build_summary(self, duplicate_selected_frame_indices: Dict[int, int]) -> Dict[str, Any]:
        """Isleme sonunda yazilacak kisa ozet raporunu uretir.

        Args:
            duplicate_selected_frame_indices: NIHAI `evidence_frames`
                ciktisinda (gercek sampler sonucu, tahmin DEGIL) ayni
                `frame_id`nin birden fazla kez gectigi durumlar - `{frame_id:
                tekrar_sayisi}`.

        Returns:
            JSON-serilestirilebilir ozet sozlugu.
        """
        return {
            "diagnostic_output_path": str(self.output_path),
            "total_sampled_frames_logged": self.total_rows,
            "selected_frame_count": sum(self.selection_reason_counts.values()),
            "selection_reason_counts": dict(self.selection_reason_counts),
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "duplicate_record_attempts": self.duplicate_record_attempts,
            "duplicate_selected_frame_indices": duplicate_selected_frame_indices,
        }

# Yogunluk durum makinesi (bkz. modul docstring'i): iki O(1) cooldown zaman
# damgasi ile turetilen, kalici olarak SAKLANMAYAN, her karede yeniden
# hesaplanan durum etiketleri (kumeleme/olay DEGIL - yalnizca secim
# yogunlugu kontrolu).
_DENSITY_CALM = "calm"
_DENSITY_EARLY = "early"
_DENSITY_STRONG = "strong"

if TYPE_CHECKING:
    # Yalnizca tip ipucu icin gereklidir (bkz. `sampler_from_config`); modul
    # ustunde zaten `from __future__ import annotations` oldugundan tip
    # ipuclari calisma zamaninda hic degerlendirilmez. Bu importu calisma
    # zamanindan (runtime) TYPE_CHECKING'e tasimak, yalnizca
    # `AdaptiveFrameSampler`/`EvidenceFrame`'i kullanan (ve
    # `sampler_from_config`'e hic ihtiyaci olmayan) cagiranlar icin gereksiz
    # `pydantic`/`PyYAML` (config_loader) import maliyetini ortadan kaldirir;
    # davranista HICBIR degisiklik yaratmaz.
    from src.utils.config_loader import SamplerConfig

logger = logging.getLogger(__name__)

_DEFAULT_EVIDENCE_OUTPUT_DIR = "outputs/evidence_frames"


def _insert_chronologically(frames: List[EvidenceFrame], new_frame: EvidenceFrame) -> None:
    """`new_frame`i, `frames` listesindeki dogru KRONOLOJIK konuma ekler.

    Coklu evidence akisinda (esik/coverage/erken/guclu-hysteresis) HER ZAMAN
    en son islenmekte olan karenin zaman damgasi eklenir - bu her zaman
    listenin sonuna dogru artan sirada olur. TEK istisna: bir
    `pending_early_candidate` (bkz. `process_video`), onaylanana kadar
    zaman icinde TUTULUR ve confirm edildiginde KENDI (eski) zaman
    damgasiyla eklenir; bu sirada araya (daha yeni zaman damgali) bir
    `temporal_coverage` karesi girmis olabilir. Bu fonksiyon, listeyi
    HER ZAMAN kronolojik tutmak icin dogru konumu geriye dogru arar
    (pratikte en fazla birkac elemanlik bir kaydirma - buyuyen bir STATE
    buffer'i DEGILDIR, yalnizca zaten var olan CIKTI listesine dogru
    sirada ekleme yapar).

    Args:
        frames: Su ana kadar biriktirilmis `EvidenceFrame` listesi (yerinde degistirilir).
        new_frame: Eklenecek yeni kare.
    """
    idx = len(frames)
    while idx > 0 and frames[idx - 1].timestamp_sec > new_frame.timestamp_sec:
        idx -= 1
    frames.insert(idx, new_frame)


@dataclass
class SamplerRunStats:
    """Tek bir `process_video` cagrisi icin CPU suzgec/GPU tasarruf istatistikleri."""

    total_frames_scanned: int
    sampled_frames_evaluated: int
    evidence_frame_count: int
    eliminated_frame_count: int
    eliminated_ratio_pct: float
    elapsed_sec: float


@dataclass
class _PendingSelectionCandidate:
    """Mevcut secim penceresi (sakin/erken/guclu-hysteresis fark etmez) icinde
    su ana kadar goeruelen EN IYI (en yuksek `net_change_score`'lu) esik-alti
    aday; pencere kapanana kadar tek bir ornek olarak tutulur (TUM pencere
    bellekte biriktirilmez - O(1) durum). `gray`, olusturuldugu andaki
    `_preprocess_frame` ciktisidir - secim aninda dedup karsilastirmasi icin
    yeniden hesaplanmasina gerek kalmaz."""

    frame: np.ndarray
    gray: np.ndarray
    frame_id: int
    timestamp_sec: float
    net_change_score: float
    motion_bbox: Optional[Tuple[int, int, int, int]]


class AdaptiveFrameSampler:
    """OpenCV tabanli, CPU uzerinde ultra hizli calisan Uyarlanabilir Kare Ornekleyici.

    Gereksiz/durgun kareleri buyuk oranda eleyerek yalnizca anlamli degisim
    iceren kareleri VLM katmanina iletir. Katman tamamen CPU uzerinde calisir;
    GPU yuku sifirdir. Olay kumelemesi YAPMAZ (bkz. modul docstring'i).
    """

    def __init__(
        self,
        min_change_threshold: float = 0.001,
        blur_kernel_size: Tuple[int, int] = (21, 21),
        history_window: int = 30,
        evidence_output_dir: str = _DEFAULT_EVIDENCE_OUTPUT_DIR,
        temporal_vote_window: int = 1,
        temporal_vote_min_count: int = 1,
        max_temporal_gap_sec: float = 15.0,
        early_change_score_ratio: float = 0.4,
        early_change_window: int = 3,
        early_change_min_count: int = 2,
        early_change_selection_interval_sec: float = 3.0,
        early_change_cooldown_sec: float = 4.0,
        significant_change_selection_interval_sec: float = 1.0,
        strong_change_cooldown_sec: float = 4.0,
        dedup_similarity_ratio: float = 0.5,
        single_frame_change_enabled: bool = True,
        single_frame_change_noise_floor_ratio: float = 2.0,
        single_frame_change_max_area_ratio: float = 0.35,
        long_baseline_change_enabled: bool = True,
        long_baseline_interval_sec: float = 0.5,
        long_baseline_slow_interval_sec: float = 3.0,
        diagnostic_enabled: bool = False,
        diagnostic_output_dir: str = "outputs/diagnostics",
        diagnostic_output_format: str = "jsonl",
    ) -> None:
        """AdaptiveFrameSampler'i esik parametreleriyle baslatir.

        Args:
            min_change_threshold: Bir karenin Kanit Karesi sayilmasi icin
                gereken, gurultu tabani dusulmus minimum piksel degisim orani.
            blur_kernel_size: Gurultu gidermek icin uygulanan Gauss bulaniklastirma
                cekirdek boyutu (tek sayilardan olusan (genislik, yukseklik) ikilisi).
            history_window: Dinamik gurultu tabanini (medyan) hesaplamak icin
                tutulan son degisim orani sayisi.
            evidence_output_dir: Her Kanit Karesinin `.jpg` olarak diske
                yazilacagi klasor (UI ve denetim/log amacli kalicilik icin).
                ONEMLI: Kanit Karesi sayisinda sabit bir ust sinir YOKTUR
                (eski `max_evidence_buffer` kaldirildi); uzun/surekli bir
                olay ne kadar sürerse sürsün hicbir Kanit Karesi burada
                sessizce atlanmaz.
            temporal_vote_window: Bir karenin candidate olarak onaylanip
                onaylanmayacagina karar verirken dikkate alinan, o anki karar
                dahil, en son kac esik-testi sonucunun tutulacagi (temporal
                voting penceresi). Varsayilan `1` ile pencere yalnizca o anki
                karari icerir; bu, mevcut (voting oncesi) davranisla BIREBIR
                AYNIDIR. `min_change_threshold` formulunu veya gurultu
                tabanini DEGISTIRMEZ; yalnizca esigi gecen bir karenin GERCEK
                candidate sayilip sayilmayacagina ek bir "sureklilik" filtresi
                uygular (bkz. `temporal_vote_min_count`). NOT: Bu, esigi
                GECEN bir kareyi asla "kumeleme" amaciyla elemez; yalnizca
                izole/tek karelik gurultuyu (kamera titremesi, sikistirma
                artefakti) esik testinin BIR PARCASI olarak filtreler.
            temporal_vote_min_count: `temporal_vote_window` penceresi
                icinde, bir karenin candidate olarak onaylanmasi icin gereken
                minimum "supheli" (esigi gecen) karar sayisi. Varsayilan `1`
                ile tek bir supheli karar yeterlidir; bu da mevcut davranisla
                BIREBIR AYNIDIR.
            max_temporal_gap_sec: Son evidence karesinden (esik-gecen VEYA
                coverage) bu yana izin verilen azami sessizlik suresi
                (saniye). Bu sure asilirsa, o ana kadar degerlendirilen
                esik-alti adaylar arasindan `net_change_score`'u en yuksek
                olan kare `selection_reason="temporal_coverage"` ile
                evidence listesine eklenir (bkz. modul docstring'i). Bu bir
                OLAY KUMELEMESI DEGILDIR; yalnizca uzun zamansal kor
                noktalari onleyen bir guvenlik agidir. Uc seviye arasinda EN
                GEVSEK (en seyrek) secim araligi budur.
            early_change_score_ratio: `net_change_score >=
                early_change_score_ratio * min_change_threshold` ise bu kare
                "supheli-erken" sayilir (`0 < oran < 1`).
            early_change_window: Erken-degisim onayinda dikkate alinan, en
                son kac "supheli-erken" karar sonucunun tutulacagi (sabit
                boyutlu pencere - `_confirm_candidate` ile ayni desen).
            early_change_min_count: `early_change_window` icinde
                erken-degisim DURUMUNUN onaylanmasi icin gereken minimum
                "supheli-erken" karar sayisi; tek bir anlik skor
                sicramasinin erken-degisim tetiklemesini engeller.
            early_change_selection_interval_sec: Erken-degisim durumu
                aktifken uygulanan azami secim araligi (saniye);
                `max_temporal_gap_sec`den KUCUK olmalidir.
            early_change_cooldown_sec: Erken-degisim sinyali kesildikten
                sonra sakin moda donmeden once beklenen sure (hysteresis).
            significant_change_selection_interval_sec: Ana esik gecildikten
                sonraki hysteresis penceresinde uygulanan azami secim araligi
                (saniye); uc seviye arasinda EN SIK olanidir.
            strong_change_cooldown_sec: Ana esik gecildikten sonra "guclu
                degisim" durumunun korunacagi sure (saniye); skor tekrar
                esigin altina dusse bile bu sure boyunca secim sikligi
                yuksek kalir.
            dedup_similarity_ratio: Esik-alti (coverage/erken/guclu-
                hysteresis) bir aday, SON SECILEN evidence karesine gore fark
                orani `dedup_similarity_ratio * min_change_threshold`den
                DUSUKSE, gorsel olarak "neredeyse ayni" sayilip SECILMEZ. Bu
                kontrol `threshold_exceeded` karelere ASLA uygulanmaz.
            single_frame_change_enabled: `True` ise, ana esigin ALTINDA
                kalan ama TEK bir karede GUCLU, LOKAL ve gurultu tabanindan
                acikca ayrisan bir degisim, `early_change_min_count` cok-
                kareli onayi BEKLENMEDEN aninda `selection_reason=
                "single_frame_change"` ile secilir (bkz. `process_video`).
                Mevcut cok-kareli `early_change` mekanizmasi (hafif ama
                surdurulen sinyaller icin) DEGISMEDEN korunur - bu, ONA EK,
                ayri ve bagimsiz bir yoldur. `False` ile bu yol tamamen
                devre disi kalir (mevcut cok-kareli davranis tek basina
                korunur).
            single_frame_change_noise_floor_ratio: Bir karenin "tek-kare
                guclu degisim" sayilmasi icin `net_change_score`, bir TABAN
                degerin en az bu kadar KATI olmalidir (`net_change_score >=
                single_frame_change_noise_floor_ratio * taban`). Taban
                normalde `adaptive_noise_floor`dur (skor, o anki tipik
                gurultu seviyesinin acikca uzerinde olmali); ancak
                `adaptive_noise_floor` cok erken/tamamen durgun bir sahnede
                sifira cok yakinken (ör. video basinda ya da gurultusuz
                sentetik goruntude), oran anlamsizca buyuyup YAVAS buyuyen
                bir rampanin ilk karesini bile "guclu" gosterebilir - bu
                durumda taban, zaten `min_change_threshold`e oranli olan
                erken-degisim esigine (`early_change_score_ratio *
                min_change_threshold`) DUSER, boylece "guclu" sayilmak icin
                skor daima ANLAMLI bir mutlak buyuklukte olmak ZORUNDADIR.
                Sabit/hard-code bir skor DEGILDIR - her iki tabanda da
                videonun o anki olcegine GORELI kalir. `net_change_score`
                zaten `raw_change_ratio - adaptive_noise_floor` oldugundan
                (bkz. `process_video`), bu oran ayni zamanda "son tipik
                seviyeye gore pozitif sicrama" sinyalini de KAPSAR - ayri bir
                EMA/onceki-skor config alanina GEREK birakmaz.
            single_frame_change_max_area_ratio: Hareket maskesinin
                sinirlayici kutu alaninin (`_bbox_from_mask`) kare alanina
                orani bu degeri ASARSA aday LOKAL sayilmaz (`(0, 1]`
                araliginda) - boylece kareyi butunuyle etkileyen ani
                parlaklik degisimi/kamera titremesi (genelde tum kareye
                yayilir, oran ~1'e yakin) buyuk olcude bastirilir; kucuk/
                lokal nesne hareketleri (dusuk oran) korunur.
            long_baseline_change_enabled: `True` ise, ardisik-ornek
                karsilastirmasina (`self.prev_gray`, DEGISMEDEN korunur) EK
                olarak, `long_baseline_interval_sec`de bir yenilenen DAHA
                ESKI bir referans kareyle de karsilastirma yapilir; bu ikinci
                sinyal yalnizca `is_early_suspicious`i (esik-alti aday
                tespiti) etkileyebilir - `is_suspicious`/`threshold_exceeded`
                yoluna ASLA dokunmaz. `sample_fps` yukseldikce ardisik-ornek
                araligi kisaldigindan, dusuk kontrastli/kademeli baslayan
                (ör. dumanin ilk gorulme ani) olaylarin ORNEK-ARASI degisimi
                gurultu-esigi/bulaniklastirma tarafindan bastirilip
                kaybolabilir - bu ikinci, daha uzun bazli sinyal o kaybi
                telafi eder. `False` ile bu yol tamamen devre disi kalir
                (yalnizca ardisik-ornek karsilastirmasi kullanilir - eski,
                bu mekanizma eklenmeden ONCEKI davranisla BIREBIR AYNI).
            long_baseline_interval_sec: Uzun-baz (hizli kanal) referans
                karenin yenilenme araligi (saniye); `0`dan buyuk olmalidir.
                Orta hizli/dusuk kontrastli baslangiclari (ör. dumanin ilk
                gorulme ani) yakalar.
            long_baseline_slow_interval_sec: Ikinci, DAHA UZUN bir uzun-baz
                referansin yenilenme araligi (saniye); `long_baseline_
                interval_sec`den BUYUK olmalidir. `long_baseline_interval_
                sec` (hizli kanal) tek basina cok YAVAS gelisen olaylari
                (ör. kademeli baslayan bir yangin) yakalamaya YETMEYEBILIR -
                cunku o araligin kendisi de bu kadar yavas bir surecte
                yeterli birikimli fark GORMEYEBILIR. Bu IKINCI, daha uzun
                bazli kanal, AYNI `is_early_suspicious`i (asagida, VEYA
                mantigiyla) besler ve yalnizca cok yavas surecler icin ek
                bir guvenlik agidir - hizli kanalin veya ardisik-ornek
                karsilastirmasinin YERINE GECMEZ, TAMAMEN BAGIMSIZ ve EK
                bir sinyaldir (bkz. `long_baseline_interval_sec`).
            diagnostic_enabled: `True` ise `process_video`, HER ornek kare
                icin tam karar izini (skorlar, esikler, durum, secim/red
                nedeni) `diagnostic_output_dir`e yazar (bkz.
                `_DiagnosticRecorder`). Varsayilan `False` ile bu mekanizma
                TAMAMEN devre disidir - hicbir ek hesaplama/IO yapilmaz,
                mevcut secim davranisi ve performans BIREBIR korunur. Bu
                bir kok-neden/hata-ayiklama aracidir; secim ALGORITMASINI
                DEGISTIRMEZ.
            diagnostic_output_dir: Tanilama dosyalarinin yazilacagi klasor
                (yalnizca `diagnostic_enabled=True` iken kullanilir).
            diagnostic_output_format: `"jsonl"` veya `"csv"` (yalnizca
                `diagnostic_enabled=True` iken kullanilir).

        Raises:
            ValueError: `temporal_vote_window` veya `temporal_vote_min_count`
                1'den kucukse, `temporal_vote_min_count`,
                `temporal_vote_window`den buyukse, `max_temporal_gap_sec`,
                `early_change_selection_interval_sec`,
                `early_change_cooldown_sec`,
                `significant_change_selection_interval_sec` veya
                `strong_change_cooldown_sec` 0'dan kucuk/esitse,
                `early_change_score_ratio` veya `dedup_similarity_ratio`
                `(0, 1]` araliginda degilse, `early_change_window`/
                `early_change_min_count` 1'den kucukse veya
                `early_change_min_count` penceresinden buyukse, ya da
                secim araliklari beklenen `significant < early < calm`
                sirasini bozuyorsa; `single_frame_change_noise_floor_ratio`
                0'dan kucuk/esitse veya `single_frame_change_max_area_ratio`
                `(0, 1]` araliginda degilse.
        """
        if temporal_vote_window < 1:
            raise ValueError(
                f"temporal_vote_window en az 1 olmalidir, verilen: {temporal_vote_window}"
            )
        if temporal_vote_min_count < 1:
            raise ValueError(
                f"temporal_vote_min_count en az 1 olmalidir, verilen: {temporal_vote_min_count}"
            )
        if temporal_vote_min_count > temporal_vote_window:
            raise ValueError(
                f"temporal_vote_min_count ({temporal_vote_min_count}) "
                f"temporal_vote_window'dan ({temporal_vote_window}) buyuk olamaz."
            )
        if max_temporal_gap_sec <= 0:
            raise ValueError(
                f"max_temporal_gap_sec 0'dan buyuk olmalidir, verilen: {max_temporal_gap_sec}"
            )
        if not (0.0 < early_change_score_ratio <= 1.0):
            raise ValueError(
                f"early_change_score_ratio (0, 1] araliginda olmalidir, verilen: {early_change_score_ratio}"
            )
        if early_change_window < 1:
            raise ValueError(
                f"early_change_window en az 1 olmalidir, verilen: {early_change_window}"
            )
        if early_change_min_count < 1:
            raise ValueError(
                f"early_change_min_count en az 1 olmalidir, verilen: {early_change_min_count}"
            )
        if early_change_min_count > early_change_window:
            raise ValueError(
                f"early_change_min_count ({early_change_min_count}) "
                f"early_change_window'dan ({early_change_window}) buyuk olamaz."
            )
        if early_change_selection_interval_sec <= 0:
            raise ValueError(
                "early_change_selection_interval_sec 0'dan buyuk olmalidir, "
                f"verilen: {early_change_selection_interval_sec}"
            )
        if early_change_cooldown_sec <= 0:
            raise ValueError(
                f"early_change_cooldown_sec 0'dan buyuk olmalidir, verilen: {early_change_cooldown_sec}"
            )
        if significant_change_selection_interval_sec <= 0:
            raise ValueError(
                "significant_change_selection_interval_sec 0'dan buyuk olmalidir, "
                f"verilen: {significant_change_selection_interval_sec}"
            )
        if strong_change_cooldown_sec <= 0:
            raise ValueError(
                f"strong_change_cooldown_sec 0'dan buyuk olmalidir, verilen: {strong_change_cooldown_sec}"
            )
        if not (0.0 < dedup_similarity_ratio <= 1.0):
            raise ValueError(
                f"dedup_similarity_ratio (0, 1] araliginda olmalidir, verilen: {dedup_similarity_ratio}"
            )
        if single_frame_change_noise_floor_ratio <= 0:
            raise ValueError(
                "single_frame_change_noise_floor_ratio 0'dan buyuk olmalidir, "
                f"verilen: {single_frame_change_noise_floor_ratio}"
            )
        if not (0.0 < single_frame_change_max_area_ratio <= 1.0):
            raise ValueError(
                "single_frame_change_max_area_ratio (0, 1] araliginda olmalidir, "
                f"verilen: {single_frame_change_max_area_ratio}"
            )
        if not (
            significant_change_selection_interval_sec
            <= early_change_selection_interval_sec
            <= max_temporal_gap_sec
        ):
            raise ValueError(
                "Secim araliklari significant_change_selection_interval_sec "
                f"({significant_change_selection_interval_sec}) <= "
                f"early_change_selection_interval_sec ({early_change_selection_interval_sec}) <= "
                f"max_temporal_gap_sec ({max_temporal_gap_sec}) sirasini izlemelidir."
            )
        if diagnostic_output_format not in ("csv", "jsonl"):
            raise ValueError(
                f"diagnostic_output_format 'csv' veya 'jsonl' olmalidir, verilen: {diagnostic_output_format!r}"
            )
        if long_baseline_interval_sec <= 0:
            raise ValueError(
                f"long_baseline_interval_sec 0'dan buyuk olmalidir, verilen: {long_baseline_interval_sec}"
            )
        if long_baseline_slow_interval_sec <= long_baseline_interval_sec:
            raise ValueError(
                "long_baseline_slow_interval_sec "
                f"({long_baseline_slow_interval_sec}) long_baseline_interval_sec'den "
                f"({long_baseline_interval_sec}) BUYUK olmalidir."
            )

        self.min_change_threshold = min_change_threshold
        self.blur_kernel_size = blur_kernel_size
        self.history_window = history_window
        self.evidence_output_dir = Path(evidence_output_dir)
        self.temporal_vote_window = temporal_vote_window
        self.temporal_vote_min_count = temporal_vote_min_count
        self.max_temporal_gap_sec = max_temporal_gap_sec
        self.early_change_score_ratio = early_change_score_ratio
        self.early_change_window = early_change_window
        self.early_change_min_count = early_change_min_count
        self.early_change_selection_interval_sec = early_change_selection_interval_sec
        self.early_change_cooldown_sec = early_change_cooldown_sec
        self.significant_change_selection_interval_sec = significant_change_selection_interval_sec
        self.strong_change_cooldown_sec = strong_change_cooldown_sec
        self.dedup_similarity_ratio = dedup_similarity_ratio
        self.single_frame_change_enabled = single_frame_change_enabled
        self.single_frame_change_noise_floor_ratio = single_frame_change_noise_floor_ratio
        self.single_frame_change_max_area_ratio = single_frame_change_max_area_ratio
        self.long_baseline_change_enabled = long_baseline_change_enabled
        self.long_baseline_interval_sec = long_baseline_interval_sec
        self.long_baseline_slow_interval_sec = long_baseline_slow_interval_sec
        self.diagnostic_enabled = diagnostic_enabled
        self.diagnostic_output_dir = Path(diagnostic_output_dir)
        self.diagnostic_output_format = diagnostic_output_format

        self.prev_gray: np.ndarray | None = None
        self.noise_floor_history: List[float] = []
        # Uzun-baz karsilastirma (bkz. `long_baseline_change_enabled`):
        # `self.prev_gray` gibi, taze bir video icin dogal olarak `None`
        # baslar; periyodik olarak yenilenir (bkz. `process_video`).
        self.long_baseline_gray: np.ndarray | None = None
        self.long_baseline_noise_floor_history: List[float] = []
        # Ikinci, DAHA UZUN bazli kanal (bkz. `long_baseline_slow_interval_sec`
        # docstring'i) - cok yavas gelisen olaylar icin, BIRINCI kanalla
        # AYNI desen, TAMAMEN BAGIMSIZ kendi durumu.
        self.long_baseline_gray_slow: np.ndarray | None = None
        self.long_baseline_noise_floor_history_slow: List[float] = []
        self.last_run_stats: Optional[SamplerRunStats] = None
        self._recent_threshold_decisions: Deque[bool] = deque(maxlen=temporal_vote_window)
        # Erken-degisim onayi icin sabit boyutlu pencere (bkz.
        # `_confirm_early_change` - `_confirm_candidate` ile ayni desen).
        self._recent_early_decisions: Deque[bool] = deque(maxlen=early_change_window)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Goruntuyu gri tonlamaya cevirip Gauss bulaniklastirma ile gurultuyu yumusatir.

        Args:
            frame: BGR formatinda ham video karesi.

        Returns:
            Gri tonlamali, bulaniklastirilmis kare.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        return blurred

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Ikili bir hareket maskesinden sinirlayici kutuyu (bounding box) cikarir.

        Args:
            mask: `cv2.threshold` ile uretilmis ikili (0/255) hareket maskesi.

        Returns:
            `(x_min, y_min, x_max, y_max)` — maskede hic hareketli piksel
            yoksa `None`.
        """
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def _confirm_candidate(self, is_suspicious: bool) -> bool:
        """Temporal voting: bir karenin GERCEK candidate olarak onaylanip
        onaylanmayacagina, son `temporal_vote_window` esik-testi sonucuna
        bakarak karar verir.

        Mevcut `min_change_threshold` formulunu DEGISTIRMEZ; yalnizca onun
        ciktisini (`is_suspicious`) girdi olarak kullanan ek bir sureklilik
        filtresidir. `is_suspicious` her cagrida (kare esigi gecsin ya da
        gecmesin) pencereye eklenir, boylece pencere daima son N kararin
        GERCEK gecmisini yansitir. Varsayilan `temporal_vote_window=1`,
        `temporal_vote_min_count=1` ile bu metod `is_suspicious` ile
        birebir ayni sonucu doner (voting-oncesi davranisla tam uyumlu).

        Args:
            is_suspicious: Bu karenin `net_change_score >= min_change_threshold`
                testini gecip gecmedigi (mevcut, degismemis esik sonucu).

        Returns:
            `True` ise kare GERCEK candidate olarak onaylanir (hem kendisi
            supheli HEM DE pencerede yeterli sureklilik var); `False` ise
            (esigi gecmedi ya da izole/yetersiz sureklilik) reddedilir.
        """
        self._recent_threshold_decisions.append(is_suspicious)
        if not is_suspicious:
            return False
        vote_count = sum(self._recent_threshold_decisions)
        return vote_count >= self.temporal_vote_min_count

    def _confirm_early_change(self, is_early_suspicious: bool) -> bool:
        """Erken-degisim onayi: ana esigin ALTINDAKI bir sinyalin, TEK bir
        anlik skor sicramasi degil, SURDURULEN bir egilim olup olmadigina
        `early_change_window` icindeki son kararlara bakarak karar verir.

        `_confirm_candidate` ile AYNI desen (sabit boyutlu deque + oy
        sayimi); `min_change_threshold`/`net_change_score` formulunu
        DEGISTIRMEZ, yalnizca `early_change_score_ratio * min_change_threshold`
        esigini gecen kareleri girdi olarak alir.

        Args:
            is_early_suspicious: Bu karenin `net_change_score >=
                early_change_score_ratio * min_change_threshold` testini
                gecip gecmedigi.

        Returns:
            `True` ise erken-degisim DURUMU onaylanir (pencere icinde
            yeterli sayida "supheli-erken" karar birikmis); `False` ise
            (izole bir sicrama ya da hic sinyal yok) onaylanmaz.
        """
        self._recent_early_decisions.append(is_early_suspicious)
        if not is_early_suspicious:
            return False
        vote_count = sum(self._recent_early_decisions)
        return vote_count >= self.early_change_min_count

    def _is_near_duplicate(
        self, candidate_gray: np.ndarray, reference_gray: Optional[np.ndarray]
    ) -> bool:
        """Bir esik-alti adayin, SON SECILEN evidence karesine gore gorsel
        olarak "neredeyse ayni" olup olmadigini kontrol eder.

        YALNIZCA coverage/early_change/significant_change (esik-alti)
        secimlerine uygulanir - `threshold_exceeded` kareler bu kontrolden
        HICBIR ZAMAN gecirilmez (cagiran taraf sorumlulugundadir), boylece
        "esigi gecen hicbir kare ... deduplication ... nedeniyle elenmesin"
        garantisi korunur. Zaman olarak uzak ama GORSEL olarak farkli iki
        kareyi (ör. bir olayin baslangic karesi, ondan once secilmis sakin
        bir kareye kiyasla) yanlislikla "duplicate" saymaz - yalnizca
        `reference_gray`ye (en son secilen kare) gore ayirt edilemeyecek
        kadar benzer olan kareleri eler.

        Args:
            candidate_gray: Aday karenin (`_preprocess_frame` ciktisi)
                gri-tonlamali hali.
            reference_gray: Son SECILEN evidence karesinin gri-tonlamali
                hali; henuz hic evidence secilmediyse `None` (bu durumda
                karsilastirilacak bir referans yoktur, duplicate SAYILMAZ).

        Returns:
            `True` ise aday, son secilen kareyle gorsel olarak ayirt
            edilemeyecek kadar benzer (SECILMEMELI); aksi halde `False`.
        """
        if reference_gray is None:
            return False
        diff = cv2.absdiff(candidate_gray, reference_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        diff_ratio = np.sum(thresh > 0) / float(thresh.size)
        return diff_ratio < (self.dedup_similarity_ratio * self.min_change_threshold)

    @staticmethod
    def _density_state(timestamp_sec: float, strong_cooldown_until: float, early_cooldown_until: float) -> str:
        """Su anki YOGUNLUK durumunu, iki O(1) cooldown zaman damgasindan turetir.

        Kalici olarak SAKLANMAZ - her karede bu fonksiyonla yeniden
        hesaplanir (bkz. modul docstring'i). Olay kumelemesi/baslangic-bitis
        zamani DEGILDIR; yalnizca secim sikligi kontrolu icindir.

        Args:
            timestamp_sec: Degerlendirilen karenin zaman damgasi.
            strong_cooldown_until: Ana esik gecisinin GUCLU DEGISIM
                durumunu koruyacagi zaman damgasi (bkz. `strong_change_cooldown_sec`).
            early_cooldown_until: Erken-degisim onayinin durumu koruyacagi
                zaman damgasi (bkz. `early_change_cooldown_sec`).

        Returns:
            `_DENSITY_STRONG` | `_DENSITY_EARLY` | `_DENSITY_CALM`.
        """
        if timestamp_sec < strong_cooldown_until:
            return _DENSITY_STRONG
        if timestamp_sec < early_cooldown_until:
            return _DENSITY_EARLY
        return _DENSITY_CALM

    def _selection_interval_and_reason(self, density_state: str) -> Tuple[float, str]:
        """Verilen yogunluk durumu icin uygulanacak azami secim araligini ve
        `selection_reason` degerini dondurur (bkz. modul docstring'i).

        Args:
            density_state: `_density_state()` ciktisi.

        Returns:
            `(secim_araligi_saniye, selection_reason)` ikilisi.
        """
        if density_state == _DENSITY_STRONG:
            return self.significant_change_selection_interval_sec, _SELECTION_REASON_SIGNIFICANT_CHANGE
        if density_state == _DENSITY_EARLY:
            return self.early_change_selection_interval_sec, _SELECTION_REASON_EARLY_CHANGE
        return self.max_temporal_gap_sec, _SELECTION_REASON_COVERAGE

    def _encode_frame_jpeg(self, frame: np.ndarray) -> bytes:
        """Kareyi tek seferde JPEG'e kodlar (disk ve base64 tarafindan ortak kullanilir).

        Args:
            frame: BGR formatinda video karesi.

        Returns:
            JPEG-kodlu ham kare baytlari.

        Raises:
            RuntimeError: Kare JPEG formatina kodlanamazsa.
        """
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("Kare JPEG formatina kodlanamadi.")
        return buffer.tobytes()

    def process_video(self, video_path: str, sample_fps: int = 30) -> List[EvidenceFrame]:
        """Videoyu okur, kare farklarini hesaplar ve suzulmus Kanit Karelerini dondurur.

        Esik-gecmis HICBIR kare burada elenmez: sabit bir ust sinir yoktur
        (eski `max_evidence_buffer` kaldirildi), kumeleme/deduplication/liste
        kesme YAPILMAZ. Her kare tam olarak BIR kez okunur ve BIR kez JPEG'e
        kodlanir (bkz. `_encode_frame_jpeg`); VLM katmani ayni baytlari
        yeniden kullanir (bkz. `src/sampler/payload_builder.py`).

        Ayrica, VLM'e gonderilecek karelerin SECIM SIKLIGI uc yogunluk
        seviyesi arasinda uyarlanir (kaynak/analiz `sample_fps`'i SABIT
        kalir - bkz. modul docstring'i): sakin bolgede `max_temporal_gap_sec`
        araligiyla (`selection_reason="temporal_coverage"`), ana esigin
        ALTINDA ama surdurulen bir onset sinyali basladiginda
        `early_change_selection_interval_sec` araligiyla
        (`"early_change"` - onaylanan ilk kare ANINDA secilir), ve ana esik
        gecildikten sonraki hysteresis penceresinde
        `significant_change_selection_interval_sec` araligiyla
        (`"significant_change"`). Bu kumeleme degildir, olay baslangic/bitis
        zamani URETMEZ.

        Args:
            video_path: `.mp4` dosya yolu veya RTSP URI'si.
            sample_fps: Videonun kac saniyede bir kare kontrol edilecegini
                belirleyen ornekleme hizi (native FPS'ten dusuk olmalidir).

        Returns:
            Zaman sirali (kronolojik), kayipsiz `EvidenceFrame` listesi.

        Raises:
            ValueError: Video dosyasi acilamazsa.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Video dosyasi acilamadi: {video_path}")

        started_at = time.perf_counter()
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_step = max(1, int(native_fps / sample_fps))

        # Temporal voting VE erken-degisim penceresi (bkz. `_confirm_candidate`/
        # `_confirm_early_change`) her yeni video icin sifirlanir; onceki bir
        # `process_video` cagrisinin (varsa) son kararlari bu videoya tasinmaz.
        self._recent_threshold_decisions.clear()
        self._recent_early_decisions.clear()

        # Uzun-baz karsilastirma icin bu videoya OZEL zaman-damgasi sayaci
        # (bkz. `long_baseline_change_enabled`); `self.long_baseline_gray`nin
        # kendisi (`prev_gray`/`noise_floor_history` ile AYNI desen geregi)
        # ornekler-arasi taze bir `AdaptiveFrameSampler` uzerinden dogal
        # olarak sifirlanir (bkz. `sampler_from_config`).
        long_baseline_timestamp: Optional[float] = None
        # Bir onceki ornegin uzun-baz net skoru (bkz. asagisi - "hala
        # GELISMEKTE mi" testi icin).
        long_baseline_prev_net_score: float = 0.0
        # Ikinci, DAHA UZUN bazli kanal icin AYNI desen (bkz.
        # `long_baseline_slow_interval_sec`) - cok yavas gelisen olaylar
        # icin, BIRINCI kanaldan TAMAMEN BAGIMSIZ.
        long_baseline_slow_timestamp: Optional[float] = None
        long_baseline_slow_prev_net_score: float = 0.0

        evidence_frames: List[EvidenceFrame] = []
        frame_id = 0
        sampled_frame_count = 0
        first_frame_raw: Optional[np.ndarray] = None
        first_frame_timestamp: float = 0.0

        # Secim durumu (TUMU O(1), buyuyen bir buffer/ring buffer DEGIL):
        # - `last_evidence_timestamp`: son SECILEN evidence karesinin zaman
        #   damgasi (gap sayacinin referansi; video BASLARKEN gecen sessiz
        #   sure de kapsama tetigine dahil olsun diye video basinda ayarlanir).
        # - `pending_best`: mevcut secim penceresinde su ana kadar goeruelen
        #   EN IYI esik-alti aday (tum pencere degil, TEK ornek).
        # - `last_selected_gray`: son SECILEN evidence karesinin gri-tonlamasi
        #   (yalnizca esik-alti secimler icin duplicate kontrolu referansi -
        #   bkz. `_is_near_duplicate`; `threshold_exceeded` kareleri ASLA etkilemez).
        # - `strong_cooldown_until`/`early_cooldown_until`: guclu/erken
        #   durumun korunacagi zaman damgasi (video ile ayni birimde, saniye);
        #   hicbir zaman tetiklenmediyse `-inf` (her zaman "gecmis").
        last_evidence_timestamp: Optional[float] = None
        pending_best: Optional[_PendingSelectionCandidate] = None
        last_selected_gray: Optional[np.ndarray] = None
        strong_cooldown_until: float = float("-inf")
        early_cooldown_until: float = float("-inf")
        # `pending_early_candidate`: SAKIN durumdayken bir erken-degisim
        # zincirinin ILK supheli-erken karesi (bkz. modul docstring'i).
        # `_PendingSelectionCandidate` (pending_best ile AYNI dataclass,
        # yeniden kullanilir - ayri bir veri modeli GEREKMEZ) tek bir
        # ornek olarak tutulur; onaylandiginda (`early_change_min_count`ye
        # ulasildiginda) BU kare secilir, doğrulama aninin karesi DEGIL.
        pending_early_candidate: Optional[_PendingSelectionCandidate] = None

        # --- Tanilama (diagnostic) modu: yalnizca `diagnostic_enabled=True`
        # iken kurulur; KAPALIYKEN `diag` `None` kalir ve asagidaki TUM
        # `if diag is not None:` bloklari atlanir - hicbir ek hesaplama/IO
        # yapilmaz (bkz. `_DiagnosticRecorder`, gorev kisiti "kapaliyken
        # mevcut davranisi/performansi degistirmesin"). `pending_best_diag`/
        # `pending_early_diag`, gercek `pending_best`/`pending_early_candidate`
        # ile AYNI yasam dongusunu (O(1), TEK ornek) izleyen, YALNIZCA
        # tanilama amacli GOLGE sozlukleridir - sampler'in secim KARARLARINI
        # hicbir sekilde ETKILEMEZ/OKUMAZ (yalniz-yazilir gozlem katmani).
        diag: Optional[_DiagnosticRecorder] = None
        pending_best_diag: Optional[Dict[str, Any]] = None
        pending_early_diag: Optional[Dict[str, Any]] = None
        if self.diagnostic_enabled:
            ext = "csv" if self.diagnostic_output_format == "csv" else "jsonl"
            run_stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
            diag_path = self.diagnostic_output_dir / f"sampler_diagnostic_{Path(video_path).stem}_{run_stamp}.{ext}"
            diag = _DiagnosticRecorder(diag_path, self.diagnostic_output_format)
            logger.info("Sampler tanilama (diagnostic) modu ACIK: %s", diag_path)

        def _diag_row(
            *,
            fid: int,
            ts: float,
            net_score: Optional[float],
            raw_ratio: Optional[float],
            noise_floor: Optional[float],
            early_exceeded: Optional[bool],
            main_exceeded: Optional[bool],
            early_confirmed: Optional[bool],
            density: Optional[str],
            vote_passed: Optional[bool],
            interval_sec: Optional[float],
            interval_elapsed: Optional[bool],
            gap_ms: Optional[int],
            dedup_checked: bool,
            dedup_is_dup: Optional[bool],
            selected: bool,
            selection_reason: Optional[str],
            rejection_reason: Optional[str],
            long_baseline_net_score: Optional[float] = None,
            long_baseline_early_suspicious: Optional[bool] = None,
            long_baseline_slow_net_score: Optional[float] = None,
            long_baseline_slow_early_suspicious: Optional[bool] = None,
        ) -> Dict[str, Any]:
            """Bir tanilama satirinin TUM alanlarini, cagiran taraftan gelen
            kare-ozel degerlerle + o anki (kapanis-uzeninden okunan, DEGISTIRMEYEN)
            cooldown/pending durumuyla birlestirir. Goruntu/base64 ICERMEZ."""
            return {
                "frame_index": fid,
                "timestamp_ms": _ms(ts),
                "timestamp_str": _readable_timestamp(ts),
                "net_change_score": net_score,
                "raw_change_ratio": raw_ratio,
                "adaptive_noise_floor": noise_floor,
                "tile_scores": None,  # Mevcut algoritma lokal/tile skoru HESAPLAMAZ - bkz. rapor notu.
                "early_threshold_value": self.early_change_score_ratio * self.min_change_threshold,
                "main_threshold_value": self.min_change_threshold,
                "early_threshold_exceeded": early_exceeded,
                "main_threshold_exceeded": main_exceeded,
                "recent_early_decisions": "".join("T" if v else "F" for v in self._recent_early_decisions),
                "recent_early_true_count": sum(self._recent_early_decisions),
                "early_change_confirmed": early_confirmed,
                "density_state": density,
                "pending_early_candidate_frame_index": (
                    pending_early_candidate.frame_id if pending_early_candidate is not None else None
                ),
                "pending_early_candidate_timestamp_ms": (
                    _ms(pending_early_candidate.timestamp_sec) if pending_early_candidate is not None else None
                ),
                "temporal_vote_passed": vote_passed,
                "temporal_vote_true_count": sum(self._recent_threshold_decisions),
                "temporal_vote_min_count": self.temporal_vote_min_count,
                "strong_cooldown_active": ts < strong_cooldown_until,
                "strong_cooldown_until_ms": _finite_ms(strong_cooldown_until),
                "early_cooldown_active": ts < early_cooldown_until,
                "early_cooldown_until_ms": _finite_ms(early_cooldown_until),
                "selection_interval_sec": interval_sec,
                "selection_interval_elapsed": interval_elapsed,
                "gap_since_last_evidence_ms": gap_ms,
                "dedup_checked": dedup_checked,
                "dedup_is_duplicate": dedup_is_dup,
                "coverage_window_start_ms": _ms(last_evidence_timestamp) if last_evidence_timestamp is not None else None,
                "coverage_window_end_ms": _ms(ts),
                "selected": selected,
                "selection_reason": selection_reason,
                "rejection_reason": rejection_reason,
                "long_baseline_net_score": long_baseline_net_score,
                "long_baseline_early_suspicious": long_baseline_early_suspicious,
                "long_baseline_slow_net_score": long_baseline_slow_net_score,
                "long_baseline_slow_early_suspicious": long_baseline_slow_early_suspicious,
            }

        def _discard_pending_early_if_same_frame(flushed_frame_id: int) -> None:
            """`pending_best` tam da `flushed_frame_id` icin evidence'e
            EKLENDIGINDE cagrilir: eger `pending_early_candidate` HALA AYNI
            `frame_id`yi tutuyorsa (ikisi de ayni orijinal kareyi, "zincirin
            ilk karesini yakala" bloğunda AYNI iterasyonda BAGIMSIZ olarak
            aday gostermis olabilir - bkz. modul rapor notu #2), onu GUVENLI
            sekilde iptal eder - aksi halde bu kare DAHA SONRA cok-kareli
            erken-degisim onayi tamamlandiginda `early_change` ile IKINCI
            kez eklenebilirdi (dedup, YALNIZCA `last_selected_gray`ye -son
            secilen kareye- gore calisir; aradan BASKA bir kare secilmisse
            bu ikinci eklemeyi YAKALAYAMAZ). O(1): tek bir tam-sayi
            karsilastirmasidir, buyuyen bir yapi GEREKMEZ."""
            nonlocal pending_early_candidate, pending_early_diag
            if (
                pending_early_candidate is not None
                and pending_early_candidate.frame_id == flushed_frame_id
            ):
                pending_early_candidate = None
                if diag is not None and pending_early_diag is not None:
                    row = dict(pending_early_diag)
                    row["selected"] = False
                    row["selection_reason"] = None
                    row["rejection_reason"] = _REJECTION_ALREADY_SELECTED_FRAME_INDEX
                    diag.record(row)
                    pending_early_diag = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % frame_step != 0:
                    frame_id += 1
                    continue

                sampled_frame_count += 1
                timestamp_sec = frame_id / native_fps
                curr_gray = self._preprocess_frame(frame)

                if first_frame_raw is None:
                    first_frame_raw = frame.copy()
                    first_frame_timestamp = timestamp_sec
                    last_evidence_timestamp = timestamp_sec

                if self.prev_gray is not None:
                    frame_diff = cv2.absdiff(curr_gray, self.prev_gray)
                    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                    change_ratio = np.sum(thresh > 0) / float(thresh.size)

                    self.noise_floor_history.append(change_ratio)
                    if len(self.noise_floor_history) > self.history_window:
                        self.noise_floor_history.pop(0)

                    adaptive_noise_floor = np.median(self.noise_floor_history)
                    net_change_score = max(0.0, change_ratio - adaptive_noise_floor)

                    # Uzun-baz karsilastirma (bkz. `long_baseline_change_enabled`
                    # docstring'i - kok neden: yuksek `sample_fps`de ardisik-ornek
                    # araligi kisaldigindan, dusuk kontrastli/kademeli baslayan bir
                    # olayin ORNEK-ARASI degisimi blur+sabit piksel-fark esigiyle
                    # (25) birlesince SIFIRA duser; `min_change_threshold` bunu
                    # DUZELTEMEZ cunku ham sinyalin kendisi kayboluyor). `self.
                    # prev_gray` (ardisik-ornek) DEGISTIRILMEDEN korunur - bu,
                    # TAMAMEN AYRI, EK bir sinyaldir. Yalnizca `is_early_
                    # suspicious`i (asagida) etkileyebilir; `is_suspicious`/
                    # `threshold_exceeded` yoluna (kosulsuz secim) ASLA dokunmaz.
                    long_net_score: Optional[float] = None
                    long_is_early_suspicious = False
                    if self.long_baseline_change_enabled:
                        if self.long_baseline_gray is None:
                            self.long_baseline_gray = curr_gray
                            long_baseline_timestamp = timestamp_sec
                        else:
                            long_diff = cv2.absdiff(curr_gray, self.long_baseline_gray)
                            _, long_thresh = cv2.threshold(long_diff, 25, 255, cv2.THRESH_BINARY)
                            long_raw_ratio = np.sum(long_thresh > 0) / float(long_thresh.size)
                            long_noise_floor = (
                                np.median(self.long_baseline_noise_floor_history)
                                if self.long_baseline_noise_floor_history
                                else 0.0
                            )
                            long_net_score = max(0.0, long_raw_ratio - long_noise_floor)
                            # ONEMLI (yanlis-pozitif koruma - bkz. gorev notu
                            # "duragan/gurultulu sahnelerde yanlis evidence
                            # uretmesin"): TEK SEFERLIK bir sicrama (ör. bir
                            # yamanin belirip SABIT kalmasi), referans HENUZ
                            # yenilenmediyse, referansa gore uzun sure "yuksek"
                            # gorunmeye devam eder - ama bu SURDURULEN bir
                            # GELISME DEGILDIR, TEK bir adimdir. Bu yuzden
                            # "supheli" saymak icin skorun bir ONCEKI ornege
                            # gore HALA ARTMAKTA olmasi da GEREKIR (gercek
                            # kademeli buyume - ör. dumanin genisemesi - HER
                            # ornekte referansa gore biraz DAHA FAZLA fark
                            # biriktirir; tek seferlik bir adim ise referans
                            # yenilenene kadar SABIT/DUZ kalir, bir DAHA
                            # ARTMAZ).
                            long_is_early_suspicious = (
                                long_net_score >= (self.early_change_score_ratio * self.min_change_threshold)
                                and long_net_score > long_baseline_prev_net_score
                            )
                            long_baseline_prev_net_score = long_net_score
                            if (
                                long_baseline_timestamp is None
                                or timestamp_sec - long_baseline_timestamp >= self.long_baseline_interval_sec
                            ):
                                # Referans yenilenmeden ONCE, o anki ham fark orani
                                # kendi gurultu-tabani gecmisine eklenir (bkz.
                                # `noise_floor_history` ile AYNI desen, ama uzun-baz
                                # araligina GORELI) - boylece bu sinyalin de
                                # KENDI adaptif tabani vardir, kisa-baz `adaptive_
                                # noise_floor`u YENIDEN KULLANMAZ (o, cok DAHA KISA
                                # bir araliga gore kalibredir).
                                self.long_baseline_noise_floor_history.append(long_raw_ratio)
                                if len(self.long_baseline_noise_floor_history) > self.history_window:
                                    self.long_baseline_noise_floor_history.pop(0)
                                self.long_baseline_gray = curr_gray
                                long_baseline_timestamp = timestamp_sec
                                # Yeni referansa gore skor SIFIRDAN baslar - bir
                                # onceki (ESKI referansa gore hesaplanmis, farkli
                                # olcekteki) skorla KARSILASTIRILAMAZ.
                                long_baseline_prev_net_score = 0.0

                    # Ikinci, DAHA UZUN bazli kanal (bkz. `long_baseline_slow_
                    # interval_sec` docstring'i): yukaridaki (hizli, ör. 0.5s)
                    # kanal orta hizli/dusuk kontrastli baslangiclari yakalar,
                    # ama COK yavas gelisen bir olayda (ör. kademeli baslayan
                    # bir yangin) o kisa araligin KENDISI de yeterli birikimli
                    # fark GORMEYEBILIR. BIRINCI kanalla BIREBIR AYNI desen -
                    # TAMAMEN BAGIMSIZ durum, KENDI gurultu tabani, KENDI
                    # "hala artmakta mi" korumasi.
                    long_slow_net_score: Optional[float] = None
                    long_slow_is_early_suspicious = False
                    if self.long_baseline_change_enabled:
                        if self.long_baseline_gray_slow is None:
                            self.long_baseline_gray_slow = curr_gray
                            long_baseline_slow_timestamp = timestamp_sec
                        else:
                            long_slow_diff = cv2.absdiff(curr_gray, self.long_baseline_gray_slow)
                            _, long_slow_thresh = cv2.threshold(long_slow_diff, 25, 255, cv2.THRESH_BINARY)
                            long_slow_raw_ratio = np.sum(long_slow_thresh > 0) / float(long_slow_thresh.size)
                            long_slow_noise_floor = (
                                np.median(self.long_baseline_noise_floor_history_slow)
                                if self.long_baseline_noise_floor_history_slow
                                else 0.0
                            )
                            long_slow_net_score = max(0.0, long_slow_raw_ratio - long_slow_noise_floor)
                            long_slow_is_early_suspicious = (
                                long_slow_net_score >= (self.early_change_score_ratio * self.min_change_threshold)
                                and long_slow_net_score > long_baseline_slow_prev_net_score
                            )
                            long_baseline_slow_prev_net_score = long_slow_net_score
                            if (
                                long_baseline_slow_timestamp is None
                                or timestamp_sec - long_baseline_slow_timestamp >= self.long_baseline_slow_interval_sec
                            ):
                                self.long_baseline_noise_floor_history_slow.append(long_slow_raw_ratio)
                                if len(self.long_baseline_noise_floor_history_slow) > self.history_window:
                                    self.long_baseline_noise_floor_history_slow.pop(0)
                                self.long_baseline_gray_slow = curr_gray
                                long_baseline_slow_timestamp = timestamp_sec
                                long_baseline_slow_prev_net_score = 0.0

                    # Temporal voting: mevcut esik sonucunu (degismedi) girdi olarak
                    # kullanip, izole/tek karelik supheli kareleri (kamera titremesi,
                    # isik degisimi, sikistirma artefakti) eleyen ek sureklilik kontrolu.
                    # Varsayilan ayarlarda (window=1, min_count=1) bu, ESKI davranisla
                    # BIREBIR AYNIDIR (bkz. `_confirm_candidate` docstring'i).
                    is_suspicious = net_change_score >= self.min_change_threshold
                    # (Diagnostic icin isimlendirilmis; karar mantigini DEGISTIRMEZ -
                    # `if self._confirm_candidate(is_suspicious):` ile BIREBIR ayni.)
                    temporal_vote_passed = self._confirm_candidate(is_suspicious)
                    if temporal_vote_passed:
                        # Sabit bir ust sinir YOKTUR: video ne kadar uzun/olay ne kadar
                        # surekli olursa olsun, hicbir Kanit Karesi burada sessizce
                        # atlanmaz. Kumeleme artik VLM katmaninda yapilir. Bu dal,
                        # yogunluk mekanizmasi eklenmeden ONCEKI davranisla BIREBIR
                        # AYNIDIR: HER esik-gecen kare kosulsuz secilir, dedup
                        # UYGULANMAZ (bkz. `_is_near_duplicate` docstring'i).
                        motion_bbox = self._bbox_from_mask(thresh)
                        diag_gap_ms = (
                            _ms(timestamp_sec) - _ms(last_evidence_timestamp)
                            if diag is not None and last_evidence_timestamp is not None
                            else None
                        )
                        diag_density_at_arrival = (
                            self._density_state(timestamp_sec, strong_cooldown_until, early_cooldown_until)
                            if diag is not None
                            else None
                        )

                        # Ana esik, erken-degisim onayi TAMAMLANMADAN gecilmis
                        # olabilir (ör. sinyal cok hizli buyudu). Eger su an
                        # onaylanmamis bir `pending_early_candidate` (ayni
                        # surekli degisim zincirinin BASLANGIC karesi) varsa,
                        # ana esik karesinin "golgesinde" kaybolmasin diye
                        # ONCE o (kendi eski zaman damgasiyla, kronolojik
                        # dogru konuma) eklenir - boylece zincirin GERCEK
                        # baslangici, dogrudan esige sicrayan olaylarda da
                        # korunur (bkz. gorev notu #6).
                        pending_early_is_dup = (
                            pending_early_candidate is not None
                            and self._is_near_duplicate(pending_early_candidate.gray, last_selected_gray)
                        )
                        if pending_early_candidate is not None and not pending_early_is_dup:
                            _insert_chronologically(
                                evidence_frames,
                                self._build_evidence_frame(
                                    pending_early_candidate.frame,
                                    pending_early_candidate.frame_id,
                                    pending_early_candidate.timestamp_sec,
                                    pending_early_candidate.net_change_score,
                                    pending_early_candidate.motion_bbox,
                                    selection_reason=_SELECTION_REASON_EARLY_CHANGE,
                                ),
                            )
                            last_selected_gray = pending_early_candidate.gray
                        if diag is not None and pending_early_diag is not None:
                            row = dict(pending_early_diag)
                            row["selected"] = not pending_early_is_dup
                            row["selection_reason"] = _SELECTION_REASON_EARLY_CHANGE if not pending_early_is_dup else None
                            row["rejection_reason"] = _REJECTION_NEAR_DUPLICATE if pending_early_is_dup else None
                            row["dedup_checked"] = True
                            row["dedup_is_duplicate"] = pending_early_is_dup
                            diag.record(row)
                            pending_early_diag = None
                        pending_early_candidate = None

                        evidence_frames.append(
                            self._build_evidence_frame(
                                frame,
                                frame_id,
                                timestamp_sec,
                                net_change_score,
                                motion_bbox,
                                selection_reason=_SELECTION_REASON_THRESHOLD,
                            )
                        )
                        if diag is not None:
                            diag.record(
                                _diag_row(
                                    fid=frame_id,
                                    ts=timestamp_sec,
                                    net_score=net_change_score,
                                    raw_ratio=change_ratio,
                                    noise_floor=adaptive_noise_floor,
                                    early_exceeded=True,
                                    main_exceeded=True,
                                    early_confirmed=None,
                                    density=diag_density_at_arrival,
                                    vote_passed=True,
                                    interval_sec=None,
                                    interval_elapsed=None,
                                    gap_ms=diag_gap_ms,
                                    dedup_checked=False,
                                    dedup_is_dup=None,
                                    selected=True,
                                    selection_reason=_SELECTION_REASON_THRESHOLD,
                                    rejection_reason=None,
                                )
                            )
                        last_evidence_timestamp = timestamp_sec
                        last_selected_gray = curr_gray
                        pending_best = None
                        if diag is not None and pending_best_diag is not None:
                            row = dict(pending_best_diag)
                            row["selected"] = False
                            row["selection_reason"] = None
                            row["rejection_reason"] = _REJECTION_NOT_BEST_COVERAGE_CANDIDATE
                            diag.record(row)
                            pending_best_diag = None
                        # Guclu-degisim hysteresis penceresini ac/uzat; erken-
                        # degisim cooldown'unu da en azindan bu kadar uzat ki
                        # olay yatisirken GUCLU -> ERKEN -> SAKIN diye kademeli
                        # dussun (dogrudan sakine sicramasin).
                        strong_cooldown_until = timestamp_sec + self.strong_change_cooldown_sec
                        early_cooldown_until = max(
                            early_cooldown_until, timestamp_sec + self.early_change_cooldown_sec
                        )
                        self._recent_early_decisions.clear()
                    else:
                        # Esik-alti aday. Once bu karenin "supheli-erken" olup
                        # olmadigini (ana esigin ALTINDA ama gurultu tabanindan
                        # belirgin sekilde yuksek) degerlendir; SURDURULEN bir
                        # egilimse (tek karelik sicrama DEGIL) erken-degisim
                        # cooldown'unu ac/uzat.
                        motion_bbox = self._bbox_from_mask(thresh)
                        pre_density_state = self._density_state(
                            timestamp_sec, strong_cooldown_until, early_cooldown_until
                        )

                        is_early_suspicious = (
                            net_change_score >= (self.early_change_score_ratio * self.min_change_threshold)
                            or long_is_early_suspicious
                            or long_slow_is_early_suspicious
                        )

                        selected_this_frame = False

                        # --- Tek-kareli GUCLU/LOKAL degisim (single_frame_change):
                        # `early_change_min_count` cok-kareli onayindan TAMAMEN
                        # BAGIMSIZ, ayri bir yol. Hafif-ama-surdurulen sinyaller
                        # HALA yalnizca asagidaki cok-kareli mekanizma ile
                        # yakalanir (bu blok onu DEGISTIRMEZ); burada yakalanan,
                        # TEK bir ornekte zaten GUCLU olan ve gorsel olarak
                        # LOKAL (kareyi butunuyle kaplayan bir parlaklik
                        # sicramasi/kamera titremesi DEGIL) bir sinyaldir. Uc
                        # sinyal birlikte degerlendirilir: (1) skorun adaptif
                        # gurultu tabanina orani (net_change_score zaten
                        # `raw_change_ratio - adaptive_noise_floor` oldugundan,
                        # bu ayni zamanda "son tipik seviyeye gore pozitif
                        # sicrama" sinyalini de tasir - ayri bir EMA alanina
                        # gerek yoktur), (2) hareket maskesinin kareye orani
                        # (lokal/global ayrimi - zaten hesaplanmis `motion_bbox`
                        # yeniden kullanilir, EK maliyet YOKTUR), (3) son
                        # secilen kareye gore dedup. Sabit hard-code bir esik
                        # KULLANILMAZ - tumu `min_change_threshold`/
                        # `adaptive_noise_floor`e ORANLI config degerleridir
                        # (bkz. `single_frame_change_noise_floor_ratio`/
                        # `single_frame_change_max_area_ratio`).
                        motion_area_ratio = 0.0
                        if motion_bbox is not None:
                            x_min, y_min, x_max, y_max = motion_bbox
                            frame_area = curr_gray.shape[0] * curr_gray.shape[1]
                            motion_area_ratio = (
                                (x_max - x_min + 1) * (y_max - y_min + 1)
                            ) / frame_area

                        # `adaptive_noise_floor` cok erken/tamamen durgun bir
                        # tabanda 0'a cok yakin/esit olabilir - bu durumda
                        # ORAN (net_change_score/noise_floor) anlamsizca
                        # buyuyup YAVAS/dogrusal buyuyen bir rampanin ilk
                        # karesini bile "guclu" gosterebilir. Bu yuzden
                        # taban, gurultu tabani SIFIRA yakinken erken-degisim
                        # esigine (zaten `min_change_threshold`e oranli,
                        # sabit/hard-code DEGIL) geri duser - boylece "guclu"
                        # sayilmak icin skor daima ANLAMLI bir mutlak
                        # buyuklukte olmak ZORUNDADIR, salt "sifirdan farkli"
                        # olmak yetmez.
                        single_frame_baseline = (
                            adaptive_noise_floor
                            if adaptive_noise_floor > 0.0
                            else self.early_change_score_ratio * self.min_change_threshold
                        )
                        is_single_frame_strong = (
                            self.single_frame_change_enabled
                            and is_early_suspicious
                            and motion_bbox is not None
                            and net_change_score
                            >= self.single_frame_change_noise_floor_ratio * single_frame_baseline
                            and motion_area_ratio <= self.single_frame_change_max_area_ratio
                        )
                        if is_single_frame_strong:
                            prior_last_evidence_ts = last_evidence_timestamp
                            # ONEMLI (kok neden duzeltmesi - "ilk guvenilir baslangic
                            # adayinin sessizce kaybolmasi"): `is_single_frame_strong`,
                            # ana esik gibi `early_cooldown_until`i ileri atarak
                            # SAKIN -> ERKEN gecisini TETIKLEYEBILIR (bkz.
                            # `_density_state`), ama asagidaki "SAKIN -> ERKEN
                            # gecisinin TAM O ANI" bloğu bu kare zaten
                            # `single_frame_change` ile secilecegi (selected_this_frame
                            # = True) icin `not selected_this_frame` korumasiyla
                            # ATLANIR. O anda hala bekleyen bir
                            # `pending_early_candidate` (zincirin GERCEK, daha ESKI
                            # baslangic karesi) varsa, `pre_density_state` bu kareden
                            # sonra bir DAHA ASLA `CALM` olmayacagi (cooldown zaten bu
                            # kareyle uzatildi) icin o aday BASKA HICBIR KOD YOLUYLA
                            # (yalnizca sonraki bir `threshold_exceeded` kurtarabilir)
                            # flush edilemez ve sessizce dusurulebilirdi. Esik-gecen
                            # dalin (yukarida, ~satir 1067) YAPTIGI GIBI, burada da
                            # mevcut kareyi eklemeden ONCE pending_early_candidate'i
                            # (kendi eski zaman damgasiyla, dogru kronolojik konuma)
                            # flush ederiz - boylece hangi yol onu "tetiklerse
                            # tetiklesin" GERCEK baslangic karesi asla kaybolmaz.
                            if pending_early_candidate is not None:
                                pending_early_is_dup = self._is_near_duplicate(
                                    pending_early_candidate.gray, last_selected_gray
                                )
                                if not pending_early_is_dup:
                                    _insert_chronologically(
                                        evidence_frames,
                                        self._build_evidence_frame(
                                            pending_early_candidate.frame,
                                            pending_early_candidate.frame_id,
                                            pending_early_candidate.timestamp_sec,
                                            pending_early_candidate.net_change_score,
                                            pending_early_candidate.motion_bbox,
                                            selection_reason=_SELECTION_REASON_EARLY_CHANGE,
                                        ),
                                    )
                                    last_selected_gray = pending_early_candidate.gray
                                if diag is not None and pending_early_diag is not None:
                                    row = dict(pending_early_diag)
                                    row["selected"] = not pending_early_is_dup
                                    row["selection_reason"] = (
                                        _SELECTION_REASON_EARLY_CHANGE if not pending_early_is_dup else None
                                    )
                                    row["rejection_reason"] = (
                                        _REJECTION_NEAR_DUPLICATE if pending_early_is_dup else None
                                    )
                                    row["dedup_checked"] = True
                                    row["dedup_is_duplicate"] = pending_early_is_dup
                                    diag.record(row)
                                    pending_early_diag = None
                                pending_early_candidate = None
                            single_frame_is_dup = self._is_near_duplicate(curr_gray, last_selected_gray)
                            if not single_frame_is_dup:
                                evidence_frames.append(
                                    self._build_evidence_frame(
                                        frame,
                                        frame_id,
                                        timestamp_sec,
                                        net_change_score,
                                        motion_bbox,
                                        selection_reason=_SELECTION_REASON_SINGLE_FRAME_CHANGE,
                                    )
                                )
                                last_evidence_timestamp = timestamp_sec
                                last_selected_gray = curr_gray
                                pending_best = None
                                # Guclu-ama-tek-kareli bir olay tespit edildi:
                                # erken-degisim cooldown'unu (ana esik gibi
                                # STRONG'a degil, ERKEN'e) ac/uzat - boylece
                                # olay devam ederse takip eden kareler daha
                                # sik ornekle secilir (bkz. modul docstring'i).
                                early_cooldown_until = max(
                                    early_cooldown_until, timestamp_sec + self.early_change_cooldown_sec
                                )
                                selected_this_frame = True
                            if diag is not None:
                                diag.record(
                                    _diag_row(
                                        fid=frame_id,
                                        ts=timestamp_sec,
                                        net_score=net_change_score,
                                        raw_ratio=change_ratio,
                                        noise_floor=adaptive_noise_floor,
                                        early_exceeded=True,
                                        main_exceeded=is_suspicious,
                                        early_confirmed=None,
                                        density=pre_density_state,
                                        vote_passed=temporal_vote_passed,
                                        interval_sec=None,
                                        interval_elapsed=None,
                                        gap_ms=(
                                            _ms(timestamp_sec) - _ms(prior_last_evidence_ts)
                                            if prior_last_evidence_ts is not None
                                            else None
                                        ),
                                        dedup_checked=True,
                                        dedup_is_dup=single_frame_is_dup,
                                        selected=not single_frame_is_dup,
                                        selection_reason=(
                                            _SELECTION_REASON_SINGLE_FRAME_CHANGE
                                            if not single_frame_is_dup
                                            else None
                                        ),
                                        rejection_reason=(
                                            _REJECTION_NEAR_DUPLICATE if single_frame_is_dup else None
                                        ),
                                    )
                                )

                        # Zincirin ILK karesini yakala: yalnizca SAKIN
                        # durumdan cikan YENI bir zincir icin (pre_density_state
                        # == CALM) ve henuz bir pending_early_candidate yokken.
                        # `_PendingSelectionCandidate` (pending_best ile AYNI
                        # dataclass) yeniden kullanilir - O(1), tek ornek.
                        # `not selected_this_frame`: bu kare zaten
                        # single_frame_change ile secildiyse (bkz. yukarisi),
                        # AYNI kare icin ikinci bir bekleyen-aday izi
                        # ACILMAZ - aksi halde ayni frame_index daha sonra
                        # `early_change` ile TEKRAR eklenebilir.
                        if (
                            not selected_this_frame
                            and is_early_suspicious
                            and pending_early_candidate is None
                            and pre_density_state == _DENSITY_CALM
                        ):
                            pending_early_candidate = _PendingSelectionCandidate(
                                frame=frame.copy(),
                                gray=curr_gray,
                                frame_id=frame_id,
                                timestamp_sec=timestamp_sec,
                                net_change_score=net_change_score,
                                motion_bbox=motion_bbox,
                            )
                            if diag is not None:
                                pending_early_diag = _diag_row(
                                    fid=frame_id,
                                    ts=timestamp_sec,
                                    net_score=net_change_score,
                                    raw_ratio=change_ratio,
                                    noise_floor=adaptive_noise_floor,
                                    early_exceeded=True,
                                    main_exceeded=is_suspicious,
                                    early_confirmed=None,
                                    density=pre_density_state,
                                    vote_passed=temporal_vote_passed,
                                    interval_sec=None,
                                    interval_elapsed=None,
                                    gap_ms=(
                                        _ms(timestamp_sec) - _ms(last_evidence_timestamp)
                                        if last_evidence_timestamp is not None
                                        else None
                                    ),
                                    dedup_checked=False,
                                    dedup_is_dup=None,
                                    selected=False,
                                    selection_reason=None,
                                    rejection_reason=None,
                                    long_baseline_net_score=long_net_score,
                                    long_baseline_early_suspicious=long_is_early_suspicious,
                                    long_baseline_slow_net_score=long_slow_net_score,
                                    long_baseline_slow_early_suspicious=long_slow_is_early_suspicious,
                                )

                        confirmed_early_now = self._confirm_early_change(is_early_suspicious)
                        if confirmed_early_now:
                            early_cooldown_until = timestamp_sec + self.early_change_cooldown_sec

                        # ONEMLI (kanita dayali minimal iyilestirme - "olay
                        # baslangicina YAKLASAN saniyelerden VLM'e daha fazla
                        # kare gitsin, sabit bir ust sinir/kota EKLENMEDEN"):
                        # bir zincir (`pending_early_candidate`) HENUZ onay
                        # bekliyorken - yani cok-kareli onay
                        # (`early_change_min_count`) tamamlanana kadar
                        # gecebilecek birden fazla saniyelik "yaklasma"
                        # penceresinde - o ana kadar YALNIZCA zincirin ILK
                        # karesi (`pending_early_candidate`in kendisi,
                        # onaylandiginda ayrica flush edilir) evidence'a
                        # girer; supheli-erken kalan ARADAKI kareler (bu
                        # kare de dahil) hicbir yere kaydedilmez. Bu blok,
                        # `pending_best`/`pending_early_candidate`in KENDI
                        # bekleme/onay mantigina HICBIR SEKILDE dokunmadan
                        # (o ikisi DEGISMEDEN, TAMAMEN BAGIMSIZ), bu ARADAKI
                        # zaman penceresinde `early_change_selection_
                        # interval_sec`den daha uzun sure evidence
                        # eklenmediyse SUPHELI-ERKEN kalan GUNCEL kareyi
                        # dogrudan (EK, ikinci bir) `"early_change"`
                        # kaniti olarak ekler - boylece VLM, olayin
                        # baslangicina yaklasirken tek bir kare degil,
                        # yoğunluga oranli SAYIDA kare gorur.
                        # `confirmed_early_now` bu kare icin `True` ISE bu
                        # blok ATLANIR - o zaten asagidaki ADANMIS
                        # SAKIN->ERKEN gecis blogu tarafindan (dogru
                        # `pending_early_candidate` ile) islenecektir; ayni
                        # kareyi iki kez eklemekten kacinilir.
                        # `pending_early_candidate.frame_id != frame_id`:
                        # zincirin ILK karesini (bu kare zaten `pending_
                        # early_candidate` olarak beklemede - onaylandiginda
                        # KENDI zaman damgasiyla eklenecek) burada ikinci
                        # kez EKLEMEZ.
                        if (
                            not selected_this_frame
                            and not confirmed_early_now
                            and is_early_suspicious
                            and pending_early_candidate is not None
                            and pending_early_candidate.frame_id != frame_id
                            and last_evidence_timestamp is not None
                            and (timestamp_sec - last_evidence_timestamp) >= self.early_change_selection_interval_sec
                        ):
                            prior_approach_evidence_ts = last_evidence_timestamp
                            approach_is_dup = self._is_near_duplicate(curr_gray, last_selected_gray)
                            if not approach_is_dup:
                                evidence_frames.append(
                                    self._build_evidence_frame(
                                        frame,
                                        frame_id,
                                        timestamp_sec,
                                        net_change_score,
                                        motion_bbox,
                                        selection_reason=_SELECTION_REASON_EARLY_CHANGE,
                                    )
                                )
                                last_evidence_timestamp = timestamp_sec
                                # ONEMLI: `last_selected_gray` BILEREK
                                # GUNCELLENMEZ. Bu, halihazirda bekleyen
                                # `pending_early_candidate`in (zincirin
                                # GERCEK ilk karesi) ONAYLANDIGINDA yapacagi
                                # KENDI `_is_near_duplicate` kontrolunu bu ARA
                                # (visually cok benzer olabilecek, cunku
                                # surekli buyuyen bir olayda ardisik kareler
                                # dogal olarak birbirine yakindir) ornekle
                                # KIRLETMEZ - aksi halde GERCEK baslangic
                                # karesi, "son secilen kareye (bu ara
                                # ornege) neredeyse ayni" gerekcesiyle
                                # yanlislikla dedup ile ELENEBILIRDI (onceki
                                # gorevde duzeltilen "sessiz kayip" hatasinin
                                # baska bir varyanti). Bu kareler zaten
                                # kendi aralarinda `curr_gray`ye karsi HER
                                # SEFERINDE dedup kontrolunden geciyor -
                                # yalnizca PAYLASILAN referans DEGISMIYOR.
                                pending_best = None
                                selected_this_frame = True
                            if diag is not None:
                                diag.record(
                                    _diag_row(
                                        fid=frame_id,
                                        ts=timestamp_sec,
                                        net_score=net_change_score,
                                        raw_ratio=change_ratio,
                                        noise_floor=adaptive_noise_floor,
                                        early_exceeded=True,
                                        main_exceeded=is_suspicious,
                                        early_confirmed=confirmed_early_now,
                                        density=pre_density_state,
                                        vote_passed=temporal_vote_passed,
                                        interval_sec=self.early_change_selection_interval_sec,
                                        interval_elapsed=True,
                                        gap_ms=_ms(timestamp_sec) - _ms(prior_approach_evidence_ts),
                                        dedup_checked=True,
                                        dedup_is_dup=approach_is_dup,
                                        selected=not approach_is_dup,
                                        selection_reason=(
                                            _SELECTION_REASON_EARLY_CHANGE if not approach_is_dup else None
                                        ),
                                        rejection_reason=(
                                            _REJECTION_NEAR_DUPLICATE if approach_is_dup else None
                                        ),
                                    )
                                )

                        # Zincir onaya ULASMADAN tamamen sonduyse (pencerede
                        # artik hicbir "supheli-erken" karar kalmadiysa),
                        # pending adayi GUVENLI sekilde temizle - aksi halde
                        # SONRAKI, ILGISIZ bir zincirin onayinda yanlislikla
                        # bu eski kare secilebilir (bkz. gorev notu #6/#7).
                        if pending_early_candidate is not None and sum(self._recent_early_decisions) == 0:
                            pending_early_candidate = None
                            if diag is not None and pending_early_diag is not None:
                                row = dict(pending_early_diag)
                                row["rejection_reason"] = _REJECTION_EARLY_NOT_CONFIRMED
                                diag.record(row)
                                pending_early_diag = None

                        density_state = self._density_state(
                            timestamp_sec, strong_cooldown_until, early_cooldown_until
                        )
                        selection_interval, selection_reason = self._selection_interval_and_reason(
                            density_state
                        )

                        # SAKIN -> ERKEN gecisinin TAM O ANI: zincirin
                        # BASLANGIC karesi (`pending_early_candidate` -
                        # ONAYIN GERCEKLESTIGI mevcut kare DEGIL) secilir; ana
                        # esik VE secim araligi beklenmeden, GERIYE DONUK bir
                        # video-buffer'i gerekmeden. `pending_best` (genel
                        # pencere biriktirici) bu ozel durumda KULLANILMAZ.
                        # `not selected_this_frame`: bu kare zaten
                        # single_frame_change ile secildiyse, `pending_early_
                        # candidate` (yukarida bilerek OLUSTURULMADI) icin
                        # savunmaci yedek burada AYNI kareyi TEKRAR eklemeye
                        # calisirdi - bu dal TAMAMEN atlanir.
                        if (
                            not selected_this_frame
                            and pre_density_state == _DENSITY_CALM
                            and density_state == _DENSITY_EARLY
                        ):
                            # Yapisal olarak `pending_early_candidate`, tam da
                            # bu gecisi tetikleyen onay anindan once (bu kare
                            # dahil, is_early_suspicious ilk True oldugunda)
                            # zaten set edilmis olmalidir; yine de savunmaci
                            # bir yedek olarak mevcut kare kullanilir.
                            onset_source = pending_early_candidate or _PendingSelectionCandidate(
                                frame=frame.copy(),
                                gray=curr_gray,
                                frame_id=frame_id,
                                timestamp_sec=timestamp_sec,
                                net_change_score=net_change_score,
                                motion_bbox=motion_bbox,
                            )
                            onset_is_dup = self._is_near_duplicate(onset_source.gray, last_selected_gray)
                            if not onset_is_dup:
                                _insert_chronologically(
                                    evidence_frames,
                                    self._build_evidence_frame(
                                        onset_source.frame,
                                        onset_source.frame_id,
                                        onset_source.timestamp_sec,
                                        onset_source.net_change_score,
                                        onset_source.motion_bbox,
                                        selection_reason=_SELECTION_REASON_EARLY_CHANGE,
                                    ),
                                )
                                # ONEMLI (kare zaman damgasi vs. secim-durumu
                                # zaman damgasi): eklenen EvidenceFrame'in
                                # KENDI icerik zaman damgasi HALA
                                # `onset_source.timestamp_sec`dir (yukarida,
                                # DEGISMEDEN) - burada `last_evidence_
                                # timestamp` ICIN kasitli olarak bunun YERINE
                                # `timestamp_sec` (bu donguce ISLEME/KARAR
                                # ani) kullanilir. `onset_source` gecmiste
                                # (pending_early_candidate'in eski karesinde)
                                # sikismis olabilir; onun eski damgasini gap
                                # sayacina yazmak, sayacin GERIYE sicramasina
                                # ve bir SONRAKI secimin kendi
                                # `selection_interval`ini ihlal edecek kadar
                                # erken tetiklenmesine yol acar (bkz. rapor
                                # notu #2).
                                last_evidence_timestamp = timestamp_sec
                                last_selected_gray = onset_source.gray
                                pending_best = None
                                selected_this_frame = True
                            if diag is not None:
                                base_row = pending_early_diag or _diag_row(
                                    fid=onset_source.frame_id,
                                    ts=onset_source.timestamp_sec,
                                    net_score=onset_source.net_change_score,
                                    raw_ratio=change_ratio,
                                    noise_floor=adaptive_noise_floor,
                                    early_exceeded=True,
                                    main_exceeded=is_suspicious,
                                    early_confirmed=True,
                                    density=pre_density_state,
                                    vote_passed=temporal_vote_passed,
                                    interval_sec=None,
                                    interval_elapsed=None,
                                    gap_ms=None,
                                    dedup_checked=False,
                                    dedup_is_dup=None,
                                    selected=False,
                                    selection_reason=None,
                                    rejection_reason=None,
                                )
                                row = dict(base_row)
                                row["early_change_confirmed"] = True
                                row["dedup_checked"] = True
                                row["dedup_is_duplicate"] = onset_is_dup
                                row["selected"] = not onset_is_dup
                                row["selection_reason"] = _SELECTION_REASON_EARLY_CHANGE if not onset_is_dup else None
                                row["rejection_reason"] = _REJECTION_NEAR_DUPLICATE if onset_is_dup else None
                                diag.record(row)
                            # Pending aday, secildi VEYA dedup ile reddedildi -
                            # her iki durumda da bu zincir icin TUKETILDI,
                            # temizlenir (gorev notu #5).
                            pending_early_candidate = None
                            pending_early_diag = None

                            # Mevcut kare (frame_id), zincirin GERCEK
                            # baslangicini (onset_source, genelde DAHA ESKI
                            # bir frame_id) tetikleyen ONAY karesidir - kendisi
                            # secilmedi. `onset_source` savunmaci yedekte
                            # (pending_early_candidate yoktu) mevcut kareyle
                            # AYNI olabilir; o durumda satiri zaten yukarida
                            # kaydedildi, tekrar kaydetmeyiz (gorev notu #2:
                            # tam olarak BIR satir/frame_index).
                            if diag is not None and not onset_is_dup and onset_source.frame_id != frame_id:
                                diag.record(
                                    _diag_row(
                                        fid=frame_id,
                                        ts=timestamp_sec,
                                        net_score=net_change_score,
                                        raw_ratio=change_ratio,
                                        noise_floor=adaptive_noise_floor,
                                        early_exceeded=is_early_suspicious,
                                        main_exceeded=is_suspicious,
                                        early_confirmed=confirmed_early_now,
                                        density=density_state,
                                        vote_passed=temporal_vote_passed,
                                        interval_sec=None,
                                        interval_elapsed=None,
                                        gap_ms=None,
                                        dedup_checked=False,
                                        dedup_is_dup=None,
                                        selected=False,
                                        selection_reason=None,
                                        rejection_reason=_REJECTION_NOT_BEST_COVERAGE_CANDIDATE,
                                    )
                                )

                        if not selected_this_frame:
                            # Genel pencere biriktirme: esitlikte (ör. sabit bir
                            # skorda) EN GUNCEL aday kazanir (`>=`) - aksi
                            # halde pending_best eski bir karede sikisip kalir
                            # ve secim zaman damgasi geriye sarkar (bkz.
                            # onceki gorev notlari).
                            current_wins_pending_best = (
                                pending_best is None or net_change_score >= pending_best.net_change_score
                            )
                            if current_wins_pending_best:
                                # ONEMLI (tanilama kaydi dogrulugu - kok neden:
                                # "duragan/gurultulu sahnelerde yanlis evidence
                                # uretmesin" testleri sirasinda kesfedildi):
                                # yerinden edilen ONCEKI `pending_best`,
                                # HALEN COZULMEMIS bir `pending_early_
                                # candidate`in TAM DA AYNI karesi olabilir
                                # (ikisi de "o ana kadarki en iyi esik-alti
                                # aday"i BAGIMSIZ olarak izler ve genellikle
                                # AYNI karede baslar). Bu durumda burada
                                # "rejected" bir satir YAZMAK, o frame_index
                                # icin `_DiagnosticRecorder`in KENDI "tek
                                # satir" guvenlik agini erken TUKETIR - daha
                                # SONRA pending_early_candidate GERCEKTEN
                                # onaylanip flush edildiginde (bkz. "SAKIN ->
                                # ERKEN gecisi" blogu), o satirin KENDI
                                # `selected=True` kaydi bu erken/yanlis
                                # kayittan dolayi "already_selected_frame_
                                # index" olarak BASTIRILIRDI - GERCEK secim
                                # (`evidence_frames`) DOGRU kalsa da,
                                # diagnostic gunlugu o kareyi (yanlislikla)
                                # hic secilmemis gosterirdi. Bu yuzden, HALA
                                # AKTIF pending_early_candidate ile AYNI
                                # frame_index icin bu "rejected" satir hic
                                # YAZILMAZ - o frame_index'in TEK satiri,
                                # pending_early_candidate'in kendi NIHAI
                                # (selected VEYA nihayetinde reddedilen)
                                # sonucuna birakilir.
                                if (
                                    diag is not None
                                    and pending_best_diag is not None
                                    and not (
                                        pending_early_candidate is not None
                                        and pending_best_diag["frame_index"] == pending_early_candidate.frame_id
                                    )
                                ):
                                    # ONCEKI pending_best (daha eski bir
                                    # frame_index), simdi kazanan bu kare
                                    # tarafindan YERINDEN edildi - genel
                                    # pencerede ayni anda tek aday olabilir.
                                    row = dict(pending_best_diag)
                                    row["selected"] = False
                                    row["selection_reason"] = None
                                    row["rejection_reason"] = _REJECTION_NOT_BEST_COVERAGE_CANDIDATE
                                    diag.record(row)
                                    pending_best_diag = None
                                elif diag is not None and pending_best_diag is not None:
                                    pending_best_diag = None
                                pending_best = _PendingSelectionCandidate(
                                    frame=frame.copy(),
                                    gray=curr_gray,
                                    frame_id=frame_id,
                                    timestamp_sec=timestamp_sec,
                                    net_change_score=net_change_score,
                                    motion_bbox=motion_bbox,
                                )
                                if diag is not None:
                                    pending_best_diag = _diag_row(
                                        fid=frame_id,
                                        ts=timestamp_sec,
                                        net_score=net_change_score,
                                        raw_ratio=change_ratio,
                                        noise_floor=adaptive_noise_floor,
                                        early_exceeded=is_early_suspicious,
                                        main_exceeded=is_suspicious,
                                        early_confirmed=confirmed_early_now,
                                        density=density_state,
                                        vote_passed=temporal_vote_passed,
                                        interval_sec=selection_interval,
                                        interval_elapsed=False,
                                        gap_ms=(
                                            _ms(timestamp_sec) - _ms(last_evidence_timestamp)
                                            if last_evidence_timestamp is not None
                                            else None
                                        ),
                                        dedup_checked=False,
                                        dedup_is_dup=None,
                                        selected=False,
                                        selection_reason=None,
                                        rejection_reason=None,
                                        long_baseline_net_score=long_net_score,
                                        long_baseline_early_suspicious=long_is_early_suspicious,
                                        long_baseline_slow_net_score=long_slow_net_score,
                                        long_baseline_slow_early_suspicious=long_slow_is_early_suspicious,
                                    )
                            elif diag is not None and not (
                                pending_early_candidate is not None
                                and pending_early_candidate.frame_id == frame_id
                            ):
                                # Bu kare pending_best yarisini KAYBETTI - genel
                                # pencerenin adayi olmadi, bir daha da olmayacak
                                # (o an sadece EN GUNCEL/EN YUKSEK skorlu aday
                                # tutulur), dolayisiyla kendi kaderi burada
                                # KESIN: reddedildi.
                                # ISTISNA (tanilama kaydi dogrulugu - bkz. yukarida
                                # "current_wins_pending_best" dalindaki AYNI
                                # gerekce): bu kare (`frame_id`), pending_best
                                # yarisini KAYBETSE BILE, AYNI anda HALA COZULMEMIS
                                # bir `pending_early_candidate` olabilir (ör.
                                # `net_change_score`u dusuk ama uzun-baz sinyali
                                # onu supheli-erken yapmis bir kare) - bu durumda
                                # burada "kesin reddedildi" satiri YAZILMAZ, aksi
                                # halde bu frame_index'in DiagnosticRecorder'daki
                                # TEK satiri erken TUKETILIR ve pending_early_
                                # candidate GERCEKTEN onaylanip flush edildiginde
                                # (`selected=True`) o dogru kayit "already_
                                # selected_frame_index" olarak BASTIRILIR - GERCEK
                                # secim (`evidence_frames`) DOGRU kalsa da,
                                # diagnostic gunlugu yanlislikla o kareyi hic
                                # secilmemis gosterirdi.
                                diag.record(
                                    _diag_row(
                                        fid=frame_id,
                                        ts=timestamp_sec,
                                        net_score=net_change_score,
                                        raw_ratio=change_ratio,
                                        noise_floor=adaptive_noise_floor,
                                        early_exceeded=is_early_suspicious,
                                        main_exceeded=is_suspicious,
                                        early_confirmed=confirmed_early_now,
                                        density=density_state,
                                        vote_passed=temporal_vote_passed,
                                        interval_sec=selection_interval,
                                        interval_elapsed=None,
                                        gap_ms=(
                                            _ms(timestamp_sec) - _ms(last_evidence_timestamp)
                                            if last_evidence_timestamp is not None
                                            else None
                                        ),
                                        dedup_checked=False,
                                        dedup_is_dup=None,
                                        selected=False,
                                        selection_reason=None,
                                        rejection_reason=_REJECTION_NOT_BEST_COVERAGE_CANDIDATE,
                                        long_baseline_net_score=long_net_score,
                                        long_baseline_early_suspicious=long_is_early_suspicious,
                                        long_baseline_slow_net_score=long_slow_net_score,
                                        long_baseline_slow_early_suspicious=long_slow_is_early_suspicious,
                                    )
                                )

                            if (
                                last_evidence_timestamp is not None
                                and (timestamp_sec - last_evidence_timestamp) >= selection_interval
                            ):
                                # ONEMLI: dedup SADECE early_change/significant_change
                                # (aktif-olay yogunlugu) secimlerine uygulanir.
                                # temporal_coverage (sakin bolge) ASLA dedup ile
                                # elenmez - aksi halde tamamen durgun/degismeyen
                                # uzun bir sessizlikte pending_best HER ZAMAN son
                                # secilen kareyle "ayni" cikar ve "uzun araliklar
                                # tamamen bos birakilmasin" garantisi (onceki
                                # gorevde eklenen coverage mekanizmasi) bozulur.
                                blocked_by_dedup = (
                                    selection_reason != _SELECTION_REASON_COVERAGE
                                    and self._is_near_duplicate(pending_best.gray, last_selected_gray)
                                )
                                if blocked_by_dedup:
                                    # Son secilen kareyle gorsel olarak ayirt
                                    # edilemeyecek kadar benzer: SECILMEZ, ama
                                    # last_evidence_timestamp/pending_best
                                    # DEGISMEZ - bir sonraki ornekte (muhtemelen
                                    # farkli bir pending_best ile) tekrar
                                    # denenir; hicbir yigilma/veri kaybi olusmaz.
                                    # `pending_best_diag` da AYNI sekilde acik
                                    # birakilir (henuz sonuclanmadi).
                                    pass
                                else:
                                    evidence_frames.append(
                                        self._build_evidence_frame(
                                            pending_best.frame,
                                            pending_best.frame_id,
                                            pending_best.timestamp_sec,
                                            pending_best.net_change_score,
                                            pending_best.motion_bbox,
                                            selection_reason=selection_reason,
                                        )
                                    )
                                    if diag is not None and pending_best_diag is not None:
                                        row = dict(pending_best_diag)
                                        row["selection_interval_sec"] = selection_interval
                                        row["selection_interval_elapsed"] = True
                                        row["dedup_checked"] = selection_reason != _SELECTION_REASON_COVERAGE
                                        row["dedup_is_duplicate"] = False
                                        row["selected"] = True
                                        row["selection_reason"] = selection_reason
                                        row["rejection_reason"] = None
                                        diag.record(row)
                                        pending_best_diag = None
                                    # ONEMLI (kare zaman damgasi vs. secim-
                                    # durumu zaman damgasi - gorev notu #2):
                                    # eklenen EvidenceFrame'in KENDI icerik
                                    # zaman damgasi HALA `pending_best.
                                    # timestamp_sec`dir (yukarida, DEGISMEDEN).
                                    # ANCAK `last_evidence_timestamp` (gap/
                                    # secim-durumu sayaci) icin BUNU
                                    # KULLANMAYIZ: `pending_best`, adaptif
                                    # gurultu tabani rampayi "geriden
                                    # takip ederken" (bkz. modul docstring'i)
                                    # skoru dususe geçtiği icin BIRDEN COK
                                    # ornek boyunca (isleme suresi olarak
                                    # SANIYELER) AYNI eski karede sikisip
                                    # kalabilir - o eski damgayi gap sayacina
                                    # yazmak sayacin GERIYE sicramasina ve
                                    # BIR SONRAKI secimin, cikan
                                    # zaman-damgali ciktida kendi
                                    # `selection_interval`ini acikca ihlal
                                    # edecek kadar erken (gercek isleme
                                    # zamaninda ise TAM zamaninda) gelmis GIBI
                                    # gorunmesine yol acar. Bunun yerine bu
                                    # dongunun KENDI `timestamp_sec`i (secim
                                    # KARARININ verildigi an) kullanilir -
                                    # asla geriye gitmez (frame_id monoton
                                    # arttigindan) ve gap sayaci HER ZAMAN
                                    # gercek isleme ilerlemesini yansitir.
                                    last_evidence_timestamp = timestamp_sec
                                    last_selected_gray = pending_best.gray
                                    _discard_pending_early_if_same_frame(pending_best.frame_id)
                                    pending_best = None
                elif diag is not None:
                    # Bu ornegin (video basindaki ILK kare) henuz bir onceki
                    # kareyle karsilastirilacak referansi yok - hicbir skor
                    # hesaplanamaz, tanimi geregi secilemez.
                    diag.record(
                        _diag_row(
                            fid=frame_id,
                            ts=timestamp_sec,
                            net_score=None,
                            raw_ratio=None,
                            noise_floor=None,
                            early_exceeded=None,
                            main_exceeded=None,
                            early_confirmed=None,
                            density=None,
                            vote_passed=None,
                            interval_sec=None,
                            interval_elapsed=None,
                            gap_ms=None,
                            dedup_checked=False,
                            dedup_is_dup=None,
                            selected=False,
                            selection_reason=None,
                            rejection_reason=_REJECTION_NO_REFERENCE_FRAME,
                        )
                    )

                self.prev_gray = curr_gray
                frame_id += 1
        finally:
            cap.release()

        # `pending_early_candidate` video sonuna kadar ONAYLANMAMISSA
        # (min_count'a hicbir zaman ulasilmadi), GUVENLI sekilde birakilir -
        # ne ayrica emit edilir ne de hataya yol acar (gorev notu #8):
        # onaylanmamis bir sinyal, tanim geregi "surdurulen bir egilim"
        # olarak dogrulanamamis demektir; sessizce dusurmek veri kaybi
        # DEGILDIR (ayni ilkeler: esik/coverage icin de "asilmadiysa
        # birakilir" - asagida). Ekstra kod GEREKMEZ - fonksiyon dondugunde
        # kapsam disina cikar.

        # Video sonu: pencere kapanmadan (bir sonraki ornek gelmeden) video
        # bitmis olabilir. Su ana kadarki en iyi aday varsa VE (o anki
        # yogunluk durumuna gore) gap zaten asilmissa, kayipsizlik icin
        # GUVENLI sekilde son bir kare olarak eklenir (dedup kontrolu dahil);
        # asilmadiysa (video, gerektirecek kadar uzun surmedi) hicbir seye
        # elenmez/hata verilmez - sessizce birakilir (bu veri kaybi degildir,
        # esik zaten hic gecilmedi).
        if pending_best is not None and last_evidence_timestamp is not None:
            final_density_state = self._density_state(
                pending_best.timestamp_sec, strong_cooldown_until, early_cooldown_until
            )
            final_interval, final_reason = self._selection_interval_and_reason(final_density_state)
            final_blocked_by_dedup = (
                final_reason != _SELECTION_REASON_COVERAGE
                and self._is_near_duplicate(pending_best.gray, last_selected_gray)
            )
            if (
                (pending_best.timestamp_sec - last_evidence_timestamp) >= final_interval
                and not final_blocked_by_dedup
            ):
                evidence_frames.append(
                    self._build_evidence_frame(
                        pending_best.frame,
                        pending_best.frame_id,
                        pending_best.timestamp_sec,
                        pending_best.net_change_score,
                        pending_best.motion_bbox,
                        selection_reason=final_reason,
                    )
                )
                if diag is not None and pending_best_diag is not None:
                    row = dict(pending_best_diag)
                    row["selection_interval_sec"] = final_interval
                    row["selection_interval_elapsed"] = True
                    row["dedup_checked"] = final_reason != _SELECTION_REASON_COVERAGE
                    row["dedup_is_duplicate"] = False
                    row["selected"] = True
                    row["selection_reason"] = final_reason
                    row["rejection_reason"] = None
                    diag.record(row)
                    pending_best_diag = None
                _discard_pending_early_if_same_frame(pending_best.frame_id)
            elif diag is not None and pending_best_diag is not None:
                # Video bitti; su ana kadarki en iyi aday hicbir zaman
                # secim araligini asamadi VEYA dedup ile engellendi - GUVENLI
                # sekilde dusuruldu (veri kaybi degil, esik zaten hic
                # gecilmedi/interval dolmadi - bkz. yukaridaki yorum).
                row = dict(pending_best_diag)
                row["selection_interval_sec"] = final_interval
                row["selection_interval_elapsed"] = (
                    (pending_best.timestamp_sec - last_evidence_timestamp) >= final_interval
                )
                row["dedup_checked"] = final_blocked_by_dedup or (final_reason != _SELECTION_REASON_COVERAGE)
                row["dedup_is_duplicate"] = final_blocked_by_dedup
                row["selected"] = False
                row["selection_reason"] = None
                row["rejection_reason"] = (
                    _REJECTION_NEAR_DUPLICATE
                    if final_blocked_by_dedup
                    else _REJECTION_SELECTION_INTERVAL_NOT_ELAPSED
                )
                diag.record(row)
                pending_best_diag = None

        if diag is not None and pending_best_diag is not None:
            # `pending_best` `None` durumundaydi (ör. hicbir kare esik-alti
            # genel pencereye hic girmedi) ama golge kaydi bir sekilde acik
            # kaldiysa (savunmaci guvenlik agi - normalde ulasilmaz):
            # kaybolmadan, GUVENLI sekilde "belirsiz" olarak kapatilir.
            row = dict(pending_best_diag)
            row["selected"] = False
            row["selection_reason"] = None
            row["rejection_reason"] = _REJECTION_NOT_BEST_COVERAGE_CANDIDATE
            diag.record(row)
            pending_best_diag = None

        if diag is not None and pending_early_diag is not None:
            # Video bitti; `pending_early_candidate` video sonuna kadar
            # onaylanmis (SAKIN -> ERKEN gecisi hic olmamis) olabilir -
            # zincir tanim geregi "surdurulen bir egilim" olarak asla
            # dogrulanamadi, GUVENLI sekilde dusuruldu (yukaridaki yorum).
            row = dict(pending_early_diag)
            row["selected"] = False
            row["selection_reason"] = None
            row["rejection_reason"] = _REJECTION_EARLY_NOT_CONFIRMED
            diag.record(row)
            pending_early_diag = None

        if not evidence_frames:
            if first_frame_raw is None:
                raise ValueError(f"Video kaynagindan hic kare okunamadi: {video_path}")

            logger.warning(
                "Esigi gecen Kanit Karesi bulunamadi; sistemin cokmemesi icin frame 0 "
                "varsayilan Kanit Karesi olarak kabul ediliyor (fallback)."
            )
            fallback = self._build_evidence_frame(
                first_frame_raw,
                frame_id=0,
                timestamp_sec=first_frame_timestamp,
                change_score=0.0,
                selection_reason=_SELECTION_REASON_FALLBACK,
            )
            fallback.is_fallback = True
            evidence_frames.append(fallback)
            if diag is not None:
                logger.info(
                    "Sampler tanilama: video hicbir esigi gecmedi, fallback kare (frame 0) "
                    "kullanildi - tanilama gunlugundeki frame 0 satiri bu son fallback KARARINI "
                    "DEGIL, orijinal donguce-ici (in-loop) degerlendirmesini yansitir."
                )

        elapsed_sec = time.perf_counter() - started_at
        eliminated = sampled_frame_count - len(evidence_frames)
        eliminated_ratio = (100.0 * eliminated / sampled_frame_count) if sampled_frame_count else 0.0
        self.last_run_stats = SamplerRunStats(
            total_frames_scanned=frame_id,
            sampled_frames_evaluated=sampled_frame_count,
            evidence_frame_count=len(evidence_frames),
            eliminated_frame_count=eliminated,
            eliminated_ratio_pct=round(eliminated_ratio, 2),
            elapsed_sec=round(elapsed_sec, 3),
        )
        logger.info(
            "AdaptiveFrameSampler tamamlandi: %d ham kare tarandi, %d kare ornekleme icin "
            "degerlendirildi, %d Kanit Karesi uretildi (%d kare elendi, %%%.1f eleme orani), "
            "sure=%.3fs",
            frame_id,
            sampled_frame_count,
            len(evidence_frames),
            eliminated,
            eliminated_ratio,
            elapsed_sec,
        )

        if diag is not None:
            # Ozet rapor icin GERCEK tekrar-secim (duplicate) tespiti:
            # SPEKULATIF golge takibi DEGIL, NIHAI `evidence_frames`
            # ciktisindan dogrudan sayilir (gorev kisiti "aynı frame_index
            # birden fazla kez secildiyse diagnostic ciktida bunu ayrica
            # ozetle").
            frame_id_counts = Counter(ef.frame_id for ef in evidence_frames)
            duplicate_selected_frame_indices = {
                fid: count for fid, count in frame_id_counts.items() if count > 1
            }
            diag.close()
            summary = diag.build_summary(duplicate_selected_frame_indices)
            summary["fallback_used"] = any(ef.is_fallback for ef in evidence_frames)
            summary_path = diag.output_path.with_name(diag.output_path.stem + "_summary.json")
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
            logger.info(
                "Sampler tanilama ozeti: %d satir, %d secilen (%s), %d reddedilen (%s), "
                "%d tekrar-secim denemesi, %d tekrar-secilen frame_index -> %s",
                summary["total_sampled_frames_logged"],
                summary["selected_frame_count"],
                dict(summary["selection_reason_counts"]),
                sum(summary["rejection_reason_counts"].values()),
                dict(summary["rejection_reason_counts"]),
                summary["duplicate_record_attempts"],
                len(duplicate_selected_frame_indices),
                summary_path,
            )

        return evidence_frames

    def _build_evidence_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_sec: float,
        change_score: float,
        motion_bbox: Optional[Tuple[int, int, int, int]] = None,
        selection_reason: str = _SELECTION_REASON_THRESHOLD,
    ) -> EvidenceFrame:
        """Bir kareden `EvidenceFrame` uretir ve JPEG/base64'e cevirir (tek seferlik encode).

        Args:
            frame: BGR formatinda ham video karesi.
            frame_id: Karenin video icindeki sirasi.
            timestamp_sec: Karenin saniye cinsinden zaman damgasi.
            change_score: Hesaplanan (gurultu-tabani-dusulmus) evidence/degisim skoru.
            motion_bbox: Bu kareyi ureten hareket maskesinin sinirlayici kutusu
                (`_bbox_from_mask`); yoksa (ör. fallback kare) `None`.
            selection_reason: Bu karenin evidence listesine GIRME NEDENI
                (`"threshold_exceeded"` | `"temporal_coverage"` |
                `"fallback"`) - konumsal bir rol DEGILDIR, bkz.
                `EvidenceFrame.selection_reason`.

        Returns:
            `image_bytes`/`base64_image` doldurulmus, `saved_path=None` olan `EvidenceFrame`.
        """
        image_bytes = self._encode_frame_jpeg(frame)
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        minutes, seconds = divmod(int(timestamp_sec), 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        return EvidenceFrame(
            evidence_id=f"ev{frame_id}",
            frame_id=frame_id,
            timestamp_sec=round(timestamp_sec, 2),
            timestamp_str=time_str,
            change_score=round(change_score, 4),
            image_bytes=image_bytes,
            base64_image=f"data:image/jpeg;base64,{base64_str}",
            image_shape=frame.shape,
            saved_path=None,
            motion_bbox=motion_bbox,
            selection_reason=selection_reason,
        )


def sampler_from_config(
    config: SamplerConfig, min_change_threshold_override: Optional[float] = None
) -> AdaptiveFrameSampler:
    """`configs/config.yaml` icindeki `sampler` blogundan bir `AdaptiveFrameSampler` uretir.

    Her cagri taze bir `AdaptiveFrameSampler` orneği doner (paylasimli/kalici
    durum tutmaz); boylece operator panelinden gelen ayarlar cagrilar arasinda
    birbirine karismaz ve her analiz temiz bir `prev_gray`/gurultu gecmisiyle
    baslar.

    Args:
        config: Dogrulanmis `SamplerConfig` nesnesi.
        min_change_threshold_override: Verilirse, config degeri yerine bu
            hassasiyet esigi kullanilir (operator panelindeki slider icin).

    Returns:
        Config (ve varsa override) parametreleriyle ilklendirilmis, taze
        `AdaptiveFrameSampler` orneği.
    """
    return AdaptiveFrameSampler(
        min_change_threshold=(
            min_change_threshold_override
            if min_change_threshold_override is not None
            else config.min_change_threshold
        ),
        blur_kernel_size=tuple(config.blur_kernel_size),
        history_window=config.history_window,
        temporal_vote_window=config.temporal_vote_window,
        temporal_vote_min_count=config.temporal_vote_min_count,
        max_temporal_gap_sec=config.max_temporal_gap_sec,
        early_change_score_ratio=config.early_change_score_ratio,
        early_change_window=config.early_change_window,
        early_change_min_count=config.early_change_min_count,
        early_change_selection_interval_sec=config.early_change_selection_interval_sec,
        early_change_cooldown_sec=config.early_change_cooldown_sec,
        significant_change_selection_interval_sec=config.significant_change_selection_interval_sec,
        strong_change_cooldown_sec=config.strong_change_cooldown_sec,
        dedup_similarity_ratio=config.dedup_similarity_ratio,
        single_frame_change_enabled=config.single_frame_change_enabled,
        single_frame_change_noise_floor_ratio=config.single_frame_change_noise_floor_ratio,
        single_frame_change_max_area_ratio=config.single_frame_change_max_area_ratio,
        long_baseline_change_enabled=config.long_baseline_change_enabled,
        long_baseline_interval_sec=config.long_baseline_interval_sec,
        long_baseline_slow_interval_sec=config.long_baseline_slow_interval_sec,
        diagnostic_enabled=config.diagnostic_enabled,
        diagnostic_output_dir=config.diagnostic_output_dir,
        diagnostic_output_format=config.diagnostic_output_format,
    )


if __name__ == "__main__":
    # Modul 1'in bagimsiz calistirilabilirlik testi:
    #   python -m src.sampler.adaptive_sampler [video_yolu]
    # Varsayilan olarak data/test.mp4 uzerinde calisir; VLM/GPU'ya veya
    # projenin geri kalanina hicbir bagimliligi yoktur.
    import json
    import sys

    logging.basicConfig(level=logging.INFO)

    demo_video_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.mp4"
    if not Path(demo_video_path).exists():
        print(
            f"Video bulunamadi: {demo_video_path}\n"
            "Kullanim: python -m src.sampler.adaptive_sampler [video_yolu]\n"
            "(Varsayilan olarak 'data/test.mp4' aranir.)"
        )
        sys.exit(1)

    demo_sampler = AdaptiveFrameSampler()
    demo_evidence_frames = demo_sampler.process_video(demo_video_path)
    demo_stats = demo_sampler.last_run_stats

    print(f"Video: {demo_video_path}")
    print(f"Taranan ham kare: {demo_stats.total_frames_scanned}")
    print(f"Degerlendirilen ornek kare: {demo_stats.sampled_frames_evaluated}")
    print(f"Uretilen Kanit Karesi: {demo_stats.evidence_frame_count}")
    print(
        f"Elenen kare sayisi: {demo_stats.eliminated_frame_count} "
        f"(%{demo_stats.eliminated_ratio_pct:.1f} eleme orani)"
    )

    demo_output_path = Path("data/mock_evidence.json")
    demo_output_path.parent.mkdir(parents=True, exist_ok=True)
    demo_payload = {
        "video_source": demo_video_path,
        "stats": {
            "total_frames_scanned": demo_stats.total_frames_scanned,
            "sampled_frames_evaluated": demo_stats.sampled_frames_evaluated,
            "evidence_frame_count": demo_stats.evidence_frame_count,
            "eliminated_frame_count": demo_stats.eliminated_frame_count,
            "eliminated_ratio_pct": demo_stats.eliminated_ratio_pct,
            "elapsed_sec": demo_stats.elapsed_sec,
        },
        "evidence_frames": [
            ef.model_dump(exclude={"image_bytes"}) for ef in demo_evidence_frames
        ],
    }
    demo_output_path.write_text(
        json.dumps(demo_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Kanit verisi yazildi: {demo_output_path}")
