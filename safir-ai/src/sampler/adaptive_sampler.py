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
     min_change_threshold`i gecen) sinyal SURDURULDUGUNDE tetiklenir; bu
     durumun BASLADIGI kare (ana esik beklenmeden) DOGRUDAN secilir - boylece
     bir olayin baslangic karesi (ör. izmaritin atildigi an, dumanin ilk
     gorulme ani) geriye donuk bir buffer gerekmeden korunur. Aktifken
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
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Deque, List, Optional, Tuple

import cv2
import numpy as np

from src.sampler.schema import EvidenceFrame

_SELECTION_REASON_THRESHOLD = "threshold_exceeded"
_SELECTION_REASON_COVERAGE = "temporal_coverage"
_SELECTION_REASON_EARLY_CHANGE = "early_change"
_SELECTION_REASON_SIGNIFICANT_CHANGE = "significant_change"
_SELECTION_REASON_FALLBACK = "fallback"

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
                sirasini bozuyorsa.
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

        self.prev_gray: np.ndarray | None = None
        self.noise_floor_history: List[float] = []
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

    def process_video(self, video_path: str, sample_fps: int = 5) -> List[EvidenceFrame]:
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

                    # Temporal voting: mevcut esik sonucunu (degismedi) girdi olarak
                    # kullanip, izole/tek karelik supheli kareleri (kamera titremesi,
                    # isik degisimi, sikistirma artefakti) eleyen ek sureklilik kontrolu.
                    # Varsayilan ayarlarda (window=1, min_count=1) bu, ESKI davranisla
                    # BIREBIR AYNIDIR (bkz. `_confirm_candidate` docstring'i).
                    is_suspicious = net_change_score >= self.min_change_threshold
                    if self._confirm_candidate(is_suspicious):
                        # Sabit bir ust sinir YOKTUR: video ne kadar uzun/olay ne kadar
                        # surekli olursa olsun, hicbir Kanit Karesi burada sessizce
                        # atlanmaz. Kumeleme artik VLM katmaninda yapilir. Bu dal,
                        # yogunluk mekanizmasi eklenmeden ONCEKI davranisla BIREBIR
                        # AYNIDIR: HER esik-gecen kare kosulsuz secilir, dedup
                        # UYGULANMAZ (bkz. `_is_near_duplicate` docstring'i).
                        motion_bbox = self._bbox_from_mask(thresh)
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
                        last_evidence_timestamp = timestamp_sec
                        last_selected_gray = curr_gray
                        pending_best = None
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

                        is_early_suspicious = net_change_score >= (
                            self.early_change_score_ratio * self.min_change_threshold
                        )
                        if self._confirm_early_change(is_early_suspicious):
                            early_cooldown_until = timestamp_sec + self.early_change_cooldown_sec

                        density_state = self._density_state(
                            timestamp_sec, strong_cooldown_until, early_cooldown_until
                        )
                        selection_interval, selection_reason = self._selection_interval_and_reason(
                            density_state
                        )

                        selected_this_frame = False
                        # SAKIN -> ERKEN gecisinin TAM O ANI: olayin baslangic
                        # karesi (ör. izmaritin atildigi an), ana esik VE secim
                        # araligi beklenmeden, GERIYE DONUK bir buffer
                        # gerekmeden dogrudan secilir - o an elimizdeki TEK
                        # kare budur (mevcut kare); `pending_best` eski/farkli
                        # bir kareyi tutuyor olabilir, o yuzden bu ozel durumda
                        # `pending_best` KULLANILMAZ.
                        if pre_density_state == _DENSITY_CALM and density_state == _DENSITY_EARLY:
                            if not self._is_near_duplicate(curr_gray, last_selected_gray):
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
                                last_selected_gray = curr_gray
                                pending_best = None
                                selected_this_frame = True

                        if not selected_this_frame:
                            # Genel pencere biriktirme: esitlikte (ör. sabit bir
                            # skorda) EN GUNCEL aday kazanir (`>=`) - aksi
                            # halde pending_best eski bir karede sikisip kalir
                            # ve secim zaman damgasi geriye sarkar (bkz.
                            # onceki gorev notlari).
                            if pending_best is None or net_change_score >= pending_best.net_change_score:
                                pending_best = _PendingSelectionCandidate(
                                    frame=frame.copy(),
                                    gray=curr_gray,
                                    frame_id=frame_id,
                                    timestamp_sec=timestamp_sec,
                                    net_change_score=net_change_score,
                                    motion_bbox=motion_bbox,
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
                                    # Gap sayaci, TETIKLEYEN ornegin degil, EKLENEN
                                    # karenin KENDI zaman damgasindan devam eder
                                    # (o kare artik "en son gorulen" evidence'tir).
                                    last_evidence_timestamp = pending_best.timestamp_sec
                                    last_selected_gray = pending_best.gray
                                    pending_best = None

                self.prev_gray = curr_gray
                frame_id += 1
        finally:
            cap.release()

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
