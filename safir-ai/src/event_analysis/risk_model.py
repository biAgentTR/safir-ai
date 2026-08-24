"""T016 v2 - SAFIR Evidence-Weighted Risk Model: deterministik, matematiksel, aciklanabilir risk skorlama.

RISK ENGINE V2 (2026-08-24): eski sistem, siddet seviyesini SABIT bir bucket
ortalamasina (`dusuk`->12, `orta`->38, `yuksek`->63, `kritik`->88) esliyordu -
skor TAMAMEN severity'den turerdi, baska hicbir kanit (confidence, sureklilik,
tekrar, PPE/koruma bosluğu, kural gucü, mevzuat destegi) skoru ETKILEMEZDI.
Bu modul o sabit-bucket mantigini TAMAMEN KALDIRIR ve yerine, mevcut,
GERCEKTEN tasinan kanittan (uydurulmus HICBIR veri OLMADAN) tureyen agirlikli-
carpimsal bir model koyar.

Fine-Kinney'nin temel fikri (Risk ~ Severity x Likelihood x Exposure)
REFERANS alinir ama DOGRUDAN kopyalanmaz - SAFIR'in video-tabanli kanit
zincirine (RuleEngine siddeti, VLM/TemporalEvent confidence, sureklilik,
tekrar, RuleEngine'in PPE-ihlali tespiti, kural/mevzuat destegi) gore
YENIDEN tasarlanmistir.

MATEMATIKSEL MODEL
-------------------
Sekiz 0.0-1.0 normalize edilmis feature:

    S  severity            - RuleEngine siddet siniflandirmasi
    L  likelihood          - olay tespit guveni (TemporalEvent.confidence)
    E  exposure            - HENUZ OLCULEMIYOR (bkz. asagida) - dokumante
                              edilmis, UYDURULMAMIS sinirlim
    D  duration             - tehlikeli durumun sureklilik suresi
    R  recurrence           - ayni/iliskili olayin bu cagridaki tekrar sayisi
    P  protection_gap      - RuleEngine'in tespit ettigi KKD/koruma ihlali
    U  rule_support         - kac BAGIMSIZ kural bu siddeti dogruluyor (COMBO > tekli)
    G  regulatory_support   - RAG'in getirdigi, GERCEKTEN dogrulanmis (source_verified)
                              mevzuat kanitinin deterministik relevance_score'u

Formul (TAMAMEN carpimsal + pozitif-katkili "boost" yapisi - bkz. asagidaki
monotoniklik gerekcesi):

    base_risk = S x (L_FLOOR + (1 - L_FLOOR) x L)

    temporal_factor  = 1 + W_DURATION x D + W_RECURRENCE x R
    exposure_factor  = 1 + W_EXPOSURE x E
    protection_factor = 1 + W_PROTECTION x P
    evidence_factor  = 1 + W_RULE_SUPPORT x U + W_REGULATORY_SUPPORT x G

    boost_factor = temporal_factor x exposure_factor x protection_factor x evidence_factor

    raw_score = base_risk x boost_factor
    final_score = 100 x raw_score / RAW_SCORE_MAX   (RAW_SCORE_MAX = tum feature'lar
                                                       maksimumdayken ulasilabilecek
                                                       raw_score - bkz. asagida)

NEDEN BU YAPI (monotoniklik ISPATI): `base_risk`, S ve L'de MONOTONIK AZALMAYAN
(turevleri >= 0). Her `*_factor`, ilgili feature(ler)inde MONOTONIK AZALMAYAN
(katsayilar hep pozitif, "1 + pozitif_terim" formunda). Negatif-olmayan,
monotonik-azalmayan fonksiyonlarin CARPIMI da monotonik azalmayandir - bu
yuzden `final_score`, SEKIZ feature'in HER BIRINDE ayri ayri monotonik
azalmayandir (bkz. `tests/event_analysis/test_risk_model.py` - bu ozellik
matematiksel olarak GARANTI, "umarim boyle calisir" degil).

`S` SIFIR OLAMAZ: siddet rank'i `(rank+1)/4` ile normalize edilir (dusuk=0.25,
orta=0.50, yuksek=0.75, kritik=1.0) - `rank/4` (dusuk=0.0) KULLANILMADI cunku
carpimsal formulde S=0, TUM diger kanitin agirligini SIFIRLARDI (bir "dusuk"
siddetli ama COK GUCLU baska kanitlarla desteklenen olay, hala 0 puan alirdi
- bu, kanitin degerini INKAR eder).

MAKSIMUM SKOR TAVANLARI (S'nin ayni-agirlikli L'yle VE tam boost ile
ulasabilecegi tavan): dusuk<=25, orta<=50, yuksek<=75, kritik<=100 - bu
SEVIYE ESIKLERI (bkz. `_LEVEL_THRESHOLDS`) TESADUFEN mevcut
`configs/config.yaml::agent.risk_thresholds` (25/50/75/100) ile AYNI cikar;
bu bir kopya DEGIL, formulun matematiksel bir SONUCUDUR (bkz. gorev tanimi
7. bolum: "mevcut terminoloji ve audit dogrultusunda").

EKSIK KANIT (gorev tanimi 12. bolum): hicbir feature icin UYDURULMUS deger
KULLANILMAZ. `likelihood` bilinmiyorsa (TemporalEvent verilmedi) NOTR bir
"yari-guven" (L_FLOOR=0.5, yani base_risk = S x 0.75) kullanilir - eksik
likelihood'u "tam guven" (S x 1.0) SAYMAK riski SISIRIR, "sifir guven" (S x
L_FLOOR=S x 0.5) SAYMAK GUCLU bir severity sinyalini HAKSIZ YERE EZER; ikisi
arasindaki DENGELI nokta secildi. `duration`/`recurrence`/`exposure`/
`regulatory_support` bilinmiyorsa KATKISI SIFIRDIR (boost_factor'a ek
YAPILMAZ) - "bilinmeyen" ASLA "ekstra risk" olarak YORUMLANMAZ (guvenli
varsayilan). `protection_gap`/`rule_support` HER ZAMAN hesaplanabilir
(RuleMatch.event_type/rule_id'den GERCEK, deterministik turetilir) - bu
ikisi icin "bilinmeyen" durumu YOKTUR.

EXPOSURE (E) - BILINEN SINIRLAMA: SAFIR'in mevcut kod tabaninda (audit
edildi) dogrudan bir proximity/occupancy/mesafe olcumu (orn. bir kisi ile
tehlike arasindaki piksel/metre mesafesi) YOKTUR - YOLO/nesne tespiti/
bounding-box altyapisi PROJEDE MEVCUT DEGIL ve bu gorev kapsaminda
EKLENMEMISTIR (gorev tanimi ACIKCA yasakliyor). Bu yuzden `E` bu surumde
HER ZAMAN `None` (bilinmiyor) doner - `exposure_factor` HER ZAMAN 1.0'dir
(notr, ek katki YOK). Gercek bir proximity/occupancy sinyali projeye
eklendiginde, bu tek nokta (`_exposure_feature`) guncellenebilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.event_analysis.schemas import RuleMatch, TemporalEvent

_SEVERITY_ORDER = ["dusuk", "orta", "yuksek", "kritik"]
_SEVERITY_RANK = {name: rank for rank, name in enumerate(_SEVERITY_ORDER)}

# --- Formul sabitleri (config'e tasinmadi - risk motoru, RAG'daki
# `RelevanceWeights` gibi ayri bir config-okuma bagimliligi eklemez;
# `src/event_analysis/` katmani BILEREK `agent/`/`utils/config_loader`e
# bagimli degildir, bkz. `rule_engine.py` modul dokustringindeki ayni ilke) ---

_L_FLOOR = 0.5
"""`likelihood` (L) icin taban dampening carpani - S x (L_FLOOR + (1-L_FLOOR)*L).
L bilinmiyorsa/dusukse bile severity kanitinin degeri EN AZ L_FLOOR kadar
korunur (guclu bir RuleEngine eslesmesi, dusuk VLM confidence'i yuzunden
SIFIRA cekilmez)."""

_W_DURATION = 0.15
_W_RECURRENCE = 0.15
_W_EXPOSURE = 0.20
_W_PROTECTION = 0.20
_W_RULE_SUPPORT = 0.15
_W_REGULATORY_SUPPORT = 0.15

_BOOST_FACTOR_MAX = (
    (1 + _W_DURATION + _W_RECURRENCE) * (1 + _W_EXPOSURE) * (1 + _W_PROTECTION) * (1 + _W_RULE_SUPPORT + _W_REGULATORY_SUPPORT)
)
_RAW_SCORE_MAX = 1.0 * _BOOST_FACTOR_MAX
"""S=1.0, L=1.0 (base_risk=1.0) VE tum boost feature'lari maksimumdayken
ulasilabilecek `raw_score` - `final_score = 100 * raw_score / _RAW_SCORE_MAX`
boylece HER ZAMAN [0, 100] araliginda kalir (matematiksel garanti, clip'e
GUVENMEZ - ama float hassasiyeti icin yine de clip edilir)."""

_DURATION_SATURATION_SEC = 30.0
"""Bu sureden UZUN suren bir olay, duration feature'inda tam (1.0) doyuma ulasir."""
_RECURRENCE_SATURATION_COUNT = 5
"""Bu sayida (veya fazla) tekrar, recurrence feature'inda tam (1.0) doyuma ulasir."""

_LEVEL_THRESHOLDS = [
    (25.0, "dusuk"),
    (50.0, "orta"),
    (75.0, "yuksek"),
    (100.0001, "kritik"),
]
"""`final_score < esik` -> seviye (ilk eslesen kazanir, artan sirali). Bkz. modul
dokustringi: bu esikler formulun S-tavanlariyla TUTARLIDIR (kopya degil, sonuc)."""


def score_to_risk_level(score: float) -> str:
    """0-100 nihai skoru, SAFIR'in mevcut Turkce risk seviyesi kelime dagarcigina (dusuk/orta/yuksek/kritik) esler."""
    for threshold, level in _LEVEL_THRESHOLDS:
        if score < threshold:
            return level
    return "kritik"


@dataclass
class RiskFeatures:
    """Sekiz normalize edilmis (0.0-1.0) risk feature'i - `None` = bu cagirida OLCULEMEDI (uydurulmadi)."""

    severity: float
    likelihood: Optional[float]
    exposure: Optional[float]
    duration: Optional[float]
    recurrence: Optional[float]
    protection_gap: float
    rule_support: float
    regulatory_support: Optional[float]

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "severity": self.severity,
            "likelihood": self.likelihood,
            "exposure": self.exposure,
            "duration": self.duration,
            "recurrence": self.recurrence,
            "protection_gap": self.protection_gap,
            "rule_support": self.rule_support,
            "regulatory_support": self.regulatory_support,
        }


@dataclass
class RiskFeatureSources:
    """Her feature icin: `"measured"` (gercek kanittan hesaplandi) veya `"unavailable_neutral"` (kanit yoktu, notr/sifir-katki varsayilan kullanildi).

    "Unknown" ile "safe/dusuk risk" birbirine KARISTIRILMASIN diye (gorev
    tanimi 12. bolum) - operator/rapor, HANGI feature'larin GERCEKTEN
    olculdugunu, hangilerinin notr varsayilan oldugunu buradan gorebilir.
    """

    severity: str = "measured"
    likelihood: str = "measured"
    exposure: str = "unavailable_neutral"
    duration: str = "measured"
    recurrence: str = "measured"
    protection_gap: str = "measured"
    rule_support: str = "measured"
    regulatory_support: str = "measured"

    def as_dict(self) -> Dict[str, str]:
        return {
            "severity": self.severity,
            "likelihood": self.likelihood,
            "exposure": self.exposure,
            "duration": self.duration,
            "recurrence": self.recurrence,
            "protection_gap": self.protection_gap,
            "rule_support": self.rule_support,
            "regulatory_support": self.regulatory_support,
        }


@dataclass
class RiskScoreBreakdown:
    """Nihai skora giden TUM ara hesaplama adimlari - "feature contribution'ları izlenebilir olmalı" gerekcesi."""

    features: RiskFeatures
    feature_sources: RiskFeatureSources
    base_risk: float
    temporal_factor: float
    exposure_factor: float
    protection_factor: float
    evidence_factor: float
    boost_factor: float
    raw_score: float
    final_score: float
    risk_level: str

    def as_contributions_dict(self) -> Dict[str, float]:
        """Her carpan bileseninin (`base_risk` ve dort `*_factor`) nihai skora giden agirlikli katkisi - aciklanabilirlik icin."""
        return {
            "base_risk": round(self.base_risk, 4),
            "temporal_factor": round(self.temporal_factor, 4),
            "exposure_factor": round(self.exposure_factor, 4),
            "protection_factor": round(self.protection_factor, 4),
            "evidence_factor": round(self.evidence_factor, 4),
            "boost_factor": round(self.boost_factor, 4),
            "raw_score": round(self.raw_score, 4),
        }

    def explanation(self) -> str:
        """Deterministik, LLM'e SORULMAMIS, formulun KENDI degerlerinden tureyen Turkce gerekce."""
        f = self.features
        parts = [
            f"severity={f.severity:.2f}",
            f"likelihood={f.likelihood:.2f}" if f.likelihood is not None else "likelihood=notr(0.50, olculemedi)",
            f"duration={f.duration:.2f}" if f.duration is not None else "duration=notr(0.00, olculemedi)",
            f"recurrence={f.recurrence:.2f}" if f.recurrence is not None else "recurrence=notr(0.00, olculemedi)",
            f"protection_gap={f.protection_gap:.2f}",
            f"rule_support={f.rule_support:.2f}",
            f"regulatory_support={f.regulatory_support:.2f}" if f.regulatory_support is not None else "regulatory_support=notr(0.00, RAG kaniti yok)",
            "exposure=notr(1.00 carpan, HENUZ OLCULEMIYOR)",
        ]
        return (
            f"SAFIR Evidence-Weighted Risk Model (safir_evidence_weighted_v2): "
            f"final_score={self.final_score:.1f}/100 ({self.risk_level}) = "
            f"base_risk({self.base_risk:.3f}) x boost_factor({self.boost_factor:.3f}); "
            f"feature'lar: {', '.join(parts)}."
        )


def _duration_feature(duration_sec: float) -> float:
    return max(0.0, min(1.0, duration_sec / _DURATION_SATURATION_SEC))


def _recurrence_feature(occurrence_count: int) -> float:
    return max(0.0, min(1.0, (occurrence_count - 1) / (_RECURRENCE_SATURATION_COUNT - 1)))


def _protection_gap_feature(contributing_matches: List[RuleMatch]) -> float:
    """RuleEngine'in bu cagirida GERCEKTEN tespit ettigi bir KKD/koruma ihlali var mi (event_type uzerinden, deterministik)."""
    for match in contributing_matches:
        event_types = match.event_type.split("+")
        if "kkd_ihlali" in event_types:
            return 1.0
    return 0.0


def _rule_support_feature(contributing_rule_ids: List[str]) -> float:
    """Kac BAGIMSIZ kural bu siddeti destekliyor - bilesik (COMBO) kural TEK BASINA en guclu kanittir."""
    if not contributing_rule_ids:
        return 0.0
    if any(rid.startswith("COMBO-") for rid in contributing_rule_ids):
        return 1.0
    return max(0.0, min(1.0, 0.5 + 0.25 * (len(contributing_rule_ids) - 1)))


def _likelihood_and_temporal_features(
    temporal_events: Optional[List[TemporalEvent]], contributing_event_ids: List[str]
) -> tuple:
    """Katkida bulunan `TemporalEvent`lerden (varsa) likelihood/duration/recurrence turetir.

    Returns:
        `(likelihood, duration, recurrence)` - hicbiri hesaplanamiyorsa (temporal_events
        verilmedi VEYA hicbiri contributing_event_ids ile eslesmiyorsa) UCU DE `None`.
    """
    if not temporal_events:
        return None, None, None
    relevant = [te for te in temporal_events if te.event_id in contributing_event_ids]
    if not relevant:
        return None, None, None
    likelihood = max(te.confidence for te in relevant)
    duration = _duration_feature(max(te.duration for te in relevant))
    recurrence = _recurrence_feature(max(te.occurrence_count for te in relevant))
    return likelihood, duration, recurrence


def _regulatory_support_feature(semantic_rag_sources: Optional[List]) -> Optional[float]:
    """RAG'in getirdigi, GERCEKTEN dogrulanmis (source_verified) kanitlarin en yuksek deterministik relevance_score'u.

    `embedding_score`/`score` KULLANILMAZ - yalnizca `relevance_score` (deterministik
    agirlikli-hibrit skor, bkz. `src/rag/deterministic_reranker.py`) kullanilir; bu
    ikisi KARISTIRILMAZ (gorev tanimi 10. bolum).
    """
    if not semantic_rag_sources:
        return None
    scores = [
        getattr(src, "relevance_score", None)
        for src in semantic_rag_sources
        if getattr(src, "source_verified", True) and getattr(src, "relevance_score", None) is not None
    ]
    scores = [s for s in scores if s is not None]
    if not scores:
        return None
    return max(0.0, min(1.0, max(scores)))


def compute_risk_score(
    risk_level: str,
    contributing_matches: List[RuleMatch],
    contributing_rule_ids: List[str],
    contributing_event_ids: List[str],
    temporal_events: Optional[List[TemporalEvent]] = None,
    semantic_rag_sources: Optional[List] = None,
) -> RiskScoreBreakdown:
    """SAFIR Evidence-Weighted Risk Model'i uygular - bkz. modul dokustringi icin TAM formul + gerekce.

    Args:
        risk_level: RuleEngine'in secili (en yuksek siddetli) `severity` degeri
            (bkz. `risk_resolver._pick_provenance`) - HER ZAMAN bir `RuleMatch`den
            gelir, LLM'den DEGIL.
        contributing_matches: En yuksek siddete karsilik gelen `RuleMatch`(ler).
        contributing_rule_ids: `contributing_matches`in `rule_id`leri (rule_support icin).
        contributing_event_ids: Katkida bulunan `TemporalEvent.event_id`leri.
        temporal_events: Bu cagriya ait TUM `TemporalEvent`ler (likelihood/duration/
            recurrence turetmek icin); `None`/bos ise bu uc feature `None` (notr) kalir.
        semantic_rag_sources: Bu cagrinin semantik RAG sonuclari (`RetrievedDocument`
            veya `RagContext` gibi `relevance_score`/`source_verified` tasiyan
            nesneler); `None`/bos ise `regulatory_support` `None` (notr) kalir.

    Returns:
        Nihai skora giden TUM ara adimlari tasiyan `RiskScoreBreakdown`.
    """
    severity = (_SEVERITY_RANK[risk_level] + 1) / len(_SEVERITY_ORDER)

    likelihood, duration, recurrence = _likelihood_and_temporal_features(temporal_events, contributing_event_ids)
    protection_gap = _protection_gap_feature(contributing_matches)
    rule_support = _rule_support_feature(contributing_rule_ids)
    regulatory_support = _regulatory_support_feature(semantic_rag_sources)
    exposure = None  # bkz. modul dokustringi "EXPOSURE (E) - BILINEN SINIRLAMA"

    features = RiskFeatures(
        severity=severity,
        likelihood=likelihood,
        exposure=exposure,
        duration=duration,
        recurrence=recurrence,
        protection_gap=protection_gap,
        rule_support=rule_support,
        regulatory_support=regulatory_support,
    )
    sources = RiskFeatureSources(
        likelihood="measured" if likelihood is not None else "unavailable_neutral",
        duration="measured" if duration is not None else "unavailable_neutral",
        recurrence="measured" if recurrence is not None else "unavailable_neutral",
        regulatory_support="measured" if regulatory_support is not None else "unavailable_neutral",
    )

    likelihood_eff = likelihood if likelihood is not None else _L_FLOOR
    duration_eff = duration if duration is not None else 0.0
    recurrence_eff = recurrence if recurrence is not None else 0.0
    exposure_eff = 0.0  # exposure_factor daima 1.0 (notr) - bkz. modul dokustringi
    regulatory_eff = regulatory_support if regulatory_support is not None else 0.0

    base_risk = severity * (_L_FLOOR + (1 - _L_FLOOR) * likelihood_eff)
    temporal_factor = 1 + _W_DURATION * duration_eff + _W_RECURRENCE * recurrence_eff
    exposure_factor = 1 + _W_EXPOSURE * exposure_eff
    protection_factor = 1 + _W_PROTECTION * protection_gap
    evidence_factor = 1 + _W_RULE_SUPPORT * rule_support + _W_REGULATORY_SUPPORT * regulatory_eff
    boost_factor = temporal_factor * exposure_factor * protection_factor * evidence_factor

    raw_score = base_risk * boost_factor
    final_score = max(0.0, min(100.0, 100.0 * raw_score / _RAW_SCORE_MAX))
    level = score_to_risk_level(final_score)

    return RiskScoreBreakdown(
        features=features,
        feature_sources=sources,
        base_risk=base_risk,
        temporal_factor=temporal_factor,
        exposure_factor=exposure_factor,
        protection_factor=protection_factor,
        evidence_factor=evidence_factor,
        boost_factor=boost_factor,
        raw_score=raw_score,
        final_score=final_score,
        risk_level=level,
    )
