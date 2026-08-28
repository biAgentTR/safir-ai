"""T016 - Deterministic Risk Resolver: `RuleMatch` listesinden VLM/LLM'den bagimsiz nihai risk cikarir.

Mimari karar: "VLM (ve 05 LangGraph Agent'taki LLM) risk KARARI vermez, yalnizca
gozlem/aciklama uretir; nihai risk seviyesi/skoru `RuleEngine`'in (T010)
urettigi deterministik `RuleMatch.severity` degerinden VE (RISK ENGINE V2,
2026-08-24) mevcut TUM diger yapilandirilmis kanittan (event confidence,
sureklilik, tekrar, PPE/koruma bosluğu, kural gucü, RAG'dan gelen dogrulanmis
mevzuat destegi) matematiksel olarak turetilir." Bu modul, o turetmeyi TEK bir
yerde (hem `EventBuilder`in event-bazli kullanimi hem `src/main.py`nin cagri-
geneli kullanimi icin) toplar - `src/event_analysis/risk_model.py` (agirlikli-
carpimsal formul, TAM gerekce icin bkz. o modulun dokustringi) TEK skorlama
implementasyonudur, ikinci bir hesaplama yolu YOKTUR.

RISK ENGINE V2 (2026-08-24) - ONCEKI SURUMDEN FARK: eski `_SEVERITY_MIDPOINT_SCORE`
sabit-bucket eslemesi (dusuk->12, orta->38, yuksek->63, kritik->88) TAMAMEN
KALDIRILDI. Skor artik SADECE severity'den turemiyor - `risk_model.compute_risk_score`
sekiz kanit-tabanli feature kullanir. `resolve_deterministic_risk`/
`resolve_deterministic_risk_with_provenance` API'leri (GERIYE-UYUMLULUK icin)
AYNEN KORUNDU; ek (opsiyonel) parametrelerle (temporal_events, semantic_rag_sources,
llm_proposed_score) cagrildiginda DAHA ZENGIN bir hesaplama yapar - bu parametreler
verilmezse (eski cagiran kod), eksik feature'lar icin `risk_model.py`nin notr/
guvenli varsayilanlari kullanilir (crash YOK, "unknown" "safe" ile KARISTIRILMAZ).

`RuleMatch.severity` ve `AgentDecision.risk_level` zaten AYNI kelime
dagarcigini kullanir ("dusuk"/"orta"/"yuksek"/"kritik", bkz.
`src/agent/langgraph_agent.py::SafirAgent._resolve_risk_level`); bu yuzden
seviye eslemesi bire-bir, yeni bir sozluk icat edilmez.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.event_analysis.risk_model import (
    _CRITICAL_HAZARD_FLOOR,
    RiskScoreBreakdown,
    compute_risk_score,
    score_to_risk_level,
)
from src.event_analysis.schemas import RuleMatch, TemporalEvent

_SEVERITY_ORDER = ["dusuk", "orta", "yuksek", "kritik"]
_SEVERITY_RANK = {name: rank for rank, name in enumerate(_SEVERITY_ORDER)}


@dataclass
class RiskProvenance:
    """`resolve_deterministic_risk`in ATLADIGI - risk kararinin HANGI kanittan VE nasil tureidigini tasiyan izlenebilirlik nesnesi.

    Operator/rapor/trace katmaninda "bu risk NEREDEN geldi?" sorusunun
    (bkz. gorev tanimi 3. bolum "Risk Explanation") LLM'e SORULMADAN,
    mevcut structured data'dan deterministik olarak cevaplanmasini saglar.

    RISK ENGINE V2 (2026-08-24): `risk_score`/`risk_level` alanlari GERIYE-
    UYUMLULUK icin KORUNDU (`risk_score` artik `round(final_score)`dir, ESKI
    sabit-bucket degeri DEGIL). Yeni alanlar (`scoring_method`, `final_score`,
    `features`, `feature_sources`, `feature_contributions`, `llm_proposed_score`,
    `regulatory_evidence_ids`), formulun TAMAMINI izlenebilir kilar.
    """

    risk_level: Optional[str]
    """NIHAI (blended) risk seviyesi - bkz. `risk_score` aciklamasi."""
    risk_score: Optional[int]
    """NIHAI, RAPORLANAN risk skoru: `deterministic_score` VE `llm_proposed_score`
    (ikisi de mevcutsa) ARITMETIK ORTALAMASIDIR - `round((deterministic_score +
    llm_proposed_score) / 2)`. `llm_proposed_score` verilmediyse (Agent
    basarisiz oldu veya cagirilmadi) dogrudan `deterministic_score`e esittir.
    Kritik bir fiziksel tehlike icin guvenlik tabani devreye girdiyse
    (`safety_floor_applied`), ortalama bu tabanin ALTINA DUSURULMEZ - Agent'in
    dusuk bir tahmini, ZATEN kanitlanmis kritik bir tehlikeyi maskeleyemez."""
    rule_ids: List[str] = field(default_factory=list)
    rule_severities: List[str] = field(default_factory=list)
    contributing_event_ids: List[str] = field(default_factory=list)

    scoring_method: str = "safir_evidence_weighted_v2"
    final_score: Optional[float] = None
    """Yuvarlanmamis, hassas 0.0-100.0 NIHAI (blended) skor (`risk_score` bunun `round()`una esittir)."""
    deterministic_score: Optional[int] = None
    """SADECE RuleEngine + risk_model formulunden turemis, Agent'in tahmininden
    TAMAMEN BAGIMSIZ skor - `risk_score` (nihai/blended) ile KARISTIRILMAMALIDIR.
    Operatore/rapora HER IKI degerin de (bu + `llm_proposed_score`) AYRI AYRI
    gosterilmesi gerekir (bkz. gorev tanimi - risk hesaplama seffafligi)."""
    deterministic_level: Optional[str] = None
    """`deterministic_score`e karsilik gelen risk seviyesi (blended `risk_level`den FARKLI olabilir)."""
    features: Optional[Dict[str, Optional[float]]] = None
    """`risk_model.RiskFeatures.as_dict()` - severity/likelihood/exposure/duration/
    recurrence/protection_gap/rule_support/regulatory_support (0.0-1.0, `None`=olculemedi)."""
    feature_sources: Optional[Dict[str, str]] = None
    """Her feature icin `"measured"` | `"unavailable_neutral"` (bkz. `risk_model.RiskFeatureSources`)."""
    feature_contributions: Optional[Dict[str, float]] = None
    """`risk_model.RiskScoreBreakdown.as_contributions_dict()` - base_risk + her `*_factor` + raw_score."""
    llm_proposed_score: Optional[int] = None
    """Agent'in (05 LangGraph) KENDI, dogrulanmamis taslak `risk_score`u - bkz. gorev
    tanimi 8. bolum. `deterministic_score` ile ORTALAMASI alinarak nihai
    `risk_score`e (blended) katkida bulunur - bkz. `risk_score` aciklamasi."""
    regulatory_evidence_ids: List[str] = field(default_factory=list)
    """Bu hesaplamada `regulatory_support` feature'ina katkida bulunan (dogrulanmis,
    source_verified) RAG kaynaklarinin `chunk_id`leri."""
    safety_floor_applied: bool = False
    """`risk_model._CRITICAL_HAZARD_FLOOR` bu cagirida devreye girdi mi (bkz.
    `risk_model.py` "CRITICAL HAZARD SAFETY FLOOR") - TAMAMEN RuleEngine siddeti
    + gercek `matched_keywords` kanitina dayanir, `llm_proposed_score`den ASLA turemez."""

    def explanation(self) -> str:
        """Risk hesabinin SEFFAF, adim-adim Turkce gerekce cumlesi uretir.

        Formul-tabanli hesaplama varsa (`deterministic_score` mevcutsa)
        `risk_model`in KENDI aciklamasini VE Agent'in taslak skoruyla nasil
        ORTALANDIGINI acikca gosterir; yoksa (hicbir kural eslesmedi) sabit
        bir "belirlenemedi" mesaji doner - yeni bir bilgi/tahmin UYDURMAZ.
        """
        if self.risk_level is None:
            return "Hicbir deterministik kural eslesmedi; risk RuleEngine tarafindan belirlenemedi."
        rule_list = ", ".join(self.rule_ids) if self.rule_ids else "(bilinmeyen kural)"
        rule_severity = self.rule_severities[0] if self.rule_severities else "(bilinmeyen)"
        base = (
            f"Deterministik (RuleEngine) skor: {self.deterministic_score}/100 "
            f"('{self.deterministic_level}') - en yuksek siddetli eslesme(ler) {rule_list} "
            f"(RuleEngine siddeti: '{rule_severity}'), '{self.scoring_method}' matematiksel "
            "modeliyle hesaplandi."
        )
        if self.llm_proposed_score is not None:
            base += (
                f" Ajanin (LLM) taslak skoru: {self.llm_proposed_score}/100. "
                f"Nihai risk skoru, bu iki degerin ORTALAMASI alinarak "
                f"{self.risk_score}/100 ('{self.risk_level}') olarak belirlendi."
            )
            if self.safety_floor_applied:
                base += (
                    f" Not: kritik bir fiziksel tehlike guvenlik tabani devrede oldugundan "
                    f"({int(_CRITICAL_HAZARD_FLOOR)}/100), nihai skor bu tabanin ALTINA "
                    "DUSURULMEDI."
                )
        else:
            base += " Ajanin bir taslak skoru olmadigi icin nihai skor dogrudan bu deterministik degere esittir."
        if self.features is not None:
            feature_bits = ", ".join(
                f"{name}={value:.2f}" if value is not None else f"{name}=notr(olculemedi)"
                for name, value in self.features.items()
            )
            base += f" Deterministik skorun feature'lari: {feature_bits}."
        return base


def _pick_provenance(
    rule_matches: List[RuleMatch],
    temporal_events: Optional[List[TemporalEvent]] = None,
    semantic_rag_sources: Optional[List[Any]] = None,
    llm_proposed_score: Optional[int] = None,
) -> RiskProvenance:
    """`resolve_deterministic_risk`in TEK kaynagi: siddet-secim VE matematiksel skorlama BURADA uygulanir.

    `resolve_deterministic_risk` ve `resolve_deterministic_risk_with_provenance`
    ikisi de bu fonksiyonu cagirir - iki ayri (birbirinden sapabilecek) risk
    hesaplama mantigi OLUSTURULMAZ.
    """
    known = [match for match in rule_matches if match.severity in _SEVERITY_RANK]
    if not known:
        return RiskProvenance(
            risk_level=None,
            risk_score=None,
            deterministic_score=None,
            deterministic_level=None,
            llm_proposed_score=llm_proposed_score,
        )

    top_rank = max(_SEVERITY_RANK[match.severity] for match in known)
    contributing = [match for match in known if _SEVERITY_RANK[match.severity] == top_rank]
    risk_level = _SEVERITY_ORDER[top_rank]
    contributing_rule_ids = [match.rule_id for match in contributing]

    contributing_event_ids: List[str] = []
    for match in contributing:
        for event_id in [match.source_event_id, *match.related_event_ids]:
            if event_id not in contributing_event_ids:
                contributing_event_ids.append(event_id)

    # 2026-08-24 (runtime data-flow audit): bir TemporalEvent'in KENDI
    # `event_type`i yoksa (VLM'in serbest `event_name`i bilinen hicbir
    # kategoriye oturmuyor - bkz. `rule_engine.py::_safe_event_type`
    # docstring'i, "bir kategoriye ZORLAMA YAPILMAZ"), RuleEngine o olay
    # icin HICBIR RuleMatch URETMEZ - bu KASITLI ve DOGRUdur (fabrikasyon
    # ONLENIR). AMA bu, ayni olayin (orn. bir yanginin ilerleyen, DAHA
    # siddetli asamasi) GERCEK kanitinin (matched_keywords/duration/
    # confidence) `hazard_escalation`/temporal feature'lara TAMAMEN
    # KAYBOLMASINA yol aciyordu - RuleEngine "bu olayi siniflandirmiyorum"
    # dedi diye, TemporalReasoner'in ZATEN kurdugu (`related_events`,
    # simetrik, tip-bagimsiz zaman-yakinligi baglantisi) iliski de
    # YOK SAYILIYORDU. Duzeltme: TAM OLARAK contributing (zaten siddeti
    # BELLI) bir olaya `related_events` ile baglı VE kendi event_type'i
    # None olan (siniflandirilmamis) TemporalEvent'lerin event_id'leri de
    # feature hesaplamasina (SADECE feature'lara - hangi RuleMatch/severity
    # kazandigini DEGISTIRMEZ) dahil edilir. Boylece "duman baslangici"
    # (siniflandirilmis, kritik) + "kontrolsuz ilerleme" (siniflandirilmamis
    # ama ZATEN iliskili) ayni olayin GERCEK kanit butunlugunu korur.
    if temporal_events:
        by_id = {te.event_id: te for te in temporal_events}
        for event_id in list(contributing_event_ids):
            source_event = by_id.get(event_id)
            if source_event is None:
                continue
            for related_id in source_event.related_events:
                related_event = by_id.get(related_id)
                if related_event is None or related_event.event_type is not None:
                    continue
                if related_id not in contributing_event_ids:
                    contributing_event_ids.append(related_id)

    breakdown: RiskScoreBreakdown = compute_risk_score(
        risk_level=risk_level,
        contributing_matches=contributing,
        contributing_rule_ids=contributing_rule_ids,
        contributing_event_ids=contributing_event_ids,
        temporal_events=temporal_events,
        semantic_rag_sources=semantic_rag_sources,
    )

    regulatory_evidence_ids = [
        getattr(src, "chunk_id", None)
        for src in (semantic_rag_sources or [])
        if getattr(src, "source_verified", True) and getattr(src, "chunk_id", None) is not None
    ]

    # RISK ENGINE V2.1 (2026-08-27, kullanici talebi): nihai raporlanan risk
    # artik SADECE deterministik (RuleEngine + risk_model) skor DEGIL -
    # Agent'in (LLM) kendi taslak skoruyla ORTALANIR. Deterministik deger
    # `deterministic_score`/`deterministic_level` alanlarinda AYRICA (blended
    # degerden BAGIMSIZ, ham) saklanir - boylece operatore/rapora HER IKI
    # deger de tek tek gosterilebilir (seffaflik/aciklanabilirlik gerekliligi).
    deterministic_score = round(breakdown.final_score)
    deterministic_level = breakdown.risk_level

    if llm_proposed_score is not None:
        blended_score = round((deterministic_score + llm_proposed_score) / 2)
    else:
        blended_score = deterministic_score

    # Guvenlik tabani: Agent'in dusuk bir tahmini, ZATEN RuleEngine siddeti +
    # gercek matched_keywords kanitiyla dogrulanmis kritik bir fiziksel
    # tehlikeyi ORTALAMA yoluyla MASKELEYEMEZ.
    if breakdown.safety_floor_applied:
        blended_score = max(blended_score, int(_CRITICAL_HAZARD_FLOOR))

    blended_score = max(0, min(100, blended_score))
    blended_level = score_to_risk_level(blended_score)

    return RiskProvenance(
        risk_level=blended_level,
        risk_score=blended_score,
        deterministic_score=deterministic_score,
        deterministic_level=deterministic_level,
        rule_ids=contributing_rule_ids,
        rule_severities=[match.severity for match in contributing],
        contributing_event_ids=contributing_event_ids,
        scoring_method="safir_evidence_weighted_v2",
        final_score=breakdown.final_score,
        features=breakdown.features.as_dict(),
        feature_sources=breakdown.feature_sources.as_dict(),
        feature_contributions=breakdown.as_contributions_dict(),
        llm_proposed_score=llm_proposed_score,
        regulatory_evidence_ids=regulatory_evidence_ids,
        safety_floor_applied=breakdown.safety_floor_applied,
    )


def resolve_deterministic_risk_with_provenance(
    rule_matches: List[RuleMatch],
    temporal_events: Optional[List[TemporalEvent]] = None,
    semantic_rag_sources: Optional[List[Any]] = None,
    llm_proposed_score: Optional[int] = None,
) -> RiskProvenance:
    """Bir olay/cagriya ait `RuleMatch` listesinden, TUM mevcut kanitla (varsa), nihai riski matematiksel olarak hesaplar.

    Args:
        rule_matches: `RuleEngine.evaluate(...)` (tum cagri) veya `EventBuilder`in
            bir olaya grupladigi (`related_rule_matches`) `RuleMatch` listesi.
        temporal_events: (Opsiyonel, GERIYE-UYUMLULUK icin YENI) Bu cagriya ait
            `TemporalEvent`ler - likelihood/duration/recurrence feature'larini
            besler. Verilmezse bu feature'lar notr (bkz. `risk_model.py`) kalir.
        semantic_rag_sources: (Opsiyonel, YENI) Bu cagrinin semantik RAG sonuclari
            (`relevance_score`/`source_verified` tasiyan nesneler) - regulatory_support
            feature'ini besler. Verilmezse notr kalir.
        llm_proposed_score: (Opsiyonel, YENI) Agent'in KENDI taslak risk_score'u -
            YALNIZCA izleme/karsilastirma icin saklanir, hesaplamayi ETKILEMEZ.

    Returns:
        `RiskProvenance` - `risk_level`/`risk_score` formul-tabanli nihai karari
        tasir; ek olarak `features`/`feature_contributions`/vb. ile TAM izlenebilirlik saglar.
    """
    return _pick_provenance(rule_matches, temporal_events, semantic_rag_sources, llm_proposed_score)


def resolve_deterministic_risk(rule_matches: List[RuleMatch]) -> Tuple[Optional[str], Optional[int]]:
    """Bir olay/cagriya ait `RuleMatch` listesinden deterministik risk cikarir (GERIYE-UYUMLU, sade imza).

    RISK ENGINE V2: `risk_score` artik SABIT bir severity-bucket DEGIL,
    `risk_model.compute_risk_score`in urettigi matematiksel skordur (bu
    imzada temporal/RAG kaniti verilmedigi icin likelihood/duration/
    recurrence/regulatory_support notr/guvenli varsayilanlarla hesaplanir -
    bkz. `risk_model.py` modul dokustringi "EKSIK KANIT"). Zengin (temporal/
    RAG-farkinda) hesaplama icin `resolve_deterministic_risk_with_provenance`
    kullanin.

    Girdi kumesi YALNIZCA `RuleMatch.severity` (+ varsa `event_type`, rule_id
    sayisi) degerlerinden olusur - VLM confidence, VLM'in kendi risk_score/
    anomaly-score ipucu, RAG/embedding benzerlik skoru veya LLM'in kendi risk
    tahmini bu fonksiyona ASLA girmez (bkz. `src/main.py::stage_finalize_risk`,
    `context_builder.py` modul dokustringi: bu kaynaklar risk_score/risk_level'i
    ETKILEMEZ).

    Hicbir kural eslesmediyse `(None, None)` doner - risk UYDURULMAZ; cagiran
    taraf bu durumda mevcut (orn. LLM Agent'tan gelen ya da "unknown") degeri
    korumalidir.

    Args:
        rule_matches: `RuleEngine.evaluate(...)` (tum cagri) veya
            `EventBuilder`in bir olaya grupladigi (`related_rule_matches`)
            `RuleMatch` listesi.

    Returns:
        `(risk_level, risk_score)`: en yuksek siddetli eslesmenin `severity`si
        ve formul-tabanli nihai skoru. Bilinmeyen/gecersiz `severity` degerleri
        (`_SEVERITY_RANK` disinda) yok sayilir.
    """
    provenance = _pick_provenance(rule_matches)
    return provenance.risk_level, provenance.risk_score


if __name__ == "__main__":
    # python -m src.event_analysis.risk_resolver
    demo_matches = [
        RuleMatch(
            rule_id="ISG-M24",
            rule_description="ISG Yonetmeligi Madde 24",
            event_type="kkd_ihlali",
            severity="orta",
            source_event_id="evt_0",
        ),
        RuleMatch(
            rule_id="COMBO-01",
            rule_description="KKD + arac-yaya yakinligi",
            event_type="kkd_ihlali+arac_yaya_yakinligi",
            severity="kritik",
            source_event_id="evt_0",
            related_event_ids=["evt_1"],
        ),
    ]
    print(resolve_deterministic_risk(demo_matches))
