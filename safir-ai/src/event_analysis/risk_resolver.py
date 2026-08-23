"""T016 - Deterministic Risk Resolver: `RuleMatch` listesinden VLM/LLM'den bagimsiz nihai risk cikarir.

Mimari karar: "VLM (ve 05 LangGraph Agent'taki LLM) risk KARARI vermez, yalnizca
gozlem/aciklama uretir; nihai risk seviyesi/skoru `RuleEngine`'in (T010)
urettigi deterministik `RuleMatch.severity` degerlerinden turetilir." Bu
modul, o turetmeyi TEK bir yerde (hem `EventBuilder`in event-bazli kullanimi
hem `src/main.py`nin cagri-geneli kullanimi icin) toplar.

`RuleMatch.severity` ve `AgentDecision.risk_level` zaten AYNI kelime
dagarcigini kullanir ("dusuk"/"orta"/"yuksek"/"kritik", bkz.
`src/agent/langgraph_agent.py::SafirAgent._resolve_risk_level`); bu yuzden
seviye eslemesi bire-bir, yeni bir sozluk icat edilmez. Sayisal skor
(`risk_score`), `configs/config.yaml::agent.risk_thresholds` (low=25,
medium=50, high=75, critical=100) ile hizali sabit bucket ortalaridir -
config'e canli bagimlilik eklemeden (bu katman `agent/`e bagimli degildir)
aynı esiklerle tutarli bir temsili skor uretir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.event_analysis.schemas import RuleMatch

_SEVERITY_ORDER = ["dusuk", "orta", "yuksek", "kritik"]
_SEVERITY_RANK = {name: rank for rank, name in enumerate(_SEVERITY_ORDER)}

_SEVERITY_MIDPOINT_SCORE = {
    "dusuk": 12,   # 0-25 bucket ortasi
    "orta": 38,    # 26-50 bucket ortasi
    "yuksek": 63,  # 51-75 bucket ortasi
    "kritik": 88,  # 76-100 bucket ortasi
}
"""`configs/config.yaml::agent.risk_thresholds` (25/50/75/100) ile hizali,
her siddet seviyesi icin temsili 0-100 skor."""


@dataclass
class RiskProvenance:
    """`resolve_deterministic_risk`in ATLADIGI - HANGI kural(lar)in bu karari
    urettigini tasiyan, salt-veri (hesaplama yapmayan) izlenebilirlik nesnesi.

    Operator/rapor/trace katmaninda "bu risk NEREDEN geldi?" sorusunun
    (bkz. gorev tanimi 3. bolum "Risk Explanation") LLM'e SORULMADAN,
    mevcut structured data'dan deterministik olarak cevaplanmasini saglar.
    Yeni bir risk hesaplama yontemi DEGILDIR - `resolve_deterministic_risk`
    ile TAM AYNI siddet-secim mantiginin (bkz. `resolve_deterministic_risk`
    docstring'i) yalnizca hangi `RuleMatch`(ler)e dayandigini da ifsa eder.
    """

    risk_level: Optional[str]
    risk_score: Optional[int]
    rule_ids: List[str] = field(default_factory=list)
    rule_severities: List[str] = field(default_factory=list)
    contributing_event_ids: List[str] = field(default_factory=list)

    def explanation(self) -> str:
        """Deterministik, LLM'e SORULMAMIS Turkce gerekce cumlesi uretir.

        Sadece bu nesnenin kendi alanlarindan (rule_ids/severities) turer -
        yeni bir bilgi/tahmin UYDURMAZ.
        """
        if self.risk_level is None:
            return "Hicbir deterministik kural eslesmedi; risk RuleEngine tarafindan belirlenemedi."
        rule_list = ", ".join(self.rule_ids) if self.rule_ids else "(bilinmeyen kural)"
        return (
            f"Risk seviyesi '{self.risk_level}' olarak belirlendi: en yuksek siddetli "
            f"eslesme(ler) {rule_list} ('{self.risk_level}' siddetinde). Bu karar "
            "VLM/LLM'in kendi tahmininden BAGIMSIZDIR - yalnizca RuleEngine "
            "eslesmelerinden deterministik olarak turetilmistir."
        )


def _pick_provenance(rule_matches: List[RuleMatch]) -> RiskProvenance:
    """`resolve_deterministic_risk`in TEK kaynagi: siddet-secim mantigini BURADA uygular.

    `resolve_deterministic_risk` ve `resolve_deterministic_risk_with_provenance`
    ikisi de bu fonksiyonu cagirir - iki ayri (birbirinden sapabilecek) risk
    hesaplama mantigi OLUSTURULMAZ.
    """
    known = [match for match in rule_matches if match.severity in _SEVERITY_RANK]
    if not known:
        return RiskProvenance(risk_level=None, risk_score=None)

    top_rank = max(_SEVERITY_RANK[match.severity] for match in known)
    contributing = [match for match in known if _SEVERITY_RANK[match.severity] == top_rank]
    risk_level = _SEVERITY_ORDER[top_rank]

    contributing_event_ids: List[str] = []
    for match in contributing:
        for event_id in [match.source_event_id, *match.related_event_ids]:
            if event_id not in contributing_event_ids:
                contributing_event_ids.append(event_id)

    return RiskProvenance(
        risk_level=risk_level,
        risk_score=_SEVERITY_MIDPOINT_SCORE[risk_level],
        rule_ids=[match.rule_id for match in contributing],
        rule_severities=[match.severity for match in contributing],
        contributing_event_ids=contributing_event_ids,
    )


def resolve_deterministic_risk_with_provenance(rule_matches: List[RuleMatch]) -> RiskProvenance:
    """`resolve_deterministic_risk` ile AYNI karari, HANGI kural(lar)a dayandigi bilgisiyle birlikte dondurur.

    Args:
        rule_matches: Bkz. `resolve_deterministic_risk`.

    Returns:
        `RiskProvenance` - `risk_level`/`risk_score` `resolve_deterministic_risk`
        ile BIREBIR AYNIDIR; ek olarak `rule_ids`/`rule_severities`/
        `contributing_event_ids` tasir.
    """
    return _pick_provenance(rule_matches)


def resolve_deterministic_risk(rule_matches: List[RuleMatch]) -> Tuple[Optional[str], Optional[int]]:
    """Bir olay/cagriya ait `RuleMatch` listesinden EN YUKSEK siddetli, deterministik risk cikarir.

    Formalizasyon (SAFIR'in nihai risk fonksiyonu, TAM olarak budur - baska
    bir yerde ikinci bir risk hesaplamasi YOKTUR):

        risk_level = argmax_{m in rule_matches} severity_rank(m.severity)
        risk_score = SEVERITY_MIDPOINT_SCORE[risk_level]

    Girdi kumesi YALNIZCA `RuleMatch.severity` degerlerinden olusur - VLM
    confidence, VLM'in kendi risk_score/anomaly-score ipucu, RAG/embedding
    benzerlik skoru veya LLM'in kendi risk tahmini bu fonksiyona ASLA
    girmez (bkz. `src/main.py::stage_finalize_risk`, `context_builder.py`
    modul dokustringi: bu kaynaklar risk_score/risk_level'i ETKILEMEZ).

    Hicbir kural eslesmediyse `(None, None)` doner - risk UYDURULMAZ; cagiran
    taraf bu durumda mevcut (orn. LLM Agent'tan gelen ya da "unknown") degeri
    korumalidir.

    Args:
        rule_matches: `RuleEngine.evaluate(...)` (tum cagri) veya
            `EventBuilder`in bir olaya grupladigi (`related_rule_matches`)
            `RuleMatch` listesi.

    Returns:
        `(risk_level, risk_score)`: en yuksek siddetli eslesmenin
        `severity`si ve buna karsilik gelen temsili skor. Bilinmeyen/gecersiz
        `severity` degerleri (`_SEVERITY_RANK` disinda) yok sayilir.
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
