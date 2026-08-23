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
    known = [match for match in rule_matches if match.severity in _SEVERITY_RANK]
    if not known:
        return None, None

    top = max(known, key=lambda match: _SEVERITY_RANK[match.severity])
    return top.severity, _SEVERITY_MIDPOINT_SCORE[top.severity]


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
