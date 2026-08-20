"""T017 - Regulation Match Resolver: "mevzuat GETIRILDI" ile "mevzuat UYGULANIR" ayrimini acikca kurar.

Kok sorun (bkz. proje sohbeti)
-------------------------------
`src/memory/context_builder.py`, onceden `EmbeddingRAGService.query(vlm_description)`
ile SERBEST METIN uzerinde FAISS benzerlik aramasi yapip top-k sonucu, hicbir
uygulanabilirlik kontrolu olmadan `SafirReport.relevant_regulations` alanina
"ilgili mevzuat" olarak yaziyordu. Vektor/embedding benzerligi, bir mevzuatin
GERCEKTEN uygulanabilir oldugunun kaniti DEGILDIR - yalnizca metinsel/anlamsal
yakinlik gosterir. Ornek yanlis pozitif: "forklift sahada hareket etti"
gozlemi, "KKD kullanimi" mevzuatiyla yalnizca kelime/konu yakinligi yuzunden
eslestirilebiliyordu (KKD ihlaline dair hicbir kanit olmadan).

Bu modul ucu ayirir:

1. OLAY GERCEGI (event fact): `EventEngine`/`TemporalReasoner` ->
   `TemporalEvent.event_type` (VLM gozlemi + deterministik siniflandirma).
2. MEVZUAT UYGULANABILIRLIGI (regulation applicability): `RuleEngine` ->
   `EVENT_TYPE_REGULATION_MAP` (tekli) + `rules/isg_rules.yaml` (kombinasyon)
   -> `RuleMatch`. Bu tablo, event_type basina ONCEDEN incelenmis/dogrulanmis
   sabit bir esleme uretir; embedding benzerligiyle "tahmin" EDILMEZ (bkz.
   `src/event_analysis/rule_engine.py`).
3. MEVZUAT GETIRME (regulation retrieval): FAISS/RAG yalnizca, ZATEN BILINEN
   bir mevzuat ETIKETININ (orn. "ISG Yonetmeligi Madde 12") TAM METNINI
   getirmek icin kullanilir (bkz. `RuleEngine._describe_regulation`); bu,
   bagimsiz bir "uygulanabilirlik" karari DEGILDIR - yalnizca zaten karari
   verilmis bir kaydin dokuman-getirme (lookup) adimidir.

`resolve_regulation_matches`, (2)'nin ciktisini "matched"/"no_match" ayrimini
acikca tasiyan bir `RegulationMatch` kaydina cevirir. `rule_matches` bossa
(RuleEngine hicbir tekli/kombinasyon kurali eslestirmediyse) SONUC BOS
LISTEDIR - bu, "Mevzuat eslestirilemedi" durumunun GECERLI ve BEKLENEN
sonucudur; yanlis bir mevzuat UYDURULMAZ.

Risk ile karistirilmamali
--------------------------
Bu modul risk skoru/seviyesi HESAPLAMAZ (bkz. `src/event_analysis/risk_resolver.py`,
degistirilmedi). `relevance_score` alani da bir "risk agirligi" DEGILDIR;
yalnizca eslesmenin deterministik/dogrulanmis (1.0) mi yoksa yok (0.0) mu
oldugunu belirtir. Bir mevzuat bulunmasi olayi daha riskli YAPMAZ; bulunmamasi
da olayi daha az riskli YAPMAZ - risk halen tek basina `RuleEngine.severity`
-> `risk_resolver.resolve_deterministic_risk`'ten gelir.
"""

from __future__ import annotations

from typing import List

from src.event_analysis.schemas import RegulationMatch, RuleMatch

NO_MATCH_REASON = (
    "Tespit edilen olay(lar) icin guvenilir/dogrulanmis bir ISG mevzuat "
    "eslestirmesi bulunamadi (RuleEngine hicbir deterministik esleme uretmedi)."
)
"""Hicbir `RuleMatch` yokken raporda/istemde gosterilecek standart gerekce metni."""


def resolve_regulation_matches(rule_matches: List[RuleMatch]) -> List[RegulationMatch]:
    """`RuleEngine.evaluate(...)` ciktisindan, yalnizca GERCEKTEN uygulanan (deterministik dogrulanmis) `RegulationMatch` kayitlari uretir.

    RuleEngine zaten yalnizca event_type (veya kombinasyon) bazinda ONCEDEN
    dogrulanmis eslesmeleri uretir - "zayif"/"muglak" bir `RuleMatch` KAVRAMI
    yoktur (bkz. `EVENT_TYPE_REGULATION_MAP`, `rules/isg_rules.yaml`). Bu
    yuzden buradaki tum donusler `match_status="matched"`dir; RAG benzerlik
    skoru burada HICBIR ROL OYNAMAZ.

    Args:
        rule_matches: `RuleEngine.evaluate(...)` ciktisi (tekli + kombinasyon,
            bu cagriya ait tum eslesmeler).

    Returns:
        Her `RuleMatch` icin bir `RegulationMatch` (matched). `rule_matches`
        bossa BOS LISTE - "Mevzuat eslestirilemedi" GECERLI sonuctur (bkz.
        `NO_MATCH_REASON`, cagiran taraf bos listeyi bu sekilde yorumlamalidir).
    """
    return [
        RegulationMatch(
            match_status="matched",
            regulation_id=match.rule_id,
            regulation_title=match.rule_description,
            relevance_score=1.0,
            reason=(
                f"Olay tipi '{match.event_type}', {match.rule_id} kuraliyla deterministik "
                "olarak eslesir (RuleEngine: event_type -> mevzuat esleme tablosu; "
                "RAG benzerlik skoruna DAYANMAZ)."
            ),
        )
        for match in rule_matches
    ]
