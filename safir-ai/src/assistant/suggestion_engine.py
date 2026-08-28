"""09D - Dinamik Takip Sorusu Onerileri: rapora OZGU, sabit-olmayan SAFIR Asistan chip'leri.

Mentor eleştirisi ("Jurinin Demo Videosu ve Beklentileri"): şartnamedeki
"Diyalog sırasında inisiyatif alma ve doğru soruları sorma" kriteri, sabit
("SAFİR nasıl çalışır?" gibi) öneri chip'leriyle KARŞILANMAZ - bunlar HER
raporda AYNIdır, hiçbir inisiyatif göstermez. Bu modul, bir `SafirReport`ın
KENDİ içeriğinden (sınıflandırılamayan olaylar, insan incelemesine düşen
eskalasyon, VLM'in kendi metninde ifade ettiği belirsizlik, kişi-içeren olay
kategorileri) o rapora ÖZGÜ takip soruları üretir - böylece mentörün verdiği
tam örnek ("İlgili personelin yüzü net mi?") gibi sorular, rapor bunu hak
ediyorsa GERÇEKTEN önerilir.

Kasıtlı olarak KUCUK-LLM/sınıflandırıcı DEĞİL, tamamen deterministik/kural-
tabanlıdır - SebEP: (1) rapor zaten VLM/RuleEngine/LLM-fallback'in ürettiği
yapılandırılmış sinyalleri taşıyor, bunları TEKRAR bir modele sormak gereksiz
bir ek EVREN çağrısı (gecikme/maliyet) olurdu; (2) öneri chip'leri güvenlik-
kritik bir karar DEĞİLDİR (yalnızca operatöre bir soru önerisi sunar, kendisi
bir sonuç üretmez) - şeffaf/açıklanabilir bir kural kümesi burada LLM
"muhakemesinden" daha uygun bir araçtır. Sonuç HER ZAMAN bir ÖNERİDİR;
operatör dilerse hiçbirini kullanmadan kendi sorusunu da sorabilir.
"""

from __future__ import annotations

import re
from typing import List

from src.schemas.report import SafirReport

# Kişi/operatör-içeren olay kategorileri (bkz. src/event_analysis/schemas.py
# EventType) - bu kategorilerden biri tespit edildiyse, mentörün kendi
# örneğiyle (video-QA'nın somut faydasını gösteren, "yüz net mi" tarzı) bir
# görsel-doğrulama sorusu ÖZELLİKLE anlamlıdır.
_PERSON_INVOLVING_EVENT_TYPES = {
    "dusme_riski",
    "kkd_ihlali",
    "arac_yaya_yakinligi",
    "dar_alan_ihlali",
    "yetkisiz_erisim",
}

# VLM'in kendi serbest metninde (natural_language_summary) GERÇEKTEN ifade
# ettiği belirsizlik/hedge dili - bir sınıflandırma DEĞİL, yalnızca modelin
# zaten yazdığı metinde bu kelimelerin GEÇİP GEÇMEDİĞİNİ arar (regex).
_HEDGE_PATTERN = re.compile(
    r"\b(net değil|net görünmüyor|belirsiz|emin değil|tam görünmüyor|"
    r"anlaşılmıyor|ayırt edilemiyor|görünmüyor)\b",
    re.IGNORECASE,
)

_MAX_SUGGESTIONS = 4


def build_dynamic_suggestions(report: SafirReport) -> List[str]:
    """Verilen rapora ÖZGÜ, en fazla `_MAX_SUGGESTIONS` takip sorusu önerisi üretir.

    Sıra, operatör için ÖNCELİK sırasıdır (en açıklayıcı/aksiyona-yönlendirici
    ilk gelir): (1) sınıflandırılamayan/başarısız olay, (2) insan incelemesine
    düşen eskalasyon, (3) VLM'in kendi metninde ifade ettiği belirsizlik,
    (4) kişi-içeren bir olay kategorisi (görsel doğrulama sorusu).

    Args:
        report: Kalıcı, tamamlanmış bir analiz raporu.

    Returns:
        0 ile `_MAX_SUGGESTIONS` arası öneri metni; hiçbir sinyal
        bulunamazsa BOŞ liste (çağıran taraf statik/genel önerilere döner -
        bu fonksiyon ASLA bir öneri UYDURMAZ).
    """
    suggestions: List[str] = []

    unresolved = [
        ev.event_name
        for ev in report.events
        if ev.event_type is None or ev.event_name == "degerlendirme_yapilamadi"
    ]
    if unresolved:
        name = unresolved[0]
        suggestions.append(
            f"'{name}' net biçimde sınıflandırılamadı; videoyu tekrar inceleyip bu anı biraz daha anlatır mısın?"
        )

    if report.escalation_tier == "pending_review" or report.risk_status != "assessed":
        suggestions.append(
            "Bu analiz insan incelemesine yönlendirildi. Videoda seni bu sonuca götüren en belirsiz an hangisiydi?"
        )

    hedge_match = _HEDGE_PATTERN.search(report.natural_language_summary or "")
    if hedge_match:
        suggestions.append(
            f"Gözleminde '{hedge_match.group(0)}' dediğin bir nokta var; videoyu tekrar bakıp netleştirir misin?"
        )

    if any(ev.event_type in _PERSON_INVOLVING_EVENT_TYPES for ev in report.events):
        suggestions.append("İlgili personelin yüzü/kimliği videoda net görünüyor mu?")

    if report.risk_level in ("yuksek", "kritik") and len(suggestions) < _MAX_SUGGESTIONS:
        suggestions.append("Bu risk seviyesini en çok hangi kare/an belirledi, videodan gösterir misin?")

    seen: set[str] = set()
    unique: List[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:_MAX_SUGGESTIONS]


__all__ = ["build_dynamic_suggestions"]
