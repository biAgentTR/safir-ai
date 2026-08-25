"""05c - Video Takip Sorusu (Ask SAFIR video-QA) istemi.

Mentor eleştirisi ("VLM Onbellegi / Prefix Caching Avantaji"): EVREN
dokumantasyonuna gore ayni video uzerine ILK sorgu ~17.8s surerken, AYNI
video ile devam eden sorular ~3.7s'de (4.8x daha hizli) yanitlaniyor - bu,
sunucu tarafindaki (vLLM-tarzi) otomatik prefix-cache davranisiyla tutarli:
AYNI video byte'lari tekrar gonderildiginde, modelin video-onek KV-cache'i
yeniden hesaplanmadan kullanilir. EVREN'in cache mekanizmasi (session/cache-key
mi, yoksa tamamen otomatik-tekrar-byte mi) dokumantasyonda ACIKCA belirtilmez;
bu yuzden burada EN SAVUNULABILIR, hicbir ozel API varsayimi GEREKTIRMEYEN
yaklasim kullanilir: `AskService`, ayni is (`job_id`) icin AYNI video dosyasini
OLDUGU GIBI tekrar EVREN'e gonderir (bkz. `EvrenVLM.answer_video_question`).
Bu istem, o ikinci (ve sonraki) cagrilarda VLM'e verilen soru-cevap
(observation-yerine Q&A) framing'ini tanimlar - `VLM_OBSERVER_SYSTEM_PROMPT`
(olay kumeleme/EVENTS_JSON) ile KARISTIRILMAMALIDIR; bu istem yapisal
JSON DEGIL, dogrudan Turkce serbest-metin bir cevap ister.
"""

from __future__ import annotations

ASK_VIDEO_SYSTEM_PROMPT = (
    "Sen SAFIR sisteminin saha guvenligi (ISG) video-analiz asistanisin. Sana "
    "az once analiz edilmis bir video ile bu videoya iliskin bir ONCEKI ANALIZ "
    "OZETI ve simdi kullanicinin bu video hakkinda sordugu YENI bir soru "
    "verilecek. Gorevin, VIDEOYU DOGRUDAN izleyip bu soruyu YALNIZCA videoda "
    "GERCEKTEN gordugune dayanarak, kisa ve net Turkce bir cevapla "
    "yanitlamaktir.\n\n"
    "## Kurallar\n"
    "- Yalnizca videoda GERCEKTEN gozlemlediklerini bildir; videoda olmayan "
    "bir detayi UYDURMA - gorunmuyorsa/emin degilsen bunu acikca belirt.\n"
    "- Onceki analiz ozeti sana baglam icin verilir ama SENIN KENDI GOZLEMIN "
    "esas alinir; ozet ile videoda gordugun celisirse, videoda GERCEKTEN "
    "gordugunu soyle.\n"
    "- Yapisal JSON/EVENTS_JSON URETME; yalnizca operatore yonelik, kisa "
    "(en fazla birkac cumle) serbest-metin bir cevap ver.\n"
    "- Ic muhakemeni veya adim adim dusunce zincirini YAZMA; yalnizca nihai cevabi ver."
)


def build_ask_video_user_prompt(question: str, analysis_summary: str) -> str:
    """Video-QA istegi icin kullanici istemini (onceki ozet + yeni soru) uretir.

    Args:
        question: Kullanicinin bu video hakkindaki yeni (takip) sorusu.
        analysis_summary: Bu videonun daha once uretilmis, kisa metin ozeti
            (`SafirReport.summary`/`natural_language_summary`) - yalniz baglam
            icin, KANIT olarak degil (bkz. modul dokustringi).

    Returns:
        VLM'e `user` mesaji icerigi olarak verilecek (video ile birlikte
        gonderilecek) metin istemi.
    """
    context_block = (
        f"ONCEKI ANALIZ OZETI (baglam icindir, videoda GORDUGUN esas alinir):\n{analysis_summary}\n\n"
        if analysis_summary
        else ""
    )
    return f"{context_block}SORU: {question}"


__all__ = ["ASK_VIDEO_SYSTEM_PROMPT", "build_ask_video_user_prompt"]
