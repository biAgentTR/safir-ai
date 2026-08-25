"""05b - Olay Siniflandirma Fallback Istemi: EventEngine'in kucuk-LLM geri-dususu icin.

Sartname kirmizi cizgisi: "Statik, yalnizca kural tabanli cozumler dusuk
puanlanacaktir". `EventEngine`in birincil yolu (VLM'in urettigi `EVENTS_JSON`
- zaten model-tabanli siniflandirmadir) VE ikincil yolu (anahtar kelime
eslestirme) HER IKISI de basarisiz olup hicbir kategori bulamadiginda
(`genel_gozlem`e dusmeden HEMEN ONCE), bu istemle KUCUK/HIZLI bir LLM'e
("llm-fast") TEK bir siniflandirma sorusu sorulur - boylece "kural motoru
%100 emin olamazsa otonom karar" mantigi kurulur (bkz.
`src/event_analysis/event_engine.py::EventEngine._classify_with_llm_fallback`).

Bu, `05 LangGraph Ajani`nin risk-karari istemiyle KARISTIRILMAMALIDIR - bu
istem yalnizca "bu gozlem hangi ISG kategorisine giriyor?" sorusuna KISA,
JSON bicimli bir cevap ister; risk skoru/aksiyon ONERMEZ.
"""

from __future__ import annotations

# `EventType`teki (schemas.py) 8 mevzuat-hizalanmis kategori + 2 operasyonel
# ozel kategori (yetkisiz_erisim, genel_gozlem) + 1 "sinirlarin disinda ama
# riskli" kategorisi (siniflandirilamadi) - schemas.py'deki docstring'lerle
# AYNI, TEK KAYNAKTAN (schemas.py) turetilmis kisa aciklamalar.
EVENT_CLASSIFIER_CATEGORIES = (
    "- dusme_riski: yukseklikte calisma / dusme onleyici ekipman eksikligi\n"
    "- kkd_ihlali: Kisisel Koruyucu Donanim (baret, yelek, is ayakkabisi) eksikligi\n"
    "- arac_yaya_yakinligi: forklift/is makinesi ile yaya gecidi/yaya yakinligi ihlali\n"
    "- sicak_calisma_ihlali: izinsiz ates/kivilcim/sicak yuzey islemi\n"
    "- yangin_duman: duman veya alev tespiti\n"
    "- dar_alan_ihlali: gaz olcumu/gozetmen olmadan kapali-dar alana giris\n"
    "- enerji_kesme_ihlali: enerji kesme (LOTO) prosedurune uyulmadan mudahale\n"
    "- agir_yuk_riski: sinyalman olmadan vinc/kren ile agir yuk kaldirma\n"
    "- yetkisiz_erisim: yasakli/yetkisiz alana giris\n"
    "- siniflandirilamadi: yukaridaki 9 kategoriden HICBIRINE oturmayan ama YINE DE "
    "anormal/riskli gorunen bir durum\n"
    "- genel_gozlem: rutin, risksiz/anormal olmayan bir gozlem (hicbir ihlal/tehlike yok)"
)

EVENT_CLASSIFIER_SYSTEM_PROMPT = (
    "Sen SAFIR sisteminin saha guvenligi (ISG) olay siniflandirma yardimcisisin. "
    "Sana bir VLM'in (gorsel-dil modeli) urettigi TEK bir saha gozlemi (serbest metin) "
    "verilecek. Gorevin, bu gozlemi ASAGIDAKI 11 kategoriden TAM OLARAK BIRINE "
    "atamaktir. Bu, deterministik anahtar-kelime eslestirmesinin BULAMADIGI (esanlamli "
    "ifade, dolayli anlatim, farkli kelime secimi gibi nedenlerle) durumlar icin "
    "SON CARE bir geri-dusustur - dikkatli ve MUHAFAZAKAR ol.\n\n"
    "## Kategoriler\n"
    f"{EVENT_CLASSIFIER_CATEGORIES}\n\n"
    "## Kurallar\n"
    "- Gozlemde ACIKCA belirtilen bir tehlike/ihlal yoksa 'genel_gozlem' sec - "
    "bir kategori UYDURMA veya zorlama YAPMA.\n"
    "- Riskli/anormal gorunen ama 9 somut kategoriden hicbirine TAM oturmayan bir "
    "durum icin 'siniflandirilamadi' sec.\n"
    "- 'confidence' alani senin KENDI emin olma derecendir (0.0-1.0); emin degilsen "
    "DUSUK bir deger ver, 1.0'i sadece gozlem tamamen ACIK ise kullan.\n"
    "- SADECE asagidaki JSON semasina uygun, gecerli bir JSON nesnesi dondur "
    "(baska hicbir metin ekleme, kod blogu isaretleyicisi kullanma):\n"
    '{\n  "event_type": "<yukaridaki 11 kategoriden biri>",\n'
    '  "confidence": <0.0-1.0 arasi sayi>,\n'
    '  "reason": "<Turkce, tek cumlelik kisa gerekce>"\n}'
)


def build_event_classifier_user_prompt(vlm_description: str) -> str:
    """Siniflandirilacak gozlemi, kullanici istemi olarak saran metni uretir.

    Args:
        vlm_description: `EventEngineInput.vlm_description` (VLM'in serbest metin gozlemi).

    Returns:
        LLM'e `HumanMessage` icerigi olarak verilecek istem metni.
    """
    return f"Siniflandirilacak gozlem:\n{vlm_description}\n\nSADECE belirtilen JSON semasiyla yanit ver."


__all__ = [
    "EVENT_CLASSIFIER_SYSTEM_PROMPT",
    "EVENT_CLASSIFIER_CATEGORIES",
    "build_event_classifier_user_prompt",
]
