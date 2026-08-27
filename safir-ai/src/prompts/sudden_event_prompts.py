"""Ani Olay Dogrulama (Sudden Event Verification) istemi.

`src/vlm/sudden_event_detector.py::detect_sudden_events`, OpenCV kare-farki
ile videoyu tarayip ortalama gurultu seviyesinden ANLAMLI OLCUDE sapan
("ani") kare-farki sicramalari bulur ve her sicrama icin ONCESI/SONRASI
kare cifti cikarir. Bu ciftler, goruntu-kabul eden bir modele ("llm-large",
EVREN dokumantasyonu SS 7.5'teki "iki kare karsilastirma" deseniyle)
gonderilip GERCEKTEN onemli bir guvenlik olayi mi (ör. bir kapinin/mekanik
parcanin aniden kapanip bir kisiye carpmasi/sikistirmasi) yoksa siradan bir
kamera titremesi/isik degisimi/normal hareket mi oldugu bu istemle
DOGRULANIR - boylece saf hareket tespiti (yanlis-pozitif orani yuksek)
hicbir zaman DOGRUDAN rapora yansimaz, yalnizca bu ikinci, gorsel
dogrulamadan GECEN adaylar yansir.
"""

from __future__ import annotations

SUDDEN_EVENT_VERIFICATION_PROMPT = (
    "Sen saha guvenligi goruntu analizinde uzman bir denetcisin. Sana bir "
    "videodan ardisik ALINMIS iki kare veriliyor: ilk goruntu 'ONCESI' "
    "(olasi ani degisimden hemen ONCE), ikinci goruntu 'SONRASI' (hemen "
    "SONRA). Bu iki kare arasinda otomatik bir hareket-tespit sistemi ani "
    "bir degisim SICRAMASI algiladi, ama bu sistem YANLIS ALARM da "
    "verebilir (kamera titremesi, isik degisimi, siradan yuruyus). "
    "GOREVIN: bu iki karedeki degisimin GERCEKTEN onemli bir guvenlik "
    "olayi olup olmadigini KESIN olarak dogrulamak.\n\n"
    "Ozellikle ARA - bir kapi/kapak/bariyer/pres/konveyor/makine kolu gibi "
    "HAREKETLI bir mekanik/yapisal parcanin aniden kapanmasi/hareket etmesi "
    "VE bunun bir kisinin eli/kolu/parmagi/ayagi/govdesiyle temas etmesi "
    "(sikisma/ezilme/darbe); ayrica ani bir dusme, carpisma veya beklenmedik "
    "bir nesnenin ortaya cikmasi da GECERLI bir 'onemli olay'dir.\n\n"
    "KURALLAR (KESINLIKLE UY):\n"
    "1. Yalnizca bu IKI karede NET GORDUGUNU yaz. Spekulasyon/varsayim "
    "YASAK - 'olabilir', 'gorunuyor gibi', 'muhtemelen' gibi hicbir hedge "
    "ifade KULLANMA.\n"
    "2. Iki kare arasinda gordugun degisim sahnenin GENEL isik/kompozisyon "
    "farkindan ibaretse (kamera titremesi, otomatik pozlama degisimi, "
    "siradan yuruyus/hareket) bunu ONEMLI bir olay SAYMA.\n"
    "3. Kararsizsan (net degilse) `is_notable_event`i false birak - "
    "'belki'/'olabilir' bir sonuc UYDURMAKTANSA guvenli tarafta kal.\n"
    "4. SADECE su JSON semasina uygun, gecerli bir JSON nesnesi dondur, "
    "baska hicbir metin/aciklama/kod blogu isaretleyicisi EKLEME:\n"
    '{"is_notable_event": <true|false>, '
    '"description": "<NET gordugunu Turkce, kisa ve nesnel anlat; '
    'is_notable_event=false ise neden onemsiz oldugunu KISACA belirt>", '
    '"event_name": "<is_notable_event=true ise kisa, snake_case bir olay '
    "adi (ör. 'kapi_sikismasi', 'ani_dusme'); false ise bos dize>\", "
    '"confidence": <0.0-1.0>}'
)

__all__ = ["SUDDEN_EVENT_VERIFICATION_PROMPT"]
