# -*- coding: utf-8 -*-
"""
PDF Turkce font destegi (Esra — polish/fix(ui))

Reportlab'in varsayilan Helvetica fontu Turkce karakterleri (g, s, i, o, u
noktali/cengelli halleri) icermez; PDF'te "yagmur/yangin" gibi bozuk cikar.
Bu modul, sistemde bulunan Turkce destekli bir TTF fontu bulup reportlab'e
"ArialTR" / "ArialTR-Bold" adlariyla kaydeder.

Font ARANMA sirasi (ilk bulunan kullanilir):
  1) Windows: Arial (C:/Windows/Fonts)         -> gelistirici makineleri
  2) Linux/Docker: DejaVuSans                  -> konteyner ortami
  3) Yerel fonts/ klasoru (opsiyonel)

Hicbiri bulunamazsa sessizce Helvetica'ya doner (PDF yine uretilir,
sadece Turkce karakterler eksik kalir) — sistem asla cokmez.
"""

import os

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

TR_FONT = "ArialTR"
TR_FONT_BOLD = "ArialTR-Bold"

_ADAYLAR = [
    # (normal, bold)
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    (os.path.join(os.path.dirname(__file__), "fonts", "arial.ttf"),
     os.path.join(os.path.dirname(__file__), "fonts", "arialbd.ttf")),
]

_kayitli = False


def register_tr_fonts() -> bool:
    """Turkce destekli fontlari reportlab'e kaydeder. Basari durumunu dondurur."""
    global _kayitli, TR_FONT, TR_FONT_BOLD
    if _kayitli:
        return True
    for normal, bold in _ADAYLAR:
        try:
            if os.path.exists(normal):
                pdfmetrics.registerFont(TTFont("ArialTR", normal))
                pdfmetrics.registerFont(
                    TTFont("ArialTR-Bold", bold if os.path.exists(bold) else normal)
                )
                _kayitli = True
                return True
        except Exception:
            continue
    # Bulunamadi: guvenli geri donus — cagiran kod bu adlari kullanmaya devam etsin
    TR_FONT = "Helvetica"
    TR_FONT_BOLD = "Helvetica-Bold"
    return False
