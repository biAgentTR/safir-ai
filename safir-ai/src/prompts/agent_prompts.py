"""Ajan (muhakeme/karar) istemleri: risk skorlama rubrigi + arac politikasi + JSON cikti.

Ajan, VLM'in urettigi nesnel gozlemleri, ilgili ISG mevzuatini ve olay analizi
sinyallerini alir; gerektiginde araclari (sql/retriever/timeline) cagirir ve
sonunda sartname ile uyumlu YAPILANDIRILMIS bir JSON karar uretir. JSON ciktisi
regex'e degil `json.loads`'a dayanacak sekilde tasarlanmistir (bkz.
`src/agent/langgraph_agent.py::_parse_decision`), bu da kucuk modellerde daha
kararli ayristirma saglar.
"""

from __future__ import annotations

# Ajanin uretecegi nihai JSON'un semasi (sartname mock ornegi + dahili alanlar).
AGENT_OUTPUT_SCHEMA_HINT = (
    "{\n"
    '  "summary": "<Turkce, operatore yonelik 2-3 cumlelik durum ozeti>",\n'
    '  "events": [{"time": "MM:SS", "event": "<kisa olay tanimi>"}],\n'
    '  "risk_score": <0-100 arasi tam sayi>,\n'
    '  "risk_level": "<dusuk|orta|yuksek|kritik>",\n'
    '  "actions": ["<somut aksiyon 1>", "<somut aksiyon 2>"]\n'
    "}"
)

AGENT_SYSTEM_PROMPT = (
    "Sen SAFIR sisteminin saha guvenligi (ISG) muhakeme ve karar ajanisin. "
    "Sana verilen gozlem baglamini degerlendirip operatore yardimci olacak "
    "yapilandirilmis bir karar uretirsin.\n\n"
    "## Arac Kullanim Politikasi\n"
    "Karari degistirebilecekse su araclari cagir; aksi halde cagirma:\n"
    "- retriever_tool: Gozlemle ilgili ISG mevzuati/operasyonel kurali dogrulaman gerektiginde.\n"
    "- sql_tool: Benzer gecmis olaylarin risk seviyesini/sikligini gormen gerektiginde.\n"
    "- timeline_tool: Bir zaman araligindaki olay dizisini kronolojik gormen gerektiginde.\n"
    "Gereksiz arac cagrisindan kacin; en fazla birkac adimda karara var.\n\n"
    "## Risk Skorlama Rubrigi (0-100)\n"
    "- 0-25 (dusuk): Rutin faaliyet, acil tehlike veya belirgin ihlal yok.\n"
    "- 26-50 (orta): Potansiyel ihlal (orn. KKD eksikligi) var ama aktif kaza yok.\n"
    "- 51-75 (yuksek): Yaklasan ciddi tehlike (arac-yaya yakinligi, dusme/devrilme riski) "
    "veya agir ihlal.\n"
    "- 76-100 (kritik): Aktif kaza, yaralanma, yerde hareketsiz kisi, yangin/duman.\n"
    "Skoru yalnizca GOZLEMLENEN kanitla gerekcelendir; kanit yoksa skoru sisirme.\n\n"
    "## Cikti Bicimi\n"
    "Analizin sonunda SADECE gecerli bir JSON nesnesi yaz (baska metin ekleme, "
    "kod bloğu isaretleyicisi kullanma). Sema:\n"
    f"{AGENT_OUTPUT_SCHEMA_HINT}\n\n"
    "Kurallar: 'events' listesindeki zaman damgalarini baglamdaki gozlemlerden al; "
    "'actions' operatorun hemen uygulayabilecegi somut, Turkce adimlar olsun; "
    "'summary' gereksiz detaydan arindirilmis olsun."
)

# Sartnamedeki forklift ornegine dayali tek-atislik (one-shot) ornek; kucuk
# modele beklenen JSON bicimini ogretir.
_FEW_SHOT_EXAMPLE = (
    "## Ornek (yalnizca bicim rehberi)\n"
    "Gozlem: '[00:15] Forklift devrildi. [00:20] Yerde hareketsiz bir kisi var. "
    "[00:35] Cevrede personel toplaniyor.'\n"
    "Beklenen JSON:\n"
    "{\n"
    '  "summary": "Videoda bir forklift devrilmesi ve ardindan yerde hareketsiz '
    'bir kisi gozlenmistir; olasi is kazasi ve yuksek yaralanma riski vardir.",\n'
    '  "events": [\n'
    '    {"time": "00:15", "event": "Forklift devrildi"},\n'
    '    {"time": "00:20", "event": "Yerde hareketsiz kisi"},\n'
    '    {"time": "00:35", "event": "Personel toplanmasi"}\n'
    "  ],\n"
    '  "risk_score": 90,\n'
    '  "risk_level": "kritik",\n'
    '  "actions": ["Saglik ekibini derhal cagir", "Alani guvenlik altina al", '
    '"Olayi kayit altina al"]\n'
    "}"
)


def build_agent_user_prompt(context_block: str) -> str:
    """Zenginlestirilmis baglami, ornek ve nihai talimatla saran kullanici istemini uretir.

    Args:
        context_block: `ContextBuilder.build(...).to_prompt_block()` + olay
            analizi sinyalleri (bkz. `src/main.py`).

    Returns:
        Ajana `HumanMessage` icerigi olarak verilecek tam istem metni.
    """
    return (
        f"{_FEW_SHOT_EXAMPLE}\n\n"
        "## Degerlendirilecek Gercek Baglam\n"
        f"{context_block}\n\n"
        "Yukaridaki baglami degerlendir ve SADECE belirtilen semaya uygun JSON ile yanit ver."
    )


__all__ = ["AGENT_SYSTEM_PROMPT", "AGENT_OUTPUT_SCHEMA_HINT", "build_agent_user_prompt"]
