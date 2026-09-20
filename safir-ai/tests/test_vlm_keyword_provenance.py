"""VLM'in KENDI urettigi anahtar kelimelerin uctan uca korundugunu dogrulayan testler.

NEDEN BU DOSYA VAR (regresyon kaydi)
------------------------------------
Sistem, VLM'in serbest bicimli gozlem terimlerini (`attributes`/`entities`)
`DetectedEvent.matched_keywords`e tasimak uzere tasarlanmistir. Ancak uc ayri
noktada bu zincir KOPMUSTU ve sonucta her olay `matched_keywords=[]` ile
ilerliyor, sistem de `event_engine._KEYWORD_RULES` icindeki SABIT 10 kategorilik
taksonomiye geri dusuyordu:

1. `base_vlm._post_chat_completion`, typed `observations` -> `structured_events`
   eslemesinde `attributes`/`entities`/`canonical_type` alanlarini DUSURUYORDU.
2. Video-dogrudan yol (`evren_vlm`) yalnizca eski `EVENTS_JSON` blogunu
   ayristiran bir parser kullaniyordu; prompt ise typed JSON dayatiyordu -
   yani `structured_events` HER ZAMAN bos donuyordu.
3. `parse_vlm_response`, typed JSON'u ham metne `json.loads` uygulayarak
   ariyordu; modelin JSON'u bir ```json citi icine sarmasi (cok yaygin)
   ciktinin TAMAMEN "unrecognized_format" sayilmasina yol aciyordu.

Bu testler, ucunun de kapali kaldigini garanti eder. `matched_keywords` bos
kalirsa yalnizca kelimeler degil, semantik RAG sorgusu da sessizce devre disi
kalir (sorgu `matched_keywords`ten kurulur), bu yuzden regresyon PAHALIDIR.
"""

from __future__ import annotations

import json

from src.event_analysis.event_engine import EventEngine, _KEYWORD_RULES
from src.event_analysis.schemas import EventEngineInput
from src.vlm.parser import observations_to_structured_events, parse_vlm_response

# Sabit taksonomideki TUM kelimeler - testler, VLM'in kendi terimlerinin
# bunlardan BAGIMSIZ oldugunu gostermek icin bu kumeyi kullanir.
_STATIC_TAXONOMY_TERMS = {kw for keywords in _KEYWORD_RULES.values() for kw in keywords}


def _vlm_typed_output(*, fenced: bool) -> str:
    """Gemini'nin typed semaya uygun, gercekci bir ciktisini uretir.

    Args:
        fenced: `True` ise JSON bir ```json kod citi icine sarilir (modellerin
            "markdown kullanma" talimatina ragmen cok sik yaptigi sey).

    Returns:
        Ham model ciktisi metni.
    """
    payload = {
        "schema_version": "1.0",
        "scene_summary": "Depo sahasinda forklift manevrasi sirasinda yaya yakinligi gozlendi.",
        "observations": [
            {
                "observed_label": "Forklift donus yaparken yayaya asiri yaklasti",
                # Model bu olayi bilinen taksonomiye eslestirMEDI -> novel.
                "canonical_type": None,
                "taxonomy_status": "novel",
                "relative_start_sec": 12.0,
                "relative_end_sec": 15.5,
                "confidence": 0.82,
                "visibility": "clear",
                "entities": ["forklift", "yaya"],
                # VLM'in KENDI kelimeleri - sabit sozlukte BULUNMAYAN ifadeler.
                "attributes": ["ani donus manevrasi", "kor nokta", "yaya cok yakin"],
                "evidence": ["ev12", "ev13"],
                "uncertainties": ["Yaya'nin yuzu net gorunmuyor"],
            }
        ],
        "quality": {"visibility": "clear", "limitations": [], "coverage_confidence": 0.9},
    }
    raw = json.dumps(payload, ensure_ascii=False)
    return f"Iste analiz sonucu:\n```json\n{raw}\n```" if fenced else raw


def _detect(raw_content: str):
    """Ham VLM ciktisini production yolundan gecirip `DetectedEvent` listesi dondurur."""
    description, chunk_res = parse_vlm_response(raw_content)
    structured_events = observations_to_structured_events(chunk_res)
    return structured_events, EventEngine().detect(
        EventEngineInput(
            vlm_description=description,
            timestamp=0.0,
            source_model="gemini-2.5-flash",
            structured_events=structured_events,
        )
    )


def test_vlm_kendi_anahtar_kelimeleri_matched_keywords_e_tasinir() -> None:
    """VLM'in `attributes`/`entities` terimleri `matched_keywords`e AYNEN ulasir."""
    structured_events, events = _detect(_vlm_typed_output(fenced=False))

    assert structured_events, "typed `observations` semasi `structured_events` uretmeliydi"
    assert structured_events[0]["keywords"], "VLM terimleri `keywords` alanina tasinmali"

    assert len(events) == 1
    keywords = events[0].matched_keywords
    # `attributes` once, `entities` sonra; ikisi de KORUNUR.
    assert "ani donus manevrasi" in keywords
    assert "kor nokta" in keywords
    assert "yaya cok yakin" in keywords
    assert "forklift" in keywords


def test_kod_citi_icindeki_json_da_ayristirilir() -> None:
    """Model JSON'u ```json citi icine sarsa bile olaylar/kelimeler KAYBOLMAZ.

    Bu, sessiz bir tam-veri-kaybi senaryosuydu: cit yuzunden ayristirma
    basarisiz olunca `structured_events` bos kaliyor ve sistem sabit
    taksonomiye dusuyordu.
    """
    structured_events, events = _detect(_vlm_typed_output(fenced=True))

    assert structured_events, "kod citi icindeki gecerli JSON ayristirilmaliydi"
    assert len(events) == 1
    assert "ani donus manevrasi" in events[0].matched_keywords


def test_sabit_taksonomiden_kelime_EKLENMEZ() -> None:
    """Her `matched_keywords` terimi, VLM'in FIILEN urettigi bir terimdir.

    Kritik ayrim: bir VLM terimi sabit sozlukteki bir kelimeyle TESADUFEN
    ayni olabilir (ör. model gercekten "forklift" yazmistir) - bu bir sorun
    DEGILDIR. Sorun, VLM'in YAZMADIGI bir terimin taksonomiden TURETILIP
    listeye eklenmesidir; eski davranista `matched_keywords` tamamen
    `_KEYWORD_RULES`tan doldurulyordu. Bu yuzden dogru iddia "taksonomiyle
    kesisim yok" degil, "kaynak yalnizca modelin ciktisi" olmalidir.
    """
    _, events = _detect(_vlm_typed_output(fenced=False))

    vlm_terms = {"ani donus manevrasi", "kor nokta", "yaya cok yakin", "forklift", "yaya"}
    keywords = set(events[0].matched_keywords)

    assert keywords, "kelimeler bos kalmamali"
    assert keywords <= vlm_terms, (
        "VLM'in URETMEDIGI terim(ler) eklenmis (sabit taksonomi sizintisi): " f"{keywords - vlm_terms}"
    )
    # Taksonomide KARSILIGI OLMAYAN serbest terimler hayatta kalmali - sistemin
    # 10 sabit kategoriye indirgenMEDIGININ dogrudan kaniti.
    assert keywords - _STATIC_TAXONOMY_TERMS, "hicbir serbest (taksonomi disi) terim korunmamis"


def test_canonical_type_null_ise_tip_zorlanmaz() -> None:
    """Model taksonomiye eslestirmediyse (`canonical_type: null`) tip UYDURULMAZ."""
    structured_events, events = _detect(_vlm_typed_output(fenced=False))

    assert structured_events[0]["canonical_event_type"] is None
    assert structured_events[0]["taxonomy_status"] == "novel"
    # `event_type` None kalir; olay yine de KAYBOLMAZ (serbest `event_name` ile durur).
    assert events[0].event_type is None
    assert events[0].event_name == "Forklift donus yaparken yayaya asiri yaklasti"
