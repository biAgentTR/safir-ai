"""T020 (src/schemas/report.py) icin `EventSummary`/`SafirReport.events` birim testleri.

`SafirReport.events`in, VLM-uretimi serbest-bicimli olay kimligini
(`event_name`, taksonomiyle SINIRLI OLMAYAN), opsiyonel canonical
`event_type`i ve keywords'u hicbir donusum/filtreleme olmadan (API payload
simulasyonu olan `model_dump()` dahil) tasidigini dogrular.
"""

from __future__ import annotations

from src.schemas.report import EventSummary, SafirReport


def _minimal_report(**overrides) -> SafirReport:
    defaults = dict(
        video_source="v.mp4",
        generated_at="2026-01-01T00:00:00Z",
        natural_language_summary="obs",
        risk_score=10,
        risk_level="dusuk",
        recommended_action="izle",
    )
    defaults.update(overrides)
    return SafirReport(**defaults)


def test_events_defaults_to_empty_list() -> None:
    report = _minimal_report()
    assert report.events == []
    assert report.detected_event_names == []


def test_event_summary_requires_event_name_but_event_type_is_optional() -> None:
    """T020'nin cekirdek iddiasi: `event_name` BIRINCIL kimliktir ve HER ZAMAN
    doludur; `event_type` (canonical) OPSIYONELDIR ve `None` GECERLI bir
    degerdir - "eslestirilemedi" anlamina gelir, otomatik doldurulmaz."""
    summary = EventSummary(event_name="yerde_hareketsiz_kisi")

    assert summary.event_name == "yerde_hareketsiz_kisi"
    assert summary.event_type is None
    assert summary.risk_level is None
    assert summary.risk_score is None


def test_events_preserve_free_form_event_name_and_keywords_verbatim() -> None:
    deliberately_non_taxonomy_keywords = ["yerde yatan kişi", "hareketsiz kişi", "olası yaralanma"]
    report = _minimal_report(
        events=[
            EventSummary(
                event_name="yerde_hareketsiz_kisi",
                event_type=None,
                keywords=deliberately_non_taxonomy_keywords,
            )
        ]
    )

    assert report.events[0].event_name == "yerde_hareketsiz_kisi"
    assert report.events[0].event_type is None
    assert report.events[0].keywords == deliberately_non_taxonomy_keywords


def test_events_survive_model_dump_api_payload_simulation() -> None:
    """`model_dump(mode='json')`, gercek API yanitinin/JSON gecmisinin urettigi
    formati simule eder; `event_name`/keywords burada da AYNEN korunmali,
    `event_type=None` JSON'da `null` olarak gorunmeli (bir kategoriye
    ZORLANMAMALI)."""
    report = _minimal_report(
        events=[EventSummary(event_name="yerde_hareketsiz_kisi", keywords=["yerde yatan kişi", "hareketsiz kişi"])]
    )

    payload = report.model_dump(mode="json")

    assert payload["events"] == [
        {
            "event_name": "yerde_hareketsiz_kisi",
            "event_type": None,
            "keywords": ["yerde yatan kişi", "hareketsiz kişi"],
            "risk_level": None,
            "risk_score": None,
            "evidence_ids": [],
            "rule_ids": [],
        }
    ]


def test_more_than_eight_keywords_per_event_are_not_truncated() -> None:
    many_keywords = [f"kanit-{i}" for i in range(12)]
    report = _minimal_report(events=[EventSummary(event_name="yangin_ve_yogun_duman", keywords=many_keywords)])

    assert report.events[0].keywords == many_keywords
    assert len(report.events[0].keywords) == 12


def test_multiple_events_can_carry_different_names_types_and_keyword_sets() -> None:
    report = _minimal_report(
        events=[
            EventSummary(event_name="yangin_duman", event_type="yangin_duman", keywords=["duman", "alev"]),
            EventSummary(event_name="yerde_hareketsiz_kisi", event_type=None, keywords=["yerde yatan kişi"]),
        ]
    )

    by_name = {ev.event_name: ev for ev in report.events}
    assert by_name["yangin_duman"].event_type == "yangin_duman"
    assert by_name["yerde_hareketsiz_kisi"].event_type is None
    assert by_name["yerde_hareketsiz_kisi"].keywords == ["yerde yatan kişi"]


def test_detected_event_names_and_types_are_independent_lists() -> None:
    """`detected_event_names` HER olayi (canonical olsun olmasin) icerir;
    `detected_event_types` yalnizca GERCEKTEN eslesenleri icerir."""
    report = _minimal_report(
        detected_event_names=["yangin_duman", "yerde_hareketsiz_kisi"],
        detected_event_types=["yangin_duman"],
    )

    assert "yerde_hareketsiz_kisi" in report.detected_event_names
    assert "yerde_hareketsiz_kisi" not in report.detected_event_types


# --------------------------- sartname cikti bicimi ---------------------------
#
# Sartnamenin ornek mock JSON'u (Bolum 5) DORT anahtar ister:
#   {"summary": ..., "events": [{"time","event"}], "risk": ..., "actions": [...]}
# `to_sartname_json()` bunlari uretir VE sartnamenin diger maddelerinin
# istedigi kanitlari (risk_accuracy, triggered_mock_actions, zaman damgalari)
# ekler. Asagidaki testler iki seyi kilitler:
#   1. Dort zorunlu alan HER ZAMAN dogru sekilde uretilir.
#   2. Anahtar kumesi, frontend'deki ikiz uygulamayla (useReportExport.ts::
#      buildSartnameJson) AYNI kalir - ikisi sessizce ayrisamaz.


SARTNAME_JSON_KEYS = {
    # sartnamenin ornek mock JSON'undaki DORT zorunlu anahtar
    "summary",
    "events",
    "risk",
    "actions",
    # sartnamenin diger maddelerine kanit olan ek alanlar
    "onset_timestamp",
    "safe_timestamps",
    "incident_timestamps",
    "risk_score",
    "risk_status",
    "risk_accuracy",
    "triggered_mock_actions",
}


def test_sartname_json_has_the_four_mandatory_keys() -> None:
    """Sartnamenin ornek ciktisindaki dort anahtar her zaman uretilir."""
    payload = _minimal_report(summary="ozet", actions=["Saglik ekibini cagir"]).to_sartname_json()

    assert payload["summary"] == "ozet"
    assert payload["risk"] == "dusuk"
    assert payload["actions"] == ["Saglik ekibini cagir"]
    assert isinstance(payload["events"], list)


def test_sartname_json_events_use_mmss_time_and_event_text() -> None:
    """`events` ogeleri sartnamedeki gibi {"time": "MM:SS", "event": ...} seklindedir."""
    from src.schemas.report import TimelineEntry

    report = _minimal_report(timeline=[TimelineEntry(timestamp=75.0, description="Forklift devrildi")])
    events = report.to_sartname_json()["events"]

    assert events == [{"time": "01:15", "event": "Forklift devrildi"}]


def test_sartname_json_falls_back_to_recommended_action() -> None:
    """`actions` bos ise tek oneri sartname `actions` listesine donusur."""
    payload = _minimal_report(actions=[], recommended_action="Alani guvenlik altina al").to_sartname_json()

    assert payload["actions"] == ["Alani guvenlik altina al"]


def test_sartname_json_carries_triggered_mock_actions() -> None:
    """Ajanin GERCEKTEN cagirdigi mock araclar cikti da yer alir.

    Bu alan ONCEDEN yalnizca frontend'in urettigi kopyada vardi; backend'in
    `sartname_json`i onsuz doneyordu. Sartnamede PUANLANAN bir maddenin
    ("mock fonksiyonlarin ajanin araclari olarak basariyla kullanilmasi")
    kaniti oldugu icin bu ayrisma ciddiydi.
    """
    report = _minimal_report(
        triggered_mock_actions=[
            {"tool": "notify_health_team_tool", "args": {"urgency": "high"}, "result": "ok"}
        ]
    )
    payload = report.to_sartname_json()

    assert payload["triggered_mock_actions"] == [
        {"tool": "notify_health_team_tool", "args": {"urgency": "high"}, "result": "ok"}
    ]


def test_sartname_json_key_set_is_locked_against_frontend_drift() -> None:
    """Anahtar kumesi, frontend ikiziyle AYNI kalmalidir.

    Ayni belge iki yerde uretiliyor: burada (backend) ve
    `desktop/app/composables/useReportExport.ts::buildSartnameJson`. Operator
    raporu nereden alirsa alsin AYNI belgeyi gormelidir. Bu test, birinde
    yapilan bir degisikligin digerine tasinmasini ZORUNLU kilar - kume
    degistiginde burasi kirilir ve TS tarafinin da guncellenmesi gerektigini
    hatirlatir.
    """
    assert set(_minimal_report().to_sartname_json()) == SARTNAME_JSON_KEYS
