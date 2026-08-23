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
