"""T019 (src/schemas/report.py) icin `EventKeywords`/`SafirReport.event_keywords` birim testleri.

`SafirReport.event_keywords`in, VLM-uretimi serbest-bicimli kanit ifadelerini
(taksonomiyle SINIRLI OLMAYAN) hicbir donusum/filtreleme olmadan (API payload
simulasyonu olan `model_dump()` dahil) tasidigini dogrular.
"""

from __future__ import annotations

from src.schemas.report import EventKeywords, SafirReport


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


def test_event_keywords_defaults_to_empty_list() -> None:
    report = _minimal_report()
    assert report.event_keywords == []


def test_event_keywords_preserves_arbitrary_non_taxonomy_strings_verbatim() -> None:
    deliberately_non_taxonomy_keywords = [
        "yoğun siyah duman",
        "alevlenme",
        "dumanın tavana doğru yükselmesi",
        "yanma belirtisi",
    ]
    report = _minimal_report(
        event_keywords=[EventKeywords(event_type="yangin_duman", keywords=deliberately_non_taxonomy_keywords)]
    )

    assert report.event_keywords[0].event_type == "yangin_duman"
    assert report.event_keywords[0].keywords == deliberately_non_taxonomy_keywords


def test_event_keywords_survive_model_dump_api_payload_simulation() -> None:
    """`model_dump(mode='json')`, gercek API yanitinin/JSON gecmisinin urettigi
    formati simule eder; string'ler burada da AYNEN korunmali."""
    deliberately_non_taxonomy_keywords = ["dumanın tavana doğru yükselmesi", "yanma belirtisi"]
    report = _minimal_report(
        event_keywords=[EventKeywords(event_type="yangin_duman", keywords=deliberately_non_taxonomy_keywords)]
    )

    payload = report.model_dump(mode="json")

    assert payload["event_keywords"] == [
        {"event_type": "yangin_duman", "keywords": deliberately_non_taxonomy_keywords}
    ]


def test_more_than_eight_keywords_per_event_are_not_truncated() -> None:
    many_keywords = [f"kanit-{i}" for i in range(12)]
    report = _minimal_report(event_keywords=[EventKeywords(event_type="yangin_duman", keywords=many_keywords)])

    assert report.event_keywords[0].keywords == many_keywords
    assert len(report.event_keywords[0].keywords) == 12


def test_multiple_events_can_carry_different_keyword_sets() -> None:
    report = _minimal_report(
        event_keywords=[
            EventKeywords(event_type="yangin_duman", keywords=["duman", "alev"]),
            EventKeywords(event_type="kkd_ihlali", keywords=["baret yok", "yelek yok"]),
        ]
    )

    by_type = {ek.event_type: ek.keywords for ek in report.event_keywords}
    assert by_type["yangin_duman"] == ["duman", "alev"]
    assert by_type["kkd_ihlali"] == ["baret yok", "yelek yok"]
