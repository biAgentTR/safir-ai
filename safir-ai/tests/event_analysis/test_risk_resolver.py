"""T016 (src/event_analysis/risk_resolver.py) icin birim testleri.

`resolve_deterministic_risk`in, `RuleMatch` listesinden VLM/LLM'e bagimsiz,
deterministik bir risk seviyesi/skoru turetme davranisini dogrular.
"""

from __future__ import annotations

from src.event_analysis.risk_resolver import resolve_deterministic_risk
from src.event_analysis.schemas import RuleMatch


def _match(rule_id: str, severity: str, source_event_id: str = "evt_0") -> RuleMatch:
    return RuleMatch(
        rule_id=rule_id,
        rule_description=f"{rule_id} aciklamasi",
        event_type="kkd_ihlali",
        severity=severity,
        source_event_id=source_event_id,
    )


def test_no_matches_returns_none_none_risk_is_not_fabricated() -> None:
    assert resolve_deterministic_risk([]) == (None, None)


def test_single_match_returns_its_own_severity_and_score() -> None:
    risk_level, risk_score = resolve_deterministic_risk([_match("ISG-M24", "orta")])

    assert risk_level == "orta"
    assert risk_score == 38


def test_multiple_matches_take_the_highest_severity() -> None:
    matches = [_match("ISG-M24", "orta"), _match("COMBO-01", "kritik"), _match("ISG-M12", "yuksek")]

    risk_level, risk_score = resolve_deterministic_risk(matches)

    assert risk_level == "kritik"
    assert risk_score == 88


def test_severity_order_is_monotonic_across_all_levels() -> None:
    for severity, expected_score in [("dusuk", 12), ("orta", 38), ("yuksek", 63), ("kritik", 88)]:
        assert resolve_deterministic_risk([_match("R", severity)]) == (severity, expected_score)


def test_unknown_severity_values_are_ignored_and_do_not_crash() -> None:
    matches = [
        RuleMatch(
            rule_id="WEIRD",
            rule_description="bilinmeyen siddet",
            event_type="kkd_ihlali",
            severity="belirsiz",
            source_event_id="evt_0",
        )
    ]

    assert resolve_deterministic_risk(matches) == (None, None)


def test_known_and_unknown_severity_mixed_uses_only_known_ones() -> None:
    matches = [
        RuleMatch(
            rule_id="WEIRD",
            rule_description="bilinmeyen siddet",
            event_type="kkd_ihlali",
            severity="belirsiz",
            source_event_id="evt_0",
        ),
        _match("ISG-M24", "orta"),
    ]

    assert resolve_deterministic_risk(matches) == ("orta", 38)
