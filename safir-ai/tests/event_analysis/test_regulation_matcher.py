"""T017 (src/event_analysis/regulation_matcher.py) icin birim testleri.

`resolve_regulation_matches`in, "mevzuat GETIRILDI" ile "mevzuat UYGULANIR"
ayrimini dogru kurdugunu; RAG benzerlik skorunun DEGIL, yalnizca RuleEngine'in
deterministik `RuleMatch` ciktisinin "matched" uretebildigini dogrular.
"""

from __future__ import annotations

from src.event_analysis.regulation_matcher import NO_MATCH_REASON, resolve_regulation_matches
from src.event_analysis.schemas import RuleMatch


def _rule_match(
    rule_id: str,
    event_type: str,
    rule_description: str = "Mevzuat aciklamasi",
    severity: str = "yuksek",
    source_event_id: str = "evt_0",
) -> RuleMatch:
    return RuleMatch(
        rule_id=rule_id,
        rule_description=rule_description,
        event_type=event_type,
        severity=severity,
        source_event_id=source_event_id,
    )


# --- 1) Acikca ilgili olay -> MATCHED --------------------------------------


def test_genuine_rule_match_is_reported_as_matched() -> None:
    matches = resolve_regulation_matches(
        [_rule_match("OK-07", "arac_yaya_yakinligi", "Operasyonel Kural OK-07: forklift/yaya ayrimi.")]
    )

    assert len(matches) == 1
    assert matches[0].match_status == "matched"
    assert matches[0].regulation_id == "OK-07"
    assert matches[0].regulation_title == "Operasyonel Kural OK-07: forklift/yaya ayrimi."
    assert matches[0].relevance_score == 1.0
    assert matches[0].reason


# --- 2/3) Eslesme yok -> NO_MATCH (bos liste), risk UYDURULMAZ -------------


def test_no_rule_matches_returns_empty_list_not_a_fabricated_match() -> None:
    assert resolve_regulation_matches([]) == []


def test_no_match_reason_constant_is_non_empty_and_explanatory() -> None:
    assert NO_MATCH_REASON
    assert "eslestirmesi bulunamadi" in NO_MATCH_REASON.lower()


# --- 5) Ortak kelime ama farkli mevzuat anlami -> NO_MATCH -----------------


def test_forklift_event_never_produces_a_ppe_regulation_match() -> None:
    """'Forklift sahada hareket etti' olayi (arac_yaya_yakinligi), KKD (kkd_ihlali)
    mevzuatiyla ASLA eslesmemeli - ayni kelime kumesini (saha, ekipman, guvenlik)
    paylassalar bile, event_type'lar farkli oldugu icin RuleEngine hicbir zaman
    bu ikisini karistirmaz (bkz. modul dokustringindeki somut ornek)."""
    # Sadece arac_yaya_yakinligi RuleMatch'i var; KKD ile ilgili HICBIR RuleMatch yok.
    matches = resolve_regulation_matches(
        [_rule_match("OK-07", "arac_yaya_yakinligi", "Operasyonel Kural OK-07: forklift/yaya ayrimi.")]
    )

    regulation_ids = {m.regulation_id for m in matches}
    assert regulation_ids == {"OK-07"}
    assert not any("KKD" in (m.regulation_title or "") for m in matches)


# --- 6) Birden fazla aday, yalnizca ilgili olanlar secilir -----------------


def test_only_genuinely_matched_rules_are_selected_among_multiple() -> None:
    matches = resolve_regulation_matches(
        [
            _rule_match("ISG-M24", "kkd_ihlali", "ISG Yonetmeligi Madde 24."),
            _rule_match("OK-07", "arac_yaya_yakinligi", "Operasyonel Kural OK-07."),
        ]
    )

    assert {m.regulation_id for m in matches} == {"ISG-M24", "OK-07"}
    assert all(m.match_status == "matched" for m in matches)


def test_combination_rule_match_is_also_reported_as_matched() -> None:
    combo = _rule_match(
        "COMBO-01",
        "kkd_ihlali+arac_yaya_yakinligi",
        "Ayni zaman penceresinde hem KKD hem arac-yaya yakinligi.",
        severity="kritik",
    )

    matches = resolve_regulation_matches([combo])

    assert len(matches) == 1
    assert matches[0].regulation_id == "COMBO-01"
    assert matches[0].match_status == "matched"


# --- 11) Bicimsizlik/kismi veri sistemi cokertmez ---------------------------


def test_does_not_crash_on_large_or_duplicate_rule_match_lists() -> None:
    matches = resolve_regulation_matches([_rule_match(f"R{i}", "kkd_ihlali") for i in range(50)])
    assert len(matches) == 50
    assert all(m.match_status == "matched" for m in matches)
