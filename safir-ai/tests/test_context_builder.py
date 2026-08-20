"""T017 (src/memory/context_builder.py) icin birim testleri.

`ContextBuilder`in artik KENDI RAG sorgusunu yapmadigini, yalnizca cagiranin
verdigi (RuleEngine-dogrulanmis) `relevant_regulations` listesini oldugu gibi
tasidigini ve bos listede acik bir "Mevzuat eslestirilemedi" mesaji
urettigini dogrular. Hicbir dis bagimlilik (FAISS/sentence-transformers)
gerektirmez - bu, T017'nin somut hedeflerinden biridir.
"""

from __future__ import annotations

from typing import List

from src.memory.context_builder import ContextBuilder


class _FakeEventStore:
    def query_recent(self, limit: int = 5) -> List[dict]:
        return []


def test_context_builder_does_not_require_a_rag_service() -> None:
    """Constructor artik yalnizca `event_store` alir (rag_service bagimliligi kaldirildi)."""
    builder = ContextBuilder(_FakeEventStore())
    assert builder is not None


def test_build_passes_through_caller_provided_relevant_regulations_verbatim() -> None:
    builder = ContextBuilder(_FakeEventStore())

    context = builder.build(
        vlm_description="Forklift sahada hareket etti.",
        user_prompt="Risk var mi?",
        timestamp=10.0,
        relevant_regulations=["Operasyonel Kural OK-07: forklift/yaya ayrimi."],
    )

    assert context.relevant_regulations == ["Operasyonel Kural OK-07: forklift/yaya ayrimi."]
    assert "OK-07" in context.to_prompt_block()


def test_build_with_no_regulations_shows_explicit_no_match_message_not_a_fabricated_one() -> None:
    builder = ContextBuilder(_FakeEventStore())

    context = builder.build(
        vlm_description="Bir kisi yerde oturuyor.",
        user_prompt="Risk var mi?",
        timestamp=5.0,
        relevant_regulations=[],
    )

    assert context.relevant_regulations == []
    prompt = context.to_prompt_block()
    assert "Mevzuat eslestirilemedi" in prompt
    assert "risk seviyesini DUSURMEZ" in prompt


def test_build_with_regulations_omitted_defaults_to_no_match() -> None:
    """`relevant_regulations` verilmezse (None), bir mevzuat UYDURULMAZ - bos listeye duser."""
    builder = ContextBuilder(_FakeEventStore())

    context = builder.build(vlm_description="obs", user_prompt="p", timestamp=0.0)

    assert context.relevant_regulations == []
    assert "Mevzuat eslestirilemedi" in context.to_prompt_block()
