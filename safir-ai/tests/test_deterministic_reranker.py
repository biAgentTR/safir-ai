"""`src/rag/deterministic_reranker.py` icin izole (FAISS/embedding'siz) birim testleri.

RAG RERANKER DETERMINIZATION: bu modul HICBIR AG/LLM cagrisi yapmaz - her
sinyal (`lexical_overlap_score`, `keyword_match_score`, `metadata_match_score`,
`phrase_match_score`) ve nihai `score_candidate()` TAMAMEN saf fonksiyonlardir;
bu dosyadaki testler FAISS/embedding gurultusunden ARINDIRILMIS, dogrudan
matematiksel davranisi dogrular.
"""

from __future__ import annotations

import pytest

from src.rag.deterministic_reranker import (
    RelevanceWeights,
    keyword_match_score,
    lexical_overlap_score,
    metadata_match_score,
    phrase_match_score,
    score_candidate,
    turkish_normalize,
)

# ---------------------------------------------------------------------------
# Turkce normalizasyon
# ---------------------------------------------------------------------------


def test_turkish_normalize_handles_dotted_i_correctly() -> None:
    """Python'un yerlesik `str.lower()`u 'İ'yi YANLIS cevirir - bu fonksiyon DOGRU cevirmeli."""
    assert turkish_normalize("İş Ekipmanları") == "iş ekipmanları"
    assert "i̇" not in turkish_normalize("İstanbul")  # bozuk (i + combining dot) OLUSMAMALI


def test_turkish_normalize_handles_dotless_i() -> None:
    assert turkish_normalize("Işık") == "ışık"


# ---------------------------------------------------------------------------
# HEDEF 3: lexical overlap
# ---------------------------------------------------------------------------


def test_lexical_overlap_rewards_shared_content_tokens() -> None:
    query = "forklift yaya güvenliği"
    chunk = "İş ekipmanlarının kullanımında forklift ile yayaların güvenli mesafede tutulması gerekir."

    score = lexical_overlap_score(query, chunk)

    assert score > 0.5  # "forklift"/"yaya" (kok) paylasiliyor


def test_lexical_overlap_is_zero_for_completely_disjoint_vocabulary() -> None:
    assert lexical_overlap_score("forklift yaya güvenliği", "roman tarihi gladyatör dövüşleri") == 0.0


def test_lexical_overlap_ignores_stopwords() -> None:
    """Sadece stopword paylasan iki metin (orn. 've', 'bir', 'ile') YUKSEK skor ALMAMALI."""
    assert lexical_overlap_score("bu ve şu ile", "bu da ve başka bir şey") == 0.0


def test_lexical_overlap_tolerates_simple_suffix_variation() -> None:
    """'forklift' ile 'forkliftler'/'forklifte' gibi basit ek varyasyonlari sufiks-toleransi ile KABACA eslesmeli."""
    assert lexical_overlap_score("forklift riski", "forkliftlerin kullanımı sırasında risk oluşur") > 0.0


def test_lexical_overlap_empty_query_returns_zero() -> None:
    assert lexical_overlap_score("", "herhangi bir metin") == 0.0
    assert lexical_overlap_score("ve ile de", "herhangi bir metin") == 0.0  # yalnizca stopword


# ---------------------------------------------------------------------------
# HEDEF 4: keyword match (VLM keywords)
# ---------------------------------------------------------------------------


def test_keyword_match_rewards_keywords_present_in_chunk() -> None:
    keywords = ["forklift", "yaya", "çarpışma riski", "güvenlik mesafesi"]
    chunk = "Forklift ile yaya arasında güvenlik mesafesi korunmalıdır."

    score = keyword_match_score(keywords, chunk)

    assert score > 0.0


def test_keyword_match_is_zero_when_no_keywords_provided() -> None:
    """HEDEF: keyword YOKSA sistem BOZULMAMALI - bu sinyal SESSIZCE 0 katkı vermeli."""
    assert keyword_match_score(None, "herhangi bir metin") == 0.0
    assert keyword_match_score([], "herhangi bir metin") == 0.0


def test_keyword_match_is_zero_when_none_of_the_keywords_appear() -> None:
    assert keyword_match_score(["kimyasal", "depolama"], "forklift yaya güvenliği hakkında metin") == 0.0


# ---------------------------------------------------------------------------
# HEDEF 5: metadata match (document/article)
# ---------------------------------------------------------------------------


def test_metadata_match_rewards_explicit_article_number_in_query() -> None:
    score_with_article = metadata_match_score(
        "İş Ekipmanları Yönetmeliği I.2.17", document_title="İş Ekipmanları Kullanımında Sağlık ve Güvenlik Şartları Yönetmeliği", article_number="I.2.17"
    )
    score_without_article = metadata_match_score(
        "başka bir sorgu", document_title="İş Ekipmanları Kullanımında Sağlık ve Güvenlik Şartları Yönetmeliği", article_number="I.2.17"
    )

    assert score_with_article > score_without_article
    assert score_with_article >= 0.5  # article_component tam eslesti (0.5 agirlik)


def test_metadata_match_article_number_requires_word_boundary_not_substring() -> None:
    """Sorgudaki '12' chunk'in '123' maddesine YANLIS eslesmemeli (kelime-siniri kontrolu)."""
    score = metadata_match_score("Yönetmelik Madde 123", document_title=None, article_number="12")
    assert score == 0.0


def test_metadata_match_is_zero_without_title_or_article() -> None:
    assert metadata_match_score("herhangi bir sorgu", document_title=None, article_number=None) == 0.0


# ---------------------------------------------------------------------------
# Phrase match
# ---------------------------------------------------------------------------


def test_phrase_match_rewards_exact_contiguous_query_in_chunk() -> None:
    query = "elektrik kilitleme etiketleme"
    chunk = "Enerji kesme islemlerinde elektrik kilitleme etiketleme prosedurune uyulmalidir."

    assert phrase_match_score(query, chunk) == 1.0


def test_phrase_match_does_not_reward_scattered_words_not_as_a_contiguous_phrase() -> None:
    """Sorgu kelimeleri chunk'a TEK TEK serpistirilerek (keyword-stuffing) bu skoru KAZANAMAZ."""
    query = "elektrik kilitleme etiketleme"
    chunk = "Elektrik hakkinda genel bilgiler. Baska bir konuda etiketleme yapilir. Kilitleme ise ayri bir islemdir."

    assert phrase_match_score(query, chunk) == 0.0


def test_phrase_match_ignores_too_short_queries() -> None:
    assert phrase_match_score("ab", "ab iceren herhangi bir metin") == 0.0


# ---------------------------------------------------------------------------
# HEDEF 1/6/13/14: score_candidate - toplam formul, determinizm, embedding != relevance
# ---------------------------------------------------------------------------


def test_score_candidate_is_deterministic_for_identical_input() -> None:
    """HEDEF 1: Ayni query + ayni corpus (chunk) -> AYNI skorlar (tekrar tekrar cagrilsa bile)."""
    args = dict(
        query="forklift yaya güvenliği",
        chunk_text="Forklift ile yayaların güvenli mesafede tutulması gerekir.",
        embedding_score=0.62,
        document_title="İş Ekipmanları Yönetmeliği",
        article_number="I.2.17",
        keywords=["forklift", "yaya"],
    )
    results = [score_candidate(**args) for _ in range(5)]
    assert len(set(r.relevance_score for r in results)) == 1
    assert all(r == results[0] for r in results)


def test_score_candidate_combines_all_five_signals_with_configured_weights() -> None:
    weights = RelevanceWeights(semantic=0.5, lexical=0.2, keyword=0.15, metadata=0.1, phrase=0.05)
    breakdown = score_candidate(
        query="forklift yaya güvenliği",
        chunk_text="Forklift ile yayaların güvenli mesafede tutulması gerekir.",
        embedding_score=0.8,
        document_title="Forklift Yönetmeliği",
        article_number="5",
        keywords=["forklift"],
        weights=weights,
    )
    expected = (
        weights.semantic * breakdown.semantic_score
        + weights.lexical * breakdown.lexical_score
        + weights.keyword * breakdown.keyword_score
        + weights.metadata * breakdown.metadata_score
        + weights.phrase * breakdown.phrase_score
    )
    assert breakdown.relevance_score == round(expected, 4)


def test_embedding_score_and_relevance_score_are_different_things() -> None:
    """HEDEF 14: yuksek embedding_score, TEK BASINA yuksek relevance_score GARANTI ETMEZ (diger sinyaller 0 olabilir)."""
    breakdown = score_candidate(
        query="forklift yaya güvenliği",
        chunk_text="Bambaşka, tamamen alakasız bir konu hakkında metin.",
        embedding_score=0.95,  # yuksek embedding benzerligi VARSAYALIM
    )
    # semantic bileşen yuksek olsa bile diger sinyaller (lexical/keyword/metadata/phrase) 0 -
    # relevance_score, saf embedding_score'dan (0.95) DUSUK olmali (agirlikli toplam yuzunden).
    assert breakdown.semantic_score == pytest.approx(0.95)
    assert breakdown.relevance_score < breakdown.semantic_score


def test_negative_embedding_similarity_is_clamped_to_zero_semantic_score() -> None:
    breakdown = score_candidate(query="sorgu", chunk_text="metin", embedding_score=-0.4)
    assert breakdown.semantic_score == 0.0


def test_relevance_score_field_name_is_not_confidence() -> None:
    """HEDEF 13: `RelevanceBreakdown`in hicbir alani 'confidence' olarak ADLANDIRILMAMIS."""
    import dataclasses

    from src.rag.deterministic_reranker import RelevanceBreakdown

    field_names = {f.name for f in dataclasses.fields(RelevanceBreakdown)}
    assert "confidence" not in field_names
    assert "relevance_score" in field_names


def test_reason_string_is_human_readable_and_explains_all_signals() -> None:
    breakdown = score_candidate(query="forklift yaya güvenliği", chunk_text="forklift yaya güvenlik mesafesi", embedding_score=0.7)
    reason = breakdown.reason()
    assert "semantic=" in reason
    assert "lexical=" in reason
    assert "keyword=" in reason
    assert "metadata=" in reason
    assert "phrase=" in reason
