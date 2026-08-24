"""`src/main.py::_unverified_citations` icin birim testleri.

RAG PIPELINE RECONSTRUCTION (gorev tanimi 10. bolum): Ajanin serbest metninde
(summary/actions) gecen mevzuat-benzeri atiflardan, BU CAGRIDA GERCEKTEN
retrieved olan `semantic_rag_sources`le eslesmeyenlerin tespiti - deterministik,
regex-tabanli (LLM'e SORULMAZ).
"""

from __future__ import annotations

from src.main import _unverified_citations
from src.schemas.report import RagContext


def _evidence(rule_title: str, article_number: str = "") -> RagContext:
    return RagContext(rule_title=rule_title, content="ornek metin", score=0.8, article_number=article_number)


def test_citation_matching_retrieved_evidence_is_not_flagged() -> None:
    """HEDEF 9: Agent, GERCEKTEN retrieved bir kaynagi dogru madde ile aniyorsa DOGRULANIR (isaretlenmez)."""
    evidence = [_evidence("İş Ekipmanları Kullanımında Sağlık ve Güvenlik Şartları Yönetmeliği", "I.3.1")]
    text = "Bu durum İş Ekipmanları Kullanımında Sağlık ve Güvenlik Şartları Yönetmeliği Madde I.3.1 kapsamındadır."

    assert _unverified_citations(text, evidence) == []


def test_fabricated_instruction_name_not_in_any_evidence_is_flagged() -> None:
    """HEDEF 10, gorev tanimindaki somut ornek: corpus'ta olmayan bir talimat adi ('Yangın Güvenliği Talimatı YG-03') UNVERIFIED isaretlenir."""
    evidence = [_evidence("İş Ekipmanları Kullanımında Sağlık ve Güvenlik Şartları Yönetmeliği", "I.3.1")]
    text = "Acil durumda Yangın Güvenliği Talimatı YG-03 uygulanmalıdır."

    unverified = _unverified_citations(text, evidence)
    assert unverified
    assert any("YG-03" in u for u in unverified)


def test_correct_document_but_wrong_article_number_is_flagged() -> None:
    """Dokuman adi retrieved evidence'la eslesse bile, UYDURULMUS bir madde numarasi DOGRULANMAZ."""
    evidence = [_evidence("İş Ekipmanları Kullanımında Sağlık ve Güvenlik Şartları Yönetmeliği", "I.3.1")]
    text = "İş Ekipmanları Kullanımında Sağlık ve Güvenlik Şartları Yönetmeliği Madde 99 uyarınca."

    unverified = _unverified_citations(text, evidence)
    assert unverified
    assert any("Madde 99" in u for u in unverified)


def test_no_evidence_at_all_flags_any_legislation_like_mention() -> None:
    """HEDEF 5 (corpus disi mevzuat): hic retrieved evidence yoksa, gecen HERHANGI bir mevzuat-benzeri ifade UNVERIFIED'dir."""
    text = "6331 Sayılı İş Sağlığı ve Güvenliği Kanunu Madde 4 uyarınca."

    unverified = _unverified_citations(text, semantic_rag_sources=[])
    assert unverified


def test_free_text_without_any_legislation_like_pattern_produces_no_flags() -> None:
    """Mevzuat-benzeri hicbir kalip gecmiyorsa (sade operasyonel ozet), BOS liste doner - yanlis-pozitif UYDURULMAZ."""
    text = "Sahada forklift yaya yakınında çalışıyor; personel derhal uzaklaştırılmalı."

    assert _unverified_citations(text, semantic_rag_sources=[]) == []


def test_empty_text_returns_empty_list() -> None:
    assert _unverified_citations("", semantic_rag_sources=[]) == []
