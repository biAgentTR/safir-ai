"""Adim 4 - src/memory/document_extraction.py birim testleri.

Ikili veri (raw bytes) -> duz metin -> chunk -> lexical retrieval zincirinin
her adimini, gercek PDF/DOCX baytlariyla (reportlab/python-docx ile uretilir,
sahte/mock DEGIL) dogrular.
"""

from __future__ import annotations

import io

import pytest

from src.memory import document_extraction as de


def _make_pdf_bytes(pages: list[str]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        c.drawString(50, 800, text)
        c.showPage()
    c.save()
    return buf.getvalue()


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    import docx

    buf = io.BytesIO()
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    d.save(buf)
    return buf.getvalue()


# --------------------------- extract_text ---------------------------


def test_extract_txt_utf8():
    pages, page_count = de.extract_text("notlar.txt", "Merhaba dunya, ISG onemlidir.".encode("utf-8"))
    assert page_count is None
    assert len(pages) == 1
    assert pages[0][0] is None
    assert "Merhaba dunya" in pages[0][1]


def test_extract_txt_latin1_fallback_does_not_raise():
    raw = "café".encode("latin-1")
    pages, _ = de.extract_text("x.txt", raw)
    assert pages  # cokmedi, bir sekilde metin uretti


def test_extract_txt_empty_raises():
    with pytest.raises(de.ExtractionError):
        de.extract_text("bos.txt", b"   \n  ")


def test_extract_unsupported_extension_raises():
    with pytest.raises(de.ExtractionError):
        de.extract_text("virus.exe", b"MZ\x90\x00")


def test_extract_pdf_real_bytes_returns_page_per_page():
    raw = _make_pdf_bytes(["Sayfa bir icerigi forklift guvenligi.", "Sayfa iki icerigi yangin sondurme CO2."])
    pages, page_count = de.extract_text("yonetmelik.pdf", raw)
    assert page_count == 2
    assert len(pages) == 2
    assert pages[0][0] == 1
    assert "forklift" in pages[0][1].lower()
    assert pages[1][0] == 2
    assert "yangin" in pages[1][1].lower()


def test_extract_pdf_corrupt_raises():
    with pytest.raises(de.ExtractionError):
        de.extract_text("bozuk.pdf", b"%PDF-1.4 bu gecerli bir PDF degil")


def test_extract_docx_real_bytes_returns_joined_paragraphs():
    raw = _make_docx_bytes(["Ilk paragraf: dar alan calismasi.", "Ikinci paragraf: enerji kesme prosedürü."])
    pages, page_count = de.extract_text("prosedur.docx", raw)
    assert page_count is None
    assert len(pages) == 1
    assert pages[0][0] is None  # DOCX'te sayfa kavrami yok
    assert "dar alan" in pages[0][1].lower()
    assert "enerji kesme" in pages[0][1].lower()


def test_extract_docx_corrupt_raises():
    with pytest.raises(de.ExtractionError):
        de.extract_text("bozuk.docx", b"bu gecerli bir docx degil")


# --------------------------- chunk_page_texts ---------------------------


def test_chunk_page_texts_short_page_single_chunk():
    chunks = de.chunk_page_texts([(1, "kisa metin")])
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].content == "kisa metin"


def test_chunk_page_texts_splits_long_page_with_overlap():
    long_text = "a" * (de.CHUNK_SIZE_CHARS * 2 + 100)
    chunks = de.chunk_page_texts([(3, long_text)])
    assert len(chunks) > 1
    assert all(c.page_number == 3 for c in chunks)
    assert all(len(c.content) <= de.CHUNK_SIZE_CHARS for c in chunks)


def test_chunk_page_texts_skips_empty_pages():
    chunks = de.chunk_page_texts([(1, "   "), (2, "gercek icerik")])
    assert len(chunks) == 1
    assert chunks[0].page_number == 2


# --------------------------- lexical_search ---------------------------


class _C:
    def __init__(self, content: str) -> None:
        self.content = content


def test_lexical_search_ranks_by_word_overlap():
    chunks = [
        _C("forklift ve yaya guvenligi hakkinda bilgi"),
        _C("yangin sondurme CO2 sistemi hakkinda bilgi"),
        _C("tamamen alakasiz bir cumle"),
    ]
    result = de.lexical_search(chunks, "forklift yaya carpisma riski nedir?", top_k=5)
    assert len(result) >= 1
    assert result[0].content == chunks[0].content


def test_lexical_search_zero_score_chunks_excluded():
    chunks = [_C("elma armut muz"), _C("forklift yaya guvenligi")]
    result = de.lexical_search(chunks, "forklift yaya nedir", top_k=5)
    contents = [c.content for c in result]
    assert "elma armut muz" not in contents


def test_lexical_search_respects_top_k():
    chunks = [_C(f"forklift guvenligi parca {i}") for i in range(10)]
    result = de.lexical_search(chunks, "forklift guvenligi", top_k=3)
    assert len(result) == 3


def test_lexical_search_empty_query_returns_empty():
    chunks = [_C("herhangi bir icerik")]
    assert de.lexical_search(chunks, "   ", top_k=5) == []


def test_lexical_search_no_match_returns_empty():
    chunks = [_C("elma armut muz")]
    assert de.lexical_search(chunks, "forklift yaya", top_k=5) == []
