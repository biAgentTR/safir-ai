"""Adim 4 - SAFIR Asistan: kullanici tarafindan yuklenen belgelerden (PDF/TXT/DOCX)
metin cikarma + chunk'lama + basit lexical (anahtar-kelime) retrieval.

Bilerek KUCUK tutuldu: vector database/embedding servisi YOK. Chunk'lar
SQLite'ta (`ConversationStore`) duz metin olarak saklanir; `lexical_search`
kelime-orutusme sayimina dayali basit, deterministik bir siralama yapar.
Ileride embedding-tabanli retrieval eklenmek istenirse, `lexical_search`in
imzasi (`chunks + query -> siralanmis chunk listesi`) DEGISMEDEN ic
implementasyonu degistirilebilir - cagiran kod (`ask_service.py`) bu
fonksiyonun NASIL siraladigini bilmez.

Ikili (binary) veri hicbir zaman LLM'e gonderilmez: bu modul yalnizca DUZ
METIN uretir; `src/assistant/ask_service.py` yalnizca bu metni kullanir.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

# --- Merkezi limitler (Adim 4 istegi: "kod icinde sabit olarak merkezi tanimla") ---

ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}
"""Ilk versiyonda desteklenen dosya uzantilari (kucuk harf, nokta yok)."""

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
"""Tek bir dosya icin azami boyut (10 MB) - buyuk PDF'ler bile bu sinira genelde sigar."""

MAX_FILES_PER_CONVERSATION = 5
"""Bir sohbete eklenebilecek azami belge sayisi."""

MAX_EXTRACTED_TEXT_CHARS = 200_000
"""Bir belgeden cikarilan metnin azami karakter sayisi (bellek/DB/chunk patlamasini onler)."""

CHUNK_SIZE_CHARS = 1500
"""Bir chunk'in azami karakter uzunlugu (basit karakter-tabanli, tokenizer bagimliligi yok)."""

CHUNK_OVERLAP_CHARS = 150
"""Ardisik chunk'lar arasi ortusme (cumle/baglam sinirinin ortasinda kesilmeyi azaltir)."""

MAX_RETRIEVED_CHUNKS = 4
"""Bir soru icin prompt'a eklenecek azami chunk sayisi (belgenin TAMAMI ASLA eklenmez)."""


class ExtractionError(Exception):
    """Dosyadan metin cikarilamadiginda (bozuk/sifreli/desteklenmeyen icerik) firlatilir."""


@dataclass
class ChunkInput:
    """Kalici olarak saklanmadan once, tek bir chunk'in ham verisi."""

    page_number: Optional[int]
    content: str


def extract_text(filename: str, raw: bytes) -> Tuple[List[Tuple[Optional[int], str]], Optional[int]]:
    """Dosya uzantisina gore metni sayfa/parca bazinda cikarir.

    Args:
        filename: Orijinal dosya adi (yalnizca uzantiyi belirlemek icin kullanilir).
        raw: Dosyanin ham baytlari.

    Returns:
        `(page_texts, page_count)` - `page_texts`, `(sayfa_no_veya_None, metin)`
        ikililerinin listesi (PDF: sayfa basina bir eleman; DOCX/TXT: sayfa
        kavrami yok, tek eleman, `sayfa_no=None`). `page_count`, yalnizca
        gercekten sayfa bazli bicimlerde (PDF) doludur.

    Raises:
        ExtractionError: Uzanti desteklenmiyorsa veya cikarim basarisiz olursa.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ExtractionError(f"Desteklenmeyen dosya turu: .{ext}")

    if ext == "txt":
        return _extract_txt(raw)
    if ext == "pdf":
        return _extract_pdf(raw)
    if ext == "docx":
        return _extract_docx(raw)
    raise ExtractionError(f"Desteklenmeyen dosya turu: .{ext}")  # pragma: no cover - ALLOWED_EXTENSIONS ile tutarli


def _extract_txt(raw: bytes) -> Tuple[List[Tuple[Optional[int], str]], Optional[int]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")
    text = text[:MAX_EXTRACTED_TEXT_CHARS]
    if not text.strip():
        raise ExtractionError("TXT dosyasi bos veya metin icermiyor.")
    return [(None, text)], None


def _extract_pdf(raw: bytes) -> Tuple[List[Tuple[Optional[int], str]], Optional[int]]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - requirements.txt'te deklare edilir
        raise ExtractionError("PDF islenemiyor (pypdf kurulu degil).") from exc

    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise ExtractionError("Sifreli PDF desteklenmiyor.")
        page_count = len(reader.pages)
        pages: List[Tuple[Optional[int], str]] = []
        total_chars = 0
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            remaining = MAX_EXTRACTED_TEXT_CHARS - total_chars
            if remaining <= 0:
                break
            text = text[:remaining]
            total_chars += len(text)
            pages.append((i, text))
    except PdfReadError as exc:
        raise ExtractionError(f"PDF okunamadi (bozuk olabilir): {exc}") from exc

    if not pages:
        raise ExtractionError("PDF'den metin cikarilamadi (taranmis/gorsel-tabanli olabilir).")
    return pages, page_count


def _extract_docx(raw: bytes) -> Tuple[List[Tuple[Optional[int], str]], Optional[int]]:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - requirements.txt'te deklare edilir
        raise ExtractionError("DOCX islenemiyor (python-docx kurulu degil).") from exc

    try:
        document = docx.Document(io.BytesIO(raw))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    except Exception as exc:  # noqa: BLE001 - python-docx bozuk dosyada cesitli hatalar firlatabilir
        raise ExtractionError(f"DOCX okunamadi (bozuk olabilir): {exc}") from exc

    text = "\n".join(paragraphs)[:MAX_EXTRACTED_TEXT_CHARS]
    if not text.strip():
        raise ExtractionError("DOCX'ten metin cikarilamadi (bos olabilir).")
    # DOCX'te guvenilir bir "sayfa" kavrami yok (goruntuleyiciye gore degisir);
    # sayfa numarasi verilmez (bkz. modul dokustringi).
    return [(None, text)], None


def chunk_page_texts(page_texts: Sequence[Tuple[Optional[int], str]]) -> List[ChunkInput]:
    """Sayfa/parca metinlerini sabit boyutlu, hafifce ortusen chunk'lara boler.

    Her chunk, hangi sayfadan geldigini korur (`page_number`) - boylece
    kaynak gosterimi ("dosya.pdf - sayfa 12") gercek veriye dayanir.

    Args:
        page_texts: `extract_text(...)` ciktisi.

    Returns:
        Bos olmayan `ChunkInput` listesi (bos/whitespace-only chunk'lar atlanir).
    """
    chunks: List[ChunkInput] = []
    step = max(1, CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS)
    for page_number, text in page_texts:
        text = text.strip()
        if not text:
            continue
        if len(text) <= CHUNK_SIZE_CHARS:
            chunks.append(ChunkInput(page_number=page_number, content=text))
            continue
        start = 0
        while start < len(text):
            piece = text[start : start + CHUNK_SIZE_CHARS].strip()
            if piece:
                chunks.append(ChunkInput(page_number=page_number, content=piece))
            start += step
    return chunks


_WORD_PATTERN = re.compile(r"[a-zA-ZçÇğĞıİöÖşŞüÜ0-9]+")


def _tokenize(text: str) -> set[str]:
    """Basit kelime tokenlestirme (Turkce harfler dahil, noktalama haric)."""
    return {w.lower() for w in _WORD_PATTERN.findall(text) if len(w) > 2}


class _HasContent(Protocol):
    content: str


def lexical_search(chunks: Sequence[_HasContent], query: str, top_k: int = MAX_RETRIEVED_CHUNKS) -> List[_HasContent]:
    """Sorguyla en cok kelime-ortusmesine sahip chunk'lari, azalan skorla dondurur.

    Basit ama deterministik bir yaklasimdir (vector/embedding YOK): sorgu ve
    her chunk kucuk-harfli kelime kumelerine ayrilir, kesisim buyuklugu
    "skor" olarak kullanilir. Skoru 0 olan (hic ortak kelimesi olmayan)
    chunk'lar SONUCA HIC DAHIL EDILMEZ - boylece ilgisiz belgeler/parcalar
    gereksiz yere prompt'a girmez.

    Args:
        chunks: Aday chunk'lar (herhangi bir `.content: str` tasiyan nesne).
        query: Kullanicinin sorusu.
        top_k: Dondurulecek azami chunk sayisi.

    Returns:
        Skora gore azalan sirali (esitlikte orijinal sira korunur), en fazla
        `top_k` elemanli liste. Sorguda anlamli kelime yoksa veya hic eslesme
        yoksa bos liste doner.
    """
    query_words = _tokenize(query)
    if not query_words:
        return []

    scored: List[Tuple[int, int, _HasContent]] = []
    for idx, chunk in enumerate(chunks):
        chunk_words = _tokenize(chunk.content)
        score = len(query_words & chunk_words)
        if score > 0:
            scored.append((score, idx, chunk))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored[:top_k]]
