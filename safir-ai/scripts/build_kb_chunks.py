"""Knowledge Base hazirlik scripti: `data/knowledge_base/official/*.pdf` icindeki
resmi ISG mevzuatini madde-bazli (ve gerekli dokumanlarda EK alt-madde-bazli)
chunk'lara ayirip `data/knowledge_base/chunks/<id>.json` altina yazar.

ONEMLI (kapsam): Bu script YALNIZCA hazirlik/chunking adimidir. Embedding
UeRETMEZ, FAISS'e YAZMAZ, `src/rag/embedding_rag_service.py`'yi cagirmaz/
degistirmez - EmbeddingRAGService/DEFAULT_ISG_REGULATIONS hala oldugu gibi
duruyor. Cikti, ileride bir ingestion adiminin girdisi olacak sekilde
tasarlanmistir.

Chunk stratejisi (bkz. 2026-08-22 "DOCUMENT INGESTION AUDIT" bulgulari):
  - Coğu dokuman: gövde metni "MADDE N" / "GEÇİCİ MADDE N" isaretlerine gore
    bolunur (regex tabanli, deterministik).
  - `is_ekipmanlari_yonetmeligi` ve `patlayici_ortamlar_yonetmeligi`: gövdeden
    SONRA gelen EK bolumu, "N.M" formatindaki alt-madde isaretlerine (orn.
    "2.13.") gore AYRICA bolunur - bunlar MADDE DEGILDIR, farkli bir
    numaralandirma semasidir.
  - `yapi_isleri_isg_yonetmeligi`: EK-1 "YAPI İŞLERİ LİSTESİ" numarali bir
    liste (madde degil); "N– " formatindaki liste ogelerine gore bolunur.
  - Sayfa araligi (page_start/page_end), her chunk'in metnin hangi PDF
    sayfa(lar)inda gectigini karsilik gelen karakter ofsetinden turetilir
    (pypdf sayfa sirasi guvenilir referans kabul edilir - bkz. audit).

Kullanim:
    python scripts/build_kb_chunks.py
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    from PyPDF2 import PdfReader  # type: ignore

_ROOT_DIR = Path(__file__).resolve().parents[1]
_KB_DIR = _ROOT_DIR / "data" / "knowledge_base"
_OFFICIAL_DIR = _KB_DIR / "official"
_CHUNKS_DIR = _KB_DIR / "chunks"

_MADDE_RE = re.compile(r"(?:^|\n)\s*(GEÇİCİ MADDE|MADDE)\s*(\d+)\s*[-–—]", re.IGNORECASE)

# Gercek EK (annex) BASLIKLARINI, metin ICI referanslardan ("EK-I'inde",
# "Ek-4'e," gibi) ayirmak icin: bir "EK ... <sayi/rakam>" eslesmesi, hemen
# ARDINDAN bir kesme isareti (' veya tipografik ’) GELMIYORSA gercek baslik
# sayilir - govde icindeki atiflar HER ZAMAN boyle bir ek ile devam eder
# (orn. "EK-I’inde", "Ek-4'e"), gercek basliklar ETMEZ (bkz. build script
# audit notlari: "EK - I \n...", "EK – 1\n \n...").
_EK_CANDIDATE_RE = re.compile(r"(?:^|\n)\s*EK\s*[-–]?\s*([IVX]+|\d+)\b")
_EK_SUBCLAUSE_RE = re.compile(r"(?:^|\n)\s*(\d+\.\d+)\.\s")
_EK_LIST_ITEM_RE = re.compile(r"(?:^|\n)\s*(\d+)[–-]\s")


def _find_real_ek_headers(text: str) -> List[Tuple[str, int]]:
    """`_EK_CANDIDATE_RE` eslesmelerinden, hemen ardindan kesme isareti
    GELMEYENLERİ (yani gercek EK basliklarini) dondurur; (numara, offset) listesi.
    """
    headers: List[Tuple[str, int]] = []
    for m in _EK_CANDIDATE_RE.finditer(text):
        next_char = text[m.end() : m.end() + 1]
        if next_char in ("'", "’"):
            continue  # govde-ici atif (orn. "EK-I’inde"), gercek baslik degil
        headers.append((m.group(1), m.start()))
    return headers

# Bu dokumanlarda gercek guvenlik icerigi govdede degil EK bolumunde,
# N.M alt-madde formatinda (bkz. audit bulgusu: is_ekipmanlari EK-I 2.13,
# patlayici_ortamlar EK 1.2).
_EK_SUBCLAUSE_DOCS = {"is_ekipmanlari_yonetmeligi", "patlayici_ortamlar_yonetmeligi"}
# Bu dokumanda EK-1 numarali bir LISTE (madde degil).
_EK_LIST_DOCS = {"yapi_isleri_isg_yonetmeligi"}


@dataclass
class PageMap:
    """Tam metindeki bir karakter ofsetini PDF sayfa numarasina (1-based) cevirir."""

    boundaries: List[int]  # boundaries[i] = i. sayfanin (0-based) metninin basladigi ofset

    def page_for_offset(self, offset: int) -> int:
        page_idx = 0
        for i, start in enumerate(self.boundaries):
            if offset >= start:
                page_idx = i
            else:
                break
        return page_idx + 1  # 1-based


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    level: str  # "madde" | "gecici_madde" | "ek_alt_madde" | "ek_liste"
    number: str
    is_annex: bool
    page_start: int
    page_end: int
    char_count: int
    text: str


def _load_pdf(path: Path) -> Tuple[str, PageMap]:
    reader = PdfReader(str(path))
    boundaries: List[int] = []
    parts: List[str] = []
    offset = 0
    for page in reader.pages:
        boundaries.append(offset)
        page_text = page.extract_text() or ""
        parts.append(page_text)
        offset += len(page_text) + 1  # "\n" join separator
    full_text = "\n".join(parts)
    return full_text, PageMap(boundaries=boundaries)


def _split_body_by_madde(text: str, end: Optional[int] = None) -> List[Tuple[str, str, int, int]]:
    """Metni (end'e kadar) MADDE/GEÇİCİ MADDE isaretlerine gore boler.

    Returns:
        (level, number, start_offset, end_offset) dörtlülerinin listesi;
        `level` "madde" veya "gecici_madde".
    """
    limit = end if end is not None else len(text)
    matches = [m for m in _MADDE_RE.finditer(text) if m.start() < limit]
    results: List[Tuple[str, str, int, int]] = []
    for i, m in enumerate(matches):
        start = m.start()
        stop = matches[i + 1].start() if i + 1 < len(matches) else limit
        level = "gecici_madde" if m.group(1).upper().startswith("GEÇİCİ") else "madde"
        number = m.group(2)
        results.append((level, number, start, stop))
    return results


def _split_ek_section_by_subclause(text: str, ek_label: str, start: int, end: int) -> List[Tuple[str, str, int, int]]:
    matches = [m for m in _EK_SUBCLAUSE_RE.finditer(text) if start <= m.start() < end]
    results: List[Tuple[str, str, int, int]] = []
    for i, m in enumerate(matches):
        seg_start = m.start()
        seg_end = matches[i + 1].start() if i + 1 < len(matches) else end
        results.append(("ek_alt_madde", f"{ek_label}.{m.group(1)}", seg_start, seg_end))
    return results


def _split_ek_section_by_list_item(text: str, ek_label: str, start: int, end: int) -> List[Tuple[str, str, int, int]]:
    matches = [m for m in _EK_LIST_ITEM_RE.finditer(text) if start <= m.start() < end]
    results: List[Tuple[str, str, int, int]] = []
    for i, m in enumerate(matches):
        seg_start = m.start()
        seg_end = matches[i + 1].start() if i + 1 < len(matches) else end
        results.append(("ek_liste", f"{ek_label}.{m.group(1)}", seg_start, seg_end))
    return results


def build_chunks_for_document(document_id: str, pdf_path: Path) -> List[Chunk]:
    text, page_map = _load_pdf(pdf_path)

    # `_find_real_ek_headers` yine de bazi tesadufi/tablo-ici "EK-N.x" gibi
    # eslesmeleri "gercek baslik" sayabilir (orn. bykhy'deki bir tablo
    # hucresi referansi); bu YALNIZCA govdeyi eksik islememek icin, EK-bazli
    # ozel bolme uygulanan dokumanlarda (_EK_SUBCLAUSE_DOCS/_EK_LIST_DOCS)
    # ONEMLIDIR - digerlerinde EK ayrimi hic YAPILMADIGI icin tam metin
    # islenir (body_end kisaltilmaz).
    needs_ek_split = document_id in (_EK_SUBCLAUSE_DOCS | _EK_LIST_DOCS)
    ek_headers = _find_real_ek_headers(text) if needs_ek_split else []
    body_end = ek_headers[0][1] if ek_headers else len(text)

    segments = _split_body_by_madde(text, end=body_end)

    if ek_headers and needs_ek_split:
        # Her EK basligi KENDI bolgesini tanimlar (bir sonraki EK basligina
        # veya belge sonuna kadar); boylece EK-I ve EK-II gibi FARKLI
        # bolumlerin numaralandirmasi (orn. ikisinde de "1", "2", ...
        # olmasi) CAKISMAZ - `ek_label` (EK numarasi) chunk numarasina
        # ON EK olarak eklenir.
        for i, (ek_label, ek_start) in enumerate(ek_headers):
            ek_end = ek_headers[i + 1][1] if i + 1 < len(ek_headers) else len(text)
            if document_id in _EK_SUBCLAUSE_DOCS:
                segments.extend(_split_ek_section_by_subclause(text, ek_label, ek_start, ek_end))
            else:
                segments.extend(_split_ek_section_by_list_item(text, ek_label, ek_start, ek_end))
        # Diger dokumanlarda (orn. bykhy, 6331) EK bolumu bu ilk surumde
        # AYRI chunk'lanmiyor - govde disinda kalan metin sessizce atlanmiyor,
        # yalnizca EK-ozel bir bolme kurali henuz tanimlanmadi (bkz. audit:
        # "hangi dokumanlarda ozel parser gerekir").

    chunks: List[Chunk] = []
    seen_ids: set = set()
    for level, number, start, stop in segments:
        raw = text[start:stop].strip()
        if not raw:
            continue
        page_start = page_map.page_for_offset(start)
        page_end = page_map.page_for_offset(max(start, stop - 1))
        chunk_id = f"{document_id}__{level}_{number}"
        if chunk_id in seen_ids:
            # Guvenlik agi: beklenmedik bir cakisma olursa SESSIZCE
            # UST USTE YAZMAK yerine ayirt edici bir sonek eklenir (veri
            # kaybetmemek onemli - bu bir 'hic olmamasi gereken durum'dur).
            chunk_id = f"{chunk_id}__dup{sum(1 for s in seen_ids if s.startswith(chunk_id))}"
        seen_ids.add(chunk_id)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                level=level,
                number=number,
                is_annex=level in ("ek_alt_madde", "ek_liste"),
                page_start=page_start,
                page_end=page_end,
                char_count=len(raw),
                text=raw,
            )
        )
    return chunks


def main() -> int:
    if not _OFFICIAL_DIR.exists():
        print(f"[HATA] {_OFFICIAL_DIR} bulunamadi.", file=sys.stderr)
        return 1

    _CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(_OFFICIAL_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[HATA] {_OFFICIAL_DIR} icinde PDF bulunamadi.", file=sys.stderr)
        return 1

    summary = []
    for pdf_path in pdf_files:
        document_id = pdf_path.stem
        chunks = build_chunks_for_document(document_id, pdf_path)

        out_path = _CHUNKS_DIR / f"{document_id}.json"
        out_path.write_text(
            json.dumps([c.__dict__ for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        madde_n = sum(1 for c in chunks if c.level == "madde")
        gecici_n = sum(1 for c in chunks if c.level == "gecici_madde")
        ek_alt_n = sum(1 for c in chunks if c.level == "ek_alt_madde")
        ek_liste_n = sum(1 for c in chunks if c.level == "ek_liste")
        summary.append((document_id, len(chunks), madde_n, gecici_n, ek_alt_n, ek_liste_n))
        print(
            f"{document_id}: {len(chunks)} chunk "
            f"(madde={madde_n} gecici_madde={gecici_n} ek_alt_madde={ek_alt_n} ek_liste={ek_liste_n}) "
            f"-> {out_path.relative_to(_ROOT_DIR)}"
        )

    total = sum(s[1] for s in summary)
    print(f"\nToplam: {len(summary)} dokuman, {total} chunk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
