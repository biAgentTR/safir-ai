"""04 - Embedding & RAG Katmani: LLM'DEN BAGIMSIZ, deterministik retrieval relevance skorlamasi.

2026-08-24 (RAG RERANKER DETERMINIZATION): onceki surumde retrieval'in ikinci
asamasi (`reranker.py::GeminiReranker`/`GroqReranker`) bir LLM'e "0.0-1.0
arasinda relevance skoru ver" diye SERBEST bir yargi soruyordu - bu skor
kalibre edilmis bir olasilik DEGILDI, ayni girdide MODEL DAVRANISINA bagli
olarak degisebilirdi, ve 429/400 gibi harici API hatalarina acikti (bkz.
`reranker.py` modul dokustringindeki "KARAR" notu - o turda bu, BILINCLI
olarak "retrieval_unavailable" ile ele alinmisti; bu turda ise SORUNUN
KENDISI - LLM'e bagimliligin KALDIRILMASI - cozuluyor).

Bu modul, FAISS'ten gelen embedding benzerligini (`semantic_score`) TEK
BASINA degil, dort EK deterministik sinyalle birlikte agirlikli bir toplama
(`relevance_score`) indirger - hicbiri bir LLM/API cagrisi YAPMAZ, TAMAMEN
yerel/matematiksel/tekrar-uretilebilirdir (bkz. `EmbeddingRAGService.query()`,
artik bu modulu KOSULSUZ cagirir - `reranker.py`nin LLM saglayicilari
PRODUCTION relevance kararindan CIKARILDI, bkz. gorev tanimi).

BEŞ SINYAL
----------
A) semantic_score  : FAISS `IndexFlatIP` cosine benzerligi (zaten [-1,1] ama
                      pratikte L2-normalize vektorlerde negatif COK NADIR;
                      [0,1]'e clip edilir - negatif "alakasiz" sayilir).
B) lexical_score    : sorgu ile chunk metni arasindaki TURKCE-normalize
                      edilmis, stopword'den ARINDIRILMIS token orusumu
                      (basit sufiks-toleransli "recall" olcusu - AGIR bir
                      stemmer/NLP bagimliligi EKLENMEDI, bkz. `_stem_bucket`).
C) keyword_score    : VLM'in `matched_keywords`inin chunk metninde GECIP
                      GECMEDIGI (keyword YOKSA bu sinyal SESSIZCE 0 katkı
                      verir - sistem BOZULMAZ, bkz. gorev tanimi).
D) metadata_score   : sorgu, chunk'in `document_title`/`article_number`ini
                      ACIKCA HEDEFLIYORSA (orn. "... Yonetmeligi I.2.17")
                      KONTROLLU bir bonus (kelime-siniri kontrolu ile - keyword
                      stuffing'e ACIK degil).
E) phrase_score     : sorgunun TAMAMI (normalize edilmis) chunk metninde
                      AYNEN (contiguous) geciyorsa ikili bonus.

`relevance_score = w_semantic*A + w_lexical*B + w_keyword*C + w_metadata*D + w_phrase*E`

Agirliklar `RerankerConfig`den (config.yaml -> memory.reranker.weights)
okunur - HARD-CODE DEGILDIR (bkz. `RelevanceWeights`).

ONEMLI: `relevance_score`, ASLA "confidence" olarak ADLANDIRILMAZ (ne bu
modulde ne cagiran kodda) - bu bir olasilik/kalibrasyon iddiasi TASIMAZ,
yalnizca agirlikli-toplam bir SIRALAMA sinyalidir.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence

# --------------------------------------------------------------------------
# Turkce metin normalizasyonu (agir bagimlilik YOK - stdlib `re`/`unicodedata`)
# --------------------------------------------------------------------------

_TURKISH_STOPWORDS = frozenset(
    {
        "ve", "veya", "ile", "de", "da", "ki", "mi", "mı", "mu", "mü",
        "bir", "bu", "şu", "o", "bunlar", "şunlar", "onlar",
        "için", "gibi", "kadar", "göre", "üzere", "dolayı", "rağmen",
        "olan", "olarak", "olup", "olması", "olduğu", "olduğunu",
        "her", "hiç", "hep", "daha", "en", "çok", "az",
        "ama", "fakat", "ancak", "çünkü", "ise", "eğer", "yani",
        "ben", "sen", "biz", "siz", "ne", "kim", "hangi", "nasıl", "niçin",
        "tüm", "tum", "bütün", "butun", "diğer", "diger",
        "içinde", "icinde", "üzerinde", "uzerinde", "altında", "altinda",
        "var", "yok", "değil", "degil",
    }
)
"""Kucuk, elle secilmis Turkce stopword kumesi - agir bir NLP paketi
(zeyrek/nltk vb.) BILEREK EKLENMEDI (gorev tanimi: "ağır NLP dependency
ekleme")."""

_TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+", re.UNICODE)


def turkish_normalize(text: str) -> str:
    """Turkce-DOGRU kucuk harfe cevirme (Python'un `str.lower()`u 'İ'yi YANLIS cevirir) + Unicode/noktalama normalizasyonu.

    `'İstanbul'.lower()` -> `'i̇stanbul'` (nokta karakteri BOZUK kalir) - bu
    fonksiyon once Turkce'ye ozgu 'İ'/'I' haritalamasini ELLE uygular, SONRA
    genel kucuk harfe cevirir.
    """
    if not text:
        return ""
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    return text


def tokenize(text: str) -> List[str]:
    """Normalize edilmis metni, yalnizca harf/rakam token'larina ayirir (noktalama/whitespace ATILIR)."""
    return _TOKEN_RE.findall(turkish_normalize(text))


def _content_tokens(text: str) -> List[str]:
    """Tokenize eder, stopword'leri ve tek-karakterli parcaciklari ELER (anlamli ICERIK token'lari)."""
    return [t for t in tokenize(text) if t not in _TURKISH_STOPWORDS and len(t) > 1]


def _stem_bucket(token: str) -> str:
    """Basit, KURALSIZ sufiks-toleransi: 5 karakterden uzun token'lari ilk 5 karaktere indirger.

    Bu GERCEK bir stemmer DEGILDIR (orn. Zemberek/zeyrek gibi morfolojik
    cozumleme YAPMAZ) - yalnizca "forklift" ile "forkliftler"/"forklifte"
    gibi basit ek varyasyonlarini KABACA aynı kefeye koyan, ucuz bir sezgiseldir.
    """
    return token[:5] if len(token) > 5 else token


def lexical_overlap_score(query: str, chunk_text: str) -> float:
    """Sorgunun ICERIK token'larindan KACININ chunk metninde (tam veya sufiks-toleransli) gectigini olcer.

    "Recall" yonelimlidir (payda SORGU token sayisidir, chunk uzunlugundan
    BAGIMSIZ) - uzun bir chunk, sirf uzun oldugu icin haksiz avantaj ALMAZ.

    Returns:
        `[0.0, 1.0]` araliginda oran; sorguda anlamli token yoksa `0.0`.
    """
    query_tokens = set(_content_tokens(query))
    if not query_tokens:
        return 0.0
    chunk_tokens = set(_content_tokens(chunk_text))
    chunk_stems = {_stem_bucket(t) for t in chunk_tokens}
    matched = sum(1 for qt in query_tokens if qt in chunk_tokens or _stem_bucket(qt) in chunk_stems)
    return matched / len(query_tokens)


def keyword_match_score(keywords: Optional[Sequence[str]], chunk_text: str) -> float:
    """VLM'in dinamik risk keyword'lerinden KACININ chunk metninde (normalize edilmis substring olarak) gectigini olcer.

    `keywords` bos/None ise (VLM hicbir keyword uretmediyse) bu sinyal
    SESSIZCE `0.0` doner - cagiran tarafin `relevance_score` hesabini
    BOZMAZ (gorev tanimi: "sistem bozulmamalı").
    """
    if not keywords:
        return 0.0
    chunk_norm = turkish_normalize(chunk_text)
    hits = sum(1 for kw in keywords if kw and turkish_normalize(kw) in chunk_norm)
    return hits / len(keywords)


def metadata_match_score(query: str, document_title: Optional[str], article_number: Optional[str]) -> float:
    """Sorgunun, chunk'in `document_title`/`article_number`ini ACIKCA HEDEFLEYIP HEDEFLEMEDIGINI KONTROLLU sekilde odullendirir.

    Iki bilesim: (1) sorgu-baslik lexical orusumu (`lexical_overlap_score` ile
    AYNI mekanizma), (2) `article_number`in sorguda KELIME-SINIRI ile (kismi
    string eslesmesi DEGIL - orn. sorgudaki "12" chunk'in "123" maddesine
    YANLIS eslesmez) TAM olarak gecip gecmedigi. Keyword-stuffing'e ACIK
    DEGILDIR - madde numarasi ya sorguda GERCEKTEN var ya yok (ikili).
    """
    title_component = lexical_overlap_score(query, document_title) if document_title else 0.0

    article_component = 0.0
    if article_number:
        norm_query = turkish_normalize(query)
        norm_article = turkish_normalize(str(article_number))
        if norm_article and re.search(rf"(?<!\w){re.escape(norm_article)}(?!\w)", norm_query):
            article_component = 1.0

    return 0.5 * title_component + 0.5 * article_component


def phrase_match_score(query: str, chunk_text: str) -> float:
    """Sorgunun TAMAMININ (normalize edilmis, tek bir ardisik ifade olarak) chunk metninde GECIP GECMEDIGINI kontrol eder.

    Tek kelimelerin toplamindan FARKLIDIR (bkz. `lexical_overlap_score`) -
    sorgu kelimeleri chunk'a TEK TEK serpistirilerek bu skoru KAZANAMAZ
    (keyword-stuffing'e karsi dogal direnc); cok kisa sorgular (4 karakterden
    az) ANLAMSIZ eslesme riski tasidigi icin HER ZAMAN `0.0`.
    """
    norm_query = turkish_normalize(query).strip()
    if len(norm_query) < 4:
        return 0.0
    return 1.0 if norm_query in turkish_normalize(chunk_text) else 0.0


@dataclass
class RelevanceWeights:
    """Bes sinyalin agirliklari (`config.yaml` -> `memory.reranker.weights`den okunur, HARD-CODE DEGIL).

    Varsayilanlar (gorev tanimindaki ONERI) - DOGMA olarak KABUL EDILMEDI,
    ancak bu turda GERCEK corpus uzerinde canli bir A/B/skor-dagilimi analizi
    yapacak bir ortam (API/skorlama altyapisi) MEVCUT DEGILDI; semantic_score
    en guclu, genel-amacli sinyal oldugu icin (iyi egitilmis bir embedding
    modeli) en yuksek agirligi tasir - bu, bilgi-getirimi literaturundeki
    yaygin pratikle (semantic/dense retrieval BASKIN sinyal, lexical/metadata
    onu DESTEKLEYICI) tutarlidir.
    """

    semantic: float = 0.60
    lexical: float = 0.15
    keyword: float = 0.15
    metadata: float = 0.05
    phrase: float = 0.05


@dataclass
class RelevanceBreakdown:
    """Bir adayin nihai `relevance_score`unu ureten TUM ara sinyalleri ayri ayri tasir (izlenebilirlik/aciklanabilirlik icin)."""

    semantic_score: float
    lexical_score: float
    keyword_score: float
    metadata_score: float
    phrase_score: float
    relevance_score: float

    def reason(self) -> str:
        """`relevance_score`un HANGI sinyallerden nasil turedigini insan-okunur, deterministik bir cumleyle aciklar."""
        return (
            f"relevance_score={self.relevance_score:.3f} "
            f"(semantic={self.semantic_score:.3f}, lexical={self.lexical_score:.3f}, "
            f"keyword={self.keyword_score:.3f}, metadata={self.metadata_score:.3f}, "
            f"phrase={self.phrase_score:.3f})"
        )


def score_candidate(
    query: str,
    chunk_text: str,
    embedding_score: float,
    document_title: Optional[str] = None,
    article_number: Optional[str] = None,
    keywords: Optional[Sequence[str]] = None,
    weights: Optional[RelevanceWeights] = None,
) -> RelevanceBreakdown:
    """Tek bir adayin bes sinyalini hesaplayip agirlikli `relevance_score`u uretir - HICBIR AG CAGRISI YAPMAZ, TAMAMEN deterministiktir.

    Args:
        query: RAG sorgu metni.
        chunk_text: Adayin GERCEK chunk metni.
        embedding_score: FAISS `IndexFlatIP` cosine benzerligi (semantic_score'un girdisi).
        document_title: Adayin kaynak dokuman basligi (varsa).
        article_number: Adayin madde/ek numarasi (varsa).
        keywords: VLM'in dinamik risk keyword'leri (varsa; yoksa keyword_score 0.0).
        weights: Sinyal agirliklari; verilmezse `RelevanceWeights()` varsayilanlari.

    Returns:
        Ayni girdi icin HER ZAMAN AYNI (deterministik) `RelevanceBreakdown`.
    """
    w = weights or RelevanceWeights()
    semantic = max(0.0, min(1.0, embedding_score))
    lexical = lexical_overlap_score(query, chunk_text)
    keyword = keyword_match_score(keywords, chunk_text)
    metadata = metadata_match_score(query, document_title, article_number)
    phrase = phrase_match_score(query, chunk_text)
    relevance_score = w.semantic * semantic + w.lexical * lexical + w.keyword * keyword + w.metadata * metadata + w.phrase * phrase
    return RelevanceBreakdown(
        semantic_score=round(semantic, 4),
        lexical_score=round(lexical, 4),
        keyword_score=round(keyword, 4),
        metadata_score=round(metadata, 4),
        phrase_score=round(phrase, 4),
        relevance_score=round(relevance_score, 4),
    )
