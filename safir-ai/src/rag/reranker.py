"""04 - Embedding & RAG Katmani: LLM-as-judge ikinci-asama siralama (Gemini veya Groq).

`EmbeddingRAGService.query()`, FAISS'ten gelen `candidate_k` adayi bu
katmana gonderir; `Reranker` implementasyonu, sorgu ile her aday arasindaki
gercek alaka skorunu (0.0-1.0) bir metin modeline SADECE yapilandirilmis
JSON dondurmesini isteyerek hesaplatir ve adaylari YENIDEN siralar.
`embedding_score` (FAISS/cosine) ile `rerank_score` (LLM-as-judge) KARISTIRILMAZ -
cagiran taraf (`EmbeddingRAGService`) ikisini ayri alanlarda tutar.

Iki saglayici: `GeminiReranker` (Google Gemini, embedding ile AYNI
`GEMINI_API_KEY`) ve `GroqReranker` (Groq'un OpenAI-uyumlu ucu, AYRI bir
`GROQ_API_KEY`/kota - Gemini'nin embedding/VLM kotasindan BAGIMSIZ). Ikisi de
AYNI prompt/JSON semasini ve dogrulama kurallarini paylasir (bkz.
`_build_rerank_prompt`/`_parse_rerank_response`) - yalnizca AG cagrisi
saglayici-ozeldir.

GUVENLIK KURALI (degismedi): reranker API cagrisi basarisiz olursa VEYA
donen JSON gecersiz/kurallara aykiri olursa (index araligi disi, tekrar eden
index, eksik alan, ayristirilamayan metin), bu modul SESSIZCE embedding-
siralamasini "final sonuc" gibi DONDURMEZ - `RerankerUnavailableError`
firlatir; cagiran taraf bunu yakalayip retrieval'i "unavailable" (bos
sonuc) olarak ele alir. Modelin serbest ACIKLAMA metni ASLA retrieval
sonucu olarak KULLANILMAZ - yalnizca ayristirilmis {index, score} ciftleri.

KARAR (2026-08-24, RAG PIPELINE RECONSTRUCTION, gorev tanimi 8. bolum -
"reranker basarisiz olunca embedding_only fallback mi, retrieval_unavailable
mi?"): SAFIR icin BILINCLI olarak **B) retrieval_unavailable** secildi -
embedding-only siralamaya (A) SESSIZCE/OTOMATIK DUSULMEZ. Gerekce: bu sistem
ISG mevzuati gibi HUKUKI/guvenlik-kritik atiflar uretiyor; embedding
benzerligi (cosine) TEK BASINA bir maddenin sorguya GERCEKTEN alakali
oldugunun kaniti DEGILDIR (bkz. `embedding_rag_service.py` - embedding_score
bir "confidence"/olasilik DEGIL, vektor benzerligidir). LLM-as-judge rerank
BASARISIZ olduysa, bu adaylarin GERCEKTEN dogrulanmis oldugunu iddia etmek
YANLIS-POZITIF risk tasir (dogrulanmamis bir maddeyi "secilmis kanit" gibi
sunmak). Bu yuzden reranker basarisiz oldugunda sistem BASARILI gibi
DAVRANMAZ: final sonuc BOS doner, `retrieval_status="reranker_unavailable"`
ACIKCA isaretlenir (bkz. `EmbeddingRAGService.query`), her aday
`relevance_status="unavailable"` damgalanir - "hic aday yoktu" ile "adaylar
bulundu ama dogrulanamadi" birbirinden AYRISTIRILIR.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import List, Tuple

logger = logging.getLogger(__name__)

_MAX_CANDIDATE_CHARS = 600
"""Prompt boyutunu kontrol altinda tutmak icin, her aday metninin prompt'a
dahil edilecek azami karakter sayisi (yalnizca PROMPT icin kirpilir; skorlama
SONRASI dondurulen sonuc orijinal `candidates` metnine indeks olarak isaret
eder - metin KISALTILMIS olarak DONMEZ)."""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class RerankerUnavailableError(Exception):
    """Reranker API cagrisi basarisiz oldu, gerekli konfigurasyon (API anahtari) eksik VEYA donen yanit gecersiz.

    Bu, "alakasiz sonuc" ile KARISTIRILMAMALIDIR - bu, rerank'in HIC
    GUVENILIR SEKILDE YAPILAMADIGI anlamina gelir; cagiran taraf bu
    durumda embedding top-k'yi asla "final, dogrulanmis" sonuc gibi
    sunmamalidir.
    """


class Reranker(ABC):
    """Tum reranker saglayicilari icin ortak sozlesme."""

    @abstractmethod
    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[int, float]]:
        """Adaylari sorguya gore yeniden siralar.

        Args:
            query: Kullanici/sistem sorgusu.
            candidates: Aday dokuman metinleri (orijinal sira - FAISS/embedding sirasi).

        Returns:
            `(orijinal_indeks, relevance_score)` ikililerinin, `relevance_score`e
            gore AZALAN sirali listesi.

        Raises:
            RerankerUnavailableError: API cagrisi basarisiz olursa, gerekli
                konfigurasyon eksikse VEYA donen yanit gecersizse.
        """


def _build_rerank_prompt(query: str, candidates: List[str]) -> str:
    """Saglayicidan bagimsiz, SADECE yapilandirilmis JSON isteyen rerank istemini uretir."""
    candidate_lines = []
    for i, text in enumerate(candidates):
        snippet = text.strip().replace("\n", " ")[:_MAX_CANDIDATE_CHARS]
        candidate_lines.append(f"[{i}] {snippet}")
    candidates_block = "\n".join(candidate_lines)

    return (
        "Sen bir arama sonucu yeniden-siralama (rerank) sistemisin. Sana bir SORGU "
        "ve numaralandirilmis ADAY metinler verilecek. Gorevin, HER adayin sorguya "
        "ne kadar ALAKALI oldugunu 0.0 (hic alakasiz) ile 1.0 (tam alakali) arasinda "
        "puanlamaktir.\n\n"
        "SADECE asagidaki JSON formatinda yanit ver, BASKA HICBIR METIN/ACIKLAMA ekleme:\n"
        '{"results": [{"index": <aday index>, "score": <0.0-1.0>}, ...]}\n\n'
        "KURALLAR:\n"
        f"- \"index\" degeri YALNIZCA 0 ile {len(candidates) - 1} arasinda (dahil), sana "
        "verilen aday index'lerinden biri olmalidir.\n"
        "- Her index EN FAZLA BIR KEZ gecmelidir (tekrar YOK).\n"
        "- \"score\" 0.0 ile 1.0 arasinda bir ondalik sayidir.\n"
        f"- TUM {len(candidates)} aday icin bir sonuc uretmelisin (hicbirini atlama).\n\n"
        f"SORGU: {query}\n\n"
        f"ADAYLAR:\n{candidates_block}"
    )


def _parse_rerank_response(raw_text: str, candidate_count: int) -> List[Tuple[int, float]]:
    """Modelin JSON yanitini ayristirir ve KURALLARA (index araligi, tekrarsizlik, skor araligi) karsi dogrular.

    Herhangi bir sapma (bozuk JSON, eksik "results", araligin disinda/
    tekrar eden index, gecersiz skor) SESSIZCE duzeltilmez/atlanmaz -
    acikca loglanip `RerankerUnavailableError` firlatilir (bkz. modul
    dokustringi).
    """
    match = _JSON_BLOCK_RE.search(raw_text)
    candidate_json = match.group(0) if match else raw_text
    try:
        parsed = json.loads(candidate_json)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Reranker: yanit gecerli JSON degil: %r (%s)", raw_text[:500], exc)
        raise RerankerUnavailableError(f"Rerank yaniti gecerli JSON degil: {exc}") from exc

    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        logger.error("Reranker: yanitta beklenen 'results' listesi yok: %r", raw_text[:500])
        raise RerankerUnavailableError("Rerank yanitinda 'results' listesi bulunamadi.")

    seen_indices: set = set()
    results: List[Tuple[int, float]] = []
    for item in parsed["results"]:
        if not isinstance(item, dict) or "index" not in item or "score" not in item:
            logger.error("Reranker: 'results' ogesi eksik alan iceriyor: %r", item)
            raise RerankerUnavailableError(f"Rerank sonucunda eksik alan: {item!r}")

        index = item["index"]
        score = item["score"]
        if not isinstance(index, int) or not (0 <= index < candidate_count):
            logger.error(
                "Reranker: gecersiz index %r (izin verilen aralik: 0-%d)", index, candidate_count - 1
            )
            raise RerankerUnavailableError(f"Rerank sonucunda gecersiz index: {index!r}")
        if index in seen_indices:
            logger.error("Reranker: tekrar eden index tespit edildi: %r", index)
            raise RerankerUnavailableError(f"Rerank sonucunda tekrar eden index: {index!r}")
        try:
            score = float(score)
        except (TypeError, ValueError):
            logger.error("Reranker: gecersiz score %r (sayisal olmali)", score)
            raise RerankerUnavailableError(f"Rerank sonucunda gecersiz score: {score!r}")
        if not (0.0 <= score <= 1.0):
            logger.error("Reranker: score araligin disinda: %r (0.0-1.0 olmali)", score)
            raise RerankerUnavailableError(f"Rerank sonucunda score araligin disinda: {score!r}")

        seen_indices.add(index)
        results.append((index, score))

    return sorted(results, key=lambda t: t[1], reverse=True)


class GeminiReranker(Reranker):
    """Gemini metin modelini "LLM-as-judge" olarak kullanan, SADECE yapilandirilmis JSON isteyen `Reranker`.

    Dedike bir rerank API'si KULLANMAZ - mevcut Gemini API anahtarini
    (aynisini embedding icin de kullanilan `GEMINI_API_KEY`) kullanir.
    """

    def __init__(self, model_name: str, api_key_env: str = "GEMINI_API_KEY") -> None:
        """GeminiReranker'i model adiyla kurar (AG CAGRISI YAPMAZ - istemci lazy olusturulur).

        Args:
            model_name: Gemini metin modeli adi (orn. "gemini-3.5-flash-lite").
            api_key_env: API anahtarinin okunacagi ortam degiskeni adi.
        """
        self._model_name = model_name
        self._api_key_env = api_key_env
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self._api_key_env, "").strip()
        if not api_key:
            raise RerankerUnavailableError(
                f"Gemini reranker icin '{self._api_key_env}' ortam degiskeni tanimli degil."
            )

        from google import genai  # gec import: paket kurulu degilse bile modul import'u patlamasin

        self._client = genai.Client(api_key=api_key)
        return self._client

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[int, float]]:
        if not candidates:
            return []

        client = self._get_client()
        prompt = _build_rerank_prompt(query, candidates)

        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            raw_text = response.text or ""
        except Exception as exc:  # noqa: BLE001 - HERHANGI bir API hatasi guvenli bir "unavailable"a cevrilir
            logger.error("GeminiReranker: generate_content cagrisi basarisiz: %s", exc)
            raise RerankerUnavailableError(f"Gemini rerank cagrisi basarisiz: {exc}") from exc

        return _parse_rerank_response(raw_text, len(candidates))


class GroqReranker(Reranker):
    """Groq'un OpenAI-uyumlu ucunu "LLM-as-judge" olarak kullanan `Reranker`.

    Gemini'den TAMAMEN AYRI bir API anahtari/kota kullanir (`GROQ_API_KEY`) -
    boylece reranker cagrilari, embedding/VLM'in Gemini kotasiyla YARISMAZ.
    Ayni JSON semasi/dogrulama kurallari (bkz. `_build_rerank_prompt`/
    `_parse_rerank_response`) - yalnizca AG cagrisi farklidir (OpenAI SDK,
    `response_format={"type": "json_object"}`).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "https://api.groq.com/openai/v1",
        api_key_env: str = "GROQ_API_KEY",
    ) -> None:
        """GroqReranker'i model adi/uc noktasiyla kurar (AG CAGRISI YAPMAZ - istemci lazy olusturulur).

        Args:
            model_name: Groq model adi (orn. "openai/gpt-oss-20b").
            base_url: Groq'un OpenAI-uyumlu taban adresi.
            api_key_env: API anahtarinin okunacagi ortam degiskeni adi.
        """
        self._model_name = model_name
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self._api_key_env, "").strip()
        if not api_key:
            raise RerankerUnavailableError(
                f"Groq reranker icin '{self._api_key_env}' ortam degiskeni tanimli degil."
            )

        from openai import OpenAI  # gec import: paket kurulu degilse bile modul import'u patlamasin

        self._client = OpenAI(api_key=api_key, base_url=self._base_url)
        return self._client

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[int, float]]:
        if not candidates:
            return []

        client = self._get_client()
        prompt = _build_rerank_prompt(query, candidates)

        try:
            response = client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw_text = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - HERHANGI bir API hatasi guvenli bir "unavailable"a cevrilir
            logger.error("GroqReranker: chat.completions.create cagrisi basarisiz: %s", exc)
            raise RerankerUnavailableError(f"Groq rerank cagrisi basarisiz: {exc}") from exc

        return _parse_rerank_response(raw_text, len(candidates))
