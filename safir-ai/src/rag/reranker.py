"""04 - Embedding & RAG Katmani: EVREN'i "LLM-as-judge" olarak kullanan ucuncu-asama rerank.

`EmbeddingRAGService.query()`, deterministik relevance gate'ten GECMIS
("accepted") adaylari, `EvrenReranker` uzerinden EVREN'in OpenAI-uyumlu LLM
ucuna (`model="llm-fast"`) gonderip sorgu ile her aday arasindaki gercek
alaka skorunu (0.0-1.0) SADECE yapilandirilmis JSON dondurmesini isteyerek
hesaplatir ve adaylari YENIDEN siralar. `embedding_score` (Qdrant/cosine)
ile `cross_encoder_score` (EVREN LLM-as-judge) KARISTIRILMAZ - cagiran taraf
(`EmbeddingRAGService`) ikisini ayri alanlarda tutar.

2026-08-25 guncellemesi (Gemini/Groq TAMAMEN KALDIRILDI): eski
`GeminiReranker`/`GroqReranker` (harici, birbirinden bagimsiz iki saglayici)
KALDIRILDI; EVREN'in OpenAI-uyumlu LLM ucunu kullanan TEK saglayici olan
`EvrenReranker` eklendi. EVREN dokumantasyonu (SS 10), dedike bir rerank
servisinin (`model="rerank"`) getirme kalitesini OLCUMLE dusurdugunu
gostermektedir - bu yuzden `EvrenReranker` o dedike ucu KULLANMAZ; standart
`/v1/chat/completions` ucunu ("llm-fast") bu modulun ONCEDEN var olan
JSON-yargi promptuyla (`_build_rerank_prompt`/`_parse_rerank_response`)
cagirir.

`EvrenReranker`, `src/rag/local_cross_encoder_reranker.py::CrossEncoderReranker`
sozlesmesini (`score(query, texts) -> List[float]`) uygular - boylece
`EmbeddingRAGService`nin ONCEDEN var olan "cross_encoder" extension point'ine
(deterministik gate'i BYPASS ETMEYEN, yalnizca ZATEN kabul edilmis adaylari
YENIDEN SIRALAYAN ucuncu asama) hicbir arayuz degisikligi olmadan baglanir.

GUVENLIK KURALI (degismedi): EVREN cagrisi basarisiz olursa VEYA donen JSON
gecersiz/kurallara aykiri olursa (index araligi disi, tekrar eden index,
eksik alan, ayristirilamayan metin, EKSIK aday), bu modul SESSIZCE embedding/
deterministik-relevance siralamasini "final sonuc" gibi DONDURMEZ -
`CrossEncoderUnavailableError` firlatir; `EmbeddingRAGService.query()` bunu
yakalayip siralamayi deterministik relevance skoruna GERI DUSURUR (KONTROLLU
degradasyon, `RagQueryTelemetry.cross_encoder_status="unavailable"` ile
ACIKCA isaretlenir - bkz. `embedding_rag_service.py`, DEGISTIRILMEDI).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Tuple

from src.rag.local_cross_encoder_reranker import CrossEncoderReranker, CrossEncoderUnavailableError

logger = logging.getLogger(__name__)

_MAX_CANDIDATE_CHARS = 600
"""Prompt boyutunu kontrol altinda tutmak icin, her aday metninin prompt'a
dahil edilecek azami karakter sayisi (yalnizca PROMPT icin kirpilir; skorlama
SONRASI dondurulen sonuc orijinal `candidates` metnine indeks olarak isaret
eder - metin KISALTILMIS olarak DONMEZ)."""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class RerankerUnavailableError(Exception):
    """EVREN rerank cagrisi basarisiz oldu VEYA donen yanit gecersiz.

    Bu, "alakasiz sonuc" ile KARISTIRILMAMALIDIR - bu, rerank'in HIC
    GUVENILIR SEKILDE YAPILAMADIGI anlamina gelir; cagiran taraf bu
    durumda embedding top-k'yi asla "final, dogrulanmis" sonuc gibi
    sunmamalidir.
    """


def _build_rerank_prompt(query: str, candidates: List[str]) -> str:
    """SADECE yapilandirilmis JSON isteyen rerank istemini uretir."""
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

    if len(results) != candidate_count:
        logger.error(
            "Reranker: yanit TUM adaylari kapsamiyor (%d/%d sonuc dondu).", len(results), candidate_count
        )
        raise RerankerUnavailableError(
            f"Rerank yaniti tum adaylari kapsamiyor ({len(results)}/{candidate_count})."
        )

    return sorted(results, key=lambda t: t[1], reverse=True)


class EvrenReranker(CrossEncoderReranker):
    """EVREN'in OpenAI-uyumlu LLM ucunu ("llm-fast") "LLM-as-judge" olarak kullanan `CrossEncoderReranker`.

    EVREN'in dedike rerank ucunu (`model="rerank"`) KULLANMAZ (bkz. modul
    dokustringi - dokumantasyon SS 10, getirme kalitesini dusurdugunu
    gostermektedir); standart `/v1/chat/completions` cagrisi + yapilandirilmis
    JSON istegi kullanir.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key_env: str,
    ) -> None:
        """EvrenReranker'i model/uc nokta bilgisiyle kurar (AG CAGRISI YAPMAZ - istemci lazy olusturulur).

        Args:
            model_name: EVREN LLM model takma adi (orn. "llm-fast").
            base_url: EVREN'in OpenAI-uyumlu taban adresi.
            api_key_env: API anahtarinin okunacagi ortam degiskeni adi (orn. "EVREN_API_KEY").
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
            raise CrossEncoderUnavailableError(
                f"EVREN reranker icin '{self._api_key_env}' ortam degiskeni tanimli degil."
            )

        from openai import OpenAI  # gec import: paket kurulu degilse bile modul import'u patlamasin

        self._client = OpenAI(api_key=api_key, base_url=self._base_url)
        return self._client

    def score(self, query: str, texts) -> List[float]:
        """`texts` ile AYNI sirada, EVREN'in LLM-as-judge alaka skorlarini dondurur.

        Raises:
            CrossEncoderUnavailableError: EVREN cagrisi basarisiz olursa veya
                donen yanit gecersiz/eksik olursa (bkz. modul dokustringi -
                `EmbeddingRAGService.query()` bunu KONTROLLU sekilde
                deterministik relevance siralamasina geri duser).
        """
        texts = list(texts)
        if not texts:
            return []

        client = self._get_client()
        prompt = _build_rerank_prompt(query, texts)

        try:
            response = client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw_text = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - HERHANGI bir API hatasi guvenli bir "unavailable"a cevrilir
            logger.error("EvrenReranker: chat.completions.create cagrisi basarisiz: %s", exc)
            raise CrossEncoderUnavailableError(f"EVREN rerank cagrisi basarisiz: {exc}") from exc

        try:
            parsed = _parse_rerank_response(raw_text, len(texts))
        except RerankerUnavailableError as exc:
            raise CrossEncoderUnavailableError(str(exc)) from exc

        score_by_index = dict(parsed)
        return [score_by_index[i] for i in range(len(texts))]
