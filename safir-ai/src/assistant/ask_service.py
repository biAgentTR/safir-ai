"""Ask SAFIR servis katmani: soru -> (opsiyonel) analiz baglami + RAG -> LLM -> dayanakli cevap.

Mevcut soyutlamalari YENIDEN KULLANIR (yeni LLM/RAG sistemi YAZILMAZ):
  - `EmbeddingRAGService.query()`  -> `RetrievedDocument{text, score}`  (ISG mevzuati)
  - `AnalysisStore.get()`          -> kalici `SafirReport` (analiz baglami)
  - `get_llm_client()` (LLMClient / MockLLMClient) -> `.invoke(messages)` / `.stream(messages)`

GUVENLIK: modele YALNIZCA kullaniciya gosterilebilir, guvenli alanlar gonderilir.
`raw_response`, internal reasoning, secret/API key, system prompt, stack trace,
binary/base64 goruntu verisi cevaba/baglama ASLA girmez. Baglamda olmayan bilgi
uydurulmaz; yetersizse model bunu belirtir.

Konusma gecmisi (SAFIR Asistan sohbetleri)
-------------------------------------------
`answer()`/`stream_answer()`, opsiyonel bir `history` (son birkac (role, content)
mesaj) alabilir — bu YALNIZCA takip sorularini ("Peki neden?" gibi) anlamak
icin, en fazla `_MAX_HISTORY_MESSAGES` mesajla SINIRLI ve ayri, acikca
etiketli bir prompt blogu olarak eklenir; SAFIR'in KENDI onceki cevaplarini
gercek analiz verisi/kanit olarak kabul ETMEMESI icin `ASK_SYSTEM_PROMPT`
bunu acikca belirtir (celiski halinde ANALIZ BAGLAMI esas alinir). `/ask`
(mevcut, degismeyen uc nokta) `history` GECIRMEZ; yalnizca yeni `/ask/stream`
(SAFIR Asistan sayfasi) opsiyonel olarak kullanir.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.schemas.report import SafirReport

logger = logging.getLogger(__name__)

# Konusma gecmisinden prompt'a eklenecek en fazla mesaj sayisi (kullanici+asistan
# birlikte). Sinirsiz buyumeyi ve gereksiz token maliyetini onler; yalnizca
# yakin takip sorularini anlamak icin yeterlidir (bkz. modul dokustringi).
_MAX_HISTORY_MESSAGES = 8

# Yeni (Ask'e ozgu) system prompt — mevcut VLM/Agent promptlari DEGISMEZ.
ASK_SYSTEM_PROMPT = (
    "Sen SAFIR'sin: saha guvenligi ve ISG (Is Sagligi ve Guvenligi) karar destek asistanisin. "
    "SADECE sana verilen ANALIZ BAGLAMI ve MEVZUAT BAGLAMI icindeki bilgileri gercek kabul et. "
    "Bu baglam disinda olay, sayi, risk veya mevzuat UYDURMA. "
    "Bir bilgi baglamda yoksa, bunu acikca belirt (ornegin: 'Bu bilgi mevcut analiz baglaminda yer almiyor.'). "
    "Kesin olmayan seyleri kesinmis gibi sunma; kaynak yoksa kaynak varmis gibi gosterme. "
    "Eger sana bir ANALIZ BAGLAMI verilmemisse, genel ISG/SAFIR sorularini (mevzuat, "
    "risk kavramlari, SAFIR'in nasil calistigi, sistemin yetenekleri/sinirlari) normal "
    "sekilde yanitlayabilirsin. Ancak kullanici belirli bir analize ozgu bir gercek/olay/ "
    "zaman/risk skoru sorarsa ve elinde ANALIZ BAGLAMI yoksa, bunu KESINLIKLE UYDURMA; "
    "su an secili bir analiz baglami olmadigini acikca belirtip kullanicidan bir analiz "
    "secmesini iste. "
    "Sana bir KONUSMA GECMISI verilirse, bunu yalnizca takip sorularini anlamak icin "
    "kullan — bu bir KANIT DEGILDIR; kendi onceki cevaplarini gercek analiz verisi gibi "
    "kabul etme, celiski halinde ANALIZ BAGLAMI esas alinir. "
    "Ic muhakemeni veya adim adim dusunce zincirini YAZMA; yalnizca kisa, net, operatore yonelik "
    "Turkce nihai cevabi ver."
)


class AskError(Exception):
    """LLM/servis genel hatasi (endpoint tarafinda guvenli 503'e cevrilir)."""


class JobNotFound(Exception):
    """Verilen job_id History'de bulunamadi (endpoint tarafinda 404)."""


@dataclass
class Source:
    """Cevabi dayandiran tek bir kaynak. Yalnizca backend'in GERCEKTEN verdigi alanlar."""

    type: str  # "analysis" | "regulation"
    text: Optional[str] = None
    score: Optional[float] = None
    label: Optional[str] = None


@dataclass
class AskResult:
    """Ask SAFIR sonucu: dayanakli cevap + kaynaklar + kullanilan baglam etiketleri."""

    answer: str
    sources: List[Source]
    job_id: Optional[str]
    context_used: List[str]


@dataclass
class AskStreamMeta:
    """`stream_answer()`in, LLM akisi BASLAMADAN ONCE bilinen kismi: `AskResult` eksi `answer`."""

    sources: List[Source]
    job_id: Optional[str]
    context_used: List[str]


class AskService:
    """Soru + (opsiyonel) analiz baglami + RAG'i birlestirip LLM'den dayanakli cevap uretir."""

    def __init__(self, analysis_store, rag_service, llm_client, rag_top_k: int = 5) -> None:
        """Servisi mevcut depo/RAG/LLM soyutlamalariyla baglar (yeni sistem kurmaz).

        Args:
            analysis_store: `AnalysisStore` (kalici SafirReport).
            rag_service: `EmbeddingRAGService` (FAISS ISG mevzuati).
            llm_client: `get_llm_client(...)` ciktisi (LLMClient/MockLLMClient).
            rag_top_k: RAG'dan getirilecek maksimum madde.
        """
        self._analysis_store = analysis_store
        self._rag = rag_service
        self._llm = llm_client
        self._rag_top_k = rag_top_k

    @staticmethod
    def _safe_analysis_context(report: SafirReport) -> Dict[str, Any]:
        """Rapordan YALNIZCA kullaniciya gosterilebilir, guvenli alanlari cikarir (base64/internal YOK)."""
        return {
            "summary": report.summary or report.natural_language_summary,
            "risk_score": report.risk_score,
            "risk_level": report.risk_level,
            "recommended_action": report.recommended_action,
            "actions": report.actions,
            "detected_event_types": report.detected_event_types,
            "escalation_tier": report.escalation_tier,
            "auto_dispatched": report.auto_dispatched,
            "timeline": [
                {"time": SafirReport._seconds_to_mmss(e.timestamp), "event": e.description}
                for e in report.timeline
            ],
            "relevant_regulations": report.relevant_regulations,
            # Kanit: yalnizca metadata (zaman + degisim skoru); base64 goruntu ASLA.
            "evidence": [
                {"time": ef.timestamp_str, "change_score": round(ef.change_score, 4)}
                for ef in report.evidence_frames
            ],
            "vlm_observation": report.natural_language_summary,
        }

    def _prepare(
        self,
        question: str,
        job_id: Optional[str] = None,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Tuple[List[BaseMessage], List[Source], List[str], Optional[str]]:
        """Ortak baglam-olusturma: analiz baglami + RAG + (opsiyonel) konusma gecmisi -> LLM mesajlari.

        `answer()` ve `stream_answer()` tarafindan ortaklasa kullanilir; ikisi de
        AYNI guvenlik/grounding mantigina tabidir (bkz. modul dokustringi).

        Args:
            question: Kullanicinin dogal dil sorusu.
            job_id: Verilirse, o analizin kalici raporu oncelikli baglam olur.
            history: Verilirse, son `_MAX_HISTORY_MESSAGES` mesaji (role, content)
                ikilileri olarak) ayri, acikca "kanit degildir" etiketli bir
                prompt blogu olarak eklenir. `None`/bos ise (varsayilan,
                `/ask`in kullandigi yol) hicbir sey eklenmez — davranis
                onceki (gecmis-siz) haliyle BIREBIR AYNI kalir.

        Returns:
            `(messages, sources, context_used, used_job)`.

        Raises:
            ValueError: Soru bos.
            JobNotFound: `job_id` History'de yok.
        """
        q = (question or "").strip()
        if not q:
            raise ValueError("Soru bos olamaz.")

        context_used: List[str] = []
        sources: List[Source] = []
        analysis_ctx: Optional[Dict[str, Any]] = None
        used_job: Optional[str] = None

        # A) Analiz baglami (oncelikli) — yalnizca kalici, gercek rapordan.
        if job_id:
            record = self._analysis_store.get(job_id)
            if record is None:
                raise JobNotFound(job_id)
            used_job = job_id
            if record.report_json:
                report = SafirReport.model_validate_json(record.report_json)
                analysis_ctx = self._safe_analysis_context(report)
                context_used.append(f"analiz raporu ({job_id})")
                sources.append(Source(type="analysis", label="SAFIR analiz raporu"))

        # B) RAG mevzuat baglami — mevcut EmbeddingRAGService (gercek text+score).
        docs: List[Any] = []
        try:
            docs = self._rag.query(q, top_k=self._rag_top_k) or []
        except Exception:  # noqa: BLE001 - RAG yoksa cevap yine de uretilir
            logger.exception("Ask SAFIR: RAG sorgusu basarisiz oldu")
        for d in docs:
            sources.append(Source(type="regulation", text=d.text, score=round(float(d.score), 4)))
        if docs:
            context_used.append(f"RAG mevzuat ({len(docs)})")

        # C) Konusma gecmisi (opsiyonel, sinirli, ayri ve acikca etiketli).
        history_lines: List[str] = []
        if history:
            for role, content in list(history)[-_MAX_HISTORY_MESSAGES:]:
                speaker = "Kullanici" if role == "user" else "SAFIR"
                history_lines.append(f"{speaker}: {content}")
        if history_lines:
            context_used.append(f"konusma gecmisi ({len(history_lines)} mesaj)")

        # D) Kontrollu insan mesaji (yalnizca guvenli baglam).
        parts: List[str] = []
        if analysis_ctx is not None:
            parts.append(
                "ANALIZ BAGLAMI (bu analize ozgu, oncelikli):\n"
                + json.dumps(analysis_ctx, ensure_ascii=False, indent=2)
            )
        if docs:
            parts.append(
                "MEVZUAT BAGLAMI (RAG ile getirilen ISG maddeleri):\n"
                + "\n".join(f"- (skor {round(float(d.score), 3)}) {d.text}" for d in docs)
            )
        if history_lines:
            parts.append(
                "KONUSMA GECMISI (yalnizca takip sorularini anlamak icindir; KANIT DEGILDIR — "
                "SAFIR'in kendi onceki cevaplarini gercek analiz verisi gibi KABUL ETME; "
                "celiski halinde ANALIZ BAGLAMI esas alinir):\n" + "\n".join(history_lines)
            )
        if not parts:
            parts.append("(Bu soru icin analiz veya mevzuat baglami bulunamadi.)")
        parts.append("SORU: " + q)

        messages: List[BaseMessage] = [
            SystemMessage(content=ASK_SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(parts)),
        ]
        return messages, sources, context_used, used_job

    def answer(
        self,
        question: str,
        job_id: Optional[str] = None,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> AskResult:
        """Soruyu (opsiyonel analiz baglami + RAG + opsiyonel konusma gecmisi ile) yanitlar.

        Args:
            question: Kullanicinin dogal dil sorusu.
            job_id: Verilirse, o analizin kalici raporu oncelikli baglam olur.
            history: Bkz. `_prepare`. `/ask` bunu HIC gecirmez — o yolun davranisi
                bu parametre eklenmeden ONCEKI haliyle birebir aynidir.

        Returns:
            `AskResult` (answer + sources + context_used).

        Raises:
            ValueError: Soru bos.
            JobNotFound: `job_id` History'de yok.
            AskError: LLM bos/gecersiz yanit verdi.
        """
        messages, sources, context_used, used_job = self._prepare(question, job_id, history)

        try:
            ai = self._llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ask SAFIR: LLM cagrisi basarisiz oldu")
            raise AskError("LLM cagrisi basarisiz") from exc

        answer = (getattr(ai, "content", "") or "").strip()
        if not answer:
            raise AskError("LLM bos cevap dondurdu")

        return AskResult(answer=answer, sources=sources, job_id=used_job, context_used=context_used)

    def stream_answer(
        self,
        question: str,
        job_id: Optional[str] = None,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Tuple[AskStreamMeta, Iterator[str]]:
        """`answer()` ile AYNI baglam/guvenlik mantigiyla, LLM yanitini parca parca (streaming) uretir.

        Baglam-olusturma (`_prepare`) TAMAMEN AYNIDIR; tek fark, cevabin
        `self._llm.invoke(...)` yerine `self._llm.stream(...)` ile uretilmesidir
        (LangChain Runnable arayuzunun standart parcasi; `LLMClient`/`MockLLMClient`
        her ikisi de bunu destekler).

        Args:
            question: Kullanicinin dogal dil sorusu.
            job_id: Verilirse, o analizin kalici raporu oncelikli baglam olur.
            history: Bkz. `_prepare`.

        Returns:
            `(meta, chunks)` — `meta` akis BASLAMADAN ONCE hazir (kaynaklar/baglam
            etiketleri); `chunks`, cagrildikca metin parcalari ureten bir
            iterator'dur (bos parcalar atlanir).

        Raises:
            ValueError: Soru bos.
            JobNotFound: `job_id` History'de yok.
        """
        messages, sources, context_used, used_job = self._prepare(question, job_id, history)
        meta = AskStreamMeta(sources=sources, job_id=used_job, context_used=context_used)

        def _chunks() -> Iterator[str]:
            stream_fn = getattr(self._llm, "stream", None)
            if stream_fn is None:
                # Guvenli geriye-donus: stream() sunmayan bir istemci icin tek
                # parcada invoke() kullan (asla exception ile sessizce cokme).
                ai = self._llm.invoke(messages)
                content = (getattr(ai, "content", "") or "").strip()
                if content:
                    yield content
                return
            for chunk in stream_fn(messages):
                piece = getattr(chunk, "content", "") or ""
                if piece:
                    yield piece

        return meta, _chunks()
