"""03 - Gorsel Dil Modeli Katmani: tum VLM implementasyonlari icin soyut taban sinif.

ONEMLI (mimari): Olay kumelemesi ARTIK bu katmanda yapilir (sampler'da
DEGIL). `BaseVLM` alt siniflari (`QwenVLM`, `GemmaVLM`)
`analyze_evidence(evidence_frames, prompt)` implemente eder (`EvrenVLM`,
production'da kullanilan tek video-tabanli implementasyon, bunun yerine
`analyze_video` kullanir - bkz. `src/vlm/evren_vlm.py`); taban sinif
bunun uzerine iki ORTAK yetenek insa eder:

- `analyze_evidence_batched`: TUM evidence karelerini TEK bir dev payload'a
  doldurmak yerine kronolojik, kayipsiz batch'lere boler (batch siniri OLAY
  SINIRI DEGILDIR); her batch icin `analyze_evidence` BAGIMSIZ cagrilir, bir
  batch'in basarisiz olmasi digerlerini ETKILEMEZ.
- `reconcile_events`: birden fazla batch varsa, her batch'in kendi ICINDE
  urettigi (batch-yerel `event_id`li) olaylari, IKINCI (metin-tabanli,
  goruntusuz) bir VLM cagrisiyla TEK bir global olay listesine birlestirir;
  hicbir evidence kaybolmaz.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import httpx

from src.prompts.vlm_prompts import VLM_RECONCILIATION_SYSTEM_PROMPT
from src.sampler.payload_builder import VLMPayloadBuilder
from src.sampler.schema import EvidenceFrame
from src.utils.config_loader import VLLMEndpointConfig

logger = logging.getLogger(__name__)

# Gecici ag hatalarinda VLM cagrisinin kac kez yeniden deneneceği ve geri-cekilme tabani.
_MAX_INFERENCE_RETRIES = 2
_RETRY_BACKOFF_BASE_SEC = 0.5


def apply_extra_body(payload: Dict[str, Any], extra_body: Dict[str, Any]) -> None:
    """Saglayici-ozel alanlari (ozellikle `enable_thinking: false`) payload'a ekler.

    Mentor eleştirisi 5 ("Akil Yurutme/Thinking Modu Tuzagindan Kacinma"):
    `enable_thinking` acikken kucuk/orta modeller `max_tokens` butcesinin
    TAMAMINI gizli akil yurutmeye harcayip goruen `content`i BOS birakabilir
    (HTTP 200, ama bos yanit - bkz. `raise_if_empty_content`). Bu fonksiyon,
    `configs/config.yaml`daki `extra_body.chat_template_kwargs.enable_thinking:
    false`i HER `/chat/completions` payload'ina TUTARLI sekilde uygulamak
    icin TEK bir yerden paylasilir (`BaseVLM._build_chat_payload`,
    `BaseVLM._build_reconciliation_payload`, `EvrenVLM`nin video istekleri).
    Cekirdek alanlari (model/messages/...) EZMEZ.

    Args:
        payload: Uzerine yazilacak istek govdesi (yerinde degistirilir).
        extra_body: `VLLMEndpointConfig.extra_body`.
    """
    for key, value in (extra_body or {}).items():
        payload.setdefault(key, value)


def raise_if_empty_content(raw_content: str, source: str, response_data: Dict[str, Any]) -> None:
    """VLM'in ham cevabi bos/yalnizca boslukdan ibaretse, sessizce devam etmek yerine ACIKCA hata firlatir.

    Mentor eleştirisi 5: "Testlerde sistemin sessizce (HTTP 200 donerek ama
    bos icerikle) cokmesi projenin elenmesine neden olabilir." `apply_extra_body`
    bu durumun EN YAYGIN KOK NEDENINI (dusunme modunun acik kalmasi) giderir;
    ancak baska bir nedenle (ag/model hatasi, beklenmedik `finish_reason`)
    yine de bos icerik gelirse, bu SESSIZCE bos bir `VLMResponse.description`/
    cevaba donusup ilerideki asamalarda "hicbir sey gozlemlenmedi" gibi
    YANLIS yorumlanmasin diye burada ACIKCA (RuntimeError ile) yakalanir -
    cagiran taraf (`_post_chat_completion`/`EvrenVLM`) bunu zaten bilinen
    "basarisiz VLM cagrisi" yoluna (degraded rapor/parca basarisizligi/retry)
    yonlendirir; risk UYDURULMAZ, hata GIZLENMEZ.

    Args:
        raw_content: `data["choices"][0]["message"]["content"]` (strip edilmemis olabilir).
        source: Teshis icin loglanacak kimlik (model adi veya video yolu).
        response_data: Ham VLM yaniti (teshis: `finish_reason`/`usage` var mi loglanir).

    Raises:
        RuntimeError: `raw_content` bos/yalnizca boslukdan olusuyorsa.
    """
    if raw_content and raw_content.strip():
        return
    try:
        finish_reason = response_data["choices"][0].get("finish_reason")
    except (KeyError, IndexError, AttributeError):
        finish_reason = None
    usage = response_data.get("usage") if isinstance(response_data, dict) else None
    logger.error(
        "VLM bos icerik dondurdu (HTTP 200, ama content bos) - olasi neden: 'dusunme' "
        "modu tum max_tokens butcesini tuketti. kaynak=%s finish_reason=%s usage=%s",
        source,
        finish_reason,
        usage,
    )
    raise RuntimeError(
        f"VLM bos yanit dondurdu (kaynak={source}, finish_reason={finish_reason}, usage={usage}) - "
        "olasi neden: 'enable_thinking' acik kalip max_tokens butcesi gizli akil yurutmeye tuketildi."
    )


@dataclass
class VLMResponse:
    """Bir VLM cagrisinin (tek bir istek/batch veya reconciliation icin) standardize edilmis ciktisi."""

    description: str
    model_name: str
    frame_count: int
    latency_ms: float
    structured_events: List[Dict[str, Any]] = field(default_factory=list)
    """Modelin dogrudan urettigi tipli olaylar (bkz. `EVENTS_JSON` blogu):
    her biri `{"event_id", "type", "start_time", "end_time", "evidence_ids",
    "description", "risk_score", "confidence"}`. Bos ise `EventEngine`
    anahtar-kelime fallback'ine duser (bkz. `event_engine.detect`)."""
    status: str = "completed"
    """Bu VLM cagrisinin durumu: `"completed"` (basarili), `"failed"`
    (bu batch icin VLM cagrisi basarisiz oldu - description'da
    `[ANALYSIS_FAILED]` notu bulunur) veya (yalnizca batch'leri birlestiren
    agrege yanitlarda) `"partial_failure"` (bazi batch'ler basarili, bazilari
    basarisiz). Asla `risk=0`/basarili gibi yorumlanmamalidir."""
    evidence_ids: List[str] = field(default_factory=list)
    """Bu yanitin kapsadigi (gonderilen) `EvidenceFrame.evidence_id` degerleri
    (bkz. `analyze_evidence_batched`); tek-cagri (eski/agrege) yanitlarda bos olabilir."""


# VLM ciktisinin sonundaki makine-okunur olay blogunu yakalar:
#   EVENTS_JSON: [ {...}, {...} ]
_EVENTS_JSON_PATTERN = re.compile(r"EVENTS_JSON:\s*(\[.*\])", re.DOTALL | re.IGNORECASE)


def parse_structured_events(content: str) -> Tuple[str, List[Dict[str, Any]]]:
    """VLM metninden `EVENTS_JSON` blogunu ayristirir ve insan-okur metinden ayirir.

    Model, insan-okur gozlem bloklarindan sonra `EVENTS_JSON: [...]` satiri
    ekler. Bu fonksiyon o JSON dizisini ayristirip dondurur ve blogu
    aciklamadan temizler; boylece rapor/panelde yalnizca temiz gozlem metni
    kalir. Blok yoksa veya JSON gecersizse, aciklama oldugu gibi kalir ve bos
    liste doner (EventEngine anahtar-kelime fallback'ine gecer).

    Args:
        content: Modelin ham metin ciktisi.

    Returns:
        `(temiz_aciklama, structured_events)` ikilisi.
    """
    match = _EVENTS_JSON_PATTERN.search(content)
    if not match:
        return content.strip(), []

    events = _loads_events_lenient(match.group(1))
    if not events:
        logger.warning("VLM EVENTS_JSON blogu ayristirilamadi/bos, anahtar-kelime fallback'ine dusulecek.")

    clean_description = content[: match.start()].strip()
    return (clean_description or content.strip()), events


def _loads_events_lenient(raw_block: str) -> List[Dict[str, Any]]:
    """`EVENTS_JSON` dizisini toleransli sekilde ayristirir (kucuk modellerin yaygin hatalarina karsi).

    Once ham metin `json.loads` ile denenir; basarisiz olursa yaygin bir hata
    olan "sondaki virgul" (`,]` / `,}`) temizlenip yeniden denenir. Yalnizca
    sozluk (dict) elemanlar dondurulur; hicbir gecerli JSON elde edilemezse bos
    liste doner (EventEngine anahtar-kelime fallback'ine gecer).

    Args:
        raw_block: `EVENTS_JSON:` isaretcisinden sonra yakalanan `[...]` metni.

    Returns:
        Sozluk elemanlardan olusan liste veya bos liste.
    """
    candidates = [raw_block, re.sub(r",(\s*[\]}])", r"\1", raw_block)]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


class BaseVLM(ABC):
    """Butun Gorsel Dil Modeli (VLM) entegrasyonlari icin soyut taban sinif.

    Somut alt siniflar (`QwenVLM`, `GemmaVLM`), yerel vLLM servisine HTTP
    uzerinden baglanarak `analyze_evidence` metodunu implemente etmelidir. Bu
    soyutlama, ajan/muhakeme katmaninin hangi VLM'in aktif oldugundan bagimsiz
    calismasini saglar.
    """

    def __init__(self, endpoint: VLLMEndpointConfig) -> None:
        """BaseVLM'i vLLM baglanti bilgileriyle baslatir.

        Args:
            endpoint: Bu modelin vLLM servis adresini ve uretim parametrelerini
                tasiyan konfigurasyon nesnesi.
        """
        self._endpoint = endpoint

    @property
    def base_url(self) -> str:
        """Bu modelin OpenAI-uyumlu servisinin taban URL'sini dondurur (yerel vLLM veya harici saglayici)."""
        return self._endpoint.resolved_base_url()

    @property
    def model_name(self) -> str:
        """Bu VLM icin yapilandirilmis Hugging Face model adini dondurur."""
        return self._endpoint.model_name

    @abstractmethod
    def analyze_evidence(
        self, evidence_frames: List[EvidenceFrame], prompt: str
    ) -> VLMResponse:
        """Kronolojik evidence karelerini analiz edip olay kumeleme + dogal dil aciklama uretir.

        Args:
            evidence_frames: `AdaptiveFrameSampler.process_video` tarafindan
                uretilen, zaman sirali (kronolojik) evidence kareleri (bir
                batch'in tamami veya alt kumesi olabilir).
            prompt: Modelin odaklanmasi istenen olay/soruyu tanimlayan istem.

        Returns:
            Kumelenmis olaylari (`EVENTS_JSON`) ve dogal dile yakin
            aciklamayi iceren `VLMResponse`.

        Raises:
            RuntimeError: vLLM servisine erisilemezse veya yanit gecersizse.
        """
        raise NotImplementedError

    def analyze_evidence_batched(
        self, evidence_frames: List[EvidenceFrame], prompt: str, batch_size: int = 40
    ) -> List[VLMResponse]:
        """Evidence karelerini TEK bir dev payload yerine kronolojik, kayipsiz batch'ler halinde analiz eder.

        `evidence_frames`, `batch_size` buyuklugunde ARDISIK (kronolojik)
        gruplara bolunur; batch siniri bir OLAY SINIRI DEGILDIR - tek bir
        gercek olay iki batch'e yayilabilir (bu durum `reconcile_events` ile
        cozulur). Her batch icin `analyze_evidence` BAGIMSIZ olarak cagrilir.
        Bir batch'in VLM cagrisi basarisiz olursa (ag hatasi, gecersiz yanit,
        vb.) o batch icin `status="failed"` ve `[ANALYSIS_FAILED]` ile
        baslayan bir `VLMResponse` uretilir; ISTISNA DIGER batch'lere
        YAYILMAZ (bir batch'in basarisiz olmasi digerlerinin sonucunu
        KAYBETMEZ). Hicbir evidence karesi bu asamada SILINMEZ; basarisiz
        bir batch'in evidence_id'leri `VLMResponse.evidence_ids`de aynen
        korunur (cagiran taraf bunlari `unassigned` olarak isaretleyebilir).

        Args:
            evidence_frames: Analiz edilecek TUM evidence kareleri (video
                geneli, kronolojik, kayipsiz).
            prompt: Kullanici/istem metni (her batch'e aynen iletilir).
            batch_size: Bir VLM istegine dahil edilecek azami evidence karesi
                sayisi.

        Returns:
            Kronolojik sirada, her biri kendi `evidence_ids` alaniyla hangi
            evidence karelerini kapsadigini belirten `VLMResponse` listesi.
            Bos `evidence_frames` icin bos liste doner.
        """
        if not evidence_frames:
            return []

        step = max(1, batch_size)
        responses: List[VLMResponse] = []
        for start in range(0, len(evidence_frames), step):
            batch = evidence_frames[start : start + step]
            batch_evidence_ids = [ef.evidence_id for ef in batch]
            try:
                response = self.analyze_evidence(batch, prompt)
                response.evidence_ids = batch_evidence_ids
                response.status = "completed"
            except Exception as exc:  # noqa: BLE001 - izole edilip diger batch'lere yayilmaz
                logger.exception(
                    "VLM batch analizi basarisiz (evidence=%s, model=%s); diger batch'ler etkilenmeyecek.",
                    batch_evidence_ids,
                    self.model_name,
                )
                response = VLMResponse(
                    description=(
                        f"[ANALYSIS_FAILED] Evidence {batch_evidence_ids} icin VLM analizi basarisiz: {exc}"
                    ),
                    model_name=self.model_name,
                    frame_count=len(batch),
                    latency_ms=0.0,
                    structured_events=[],
                    status="failed",
                    evidence_ids=batch_evidence_ids,
                )
            responses.append(response)
        return responses

    def reconcile_events(self, batch_responses: List[VLMResponse], prompt: str) -> VLMResponse:
        """Batch-yerel olaylari IKINCI, metin-tabanli bir VLM cagrisiyla TEK global olay listesine birlestirir.

        Yalnizca `analyze_evidence_batched` BIRDEN FAZLA batch urettiginde
        gereklidir (tek batch zaten globaldir). Basarisiz batch'ler
        reconciliation girdisine DAHIL EDILMEZ (analiz edilememis, dolayisiyla
        birlestirilecek bir olay bilgisi yok) ama evidence_id'leri
        `VLMResponse.evidence_ids`de KORUNUR; cagiran taraf (bkz.
        `src/main.py::_reconcile_unassigned_evidence`) bunlari acikca
        `unassigned` olarak isaretler - evidence hicbir zaman sessizce
        kaybolmaz.

        Args:
            batch_responses: `analyze_evidence_batched` ciktisi (birden fazla
                batch, karisik basarili/basarisiz olabilir).
            prompt: Kullanici/istem metni (baglam icin reconciliation
                promptuna eklenir).

        Returns:
            Tum basarili batch'lerin evidence_id'lerini kapsayan, GLOBAL
            `event_id`li tek bir `VLMResponse`. Basarili batch yoksa
            `status="failed"` doner.

        Raises:
            RuntimeError: Reconciliation VLM cagrisi basarisiz olursa.
        """
        succeeded = [r for r in batch_responses if r.status != "failed"]
        if not succeeded:
            all_ids = [eid for r in batch_responses for eid in r.evidence_ids]
            return VLMResponse(
                description="[HATA] Hicbir batch basariyla analiz edilemedi; birlestirilecek olay yok.",
                model_name=self.model_name,
                frame_count=0,
                latency_ms=0.0,
                structured_events=[],
                status="failed",
                evidence_ids=all_ids,
            )
        if len(succeeded) == 1 and len(batch_responses) == 1:
            return succeeded[0]

        payload = self._build_reconciliation_payload(batch_responses, prompt)
        response = self._post_chat_completion(payload)
        response.evidence_ids = [eid for r in batch_responses for eid in r.evidence_ids]
        response.frame_count = sum(r.frame_count for r in batch_responses)
        return response

    def _build_reconciliation_payload(
        self, batch_responses: List[VLMResponse], prompt: str
    ) -> Dict[str, Any]:
        """Reconciliation icin METIN-TABANLI (goruntusuz) bir chat payload'i kurar.

        Her batch'in `structured_events`ini (zaten ayristirilmis EVENTS_JSON)
        JSON metni olarak reconciliation promptuna gomer; basarisiz
        batch'lerin evidence_id'lerini de ayri bir listede belirtir (boylece
        model bunlari da `unassigned` kaydina dahil edebilir).

        Args:
            batch_responses: `analyze_evidence_batched` ciktisi.
            prompt: Kullanici/istem metni (baglam icin eklenir).

        Returns:
            `/v1/chat/completions` icin JSON-serilestirilebilir, YALNIZCA
            METIN iceren istek govdesi (resim yok).
        """
        batch_summaries = [
            {"batch_index": i, "events": r.structured_events}
            for i, r in enumerate(batch_responses)
            if r.status != "failed"
        ]
        failed_evidence_ids = [eid for r in batch_responses if r.status == "failed" for eid in r.evidence_ids]

        text = (
            f"Kullanici istemi (baglam): {prompt}\n\n"
            f"Batch-yerel olay listeleri (JSON):\n{json.dumps(batch_summaries, ensure_ascii=False, indent=2)}\n\n"
            f"Analiz EDILEMEYEN (basarisiz batch) evidence_id'leri (bunlari da "
            f"'unassigned' kaydina ekle, KAYBETME): {json.dumps(failed_evidence_ids)}"
        )
        content = [
            {"type": "text", "text": VLM_RECONCILIATION_SYSTEM_PROMPT},
            {"type": "text", "text": text},
        ]
        payload: Dict[str, Any] = {
            "model": self._endpoint.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self._endpoint.max_new_tokens,
            "temperature": self._endpoint.temperature,
            "top_p": self._endpoint.top_p,
        }
        apply_extra_body(payload, self._endpoint.extra_body)
        return payload

    def answer_video_question(self, video_source: str, question: str, analysis_summary: str) -> str:
        """Ayni videoya, kalici raporun onceki analizinden SONRA yeni bir soru sorar.

        Varsayilan olarak DESTEKLENMEZ - yalnizca `EvrenVLM` (video-tabanli,
        prefix-cache avantajina sahip tek saglayici, bkz. o modulun
        dokustringi) bunu implemente eder. `QwenVLM`/`GemmaVLM`/mock
        istemciler icin cagiran taraf (`AskService`) bu istisnayi YAKALAYIP
        metin-tabanli soru-cevap akisina SESSIZCE geri doner - bu asla
        kullaniciya gorunen bir hataya donusmez.

        Args:
            video_source: Daha once analiz edilmis videonun yerel dosya yolu.
            question: Kullanicinin bu video hakkindaki yeni (takip) sorusu.
            analysis_summary: Videonun daha once uretilmis kisa metin ozeti
                (baglam icindir, KANIT degildir).

        Returns:
            Modelin videoyu dogrudan izleyerek urettigi Turkce serbest-metin cevap.

        Raises:
            NotImplementedError: Bu saglayici video-QA'yi desteklemiyorsa (varsayilan).
        """
        raise NotImplementedError(
            f"{type(self).__name__} video-tabanli takip sorusu (answer_video_question) desteklemiyor."
        )

    @abstractmethod
    def health_check(self) -> bool:
        """Modelin vLLM servisinin ayakta olup olmadigini kontrol eder.

        Returns:
            Servis erisilebilir ve saglikliysa `True`.
        """
        raise NotImplementedError

    def _build_chat_payload(
        self, evidence_frames: List[EvidenceFrame], prompt: str
    ) -> Dict[str, Any]:
        """vLLM'in OpenAI-uyumlu `/chat/completions` uc noktasi icin istek govdesi kurar.

        Icerik bloklari (evidence karelerinin base64 goruntusu + kimlik/zaman
        metadatasi) `VLMPayloadBuilder` tarafindan uretilir; bu metod yalnizca
        model-ozel alanlarla (model adi, sicaklik, token siniri) sarmalar.

        Args:
            evidence_frames: Modele gonderilecek evidence kareleri.
            prompt: Kullanici/istem metni.

        Returns:
            `/v1/chat/completions` icin JSON-serilestirilebilir istek govdesi.
        """
        content = VLMPayloadBuilder.build_content_blocks(evidence_frames, prompt)

        payload: Dict[str, Any] = {
            "model": self._endpoint.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self._endpoint.max_new_tokens,
            "temperature": self._endpoint.temperature,
            "top_p": self._endpoint.top_p,
        }
        apply_extra_body(payload, self._endpoint.extra_body)
        return payload

    def _post_chat_completion(self, payload: Dict[str, Any]) -> VLMResponse:
        """Hazirlanan istegi vLLM servisine gonderir ve yaniti `VLMResponse`'a cevirir.

        Args:
            payload: `_build_chat_payload` (veya `_build_reconciliation_payload`)
                ile uretilmis istek govdesi.

        Returns:
            Model ciktisini iceren `VLMResponse`.

        Raises:
            RuntimeError: HTTP istegi basarisiz olursa veya yanit beklenmedik
                bicimde gelirse.
        """
        started_at = time.perf_counter()
        # Gecici ag hatalarina (baglanti/timeout/5xx) karsi ustel geri-cekilmeli
        # yeniden deneme; bozuk yanit (KeyError/IndexError) yeniden denenmez.
        last_exc: Exception | None = None
        for attempt in range(_MAX_INFERENCE_RETRIES + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._endpoint.auth_headers(),
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                raw_content = data["choices"][0]["message"]["content"]
                raise_if_empty_content(raw_content, self.model_name, data)
                break
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"VLM yaniti beklenmedik bicimde ({self.model_name}): {exc}") from exc
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < _MAX_INFERENCE_RETRIES:
                    backoff = _RETRY_BACKOFF_BASE_SEC * (2**attempt)
                    logger.warning(
                        "VLM cagrisi basarisiz (deneme %d/%d, %s): %s — %.1fs sonra yeniden denenecek",
                        attempt + 1,
                        _MAX_INFERENCE_RETRIES + 1,
                        self.model_name,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
        else:
            raise RuntimeError(
                f"VLM cagrisi {_MAX_INFERENCE_RETRIES + 1} denemede basarisiz ({self.model_name}): {last_exc}"
            ) from last_exc

        # Modelin urettigi makine-okunur EVENTS_JSON blogunu ayristir; insan-okur
        # aciklamadan ayir (bos ise EventEngine anahtar-kelime fallback'ine duser).
        description, structured_events = parse_structured_events(raw_content)

        latency_ms = (time.perf_counter() - started_at) * 1000
        image_count = sum(
            1 for item in payload["messages"][0]["content"] if item["type"] == "image_url"
        )
        return VLMResponse(
            description=description,
            model_name=self.model_name,
            frame_count=image_count,
            latency_ms=latency_ms,
            structured_events=structured_events,
        )

    def health_check_impl(self) -> bool:
        """`/v1/models` uc noktasina istek atarak servisin ayakta olup olmadigini dogrular.

        Returns:
            Servis 200 ile yanit veriyorsa `True`, aksi halde `False`.
        """
        try:
            response = httpx.get(
                f"{self.base_url}/models", headers=self._endpoint.auth_headers(), timeout=5.0
            )
            return response.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("VLM saglik kontrolu basarisiz (%s): %s", self.model_name, exc)
            return False
