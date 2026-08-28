"""03 - Gorsel Dil Modeli Katmani: EVREN (TEKNOFEST yarisma cikarim servisi) implementasyonu.

EVREN'in video-analiz ucu (`model="vlm"`), diger VLM implementasyonlarindan
(`QwenVLM`/`GemmaVLM`) FARKLI olarak ayri evidence kareleri (resim) DEGIL,
DOGRUDAN tek bir video dosyasini (base64, `video_url` icerik blogu) kabul
eder (bkz. katilimci dokumantasyonu SS 7.1). Bu nedenle `EvrenVLM`,
`BaseVLM._build_chat_payload`in frame-tabanli akisini KULLANMAZ;
`analyze_video(video_source, evidence_frames, prompt)` ile videoyu (gerekirse
parcalara bolunerek, bkz. asagida) gonderir - sampler'in urettigi kareler
yalnizca (mock istemciyle PARITE icin) imzada tasinir, gercek saglayicida
KULLANILMAZ.

`analyze_evidence` (frame-tabanli, eski/coklu-goruntu akisi) yalnizca
`BaseVLM` soyut sozlesmesini karsilamak icin tanimlanmistir; production
akisinda (`src/main.py::SafirPipeline.stage_vlm`) CAGRILMAZ - EVREN'in
istek basina en fazla 2 goruntu kabul etmesi nedeniyle (bkz. dokumantasyon
SS 7.5) kare-bolme/coklu-istek deseni kasitli olarak yeniden KURULMAZ.

2026-08-25 ("video cozunurluk zarfi" duzeltmesi): EVREN, gonderilen videonun
TAMAMINA TEK bir toplam piksel butcesi uygular (dokumantasyon) - 180 saniyelik
bir klip ile 60 saniyelik AYNI cozunurlukteki bir klip AYNI oranda
kucultulmez; uzun klip cok daha agresif kucultulur ve detaylar (baret, kucuk
alev/duman baslangici vb.) kaybolabilir. `endpoint.chunk_duration_sec`
config'te tanimliysa (bkz. `configs/config.yaml` -> `vlm.models.evren`),
`analyze_video` videoyu bu sureden uzunsa `src/vlm/video_chunker.py` ile
ardisik parcalara boler, HER parcayi AYRI bir istekte gonderir ve sonuclari
(zaman-damgasi kaydirmasiyla) birlestirir - EVREN dokumantasyonunun "klip
kisa parcalara bolunmeli" onerisini uygular. `None`/`<=0` ise (varsayilan)
davranis DEGISMEZ - video eskisi gibi tek istekte gonderilir.

2026-08-25 (mentor eleştirisi 5, "Dusunme Modu Tuzagi"): bu modulun video
istekleri `BaseVLM._build_chat_payload` (Qwen/Gemma frame-tabanli yolun
kullandigi, `extra_body`/`enable_thinking:false`i ZATEN merge eden ortak
kurucu) KULLANMIYORDU - kendi payload'larini DOGRUDAN kuruyorlardi ve bu
yuzden `enable_thinking: false` HICBIR video isteginde ACIKCA gonderilmiyordu
(tam olarak P0 LLM duzeltmesindeki ayni tuzak, ama VLM tarafinda). Artik
`apply_extra_body` ile HER video istegine (`_send_single_video`,
`answer_video_question`) ayni mekanizma uygulanir; ayrica `raise_if_empty_content`
ile EVREN'in HTTP 200 + bos content donmesi ARTIK SESSIZCE gecmez.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from src.prompts import ASK_VIDEO_SYSTEM_PROMPT, VLM_OBSERVER_SYSTEM_PROMPT, build_ask_video_user_prompt
from src.sampler.schema import EvidenceFrame
from src.vlm.base_vlm import (
    BaseVLM,
    VLMResponse,
    apply_extra_body,
    parse_structured_events,
    raise_if_empty_content,
)
from src.vlm.video_chunker import VideoChunk, cleanup_chunks, split_video_into_chunks

logger = logging.getLogger(__name__)

# EVREN dokumantasyonu (SS 7.1 "Uyari"): sistem video isteklerinde 1800s'ye
# kadar calisabiliyor; istemci zaman asimi da BUNUNLA UYUMLU ayarlanmalidir
# (aksi halde baglanti modelden ONCE kesilir, sonuc GORUNTULENEMEZ).
_EVREN_VIDEO_TIMEOUT_SEC = 1800.0

# Teshis (2026-08-26): tek bir float timeout httpx'te connect/write/read/pool
# asamalarinin HEPSINE ayni deger olarak uygulanir - bu, "EVREN gercekten
# meşgul/yavas" ile "baglanti hic kurulamadi/agda bir sorun var" durumlarini
# AYIRT EDILEMEZ hale getirir (ikisi de operatore ayni "sessiz 30 dakika"
# gibi gorunur). Connect/pool KISA tutulur (baglanti kurulamiyorsa saniyeler
# icinde HATA verir, 30 dakika beklemez); write/read UZUN kalir (buyuk video
# gövdesinin yuklenmesi VE EVREN'in fiilen uzun surebilen isleme suresi icin
# - dokumantasyonun izin verdigi tam butce).
_EVREN_VIDEO_TIMEOUT = httpx.Timeout(connect=20.0, write=_EVREN_VIDEO_TIMEOUT_SEC, read=_EVREN_VIDEO_TIMEOUT_SEC, pool=20.0)

_VIDEO_MODE_NOTE = (
    "\n\nNOT (VIDEO-DOGRUDAN MOD): Yukaridaki istemde bahsedilen ayri, "
    "etiketli 'evidence kareleri' bu istekte YOKTUR - sana tek, surekli bir "
    "video dogrudan gonderilmektedir; kare kare ayri zaman etiketi "
    "SAGLANMAMISTIR. `evidence_ids` alanini bu nedenle HER ZAMAN BOS LISTE "
    "([]) birak.\n"
    "`start_time`/`end_time`, videonun basindan (00:00) itibaren SANIYE "
    "CINSINDEN GERCEK zaman damgalari olmalidir - bunlari TAHMIN ETMEK "
    "SENIN gorevindir, sana hazir bir zaman bilgisi VERILMEZ. Bu, bir VLM "
    "icin en hataya acik adimdir; bu yuzden ozellikle DIKKATLI ol:\n"
    "- Zamani, videoyu ZIHNINDE sabit bir hizda oynatiyormus gibi TAKIP ET; "
    "sahnedeki somut degisim anlarini (bir hareketin baslamasi, bir "
    "nesnenin ortaya cikmasi/kaybolmasi, ani bir durum degisikligi) kendi "
    "ic saatinle ILISKILENDIR.\n"
    "- KESIN bir saniyeden EMIN DEGILSEN, TEK bir yanlis saniye vermektense "
    "start_time/end_time araligini 1-2 saniye GENISLETEREK ver - yanlis "
    "kesinlik, dogru ama biraz genis bir araliktan DAHA KOTUDUR.\n"
    "- Video suresini (bilgin varsa) asan veya negatif bir zaman ASLA verme."
)


def _format_mmss(seconds: float) -> str:
    """Saniyeyi `MM:SS` bicimine cevirir (birlestirilmis aciklamada parca zaman etiketi icin)."""
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


# Video analizi sirasinda ADIM ADIM (kullaniciya gosterilebilir) ilerleme
# bildirimi - "video parcalara ayrilirken/gonderilirken hicbir sey
# gorunmuyor" geri bildiriminin dogrudan cevabi. `on_progress`, `stage_vlm`
# (src/main.py) tarafindan pipeline'in GERCEK trace/SSE kancasina baglanir;
# burasi (src/vlm) trace/JobState kavramindan TAMAMEN habersizdir - yalnizca
# duz bir dict callback'i cagirir (ayri katman sorumluluk ayrimi korunur).
# `phase` degerleri: "chunking" (video parcalara bolunuyor) | "chunk_start"
# (bir parca EVREN'e gonderiliyor) | "chunk_done" (parca basarili) |
# "chunk_failed" (parca basarisiz, digerleri devam eder).
VlmProgressCallback = Callable[[Dict[str, Any]], None]


def _report(on_progress: Optional[VlmProgressCallback], **fields: Any) -> None:
    """`on_progress` verilmisse cagirir; verilmemisse sessizce no-op (varsayilan davranis DEGISMEZ)."""
    if on_progress is not None:
        on_progress(fields)


class EvrenVLM(BaseVLM):
    """TEKNOFEST EVREN yarisma servisinin video-tabanli VLM ucunu kullanan implementasyon."""

    # Video DOGRUDAN gonderilir (bkz. `analyze_video`) - Adaptive Frame
    # Sampler'in ciktisi bu saglayicida HICBIR ZAMAN VLM girdisi olarak
    # kullanilmaz (bkz. `BaseVLM.requires_frame_sampling` dokustringi).
    requires_frame_sampling = False

    def analyze_video(
        self,
        video_source: str,
        evidence_frames: List[EvidenceFrame],
        prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
    ) -> VLMResponse:
        """Videoyu EVREN'e gonderip olay kumeleme + Turkce gozlem uretir (gerekirse parcalara bolerek).

        Args:
            video_source: Yerel bir video dosyasinin yolu. RTSP/canli akis
                adresleri DESTEKLENMEZ - EVREN yalnizca base64 kodlanmis bir
                video govdesi kabul eder (bkz. dokumantasyon SS 7.1).
            evidence_frames: Yalnizca `MockVLMClient` ile arayuz PARITESI
                icin alinir; gercek EVREN cagrisinda KULLANILMAZ (EVREN
                videoyu kendisi analiz eder, sampler kareleri gerekmez).
            prompt: Analiz odagini belirten kullanici istemi.
            on_progress: Opsiyonel adim-adim ilerleme bildirimi (bkz. modul
                seviyesi `VlmProgressCallback` dokustringi). `None` ise
                (varsayilan) davranis TAMAMEN degismez.

        Returns:
            EVREN tarafindan uretilen, kumelenmis olaylari ve dogal dil
            gozlemini iceren `VLMResponse` (birden fazla parca varsa
            birlestirilmis/zaman-kaydirmali).

        Raises:
            RuntimeError: `video_source` bir RTSP/canli akis adresiyse, video
                dosyasi okunamazsa/acilamazsa veya HICBIR parca icin EVREN
                cagrisi basarili olmazsa.
        """
        del evidence_frames  # yalnizca arayuz parametresi; bkz. docstring

        lowered = video_source.strip().lower()
        if lowered.startswith(("rtsp://", "http://", "https://")):
            raise RuntimeError(
                f"EVREN VLM yalnizca yerel video dosyalarini destekler (RTSP/canli akis DESTEKLENMEZ): {video_source}"
            )

        chunk_duration_sec = getattr(self._endpoint, "chunk_duration_sec", None) or 0.0
        if chunk_duration_sec <= 0:
            response = self._send_single_video(video_source, prompt, on_progress, 1, 1)
            response.chunking_summary = {
                "total_chunks": 1,
                "encoder": "none",
                "chunk_duration_sec": None,
                "video_duration_sec": None,
                "per_chunk_elapsed_sec": [round(response.latency_ms / 1000, 1)],
            }
            return response

        chunks = split_video_into_chunks(video_source, chunk_duration_sec)
        if len(chunks) == 1:
            response = self._send_single_video(video_source, prompt, on_progress, 1, 1)
            response.chunking_summary = {
                "total_chunks": 1,
                "encoder": chunks[0].encoder,
                "chunk_duration_sec": chunk_duration_sec,
                "video_duration_sec": round(chunks[0].end_offset_sec, 1) or None,
                "per_chunk_elapsed_sec": [round(response.latency_ms / 1000, 1)],
            }
            return response

        _report(on_progress, phase="chunking", total_chunks=len(chunks))
        logger.info(
            "EVREN VLM: video %d parcaya bolundu (chunk_duration_sec=%.1f), her parca AYRI istekte gonderilecek.",
            len(chunks),
            chunk_duration_sec,
        )
        try:
            return self._analyze_video_chunks(chunks, prompt, on_progress)
        finally:
            cleanup_chunks(chunks)

    def _send_single_video(
        self,
        video_path: str,
        prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
        chunk_index: int = 1,
        total_chunks: int = 1,
        range_label: Optional[str] = None,
    ) -> VLMResponse:
        """Tek bir video dosyasini (TAM veya bir parcasini) DOGRUDAN, tek istekte EVREN'e gonderir.

        Args:
            video_path: Gonderilecek `.mp4` dosyasinin yolu (tam video veya
                `video_chunker`in urettigi bir parca).
            prompt: Analiz odagini belirten kullanici istemi.
            on_progress: Bkz. `analyze_video`. `chunk_index`/`total_chunks`/
                `range_label`, bu tekil istegin genel video icindeki yerini
                (kullanici arayuzunde "Parca 1/2" gibi gosterilmek uzere)
                bildirir - tek-istekli (bolunmemis) videoda 1/1'dir.

        Returns:
            Bu tekil istegin `VLMResponse`i - `structured_events`teki
            `start_time`/`end_time` bu dosyanin KENDI basindan (0) itibaren
            saniyedir (parca ise cagiran taraf, bkz. `_analyze_video_chunks`,
            bunu orijinal videodaki gercek zamana KAYDIRIR).

        Raises:
            RuntimeError: Video dosyasi okunamazsa veya EVREN cagrisi basarisiz olursa.
        """
        try:
            with open(video_path, "rb") as fh:
                video_b64 = base64.b64encode(fh.read()).decode("ascii")
        except OSError as exc:
            raise RuntimeError(f"EVREN icin video dosyasi okunamadi: {video_path} ({exc})") from exc

        full_prompt = f"{VLM_OBSERVER_SYSTEM_PROMPT}\n\nEk istem: {prompt}{_VIDEO_MODE_NOTE}".strip()
        payload = {
            "model": self._endpoint.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                    ],
                }
            ],
            "max_tokens": self._endpoint.max_new_tokens,
            "temperature": self._endpoint.temperature,
        }
        apply_extra_body(payload, self._endpoint.extra_body)
        video_b64_mb = len(video_b64) / (1024 * 1024)
        logger.info(
            "EVREN VLM video cagrisi yapiliyor: video=%s model=%s video_b64_mb=%.1f",
            video_path,
            self.model_name,
            video_b64_mb,
        )
        _report(
            on_progress,
            phase="chunk_start",
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            range_label=range_label,
            video_mb=round(video_b64_mb, 1),
        )

        started_at = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._endpoint.auth_headers(),
                timeout=_EVREN_VIDEO_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            elapsed = time.perf_counter() - started_at
            logger.error(
                "EVREN VLM video cagrisi basarisiz (video=%s, %.1fs sonra): %s",
                video_path,
                elapsed,
                exc,
            )
            _report(
                on_progress,
                phase="chunk_failed",
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                range_label=range_label,
                elapsed_sec=round(elapsed, 1),
                error=str(exc),
            )
            raise
        elapsed = time.perf_counter() - started_at
        logger.info("EVREN VLM video cagrisi tamamlandi: video=%s sure=%.1fs", video_path, elapsed)
        _report(
            on_progress,
            phase="chunk_done",
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            range_label=range_label,
            elapsed_sec=round(elapsed, 1),
        )
        response.raise_for_status()
        data = response.json()
        try:
            raw_content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"EVREN yaniti beklenmedik bicimde: {exc}") from exc

        raise_if_empty_content(raw_content, video_path, data)
        description, structured_events = parse_structured_events(raw_content)
        latency_ms = (time.perf_counter() - started_at) * 1000
        return VLMResponse(
            description=description,
            model_name=self.model_name,
            frame_count=0,
            latency_ms=latency_ms,
            structured_events=structured_events,
        )

    def _analyze_video_chunks(
        self, chunks: List[VideoChunk], prompt: str, on_progress: Optional[VlmProgressCallback] = None
    ) -> VLMResponse:
        """Her video parcasini SIRAYLA, AYRI bir istekte gonderir ve sonuclari zaman-kaydirmasiyla birlestirir.

        Parcalar SIRAYLA (paralel DEGIL) gonderilir - EVREN uzerinde ayni
        anda birden fazla agir video istegi acmamak icin (rate-limit/kota
        riski, bkz. dokumantasyon SS 7.1 "1800s'ye kadar surebilir" notu).
        Bir parcanin istegi basarisiz olursa DIGER parcalar ETKILENMEZ - o
        parca icin `[ANALYSIS_FAILED]` notlu bir aciklama eklenir ve
        `structured_events`i BOS sayilir; TUM parcalar basarisiz olursa
        `RuntimeError` firlatilir (cagiran `stage_vlm` bunu degraded rapora cevirir).

        Args:
            chunks: `split_video_into_chunks` ciktisi (>= 2 eleman).
            prompt: Kullanici istemi (her parcaya AYNEN iletilir).

        Returns:
            Tum parcalarin `structured_events`ini (orijinal videodaki GERCEK
            zamana kaydirilmis) ve zaman-etiketli birlestirilmis aciklamayi
            iceren tek bir `VLMResponse`.

        Raises:
            RuntimeError: HICBIR parca basarili olmazsa.
        """
        merged_events: List[Dict[str, Any]] = []
        description_parts: List[str] = []
        total_latency_ms = 0.0
        succeeded_count = 0
        per_chunk_elapsed_sec: List[Optional[float]] = []

        for chunk in chunks:
            label = f"[{_format_mmss(chunk.start_offset_sec)}-{_format_mmss(chunk.end_offset_sec)}]"
            try:
                response = self._send_single_video(
                    chunk.path,
                    prompt,
                    on_progress=on_progress,
                    chunk_index=chunk.index + 1,
                    total_chunks=len(chunks),
                    range_label=label,
                )
            except Exception as exc:  # noqa: BLE001 - bir parcanin hatasi digerlerine YAYILMAZ
                logger.exception(
                    "EVREN VLM: video parcasi basarisiz (index=%d, %s); diger parcalar etkilenmeyecek.",
                    chunk.index,
                    label,
                )
                description_parts.append(f"{label} [ANALYSIS_FAILED] Bu parca icin VLM analizi basarisiz: {exc}")
                per_chunk_elapsed_sec.append(None)
                continue

            succeeded_count += 1
            total_latency_ms += response.latency_ms
            per_chunk_elapsed_sec.append(round(response.latency_ms / 1000, 1))
            description_parts.append(f"{label} {response.description}".strip())
            for event in response.structured_events:
                shifted = dict(event)
                for key in ("start_time", "end_time"):
                    value = shifted.get(key)
                    if isinstance(value, (int, float)):
                        shifted[key] = value + chunk.start_offset_sec
                merged_events.append(shifted)

        if succeeded_count == 0:
            raise RuntimeError(
                f"EVREN VLM: {len(chunks)} video parcasinin HICBIRI basarili olmadi (bkz. loglar)."
            )

        merged_events.sort(key=lambda e: e.get("start_time") or 0.0)
        return VLMResponse(
            description="\n".join(description_parts).strip(),
            model_name=self.model_name,
            frame_count=0,
            latency_ms=total_latency_ms,
            structured_events=merged_events,
            chunking_summary={
                "total_chunks": len(chunks),
                "encoder": chunks[0].encoder,
                "chunk_duration_sec": chunks[0].end_offset_sec - chunks[0].start_offset_sec,
                "video_duration_sec": round(chunks[-1].end_offset_sec, 1),
                "per_chunk_elapsed_sec": per_chunk_elapsed_sec,
            },
        )

    def answer_video_question(self, video_source: str, question: str, analysis_summary: str) -> str:
        """Ayni video dosyasini TEKRAR EVREN'e gonderip yeni bir soruyu dogrudan yanitlar.

        Mentor eleştirisi ("VLM Onbellegi/Prefix Caching Avantaji", bkz.
        `src/prompts/ask_video_prompts.py` modul dokustringi): EVREN
        dokumantasyonuna gore AYNI video icin ilk sorgu ~17.8s, sonraki
        sorgular ~3.7s suruyor - bu, sunucu tarafi otomatik prefix-cache'e
        isaret eder. Bu metod, ozel bir cache-key/session API'si UYDURMADAN,
        en dogrudan ve savunulabilir yolu izler: videoyu OLDUGU GIBI (ayni
        byte'lar) tekrar gonderir; EVREN sunucusu bu tekrari kendi tespit
        ederse hizlanma DOGAL olarak gerceklesir.

        Args:
            video_source: Daha once analiz edilmis videonun yerel dosya yolu.
            question: Kullanicinin bu video hakkindaki yeni sorusu.
            analysis_summary: Videonun daha once uretilmis kisa metin ozeti.

        Returns:
            Modelin ham (yapisal olmayan) Turkce serbest-metin cevabi.

        Raises:
            RuntimeError: `video_source` bir RTSP/canli akis adresiyse, video
                dosyasi okunamazsa/acilamazsa veya EVREN cagrisi basarisiz olursa.
        """
        lowered = video_source.strip().lower()
        if lowered.startswith(("rtsp://", "http://", "https://")):
            raise RuntimeError(
                f"EVREN video-QA yalnizca yerel video dosyalarini destekler (RTSP/canli akis DESTEKLENMEZ): {video_source}"
            )

        try:
            with open(video_source, "rb") as fh:
                video_b64 = base64.b64encode(fh.read()).decode("ascii")
        except OSError as exc:
            raise RuntimeError(f"EVREN icin video dosyasi okunamadi: {video_source} ({exc})") from exc

        user_text = build_ask_video_user_prompt(question, analysis_summary)
        payload = {
            "model": self._endpoint.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{ASK_VIDEO_SYSTEM_PROMPT}\n\n{user_text}"},
                        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                    ],
                }
            ],
            "max_tokens": self._endpoint.max_new_tokens,
            "temperature": self._endpoint.temperature,
        }
        apply_extra_body(payload, self._endpoint.extra_body)
        logger.info("EVREN video-QA cagrisi yapiliyor: video=%s model=%s", video_source, self.model_name)

        started_at = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._endpoint.auth_headers(),
                timeout=_EVREN_VIDEO_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            elapsed = time.perf_counter() - started_at
            logger.error(
                "EVREN video-QA cagrisi basarisiz (video=%s, %.1fs sonra): %s", video_source, elapsed, exc
            )
            raise
        logger.info(
            "EVREN video-QA cagrisi tamamlandi: video=%s sure=%.1fs",
            video_source,
            time.perf_counter() - started_at,
        )
        response.raise_for_status()
        data = response.json()
        try:
            raw_content = str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"EVREN yaniti beklenmedik bicimde: {exc}") from exc

        raise_if_empty_content(raw_content, video_source, data)
        return raw_content

    def analyze_evidence(self, evidence_frames: List[EvidenceFrame], prompt: str) -> VLMResponse:
        """Bu saglayicida DESTEKLENMEZ - production akisi `analyze_video` kullanir.

        Raises:
            NotImplementedError: her zaman - EVREN'de frame-tabanli/coklu-
                istek analiz kasitli olarak yeniden kurulmamistir (gorev
                kapsami: sampler'in VLM'e frame beslemesi kaldirildi).
        """
        raise NotImplementedError(
            "EvrenVLM frame-tabanli analyze_evidence'i desteklemez; production akisi "
            "SafirPipeline.stage_vlm uzerinden analyze_video(video_source, evidence_frames, prompt) kullanir."
        )

    def health_check(self) -> bool:
        """EVREN uc noktasinin (models listesi) erisilebilir olup olmadigini kontrol eder."""
        return self.health_check_impl()


# EVREN dokumantasyonu SS 5/7.3/7.5: "vlm" modeli GORUNTU KABUL ETMEZ (goruntu
# gonderilirse HTTP 400 "At most 0 image(s) may be provided"); goruntu/kare
# girdisi icin "llm-fast"/"llm-large" onerilir, ve bu modeller istek basina EN
# FAZLA 2 goruntu kabul eder. `EvrenVLM.analyze_video` (yukarida) videoyu
# DOGRUDAN "vlm"e gonderen ana (varsayilan) yoldur; asagidaki `EvrenFramesVLM`
# bunun YANINDA duran, AYRI bir aktif-model secenegidir (`configs/config.yaml`
# -> `vlm.active_model: "evren_frames"`) - "dusuk butceli" (kare-tabanli,
# Adaptive Frame Sampler ciktisini KULLANAN) analiz katmani icindir; video-
# dogrudan yol (`vlm.active_model: "evren"`) bundan HICBIR SEKILDE ETKILENMEZ.
_EVREN_FRAMES_MODEL_NAME = "llm-large"
_EVREN_FRAMES_MAX_IMAGES_PER_REQUEST = 2


class EvrenFramesVLM(BaseVLM):
    """EVREN'in kare-tabanli (dusuk butceli) yolu: Adaptive Frame Sampler'in urettigi

    evidence karelerini (goruntu), istek basina EN FAZLA 2 kareyle "llm-large"
    modeline gonderir (bkz. modul-seviyesi NOT, dokumantasyon SS 7.5 tablosu -
    goruntu/kare girdisi + "en fazla 2 goruntu" siniri ACIKCA "llm-large" icin
    olculmustur). `EvrenVLM` (video-dogrudan, model="vlm") ile PAYLASILAN kod
    yalnizca ortak HTTP/JSON-ayristirma altyapisidir (`BaseVLM.
    _build_chat_payload`/`_post_chat_completion` - `QwenVLM`/`GemmaVLM` ile
    AYNI ortak yol); video-ozel metotlar (`analyze_video`/
    `answer_video_question`) burada KASITLI olarak IMPLEMENTE EDILMEZ - bu
    sinifin evidence_frames disinda bir video baglami yoktur.
    """

    requires_frame_sampling = True

    def analyze_evidence(self, evidence_frames: List[EvidenceFrame], prompt: str) -> VLMResponse:
        """Verilen (en fazla 2 kareli) batch'i TEK istekte "llm-large"a gonderir.

        Cagiran taraf (`SafirPipeline._stage_vlm_frames` ->
        `BaseVLM.analyze_evidence_batched`) `configs/config.yaml ->
        vlm.batch_size` (2 olmalidir) buyuklugunde batch'ler halinde cagirir;
        bu metot kendisi de boyutu DOGRULAR - yanlislikla daha buyuk bir
        batch verilirse (ör. `vlm.batch_size` yanlis ayarlanmissa) SESSIZCE
        kirpmak yerine ACIKCA hata verir (EVREN dogrudan HTTP 400 donerdi;
        burada daha erken ve daha acik bir hata tercih edilir).

        Args:
            evidence_frames: Bu istege dahil edilecek evidence kareleri (1-2 adet).
            prompt: Kullanici/istem metni.

        Returns:
            `VLMResponse` (kumelenmis olaylar + dogal dil aciklama).

        Raises:
            ValueError: `evidence_frames` bos veya 2'den fazla kare icerirse.
            RuntimeError: EVREN cagrisi basarisiz olursa veya yanit gecersizse
                (bkz. `BaseVLM._post_chat_completion`).
        """
        if not evidence_frames:
            raise ValueError("EvrenFramesVLM.analyze_evidence: evidence_frames bos olamaz.")
        if len(evidence_frames) > _EVREN_FRAMES_MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"EvrenFramesVLM: istek basina en fazla {_EVREN_FRAMES_MAX_IMAGES_PER_REQUEST} "
                f"goruntu kabul edilir (EVREN dokumantasyonu SS 7.5); {len(evidence_frames)} kare "
                f"verildi. configs/config.yaml -> vlm.batch_size <= "
                f"{_EVREN_FRAMES_MAX_IMAGES_PER_REQUEST} olmalidir."
            )
        payload = self._build_chat_payload(evidence_frames, prompt)
        payload["model"] = _EVREN_FRAMES_MODEL_NAME
        return self._post_chat_completion(payload)

    def health_check(self) -> bool:
        """EVREN uc noktasinin (models listesi) erisilebilir olup olmadigini kontrol eder."""
        return self.health_check_impl()
