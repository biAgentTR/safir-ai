"""03 - Gorsel Dil Modeli Katmani: Google Gemini implementasyonu (video + kare tabanli).

NEDEN AYRI BIR MODUL (ve neden `EvrenVLM`in transport'u yeniden kullanilamaz)
----------------------------------------------------------------------------
`EvrenVLM`, videoyu OpenAI-uyumlu `/chat/completions` ucuna `{"type":
"video_url"}` icerik blogu ile gonderir. Gemini'nin OpenAI-uyumlu uyumluluk
katmani (`/v1beta/openai/chat/completions`) GORUNTU kabul eder ama VIDEO
KABUL ETMEZ - `video_url` diye bir icerik tipi YOKTUR. Bu yuzden video-
dogrudan yol, Gemini'nin KENDI (native) `:generateContent` ucunu kullanmak
ZORUNDADIR. Kare-tabanli yol ise tamamen OpenAI-uyumludur ve `BaseVLM`in
ortak `_build_chat_payload`/`_post_chat_completion` akisini (retry + typed
parser + zaman normalizasyonu dahil) OLDUGU GIBI yeniden kullanir.

Bu modul, `EvrenVLM`in video ORKESTRASYONUNU (parcalama/chunking, zaman-
damgasi kaydirmasi, parca basarisizliginda digerlerinin devam etmesi,
`on_progress` bildirimleri) yeniden YAZMAZ - `GeminiVLM`, `EvrenVLM`den
turer ve yalnizca TEK bir istegin nasil gonderilecegini (`_send_single_video`)
ve video-QA cagrisini (`answer_video_question`) override eder. Boylece
parcalama/birlestirme mantigi TEK bir yerde kalir.

Inline vs Files API
-------------------
Gemini, `:generateContent` istek GOVDESININ tamami icin ~20 MB'lik bir sinir
uygular. Bu sinirin ALTINDA kalan videolar `inline_data` (base64) ile TEK
istekte gonderilir; UZERINDE kalanlar once Files API'ye (resumable upload)
yuklenip `file_data.file_uri` ile referans verilir. Esik, base64'un ~4/3
sisirme oranini hesaba katacak sekilde HAM BAYT uzerinden secilmistir
(bkz. `_INLINE_MAX_BYTES`).
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlsplit

import httpx

from src.prompts import ASK_VIDEO_SYSTEM_PROMPT, VLM_OBSERVER_SYSTEM_PROMPT, build_ask_video_user_prompt
from src.sampler.schema import EvidenceFrame
from src.vlm.base_vlm import BaseVLM, VLMResponse, raise_if_empty_content, send_with_retry
from src.vlm.evren_vlm import _VIDEO_MODE_NOTE, EvrenVLM, VlmProgressCallback, _report

logger = logging.getLogger(__name__)

# Gemini video isteklerinin tamamlanmasi uzun surebilir (uzun klip + sunucu
# kuyrugu). `EvrenVLM`deki ayrimin AYNISI: connect/pool KISA (ag/DNS sorunu
# saniyeler icinde hata versin, dakikalarca sessiz kalmasin), write/read UZUN
# (buyuk govdenin yuklenmesi + modelin fiilen isleme suresi).
_GEMINI_VIDEO_TIMEOUT_SEC = 900.0
_GEMINI_VIDEO_TIMEOUT = httpx.Timeout(
    connect=20.0, write=_GEMINI_VIDEO_TIMEOUT_SEC, read=_GEMINI_VIDEO_TIMEOUT_SEC, pool=20.0
)

# `:generateContent` istek govdesi ~20 MB ile sinirlidir ve base64 kodlama
# ham boyutu ~4/3 sisirir. 12 MB ham (~16 MB base64) guvenli bir esiktir;
# ustundeki her video Files API'ye yuklenir.
_INLINE_MAX_BYTES = 12 * 1024 * 1024

# Files API'ye yuklenen bir video ACTIVE olana kadar beklenir (Gemini video
# dosyalarini once kendi tarafinda isler). Bu sure asilirsa hata verilir -
# SESSIZCE "video yokmus gibi" devam EDILMEZ.
_FILE_ACTIVATION_TIMEOUT_SEC = 300.0
_FILE_ACTIVATION_POLL_SEC = 2.0


def _api_root(base_url: str) -> str:
    """OpenAI-uyumlu taban adresten Gemini'nin NATIVE API kokunu (scheme://host) turetir.

    Config'te `memory`/`llm`/`vlm.gemini_frames` icin verilen taban adres
    OpenAI-uyumluluk katmanini gosterir (`.../v1beta/openai`). Native
    `:generateContent` ve Files API ise kokte (`https://host`) yasar; bu
    fonksiyon iki adresi TEK bir config alanindan turetir, boylece
    `config.yaml`a ikinci bir taban adres alani eklemek GEREKMEZ.

    Args:
        base_url: `VLLMEndpointConfig.resolved_base_url()` ciktisi.

    Returns:
        `scheme://host[:port]` bicimindeki native API koku (sonda `/` yoktur).

    Raises:
        RuntimeError: `base_url` mutlak bir HTTP(S) adresi degilse.
    """
    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        raise RuntimeError(
            f"Gemini icin gecersiz base_url: {base_url!r} - 'https://host/...' biciminde olmalidir."
        )
    return f"{parts.scheme}://{parts.netloc}"


def _extract_text(data: Dict[str, Any]) -> str:
    """Gemini `:generateContent` yanitindan duz metni toplar.

    Gemini, cevabi tek bir string olarak DEGIL, `candidates[0].content.parts`
    listesindeki `text` parcalarinin birlesimi olarak dondurur. Ayrica istek
    guvenlik filtresine takildiysa `candidates` HIC gelmeyebilir - bu durumda
    bos string donulur ve cagiran taraf `raise_if_empty_content` ile durumu
    ACIK bir hataya cevirir (sessiz "hicbir sey gozlemlenmedi" URETILMEZ).

    Args:
        data: `:generateContent` ham JSON yaniti.

    Returns:
        Birlestirilmis metin (bulunamazsa bos string).
    """
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
    return "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))


class GeminiVLM(EvrenVLM):
    """Videoyu Gemini'nin native `:generateContent` ucuna gonderen VLM implementasyonu.

    `EvrenVLM`den turer: parcalama (chunking), zaman-damgasi kaydirmasi,
    parca-basarisizligi izolasyonu ve `on_progress` bildirimleri AYNEN
    devralinir (bkz. modul dokustringi); yalnizca TEK bir istegin transport'u
    ve video-QA cagrisi bu sinifta yeniden tanimlanir.
    """

    # Video DOGRUDAN gonderilir; Adaptive Frame Sampler bu saglayicida
    # VLM girdisi olarak KULLANILMAZ (bkz. `BaseVLM.requires_frame_sampling`).
    requires_frame_sampling = False

    @property
    def _api_key(self) -> str:
        """Endpoint config'inde tanimli ortam degiskeninden Gemini API anahtarini okur."""
        return self._endpoint.resolved_api_key()

    @property
    def _native_root(self) -> str:
        """Native Gemini API koku (bkz. `_api_root`)."""
        return _api_root(self.base_url)

    def _auth_headers(self) -> Dict[str, str]:
        """Gemini native API'sinin bekledigi anahtar header'ini dondurur.

        Gemini, OpenAI'in `Authorization: Bearer` desenini yalnizca
        uyumluluk katmaninda kabul eder; native uc `x-goog-api-key` bekler.
        """
        return {"x-goog-api-key": self._api_key}

    # ---------------------------------------------------------------- Files API

    def _upload_via_files_api(self, video_path: str) -> str:
        """Videoyu Gemini Files API'ye (resumable protokol) yukler ve ACTIVE olmasini bekler.

        `_INLINE_MAX_BYTES` uzerindeki videolar istek govdesine sigmadigi icin
        once buraya yuklenir; donen `file_uri`, `:generateContent` istegine
        `file_data` blogu olarak eklenir.

        Args:
            video_path: Yuklenecek yerel video dosyasinin yolu.

        Returns:
            `file_data.file_uri` alaninda kullanilacak dosya URI'si.

        Raises:
            RuntimeError: Yukleme basarisiz olursa veya dosya
                `_FILE_ACTIVATION_TIMEOUT_SEC` icinde ACTIVE duruma gecmezse.
        """
        size = os.path.getsize(video_path)
        mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
        display_name = os.path.basename(video_path)

        start_headers = {
            **self._auth_headers(),
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        logger.info(
            "Gemini Files API: yukleme baslatiliyor (video=%s, %.1f MB, mime=%s)",
            video_path,
            size / (1024 * 1024),
            mime_type,
        )
        with httpx.Client(timeout=_GEMINI_VIDEO_TIMEOUT) as client:
            # Yukleme adimlarinin HEPSI ortak yeniden-deneme politikasindan
            # gecer: buyuk bir videonun yuklenmesi, tek bir gecici ag hatasi
            # yuzunden TUM analizi dusurmemelidir (bkz. `send_with_retry`).
            start = send_with_retry(
                lambda: client.post(
                    f"{self._native_root}/upload/v1beta/files",
                    headers=start_headers,
                    json={"file": {"display_name": display_name}},
                ),
                source=f"{display_name} (files-api:start)",
            )
            upload_url = start.headers.get("X-Goog-Upload-URL") or start.headers.get("x-goog-upload-url")
            if not upload_url:
                raise RuntimeError(
                    "Gemini Files API yukleme adresi (X-Goog-Upload-URL) donmedi - yukleme baslatilamadi."
                )

            # Dosya BIR KEZ okunur: yeniden denemede tekrar disk okumasi
            # yapilmaz ve her denemede AYNI govde gonderilir.
            with open(video_path, "rb") as fh:
                video_bytes = fh.read()
            upload = send_with_retry(
                lambda: client.post(
                    upload_url,
                    headers={
                        "Content-Length": str(size),
                        "X-Goog-Upload-Offset": "0",
                        "X-Goog-Upload-Command": "upload, finalize",
                    },
                    content=video_bytes,
                ),
                source=f"{display_name} (files-api:upload)",
            )
            file_info = (upload.json() or {}).get("file") or {}
            file_uri = file_info.get("uri")
            file_name = file_info.get("name")
            if not file_uri or not file_name:
                raise RuntimeError(f"Gemini Files API beklenmedik yanit dondurdu: {upload.text[:300]}")

            # Gemini videoyu once KENDI tarafinda isler; PROCESSING durumundaki
            # bir dosya `:generateContent`e verilirse hata doner. ACTIVE olana
            # kadar beklenir (sinirli sure).
            deadline = time.monotonic() + _FILE_ACTIVATION_TIMEOUT_SEC
            state = file_info.get("state", "PROCESSING")
            while state == "PROCESSING":
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Gemini Files API: '{display_name}' {_FILE_ACTIVATION_TIMEOUT_SEC:.0f}s icinde "
                        "ACTIVE duruma gecmedi (video isleme zaman asimi)."
                    )
                time.sleep(_FILE_ACTIVATION_POLL_SEC)
                probe = send_with_retry(
                    lambda: client.get(
                        f"{self._native_root}/v1beta/{file_name}", headers=self._auth_headers()
                    ),
                    source=f"{display_name} (files-api:poll)",
                )
                state = (probe.json() or {}).get("state", "PROCESSING")

            if state != "ACTIVE":
                raise RuntimeError(f"Gemini Files API: dosya kullanilabilir degil (state={state}).")

        logger.info("Gemini Files API: yukleme tamamlandi (uri=%s)", file_uri)
        return file_uri

    def _video_part(self, video_path: str) -> Dict[str, Any]:
        """Video icin uygun `:generateContent` icerik blogunu (inline veya Files API) uretir.

        Args:
            video_path: Gonderilecek yerel video dosyasinin yolu.

        Returns:
            `inline_data` (kucuk videolar) veya `file_data` (buyuk videolar) blogu.

        Raises:
            RuntimeError: Video dosyasi okunamazsa.
        """
        try:
            size = os.path.getsize(video_path)
        except OSError as exc:
            raise RuntimeError(f"Gemini icin video dosyasi okunamadi: {video_path} ({exc})") from exc

        mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
        if size > _INLINE_MAX_BYTES:
            return {"file_data": {"mime_type": mime_type, "file_uri": self._upload_via_files_api(video_path)}}

        try:
            with open(video_path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("ascii")
        except OSError as exc:
            raise RuntimeError(f"Gemini icin video dosyasi okunamadi: {video_path} ({exc})") from exc
        return {"inline_data": {"mime_type": mime_type, "data": encoded}}

    # ------------------------------------------------------------- Transport

    def _generate(
        self,
        parts: List[Dict[str, Any]],
        source_label: str,
        on_retry: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        """Verilen icerik bloklarini Gemini'ye gonderip ham metin cevabini dondurur.

        Yeniden deneme, kare-tabanli ("hafif") yolla AYNI ortak politikadan
        (`base_vlm.send_with_retry`) gelir: gecici hatalar (baglanti/timeout,
        408/409/425/429 ve 5xx) ustel geri-cekilmeyle - saglayici `Retry-After`
        verdiyse ONUN suresiyle - yeniden denenir; kalici istemci hatalari
        (400/401/403/413/422) ANINDA yukseltilir.

        Args:
            parts: `contents[0].parts` listesi (metin + video bloklari).
            source_label: Teshis/loglama icin kaynak kimligi (video yolu).
            on_retry: Her yeniden denemeden once cagrilan opsiyonel bildirim;
                `_send_single_video` bunu operator paneline (`on_progress`)
                baglar, boylece uzun bir video sirasinda yeniden deneme
                SESSIZ kalmaz.

        Returns:
            Modelin ham metin ciktisi.

        Raises:
            RuntimeError: Yanit beklenmedik bicimdeyse, icerik BOS gelirse veya
                tum yeniden denemeler tukenirse.
            httpx.HTTPError: KALICI bir HTTP hatasinda (cagiran taraf raporlar).
        """
        # `extra_body`, Gemini'de istek govdesinin KOKUNE degil
        # `generationConfig` icine girer (ör. `thinkingConfig.thinkingBudget: 0`).
        # Cekirdek alanlar (temperature/maxOutputTokens) EZILMEZ - `setdefault`
        # deseni `base_vlm.apply_extra_body` ile AYNI sozlesmeyi izler.
        generation_config: Dict[str, Any] = {
            "temperature": self._endpoint.temperature,
            "maxOutputTokens": self._endpoint.max_new_tokens,
        }
        for key, value in (self._endpoint.extra_body or {}).items():
            generation_config.setdefault(key, value)

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        url = f"{self._native_root}/v1beta/models/{self._endpoint.model_name}:generateContent"
        # ORTAK yeniden-deneme politikasi (bkz. metot dokustringi). `raise_for_
        # status()` BILEREK burada cagrilmaz - `send_with_retry` icinde cagrilir,
        # aksi halde HTTP durum hatalari yeniden-deneme karari verilemeden
        # yukselirdi.
        response = send_with_retry(
            lambda: httpx.post(
                url,
                json=payload,
                headers={**self._auth_headers(), "Content-Type": "application/json"},
                timeout=_GEMINI_VIDEO_TIMEOUT,
            ),
            source=source_label,
            on_retry=on_retry,
        )
        data = response.json()
        raw_content = _extract_text(data)
        # Bos icerik SESSIZCE "hicbir sey gozlemlenmedi"ye donusmesin diye
        # ACIKCA hataya cevrilir (bkz. `base_vlm.raise_if_empty_content`).
        raise_if_empty_content(raw_content, source_label, data)
        return raw_content

    def _send_single_video(
        self,
        video_path: str,
        prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
        chunk_index: int = 1,
        total_chunks: int = 1,
        range_label: Optional[str] = None,
    ) -> VLMResponse:
        """Tek bir videoyu (tam veya bir parcasini) Gemini'ye gonderip `VLMResponse` uretir.

        `EvrenVLM._send_single_video` ile AYNI sozlesmeyi uygular (ayni
        `on_progress` fazlari, ayni donus tipi); yalnizca transport Gemini'nin
        native ucudur ve ayristirma, HER IKI cikti semasini da (typed
        `observations` ve eski `EVENTS_JSON`) anlayan ORTAK parser'a
        (`src/vlm/parser.py::parse_vlm_response`) yapilir.

        Args:
            video_path: Gonderilecek `.mp4` dosyasinin yolu (tam video veya parca).
            prompt: Analiz odagini belirten kullanici istemi.
            on_progress: Bkz. `EvrenVLM.analyze_video`.
            chunk_index: Bu istegin genel video icindeki sirasi (1 tabanli).
            total_chunks: Toplam parca sayisi.
            range_label: Kullaniciya gosterilecek zaman araligi etiketi.

        Returns:
            Bu tekil istegin `VLMResponse`i; `structured_events` icindeki
            zamanlar BU dosyanin basindan (0) itibaren saniyedir.

        Raises:
            RuntimeError: Video okunamazsa veya Gemini cagrisi basarisiz olursa.
        """
        full_prompt = f"{VLM_OBSERVER_SYSTEM_PROMPT}\n\nEk istem: {prompt}{_VIDEO_MODE_NOTE}".strip()
        video_part = self._video_part(video_path)
        transport = "files_api" if "file_data" in video_part else "inline"
        size_mb = os.path.getsize(video_path) / (1024 * 1024)

        logger.info(
            "Gemini VLM video cagrisi yapiliyor: video=%s model=%s boyut=%.1f MB transport=%s",
            video_path,
            self.model_name,
            size_mb,
            transport,
        )
        _report(
            on_progress,
            phase="chunk_start",
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            range_label=range_label,
            video_mb=round(size_mb, 1),
        )

        # Yeniden deneme operatore BILDIRILIR: uzun bir videoda 2 denemelik
        # geri-cekilme sessiz gecerse panel "donmus" gibi gorunur. Frontend
        # tanimadigi `phase` degerlerini sessizce yok sayar (if/else zinciri),
        # bu yuzden yeni bir faz eklemek arayuzu KIRMAZ.
        def _report_retry(info: Dict[str, Any]) -> None:
            _report(
                on_progress,
                phase="chunk_retry",
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                range_label=range_label,
                attempt=info.get("attempt"),
                max_attempts=info.get("max_attempts"),
                delay_sec=info.get("delay_sec"),
                error=info.get("error"),
            )

        started_at = time.perf_counter()
        try:
            raw_content = self._generate(
                [{"text": full_prompt}, video_part], video_path, on_retry=_report_retry
            )
        except httpx.HTTPError as exc:
            elapsed = time.perf_counter() - started_at
            logger.error(
                "Gemini VLM video cagrisi basarisiz (video=%s, %.1fs sonra): %s", video_path, elapsed, exc
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
        logger.info("Gemini VLM video cagrisi tamamlandi: video=%s sure=%.1fs", video_path, elapsed)
        _report(
            on_progress,
            phase="chunk_done",
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            range_label=range_label,
            elapsed_sec=round(elapsed, 1),
        )

        # ORTAK parser: typed `observations` semasini da, eski `EVENTS_JSON`
        # blogunu da anlar ve VLM'in KENDI urettigi serbest anahtar
        # kelimelerini (`attributes`/`entities`) korur - sabit bir taksonomiye
        # DUSULMEZ (bkz. `src/vlm/parser.py::observations_to_structured_events`).
        from src.vlm.parser import observations_to_structured_events, parse_vlm_response

        description, chunk_res = parse_vlm_response(raw_content)
        structured_events = observations_to_structured_events(chunk_res)
        latency_ms = (time.perf_counter() - started_at) * 1000
        return VLMResponse(
            description=description,
            model_name=self.model_name,
            frame_count=0,
            latency_ms=latency_ms,
            structured_events=structured_events,
        )

    def answer_video_question(self, video_source: str, question: str, analysis_summary: str) -> str:
        """Ayni videoyu Gemini'ye tekrar gonderip operatorun yeni sorusunu yanitlar.

        Args:
            video_source: Daha once analiz edilmis videonun yerel dosya yolu.
            question: Kullanicinin bu video hakkindaki yeni sorusu.
            analysis_summary: Videonun daha once uretilmis kisa metin ozeti.

        Returns:
            Modelin ham (yapisal olmayan) Turkce serbest-metin cevabi.

        Raises:
            RuntimeError: `video_source` bir RTSP/canli akis adresiyse, dosya
                okunamazsa veya Gemini cagrisi basarisiz olursa.
        """
        lowered = video_source.strip().lower()
        if lowered.startswith(("rtsp://", "http://", "https://")):
            raise RuntimeError(
                "Gemini video-QA yalnizca yerel video dosyalarini destekler "
                f"(RTSP/canli akis DESTEKLENMEZ): {video_source}"
            )

        user_text = build_ask_video_user_prompt(question, analysis_summary)
        parts = [
            {"text": f"{ASK_VIDEO_SYSTEM_PROMPT}\n\n{user_text}"},
            self._video_part(video_source),
        ]
        logger.info("Gemini video-QA cagrisi yapiliyor: video=%s model=%s", video_source, self.model_name)
        return self._generate(parts, video_source).strip()

    def health_check(self) -> bool:
        """Gemini native API'sinin (model listesi) erisilebilir olup olmadigini kontrol eder.

        Returns:
            Uc nokta erisilebilir ve anahtar gecerliyse `True`, aksi halde `False`.
        """
        try:
            response = httpx.get(
                f"{self._native_root}/v1beta/models",
                headers=self._auth_headers(),
                timeout=10.0,
            )
            return response.status_code == 200
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning("Gemini health_check basarisiz: %s", exc)
            return False


class GeminiFramesVLM(BaseVLM):
    """Kare-tabanli (Dusuk Butceli mod) Gemini implementasyonu - OpenAI-uyumlu uc uzerinden.

    Gemini'nin uyumluluk katmani GORUNTU bloklarini (`image_url`, base64 data
    URI) kabul ettigi icin bu sinif, `BaseVLM`in ortak kare akisini
    (`_build_chat_payload` + `_post_chat_completion`: retry, typed parser,
    zaman normalizasyonu) OLDUGU GIBI kullanir - `GemmaVLM`/`QwenVLM` ile
    AYNI desen. Video-dogrudan yol icin `GeminiVLM`e bakiniz.
    """

    def analyze_evidence(self, evidence_frames: List[EvidenceFrame], prompt: str) -> VLMResponse:
        """Evidence karelerini Gemini'ye gonderip olay kumeleme + Turkce gozlem uretir.

        Args:
            evidence_frames: Adaptive Frame Sampler'in urettigi kronolojik
                evidence kareleri (bir batch).
            prompt: Analiz odagini belirten ek istem (bos olabilir).

        Returns:
            Kumelenmis olaylari ve dogal dil gozlemini iceren `VLMResponse`.

        Raises:
            RuntimeError: Gemini ucuna erisilemezse veya evidence bulunamazsa.
        """
        if not evidence_frames:
            raise RuntimeError("Gemini'ye gonderilecek evidence karesi bulunamadi.")

        full_prompt = f"{VLM_OBSERVER_SYSTEM_PROMPT}\n\nEk istem: {prompt}".strip()
        payload = self._build_chat_payload(evidence_frames, full_prompt)
        logger.info(
            "Gemini (kare-tabanli) cagrisi yapiliyor: %d evidence karesi, model=%s",
            len(evidence_frames),
            self.model_name,
        )
        return self._post_chat_completion(payload)

    def health_check(self) -> bool:
        """Gemini OpenAI-uyumlu ucunun erisilebilir olup olmadigini kontrol eder."""
        return self.health_check_impl()
