"""Merkezi YAML konfigurasyonunu tipli (pydantic) modellere donusturen yukleyici."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# EVREN/Qdrant (ve diger tum) ortam degiskenleri, bu modulun ILK import
# edildigi anda - CWD'den veya hangi giris noktasinin (src.main, bir CLI
# scripti gibi `python -m src.rag.build_knowledge_index`, vb.) calistigindan
# BAGIMSIZ olarak - proje kokundeki (`PROJECT_ROOT`, yukarida `DEFAULT_
# CONFIG_PATH` ile AYNI referans noktasi) `.env` dosyasindan yuklenir. Bu
# modul, EVREN_API_KEY/EVREN_QDRANT_KEY gibi degerleri okuyan TUM config
# siniflarindan (VLLMEndpointConfig.resolved_api_key, QdrantMemoryConfig
# kullanicilari, vb.) ONCE, her giris noktasinda (dogrudan veya `load_config`
# uzerinden dolayli) import edildigi icin bu, EVREN/Qdrant konfigurasyonu
# COZULMEDEN ONCE calismasini garantiler. Zaten tanimli GERCEK ortam
# degiskenlerinin (orn. Docker/CI'da set edilenler) uzerine SESSIZCE
# YAZILMAZ (`override=False`, python-dotenv varsayilani).
load_dotenv(PROJECT_ROOT / ".env")


class SystemConfig(BaseModel):
    """Genel sistem ayarlari (donanim, ortam, loglama)."""

    name: str
    environment: str
    device: str
    cuda_device_index: int
    log_level: str
    random_seed: int


class AppConfig(BaseModel):
    """Uygulama kimligi ve GPU'suz gelistirme icin Mock mod anahtarlari.

    `use_mock_vlm`/`use_mock_llm` `true` oldugunda `src/vlm/factory.py`
    (`get_vlm_client`/`get_llm_client`) gercek vLLM GPU servislerine hic
    baglanmadan sahte (mock) istemcileri dondurur; boylece GPU'su olmayan
    gelistiriciler tum pipeline'i uctan uca calistirabilir.
    """

    name: str = "SAFIR"
    version: str = "2.0.0"
    use_mock_vlm: bool = False
    use_mock_llm: bool = False


class SamplerConfig(BaseModel):
    """Adaptive Frame Sampler (CPU) esik ve pencere ayarlari.

    `min_change_threshold`/`blur_kernel_size`/`history_window`/`sample_fps`
    `AdaptiveFrameSampler` tarafindan dogrudan kullanilan aktif
    parametrelerdir (bkz. `sampler_from_config`). Geri kalan alanlar
    (idle_interval_sec, active_fps, noise_floor, motion_threshold,
    scene_change_threshold, resize_width, warmup_frames), sahadaki
    alternatif/eski sampler konfigurasyonlariyla geriye donuk uyumluluk ve
    ince ayar icin saklanir.

    ONEMLI (mimari): Sampler artik hicbir OLAY KUMELEMESI yapmaz; bu yuzden
    eski kumeleme-ozel alanlar (`min_event_interval_sec`,
    `cluster_merge_gap_sec`, `bbox_iou_merge_threshold`,
    `max_cluster_duration_sec`) ve `max_evidence_buffer` (Kanit Karesi
    sayisinda sabit ust sinir) KALDIRILMISTIR: kare sayisinda video geneli
    sabit bir limit YOKTUR, esik-gecmis TUM kareler VLM'e gonderilir.
    Kumeleme + VLM istek boyutu kontrolu artik `vlm.batch_size` (bkz.
    `VLMConfig`) ve VLM katmaninin kendisi tarafindan yonetilir.

    ONEMLI (zamansal kapsama): `max_temporal_gap_sec`, esigi hicbir zaman
    gecemeyen uzun sessiz araliklarda (ör. 00:15 -> 01:45 gibi) sistemin
    kor kalmasini onleyen bir GUVENLIK AGIDIR - kumeleme DEGILDIR ve
    pre/peak/post gibi bir konumsal rol getirmez (bkz.
    `AdaptiveFrameSampler.process_video`, `EvidenceFrame.selection_reason`).

    ONEMLI (yogunluk-uyarlamali secim): Sampler videoyu HER ZAMAN tek bir
    sabit `sample_fps` ile okur (kaynak/analiz hizi degismez); adaptif olan
    yalnizca VLM'e GONDERILECEK karelerin secim SIKLIGIDIR. Uc yogunluk
    seviyesi vardir - sakin (`max_temporal_gap_sec`), erken degisim
    (`early_change_*`) ve guclu degisim/hysteresis (`significant_change_*`,
    `strong_change_cooldown_sec`). Bu, ana esik gecilmeden ONCE baslayan
    kucuk-ama-surekli degisimlerin (ör. bir izmaritin atilma ani, hafif
    dumanin ilk gorulme ani) tamamen kacirilmasini onler; hicbiri olay
    kumelemesi veya pre/peak/post degildir - sadece sampler'in KENDI secim
    gerekcesidir (bkz. `EvidenceFrame.selection_reason`).
    """

    min_change_threshold: float
    blur_kernel_size: List[int]
    history_window: int
    sample_fps: int

    # --- Zamansal oylama ayarlari (bkz. AdaptiveFrameSampler._confirm_candidate) ---
    temporal_vote_window: int = 1
    temporal_vote_min_count: int = 1

    # --- Zamansal kapsama (coverage) ayari: SAKIN bolgede secim araligi ---
    max_temporal_gap_sec: float = 15.0
    """Son evidence karesinden (esik-gecen VEYA coverage) bu yana gecen sure
    bu degeri asarsa, o ana kadar degerlendirilen esik-alti adaylar arasindan
    `net_change_score`'u en yuksek olan kare `selection_reason=
    "temporal_coverage"` ile evidence listesine eklenir. Rastgele veya sabit
    periyodik bir kare DEGILDIR; pencere icindeki en bilgi-degeri yuksek
    adaydir (bkz. `AdaptiveFrameSampler.process_video`). Bu, en GEVSEK
    (en seyrek) secim araligidir - erken/guclu degisimde daha SIK secim
    yapilir (bkz. asagidaki alanlar)."""

    # --- Erken degisim (onset) ayarlari: ana esik gecilmeden ONCE, kucuk-ama-
    # -surekli bir sinyal baslarsa secim sikligini artiran esikler. Sabit,
    # olceksiz bir sayi DEGIL - `min_change_threshold`in bir ORANI olarak
    # tanimlanir, boylece hangi hassasiyet secilirse secilsin ayni oranti
    # korunur. ---
    early_change_score_ratio: float = 0.4
    """`net_change_score >= early_change_score_ratio * min_change_threshold`
    ise bu kare 'supheli-erken' sayilir (ana esigin ALTINDA ama gurultu
    tabanindan da belirgin sekilde yuksek bir sinyal). `0 < oran < 1`
    olmalidir."""
    early_change_window: int = 3
    """Erken-degisim onayinda dikkate alinan, en son kac 'supheli-erken'
    karar sonucunun tutulacagi (sabit boyutlu pencere - bkz.
    `AdaptiveFrameSampler._confirm_candidate` ile ayni desen)."""
    early_change_min_count: int = 2
    """`early_change_window` icinde erken-degisim DURUMUNUN onaylanmasi icin
    gereken minimum 'supheli-erken' karar sayisi. `1`den buyuk tutulmasi,
    TEK bir anlik skor sicramasinin (kamera titremesi vb.) erken-degisim
    tetiklemesini engeller - surdurulen bir egilim gerekir."""
    early_change_selection_interval_sec: float = 3.0
    """Erken-degisim durumu aktifken (onaylandiktan sonra) uygulanan azami
    secim araligi (saniye); `max_temporal_gap_sec`den KUCUK olmalidir -
    boylece olayin gelisimi sakin moddan daha sik izlenir."""
    early_change_cooldown_sec: float = 4.0
    """Erken-degisim sinyali kesildikten (artik 'supheli-erken' gelmemeye
    basladiktan) sonra, sakin moda donmeden once beklenen sure (hysteresis).
    Tek bir dususu hemen sakin secime donusturmez; ancak sonsuza kadar da
    acik tutmaz."""

    # --- Guclu degisim (ana esik + hysteresis) ayarlari: ana esik gecildikten
    # sonra secim sikligini en fazla artiran ve kisa bir sure koruyan esikler. ---
    significant_change_selection_interval_sec: float = 1.0
    """Ana esik gecildikten sonraki hysteresis penceresinde (skor tekrar
    esigin altina dustugunde bile) uygulanan azami secim araligi (saniye);
    uc seviye arasinda EN SIK olanidir (`< early_change_selection_interval_sec
    < max_temporal_gap_sec` olmalidir)."""
    strong_change_cooldown_sec: float = 4.0
    """Ana esik gecildikten sonra 'guclu degisim' durumunun korunacagi sure
    (saniye); bu sure icinde skor esigin altina dusse bile secim sikligi
    yuksek kalir (Skor dustugunde TEK kareye bakip hemen sakin secime
    donulmez - kisa bir hysteresis/cooldown uygulanir)."""

    # --- Duplicate onleme: yalnizca coverage/early_change/significant_change
    # (esik-alti) secimlerine uygulanir - `threshold_exceeded` kareler ASLA
    # bu kontrolle elenmez (bkz. AdaptiveFrameSampler._is_near_duplicate). ---
    dedup_similarity_ratio: float = 0.5
    """Bir esik-alti aday, SON SECILEN evidence karesine gore fark orani
    `dedup_similarity_ratio * min_change_threshold`den DUSUKSE, gorsel
    olarak 'neredeyse ayni' sayilip SECILMEZ (zaman olarak uzak, gercekten
    farkli iki durumu SILMEZ - yalnizca ardisik, gorsel olarak ayirt
    edilemeyen tekrarlari engeller)."""

    # --- Tek-kareli guclu/lokal degisim (single_frame_change): cok-kareli
    # `early_change` onayindan (early_change_min_count) TAMAMEN BAGIMSIZ,
    # ayri bir yol - hafif-ama-surdurulen sinyaller HALA yalnizca cok-kareli
    # mekanizma ile yakalanir (bkz. AdaptiveFrameSampler.process_video). ---
    single_frame_change_enabled: bool = True
    """`True` ise, ana esigin ALTINDA kalan ama TEK bir karede GUCLU, LOKAL
    ve gurultu tabanindan acikca ayrisan bir degisim, cok-kareli onay
    BEKLENMEDEN aninda `selection_reason="single_frame_change"` ile secilir.
    `False` ile bu yol tamamen devre disi kalir."""
    single_frame_change_noise_floor_ratio: float = 2.0
    """Bir karenin 'tek-kare guclu degisim' sayilmasi icin `net_change_score`,
    bir tabanin (normalde `adaptive_noise_floor`, o cok dusukken erken-degisim
    esigine duser - bkz. `AdaptiveFrameSampler.process_video`) en az bu kadar
    kati olmalidir - sabit bir skor DEGIL, videonun o anki olcegine GORELI
    kalir."""
    single_frame_change_max_area_ratio: float = 0.35
    """Hareket maskesinin sinirlayici kutu alaninin kare alanina orani bu
    degeri ASARSA aday LOKAL sayilmaz (`(0, 1]`) - kareyi butunuyle etkileyen
    ani parlaklik degisimi/kamera titremesi buyuk olcude bastirilir."""

    # --- Uzun-baz karsilastirma (long baseline): `sample_fps` art arda
    # yukseltildikce (kisa/ani olaylari yakalamak icin - bkz. `sample_fps`
    # alani), ARDISIK ORNEKLER arasindaki gercek zaman farki kuculur; bu,
    # DUSUK KONTRASTLI/KADEMELI baslayan olaylarda (ör. dumanin ilk
    # gorulme ani) HER ORNEK ARASI degisimi o kadar KUCULTUR ki (blur +
    # sabit piksel-fark esigi 25 ile birlesince) `raw_change_ratio`
    # SIFIRA duser - bu, `min_change_threshold` NE KADAR dusurulurse
    # dusurulsun DUZELTILEMEZ, cunku sorun oran DEGIL, HAM sinyalin
    # kendisinin kaybolmasidir. Bu blok, `self.prev_gray`e (kisa/ardisik-
    # ornek karsilastirmasi - ani/kisa olaylar icin DEGISTIRILMEDEN korunur)
    # EK olarak, periyodik olarak yenilenen DAHA ESKI bir referans kareyle
    # de karsilastirma yapar; bu, kucuk-ama-birikimli degisimin daha uzun
    # bir zaman araliginda GORUNUR hale gelmesini saglar. Yalnizca
    # `is_early_suspicious`i (esik-alti aday tespiti, HALA cok-kareli
    # onaydan - `early_change_min_count` - GECMEK ZORUNDA) etkiler;
    # `is_suspicious`/`threshold_exceeded` (kosulsuz secim) yoluna ASLA
    # dokunmaz - bkz. `AdaptiveFrameSampler.process_video`. ---
    long_baseline_change_enabled: bool = True
    """`True` ise, ardisik-ornek karsilastirmasina EK olarak periyodik
    yenilenen bir referans kareyle de karsilastirma yapilir (bkz. yukarisi).
    `False` ile bu yol tamamen devre disi kalir (yalnizca ardisik-ornek
    karsilastirmasi kullanilir)."""
    long_baseline_interval_sec: float = 0.5
    """Uzun-baz (hizli kanal) referans karenin yenilenme araligi (saniye).
    `0`dan buyuk olmalidir. Orta hizli/dusuk kontrastli baslangiclari
    yakalar."""
    long_baseline_slow_interval_sec: float = 3.0
    """Ikinci, DAHA UZUN bir uzun-baz referansin yenilenme araligi
    (saniye); `long_baseline_interval_sec`den BUYUK olmalidir. Hizli kanal
    tek basina COK yavas gelisen olaylari (ör. kademeli baslayan bir
    yangin) yakalamaya yetmeyebilir - bu ikinci kanal, AYNI `is_early_
    suspicious`i besleyen, TAMAMEN BAGIMSIZ ek bir guvenlik agidir."""

    # --- Tanilama (diagnostic) modu: varsayilan KAPALI, sampler secim
    # davranisini/performansini HICBIR sekilde degistirmez. Acildiginda HER
    # ORNEKLENEN kare icin tam karar izini (skorlar, esikler, durum, secim/
    # red nedeni) diske yazar - goruntu/base64 veri ASLA loglanmaz (bkz.
    # AdaptiveFrameSampler._DiagnosticRecorder). Kok neden tespiti/hata
    # ayiklama amaclidir; secim ALGORITMASINI DEGISTIRMEZ. ---
    diagnostic_enabled: bool = False
    """`True` ise `process_video`, her ornek kare icin bir tanilama satiri
    (`diagnostic_output_format`e gore CSV veya JSONL) yazar. Varsayilan
    `False` ile bu mekanizma tamamen devre disidir; hicbir ek hesaplama
    yapilmaz, mevcut davranis/performans BIREBIR korunur."""
    diagnostic_output_dir: str = "outputs/diagnostics"
    """Tanilama dosyalarinin yazilacagi klasor (mevcut `outputs/` duzenine
    uygun - bkz. `configs/config.yaml` `output:` blogu)."""
    diagnostic_output_format: str = "jsonl"
    """`"jsonl"` veya `"csv"`."""

    idle_interval_sec: float
    active_fps: float
    noise_floor: float
    motion_threshold: float
    scene_change_threshold: float
    resize_width: int
    warmup_frames: int


class VLLMEndpointConfig(BaseModel):
    """Tek bir model uc noktasi icin baglanti bilgileri (yerel vLLM veya harici saglayici).

    Varsayilan `provider="vllm"` ile davranis, `http://<vllm_host>:<vllm_port>/v1`
    adresindeki yerel OpenAI-uyumlu vLLM servisine baglanmaktir (yarismanin
    offline/yerel gereksinimi). `provider="gemini"` (veya baska bir harici
    saglayici) secildiginde `base_url` ile tam OpenAI-uyumlu taban adres ve
    `api_key_env` ile anahtarin okunacagi ortam degiskeni adi verilir; boylece
    VRAM'i yetmeyen gelistiriciler ayni pipeline'i tek config anahtariyla
    Gemini API uzerinden test edebilir.

    ONEMLI: Harici API kullanimi (Gemini) yalnizca GELISTIRME/TEST amaclidir ve
    sartnamenin "tamamen yerel/offline calisma" gereksinimini ihlal eder;
    yarisma teslimi icin `provider` her zaman `vllm` (yerel) olmalidir.
    """

    model_name: str
    vllm_host: str = "localhost"
    vllm_port: int = 0
    max_new_tokens: int
    temperature: float
    top_p: float = 1.0

    provider: str = "vllm"                       # vllm | gemini
    base_url: Optional[str] = None               # verilirse host:port yerine bu tam OpenAI-uyumlu taban kullanilir
    api_key_env: Optional[str] = None            # anahtarin okunacagi ortam degiskeni adi (orn. "GEMINI_API_KEY")
    extra_body: Dict[str, Any] = Field(default_factory=dict)
    """Istek govdesine eklenecek saglayici-ozel alanlar (vLLM guided decoding icin;
    orn. `{"guided_json": {...}}` veya `{"guided_regex": "..."}`). Varsayilan bos
    (davranis degismez). Cekirdek alanlari (model/messages/...) EZMEZ."""
    chunk_duration_sec: Optional[float] = None
    """2026-08-25 (EVREN "video cozunurluk zarfi" duzeltmesi): yalnizca video-
    tabanli saglayicilar (bkz. `src/vlm/evren_vlm.py::EvrenVLM`) tarafindan
    okunur - EVREN, gonderilen videonun TAMAMINA TEK bir piksel butcesi
    uyguladigi icin (dokumantasyon onerisi: "klip kisa parcalara bolunmeli"),
    bu deger verildiginde video, `src/vlm/video_chunker.py` ile bu sureden
    (saniye) uzun ise otomatik olarak ardisik parcalara bolunup HER parca
    AYRI bir istekte gonderilir, sonuclar zaman-damgasi kaydirmasiyla
    birlestirilir. `None`/`<=0` (varsayilan) = bolme YAPILMAZ (eski davranis,
    video tek istekte gonderilir); qwen/gemma (frame-tabanli) icin ETKISIZDIR."""

    def resolved_base_url(self) -> str:
        """Bu uc nokta icin kullanilacak OpenAI-uyumlu taban URL'yi dondurur.

        `base_url` verilmisse (harici saglayici) sondaki `/` temizlenerek o
        kullanilir; aksi halde yerel vLLM adresi (`http://host:port/v1`) uretilir.
        """
        if self.base_url:
            return self.base_url.rstrip("/")
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"

    def resolved_api_key(self) -> str:
        """Bu uc nokta icin API anahtarini dondurur.

        `api_key_env` tanimliysa ilgili ortam degiskeninden okunur; yerel vLLM
        icin anahtar gerekmediginden `"EMPTY"` doner (OpenAI istemcileri bos
        olmayan bir deger bekler).

        Raises:
            RuntimeError: `api_key_env` tanimli ama ortam degiskeni bos/yoksa.
        """
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "").strip()
            if not key:
                raise RuntimeError(
                    f"'{self.model_name}' uc noktasi icin API anahtari bulunamadi: "
                    f"'{self.api_key_env}' ortam degiskenini tanimlayin."
                )
            return key
        return "EMPTY"

    def auth_headers(self) -> Dict[str, str]:
        """Harici saglayici icin `Authorization` header'ini dondurur (yerel vLLM icin bos)."""
        if self.api_key_env:
            return {"Authorization": f"Bearer {self.resolved_api_key()}"}
        return {}


class VLMConfig(BaseModel):
    """Aktif VLM secimini ve tum VLM tanimlarini tutar (Factory Pattern icin)."""

    active_model: str
    models: Dict[str, VLLMEndpointConfig]
    batch_size: int = 40
    """`SafirPipeline.stage_vlm` icinde TEK bir VLM istegine dahil edilecek
    azami EVIDENCE KARESI sayisi (bkz. `BaseVLM.analyze_evidence_batched`).
    Tum evidence kareleri TEK dev payload'a doldurulmaz: kronolojik, kayipsiz
    batch'lere bolunur (batch siniri OLAY SINIRI DEGILDIR - bir olay iki
    batch'e bolunebilir). Birden fazla batch olustuysa, batch-local olaylar
    ikinci bir VLM 'reconciliation' cagrisiyla (bkz. `BaseVLM.reconcile_events`)
    global olaylara birlestirilir; bir batch'in basarisiz olmasi digerlerini
    etkilemez ve hicbir evidence silinmez."""

    def active_endpoint(self) -> VLLMEndpointConfig:
        """Config icinde secilen aktif VLM'in baglanti bilgisini dondurur."""
        if self.active_model not in self.models:
            raise KeyError(f"Tanimsiz VLM secimi: '{self.active_model}'")
        return self.models[self.active_model]


class LLMConfig(BaseModel):
    """Ajan/muhakeme katmani icin aktif LLM secimini ve tanimlarini tutar."""

    active_model: str
    models: Dict[str, VLLMEndpointConfig]
    decision_model: Optional[str] = None
    """Model HIYERARSISI (mentor eleştirisi: "her gorev icin buyuk modeli
    kullanmayin, hiyerarsi kurun" - EVREN dokumantasyonu SS 6). `active_model`
    (`self._llm`, arac-secimi/JSON-uretimi/tool-routing icin "hizli" model,
    bkz. `SafirAgent`) TUM muhakeme donguleri boyunca DEGISMEDEN kullanilir;
    yalnizca dongu bittiginde (arac cagrisi kalmadi/iterasyon siniri asildi)
    tek bir "nihai karar sentezi" cagrisi icin BURADA belirtilen model
    ("buyuk"/otonom karar modeli) devreye girer - `models` icindeki bir
    anahtar olmali. `None` (varsayilan) veya `active_model` ile AYNIysa
    hiyerarsi DEVRE DISIDIR - davranis/cagri sayisi ONCEKI haliyle BIREBIR
    AYNI kalir (bkz. `SafirAgent._build_decision_llm`)."""

    def active_endpoint(self) -> VLLMEndpointConfig:
        """Config icinde secilen aktif LLM'in baglanti bilgisini dondurur."""
        if self.active_model not in self.models:
            raise KeyError(f"Tanimsiz LLM secimi: '{self.active_model}'")
        return self.models[self.active_model]


class SQLiteMemoryConfig(BaseModel):
    db_path: str


class EmbeddingConfig(BaseModel):
    """Embedding & RAG Katmani icin EVREN (harici, OpenAI-uyumlu) embedding model ayarlari.

    2026-08-25 guncellemesi: LOKAL (`sentence-transformers`) embedding
    TAMAMEN KALDIRILDI - TEK saglayici artik "evren"dir: EVREN'in
    `/v1/embeddings` ucu (`model_name="bge-m3-embed"`, 1024 boyut, bkz.
    dokumantasyon SS 5/10) uzerinden calisir (bkz.
    `src/rag/embedding_providers.py::EvrenEmbeddingProvider`).
    """

    provider: str = "evren"             # su an yalnizca "evren" destekleniyor
    model_name: str                     # orn. "bge-m3-embed"
    output_dimensionality: Optional[int] = None  # HARD-CODE edilmez, config'ten gelir
    device: str = "cpu"                 # ARTIK KULLANILMIYOR (geriye-uyum icin durur - evren uzak API'dir)
    normalize_embeddings: bool = True
    base_url: Optional[str] = None      # yalnizca provider="evren" icin (EVREN taban adresi)
    api_key_env: Optional[str] = None   # yalnizca provider="evren" icin (orn. "EVREN_API_KEY")
    max_batch_tokens: Optional[int] = None
    """`embed_documents()`in her `embeddings.create()` istegi icin hedefleyecegi
    azami TAHMINI toplam giris token butcesi (bkz. `EvrenEmbeddingProvider`
    modul dokustringi - EVREN `bge-m3-embed` istek basina azami 8192 token
    kabul eder). `None` ise saglayicinin kendi guvenli varsayilani (7000)
    kullanilir; bu deger `_MAX_CONTEXT_TOKENS`in (8192) UZERINE CIKMAMALIDIR."""


class QdrantMemoryConfig(BaseModel):
    """EVREN'e tahsis edilen izole Qdrant ornegi icin baglanti ayarlari (FAISS'in yerini alir).

    Dokumantasyon SS 11: her takima ayri, izole bir Qdrant SURECI/DISK
    HACMI tahsis edilir; erisim bir yol on-eki (takim kodu) uzerinden
    saglanir ve REST portu HER ZAMAN 443 olmalidir (aksi halde istemci
    kendi varsayilanina yonelip "Connection refused" verir - bkz.
    `src/rag/embedding_rag_service.py::_build_qdrant_client`).
    `url=":memory:"` verilirse (test/offline kullanim) tamamen bellek-ici,
    agsiz bir Qdrant orneği kullanilir - GERCEK Qdrant istemci kodu calisir,
    hicbir ag baglantisi gerekmez.
    """

    url: str                            # ":memory:" veya "https://evren-vektor.ssyz.org.tr"
    api_key_env: str = "EVREN_QDRANT_KEY"
    prefix_env: str = "EVREN_TEAM"      # takim kodu (orn. "team33") - Qdrant yol on-eki
    collection_name: str = "safir_regulations"
    top_k: int                          # RERANK SONRASI nihai sonuc sayisi
    candidate_k: int = 20               # Qdrant'tan cekilecek ADAY sayisi (rerank ONCESI)
    similarity_threshold: Optional[float] = None  # embedding-seviyesi filtre (opsiyonel, genelde None)


class RelevanceWeightsConfig(BaseModel):
    """`src/rag/deterministic_reranker.py::RelevanceWeights` icin config-tabanli agirliklar (HARD-CODE DEGIL)."""

    semantic: float = 0.60
    lexical: float = 0.15
    keyword: float = 0.15
    metadata: float = 0.05
    phrase: float = 0.05


class RerankerConfig(BaseModel):
    """Ikinci-asama (deterministik) VE ucuncu-asama (AI/LLM-as-judge) retrieval skorlama ayarlari.

    2026-08-24 (RAG RERANKER DETERMINIZATION): ikinci-asama relevance karari
    bir LLM'e SORULMAZ - `src/rag/deterministic_reranker.py`nin TAMAMEN
    yerel, agirlikli-toplam algoritmasi kullanilir (bkz. `weights`).
    2026-08-25 guncellemesi: `provider`/`model_name`/`api_key_env`/`base_url`
    alanlari YENIDEN KULLANIMDA - artik EVREN'in OpenAI-uyumlu LLM ucunu
    (`model_name="llm-fast"`) "LLM-as-judge" olarak calistiran UCUNCU,
    OPSIYONEL asamayi (`src/rag/evren_reranker.py::EvrenReranker`) yapilandirir;
    bu asama deterministik gate'i BYPASS ETMEZ, yalnizca gate'ten GECMIS
    adaylari yeniden siralar (bkz. `EmbeddingRAGService.query()`
    "cross_encoder" extension point'i - isim tarihseldir, artik LOKAL bir
    cross-encoder DEGIL, EVREN LLM-as-judge kullanir).
    """

    enabled: bool = False
    provider: str = "evren"             # EVREN'in OpenAI-uyumlu LLM ucu (dedike bir rerank endpoint'i DEGIL)
    model_name: str = "llm-fast"
    candidate_k: int = 20
    top_k: int = 5
    score_threshold: float = 0.10       # bu skorun ALTINDAKI sonuclar ELENIR (0 sonuc GECERLIDIR) - deterministik relevance_score'a uygulanir
    api_key_env: str = "EVREN_API_KEY"
    base_url: Optional[str] = None      # EVREN taban adresi (orn. "https://evren-llmapi.ssyz.org.tr/v1")
    weights: RelevanceWeightsConfig = Field(default_factory=RelevanceWeightsConfig)


class MemoryConfig(BaseModel):
    """Yapilandirilmis olay bellegi (SQLite) ve anlamsal bellek (EVREN Embedding+Qdrant+Rerank) ayarlari."""

    sqlite: SQLiteMemoryConfig
    embedding: EmbeddingConfig
    qdrant: QdrantMemoryConfig
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class RiskThresholds(BaseModel):
    """0-100 risk skorunu risk seviyelerine esleyen ust sinirlar."""

    low: int
    medium: int
    high: int
    critical: int


class AgentToolsConfig(BaseModel):
    sql_tool_enabled: bool
    rag_tool_enabled: bool              # bkz. retriever_tool_enabled (ayni RAG aracina isaret eder)
    retriever_tool_enabled: bool
    timeline_tool_enabled: bool
    verification_tool_enabled: bool


class AgentConfig(BaseModel):
    """LangGraph durum makinesi ve arac yonlendirme ayarlari."""

    max_iterations: int
    risk_thresholds: RiskThresholds
    tools: AgentToolsConfig
    guided_json: bool = True
    """Ajanin nihai karari gecerli JSON degilse, JSON-modu (response_format=
    json_object; vLLM/Gemini destekler) ile TEK bir yeniden-deneme yapilir.
    Kucuk yerel modellerde bozuk JSON'u kurtarir. `false` ile devre disi."""


class EscalationConfig(BaseModel):
    """Otomatik eskalasyon esikleri (Human-on-the-Loop; bloke edici operator kapisi yok).

    `notify_score` ve uzeri risk skoru -> bildirim (NOTIFY); `auto_alarm_score`
    ve uzeri -> saha alarmi OTOMATIK tetiklenir (ALARM). Varsayilanlar
    `agent.risk_thresholds` ile hizalidir (orta>25, yuksek>50).
    """

    notify_score: int = 26
    auto_alarm_score: int = 51


class ApiConfig(BaseModel):
    host: str
    port: int
    reload: bool
    cors_origins: List[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    language: str
    json_report_dir: str
    timeline_export_dir: str
    pdf_report_dir: str
    streamlit_port: int


class GuardConfig(BaseModel):
    """Prompt Injection Guard ayarlari (bkz. `src/security/prompt_injection_guard.py`).

    Guard AYRI bir guvenlik katmanidir: risk_score/risk_level/event_type
    hesaplamaz, RuleEngine kararini VEYA RAG retrieval kararini degistirmez -
    yalnizca Agent'a giden guvenilmeyen serbest metni (user_prompt, VLM
    aciklamasi) DETECT -> CLASSIFY -> QUARANTINE/PASS akisindan gecirir.
    `provider`: "evren" (AKTIF/production - EVREN'in OpenAI-uyumlu LLM ucu,
    `EVREN_API_KEY`) - "gemini"/"groq" eski, gecici GELISTIRME/TEST
    backend'leri olarak KORUNUR (aktif degil, silinmedi).
    """

    enabled: bool = False
    provider: str = "evren"
    model_name: str = "llm-fast"
    fail_closed: bool = True
    confidence_threshold: float = 0.80
    api_key_env: str = "EVREN_API_KEY"
    base_url: Optional[str] = None      # provider="evren"/"groq" icin ZORUNLU (gemini'de kullanilmaz)


class SafirConfig(BaseModel):
    """SAFIR sisteminin butun katmanlarini kapsayan kok konfigurasyon modeli."""

    app: AppConfig = Field(default_factory=AppConfig)
    system: SystemConfig
    sampler: SamplerConfig
    vlm: VLMConfig
    memory: MemoryConfig
    llm: LLMConfig
    agent: AgentConfig
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    api: ApiConfig
    output: OutputConfig


@functools.lru_cache(maxsize=8)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> SafirConfig:
    """`config.yaml` dosyasini okuyup dogrulanmis `SafirConfig` nesnesi olarak dondurur.

    Args:
        config_path: YAML konfigurasyon dosyasinin yolu.

    Returns:
        Pydantic ile dogrulanmis, tipli konfigurasyon nesnesi.

    Raises:
        FileNotFoundError: Belirtilen yolda konfigurasyon dosyasi bulunamazsa.
        ValueError: YAML gecersiz ya da beklenen semaya uymuyorsa.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Konfigurasyon dosyasi bulunamadi: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML ayristirma hatasi ({path}): {exc}") from exc

    try:
        return SafirConfig(**raw)
    except Exception as exc:  # pydantic.ValidationError dahil
        raise ValueError(f"Konfigurasyon semasi gecersiz ({path}): {exc}") from exc
