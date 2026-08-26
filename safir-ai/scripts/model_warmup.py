"""Model Isindirma + Genel Model Test Scripti (EVREN).

Iki kullanim bicimi:

1) Otomatik ISINDIRMA (`warmup_all_models`) - `src/main.py` FastAPI
   `startup` olayinda HER baslangicta cagrilir: `configs/config.yaml`de
   fiilen KULLANILAN dort EVREN model takma adina (`vlm`, `llm-fast`,
   `llm-large`, `bge-m3-embed`) kucuk, ucuz birer test istegi gonderir.
   Amac, bu modellerin EVREN tarafinda "soguk" olmasindan kaynaklanan ilk-
   istek gecikmesini operatorun GERCEK ilk analizinden ONCE, uygulama
   baslarken absorbe etmektir. Bir modelin isinmasi BASARISIZ olursa
   (ag hatasi, eksik `EVREN_API_KEY`, vb.) uygulama COKMEZ - yalnizca
   ACIKCA loglanir (bkz. `warmup_all_models` durusu, projenin genelindeki
   "guvenli degradasyon" deseniyle TUTARLI).

2) Bagimsiz GENEL TEST (`python scripts/model_warmup.py [--full]`): ayni
   kontrolleri komut satirindan calistirip okunabilir bir rapor basar;
   `--full` ile EK olarak EVREN dokumantasyonunun (bkz. katilimci
   dokumantasyonu SS 5/7.3/7.5/10) ACIKCA belirttigi iki DAVRANIS
   sozlesmesini de dogrular:
     - `vlm` modeline GORUNTU gonderilirse HTTP 400 doner (video-only).
     - `EvrenFramesVLM` (kare-tabanli, "llm-large") istek basina en fazla
       2 goruntuyu KABUL EDER, 3.sunde ACIKCA hata verir (bkz.
       `src/vlm/evren_vlm.py::EvrenFramesVLM`).
   Herhangi bir REQUIRED model basarisiz olursa exit code != 0 doner -
   CI/on-kontrol (demo/yarisma oncesi "EVREN ayakta mi?" sağlamasi) icin
   uygundur.

NOT (rerank): EVREN'in dedike rerank ucu (`model="rerank"`) ve
`EvrenReranker` (LLM-as-judge) BILEREK isindirilmaz/test edilmez -
dokumantasyon (SS 10) bu ucun getirme kalitesini dusurdugunu
gostermektedir ve production artik onu CAGIRMAMAKTADIR (bkz.
`src/main.py::SafirPipeline.__init__`, `configs/config.yaml ->
memory.reranker` yorumu). Bu script yalnizca FIILEN KULLANILAN modelleri
kapsar.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

logger = logging.getLogger(__name__)

_WARMUP_PROMPT = "Bu bir isinma (warmup) test mesajidir; tek kelimeyle 'hazir' diye yanitla."


@dataclass
class ModelCheckResult:
    """Tek bir model icin test/isinma sonucu (rapor + `warmup_all_models` donus degeri)."""

    model_alias: str  # EVREN'e gonderilen GERCEK model adi (ör. "llm-fast")
    role: str  # bu takma adin sistemdeki rolu (ör. "vlm.active_model=evren")
    ok: bool
    latency_ms: float
    detail: str
    required: bool = True  # `--full` negatif-test senaryolari icin False


@dataclass
class WarmupReport:
    results: List[ModelCheckResult] = field(default_factory=list)

    @property
    def all_required_ok(self) -> bool:
        return all(r.ok for r in self.results if r.required)


def _make_tiny_video(path: Path) -> None:
    """5 kareli, 32x32 boyutunda, KUCUK bir sentetik `.mp4` uretir (yalnizca "vlm" isinmasi icindir).

    `scripts/e2e_smoke.py::_make_synthetic_video` ile AYNI teknik (cv2), ama
    kasitli olarak cok daha kucuk - bu, gercek bir analiz DEGIL, EVREN'in
    "vlm" modelini isindirmak icin en ucuz gecerli video-istek payload'idir.
    """
    import cv2
    import numpy as np

    frames = [np.full((32, 32, 3), 128, dtype=np.uint8) for _ in range(5)]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 32))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _make_tiny_evidence_frame(evidence_id: str, frame_id: int, timestamp_sec: float):
    """`EvrenFramesVLM` testleri icin KUCUK, gecerli bir `EvidenceFrame` uretir."""
    import base64

    import cv2
    import numpy as np

    from src.sampler.schema import EvidenceFrame

    frame = np.full((32, 32, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Test karesi JPEG'e kodlanamadi.")
    raw = buf.tobytes()
    b64 = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    return EvidenceFrame(
        evidence_id=evidence_id,
        frame_id=frame_id,
        timestamp_sec=timestamp_sec,
        timestamp_str=f"00:{int(timestamp_sec):02d}",
        change_score=0.5,
        image_bytes=raw,
        base64_image=b64,
        image_shape=(32, 32, 3),
    )


def _check_vlm_video(config) -> ModelCheckResult:
    """"vlm" modelini (video-dogrudan, VLM Direct - `EvrenVLM`) kucuk bir video ile isindirir."""
    from src.vlm.factory import get_vlm_client

    role = "vlm.active_model=evren (VLM Direct, video-dogrudan)"
    started = time.perf_counter()
    try:
        vlm_config = config.vlm.model_copy(update={"active_model": "evren"})
        vlm = get_vlm_client(vlm_config, use_mock=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "warmup.mp4"
            _make_tiny_video(tmp_path)
            response = vlm.analyze_video(str(tmp_path), [], prompt=_WARMUP_PROMPT)
        latency_ms = (time.perf_counter() - started) * 1000
        ok = bool(response.description and response.description.strip())
        detail = "content dondu" if ok else "content BOS (beklenmiyordu)"
        return ModelCheckResult("vlm", role, ok, latency_ms, detail)
    except Exception as exc:  # noqa: BLE001 - isinma basarisizligi ACIKCA raporlanir, uygulama COKMEZ
        latency_ms = (time.perf_counter() - started) * 1000
        return ModelCheckResult("vlm", role, False, latency_ms, f"basarisiz: {exc}")


def _check_text_model(config, llm_active_model: str, role: str) -> ModelCheckResult:
    """"llm-fast" veya "llm-large" modelini metin-tabanli tek bir chat completion ile isindirir."""
    from langchain_core.messages import HumanMessage

    from src.vlm.factory import get_llm_client

    started = time.perf_counter()
    model_alias = "?"
    try:
        llm_config = config.llm.model_copy(update={"active_model": llm_active_model})
        model_alias = llm_config.active_endpoint().model_name
        llm = get_llm_client(llm_config, use_mock=False)
        response = llm.invoke([HumanMessage(content=_WARMUP_PROMPT)])
        latency_ms = (time.perf_counter() - started) * 1000
        content = str(response.content or "").strip()
        ok = bool(content)
        detail = "content dondu" if ok else "content BOS (dusunme-modu tuzagi olabilir, bkz. enable_thinking)"
        return ModelCheckResult(model_alias, role, ok, latency_ms, detail)
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000
        return ModelCheckResult(model_alias, role, False, latency_ms, f"basarisiz: {exc}")


def _check_embedding_model(config) -> ModelCheckResult:
    """"bge-m3-embed" modelini tek bir `/v1/embeddings` istegiyle isindirir; vektor boyutunu dogrular."""
    from src.rag.embedding_providers import EvrenEmbeddingProvider

    emb_config = config.memory.embedding
    role = "memory.embedding (RAG semantik arama)"
    started = time.perf_counter()
    try:
        provider = EvrenEmbeddingProvider(
            model_name=emb_config.model_name,
            base_url=emb_config.base_url,
            api_key_env=emb_config.api_key_env,
            output_dimensionality=emb_config.output_dimensionality,
        )
        vector = provider.embed_query(_WARMUP_PROMPT)
        latency_ms = (time.perf_counter() - started) * 1000
        expected_dim = emb_config.output_dimensionality
        ok = len(vector) > 0 and (expected_dim is None or len(vector) == expected_dim)
        detail = f"boyut={len(vector)}" + (f" (beklenen={expected_dim})" if not ok else "")
        return ModelCheckResult(emb_config.model_name, role, ok, latency_ms, detail)
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000
        return ModelCheckResult(emb_config.model_name, role, False, latency_ms, f"basarisiz: {exc}")


def warmup_all_models(config) -> WarmupReport:
    """Fiilen kullanilan 4 EVREN model takma adina ("vlm", "llm-fast" x2 rol, "bge-m3-embed") kucuk test istekleri gonderir.

    `src/main.py` FastAPI `startup` olayindan cagrilir (bkz. modul
    dokustringi). `EVREN_API_KEY` tanimli degilse (ör. yerel/mock
    gelistirme) TUM kontroller ACIKCA basarisiz doner ama uygulama
    ACILMAYA DEVAM EDER - bu fonksiyon YALNIZCA gozlemlenebilirlik icindir,
    davranis KAPISI DEGILDIR.

    Returns:
        Her model icin sonucu iceren `WarmupReport`.
    """
    report = WarmupReport()
    report.results.append(_check_vlm_video(config))
    report.results.append(
        _check_text_model(config, "evren", "llm.active_model=evren (ajan arac-secimi/JSON, guard, evren_frames reconciliation)")
    )
    report.results.append(
        _check_text_model(config, "evren_large", "llm.decision_model=evren_large (nihai karar sentezi)")
    )
    report.results.append(_check_embedding_model(config))
    return report


def _check_frames_max_two_images(config) -> ModelCheckResult:
    """`--full`: `EvrenFramesVLM`in 2 goruntuyle BASARILI, 3 goruntuyle ACIKCA hatali oldugunu dogrular."""
    from src.vlm.vlm_factory import VLMFactory

    role = "vlm.frames_model (Dusuk Butceli, kare-tabanli, EvrenFramesVLM)"
    started = time.perf_counter()
    model_alias = config.vlm.frames_model
    try:
        frames_config = config.vlm.model_copy(update={"active_model": config.vlm.frames_model})
        model_alias = frames_config.active_endpoint().model_name
        vlm = VLMFactory.create(frames_config)
        ef1 = _make_tiny_evidence_frame("warmup_ev1", 1, 1.0)
        ef2 = _make_tiny_evidence_frame("warmup_ev2", 2, 2.0)
        ef3 = _make_tiny_evidence_frame("warmup_ev3", 3, 3.0)

        try:
            vlm.analyze_evidence([ef1, ef2, ef3], _WARMUP_PROMPT)
            latency_ms = (time.perf_counter() - started) * 1000
            return ModelCheckResult(
                model_alias, role, False, latency_ms,
                "3 goruntu ile cagri BASARILI oldu (EVREN dokumantasyonu SS 7.5 ile CELISIR - beklenen: hata)",
            )
        except ValueError:
            pass  # beklenen: yerel dogrulama 3 goruntuyu reddetti

        response = vlm.analyze_evidence([ef1, ef2], _WARMUP_PROMPT)
        latency_ms = (time.perf_counter() - started) * 1000
        ok = bool(response.description and response.description.strip())
        detail = "2 goruntu ile content dondu; 3 goruntu ACIKCA reddedildi" if ok else "2 goruntuyle content BOS"
        return ModelCheckResult(model_alias, role, ok, latency_ms, detail)
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000
        return ModelCheckResult(model_alias, role, False, latency_ms, f"basarisiz: {exc}")


def _check_vlm_rejects_image(config) -> ModelCheckResult:
    """`--full`: "vlm" modeline goruntu gonderilirse EVREN dokumantasyonunun (SS 7.5) belirttigi HTTP 400'u dogrular.

    Bu BASARISIZLIK BEKLENEN davranistir (`required=False`) - test, HTTP
    400 DISINDA bir sonuc (ör. 200 OK) alinirsa BASARISIZ sayilir, cunku bu
    EVREN'in dokumante davranisinin DEGISTIGI anlamina gelir.
    """
    import httpx

    role = "vlm + goruntu (EVREN dokumantasyonu SS 7.5 - beklenen: HTTP 400)"
    started = time.perf_counter()
    try:
        endpoint = config.vlm.models["evren"]
        ef = _make_tiny_evidence_frame("warmup_img", 1, 1.0)
        payload = {
            "model": endpoint.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _WARMUP_PROMPT},
                        {"type": "image_url", "image_url": {"url": ef.base64_image}},
                    ],
                }
            ],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        response = httpx.post(
            f"{endpoint.resolved_base_url()}/chat/completions",
            json=payload,
            headers=endpoint.auth_headers(),
            timeout=30.0,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        ok = response.status_code == 400
        detail = f"HTTP {response.status_code} (beklenen: 400)"
        return ModelCheckResult("vlm", role, ok, latency_ms, detail, required=False)
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000
        return ModelCheckResult("vlm", role, False, latency_ms, f"basarisiz: {exc}", required=False)


def _print_report(report: WarmupReport) -> None:
    print("=" * 100)
    print(f"{'MODEL':<14} {'ROL':<62} {'DURUM':<8} {'SURE(ms)':>9}  DETAY")
    print("-" * 100)
    for r in report.results:
        status = "OK" if r.ok else "HATA"
        marker = "" if r.required else " (opsiyonel/negatif test)"
        print(f"{r.model_alias:<14} {r.role[:62]:<62} {status:<8} {r.latency_ms:>9.0f}  {r.detail}{marker}")
    print("=" * 100)


def main() -> int:
    parser = argparse.ArgumentParser(description="EVREN model isinma/test scripti.")
    parser.add_argument(
        "--full", action="store_true",
        help="Isinmaya EK olarak, EVREN dokumantasyonunun negatif senaryolarini da dogrula (vlm+goruntu=400, EvrenFramesVLM 2-goruntu siniri).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from src.utils.config_loader import load_config

    config = load_config()

    if not os.environ.get("EVREN_API_KEY", "").strip():
        print("[UYARI] EVREN_API_KEY tanimli degil; tum kontroller basarisiz donecek.")

    report = warmup_all_models(config)
    if args.full:
        report.results.append(_check_frames_max_two_images(config))
        report.results.append(_check_vlm_rejects_image(config))

    _print_report(report)

    if report.all_required_ok:
        print("[OK] Tum ZORUNLU modeller isindi/dogrulandi.")
        return 0
    print("[HATA] En az bir ZORUNLU model basarisiz oldu (yukaridaki tabloya bakin).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
