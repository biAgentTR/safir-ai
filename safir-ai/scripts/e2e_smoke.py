"""SAFIR uctan uca (E2E) duman testi: sampler -> VLM -> ajan -> eskalasyon -> rapor.

Tum pipeline'i TEK bir cagride, gercek `SafirPipeline` kodu uzerinden calistirir
ve nihai `SafirReport` ile sartname-uyumlu ozet JSON'u yazar. Boylece bagimsiz
gelistirilen katmanlarin GERCEKTEN birlikte calistigi tek komutla dogrulanir.

Kullanim (repo kokunden veya safir-ai/ icinden):
    # 1) Tamamen offline, deterministik (mock VLM+LLM) — ag/GPU/anahtar gerekmez:
    python scripts/e2e_smoke.py --mock

    # 2) GERCEK E2E (Gemini test backend'i uzerinden):
    #    configs/config.yaml -> vlm.active_model: "gemini", llm.active_model: "gemini"
    #    ve GEMINI_API_KEY tanimli olmali (bkz. .env.example)
    export GEMINI_API_KEY=...   # (yalnizca gelistirme/test; sartname offline ister)
    python scripts/e2e_smoke.py --video data/ornek.mp4

Bayraklar:
    --mock        VLM ve LLM'i mock'a zorlar (config'ten bagimsiz, offline).
    --video PATH  Analiz edilecek video; verilmezse sentetik bir video uretilir.
    --real-rag    Gercek Embedding+FAISS RAG'i kullanir (model indirir); varsayilan
                  olarak ag bagimliligi olmayan hafif sahte RAG kullanilir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))


@dataclass
class _FakeDoc:
    text: str
    score: float = 1.0


class _FakeRagService:
    """`ContextBuilder`/`RetrieverTool`nin bekledigi `.query(question, top_k)` sozlesmesine uyan sahte RAG.

    Gercek `EmbeddingRAGService`nin `sentence-transformers` model indirme
    gereksinimini atlar; boylece E2E duman testi ag bagimliligi olmadan kosar.
    """

    def seed_default_regulations(self) -> None:
        return None

    def query(
        self, question: str, top_k: Optional[int] = None, keywords: Optional[List[str]] = None
    ) -> List[_FakeDoc]:
        del keywords  # gercek EmbeddingRAGService.query ile arayuz PARITESI icin kabul edilir, kullanilmaz
        return [
            _FakeDoc(text="ISG Yonetmeligi Madde 12: KKD (baret/yelek) zorunludur."),
            _FakeDoc(text="Operasyonel Kural OK-07: Forklift trafiginde yaya gecitleri acik tutulmalidir."),
        ][: (top_k or 2)]


def _make_synthetic_video(path: Path) -> None:
    """20-30. kareler arasi hareket iceren, tamamen sentetik bir .mp4 uretir (cv2 ile)."""
    import cv2
    import numpy as np

    frames = []
    for i in range(60):
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        if 20 <= i < 30:
            cv2.rectangle(frame, (5, 5), (58, 42), (255, 255, 255), -1)
        frames.append(frame)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="SAFIR uctan uca duman testi.")
    parser.add_argument("--mock", action="store_true", help="VLM+LLM'i mock'a zorla (offline).")
    parser.add_argument("--video", type=str, default=None, help="Analiz edilecek video yolu.")
    parser.add_argument("--real-rag", action="store_true", help="Gercek Embedding+FAISS RAG kullan (model indirir).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import src.main as main_module
    from src.utils.config_loader import load_config

    if not args.real_rag:
        # Ag bagimliligi olmadan kosmak icin gercek RAG'i sahteyle degistir.
        main_module.EmbeddingRAGService = lambda *a, **k: _FakeRagService()  # type: ignore[assignment]

    config = load_config()
    if args.mock:
        # `use_mock_vlm`/`use_mock_llm` yalnizca VLM/LLM istemcilerini mock'lar;
        # `PromptInjectionGuard` (src/security/prompt_injection_guard.py) bu
        # bayraklardan BAGIMSIZ kurulur ve `guard.enabled=true` ise (varsayilan
        # config.yaml) gercek EVREN agina baglanmaya calisirdi - bu docstring'in
        # vaat ettigi "tamamen offline, ag/anahtar gerekmez" davranisiyla
        # CELISIRDI. --mock guard'i da devre disi birakir.
        config = config.model_copy(
            update={
                "app": config.app.model_copy(update={"use_mock_vlm": True, "use_mock_llm": True}),
                "guard": config.guard.model_copy(update={"enabled": False}),
            }
        )

    print("=" * 72)
    print(f"Backend -> VLM: {config.vlm.active_model} | LLM: {config.llm.active_model} | mock={args.mock}")
    print("=" * 72)

    # Video kaynagi: verilmisse onu, yoksa gecici sentetik video kullan.
    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    if args.video:
        video_path = args.video
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        video_path = str(Path(tmp_dir.name) / "synthetic.mp4")
        _make_synthetic_video(Path(video_path))
        print(f"[i] Sentetik video uretildi: {video_path}")

    try:
        pipeline = main_module.SafirPipeline(config)
        report = pipeline.run(video_path, "Sahnede KKD eksikligi, forklift veya riskli durum var mi?")
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    print("\n--- NIHAI RAPOR (ozet) ---")
    print(f"risk_score      : {report.risk_score} ({report.risk_level})")
    print(f"escalation_tier : {report.escalation_tier} | auto_dispatched={report.auto_dispatched}")
    print(f"alert_id        : {report.alert_id}")
    print(f"summary         : {report.summary}")
    print(f"actions         : {report.actions}")
    print(f"vlm_model       : {report.vlm_model} | llm_model: {report.llm_model}")

    print("\n--- SARTNAME UYUMLU JSON ---")
    print(json.dumps(report.to_sartname_json(), ensure_ascii=False, indent=2))

    # Basit dogrulama: pipeline bir olay uretmis ve raporu tamamlamis olmali.
    assert report.event_id is not None, "Beklenen: en az bir olay kaydi."
    assert report.escalation_tier in {"monitor", "notify", "alarm"}
    print("\n[OK] Uctan uca duman testi basariyla tamamlandi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
