"""SAFIR adim-adim Jupyter walkthrough notebook'unu uretir (nbformat ile).

Kullanim:
    python scripts/build_notebook.py                 # notebooks/SAFIR_walkthrough.ipynb uretir
    python scripts/build_notebook.py --mock          # USE_MOCK=True olarak uretir (offline dogrulama)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(use_mock_default: bool) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s))

    md(
        "# 🛡️ SAFIR — Adim Adim Calisma (Jupyter Walkthrough)\n"
        "\n"
        "Bu defter, **Saha Analiz ve Farkindalik Icin Yapay Zeka Destekli Karar Sistemi (SAFIR)** "
        "pipeline'ini asama asama calistirir ve her adimin ciktisi gorulur:\n"
        "\n"
        "1. **Adaptive Frame Sampler (CPU)** — videodan yalnizca 'kanit karelerini' suzer (GPU tasarrufu).\n"
        "2. **Temsili Kareler** — her olayin oncesi/zirvesi/sonrasi (pre/peak/post).\n"
        "3. **VLM (Gorsel Dil Modeli)** — kareleri Turkce betimler + yapilandirilmis olaylar uretir.\n"
        "4. **Olay Tespiti** — VLM ciktisindan tipli olaylar (model-tabanli, keyword yedekli).\n"
        "5. **Hibrit Bellek / RAG** — ilgili ISG mevzuati.\n"
        "6. **LangGraph Ajani** — muhakeme -> risk skoru, ozet, aksiyon onerileri (JSON).\n"
        "7. **Otomatik Eskalasyon** — risk'e gore aksiyon kademesi (Human-on-the-Loop).\n"
        "8. **Nihai Rapor** — sartname-uyumlu JSON (uctan uca entegre kosu).\n"
    )

    md(
        "## 0) Kurulum ve Ayarlar\n"
        "Once proje kokunu Python yoluna ekleriz ve konfigurasyonu yukleriz.\n"
        "\n"
        "- `USE_MOCK`: `True` -> tamamen offline (Gemini/GPU gerekmez, sabit ornek cikti). "
        "`False` -> `config.yaml`'daki backend (Gemini). Gemini icin `GEMINI_API_KEY` tanimli olmali.\n"
        "- `USE_FAKE_RAG`: `True` -> agir embedding modelini (bge-m3, ~2GB) indirmez; hafif sahte "
        "mevzuat kullanir (demo icin hizli). `False` -> gercek FAISS RAG."
    )
    code(
        "import sys, os\n"
        "from pathlib import Path\n"
        "\n"
        "# Proje kokunu bul (notebook notebooks/ altinda ya da kok dizinde olabilir)\n"
        "_here = Path.cwd()\n"
        "PROJECT_ROOT = _here if (_here / 'src').exists() else _here.parent\n"
        "assert (PROJECT_ROOT / 'src').exists(), 'Proje koku bulunamadi (safir-ai/).'\n"
        "sys.path.insert(0, str(PROJECT_ROOT))\n"
        "os.chdir(PROJECT_ROOT)\n"
        "\n"
        "# ---- AYARLAR ----\n"
        f"USE_MOCK = {use_mock_default}       # True: offline | False: Gemini (config.yaml)\n"
        "USE_FAKE_RAG = True    # True: bge-m3 indirmez (hizli) | False: gercek FAISS RAG\n"
        "\n"
        "from src.utils.config_loader import load_config\n"
        "config = load_config()\n"
        "if USE_MOCK:\n"
        "    config = config.model_copy(update={'app': config.app.model_copy(\n"
        "        update={'use_mock_vlm': True, 'use_mock_llm': True})})\n"
        "print('VLM backend :', config.vlm.active_model)\n"
        "print('LLM backend :', config.llm.active_model)\n"
        "print('MOCK        :', USE_MOCK, '| FAKE_RAG:', USE_FAKE_RAG)"
    )

    md(
        "### Video kaynagi\n"
        "Kendi videon icin `VIDEO_PATH`'i `data/` altindaki dosyaya ayarla. Yoksa asagidaki hucre "
        "kucuk bir **sentetik** video uretir (hareketli bir dikdortgen) — pipeline'i gostermek icin yeterli."
    )
    code(
        "import cv2, numpy as np, tempfile\n"
        "\n"
        "VIDEO_PATH = 'data/ornek.mp4'   # <-- kendi videon varsa buraya\n"
        "if not Path(VIDEO_PATH).exists():\n"
        "    _tmp = tempfile.mkdtemp()\n"
        "    VIDEO_PATH = str(Path(_tmp) / 'synthetic.mp4')\n"
        "    frames = [np.full((120, 160, 3), 30, np.uint8) for _ in range(60)]\n"
        "    for i in range(20, 38):\n"
        "        cv2.rectangle(frames[i], (20, 20), (140, 100), (210, 210, 210), -1)\n"
        "    _w = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (160, 120))\n"
        "    for f in frames: _w.write(f)\n"
        "    _w.release()\n"
        "    print('Sentetik video uretildi:', VIDEO_PATH)\n"
        "else:\n"
        "    print('Video:', VIDEO_PATH)"
    )

    md(
        "## 1) Adaptive Frame Sampler (CPU)\n"
        "Video tamamen CPU'da taranir; gurultu tabani dusulmus piksel degisimine gore yalnizca "
        "**kanit kareleri** secilir. Boylece VLM'e (pahali kismi) yalnizca onemli kareler gider — "
        "GPU tasarrufu buradan gelir."
    )
    code(
        "from src.sampler.adaptive_sampler import sampler_from_config\n"
        "\n"
        "sampler = sampler_from_config(config.sampler)\n"
        "evidence = sampler.process_video(VIDEO_PATH, sample_fps=config.sampler.sample_fps)\n"
        "stats = sampler.last_run_stats\n"
        "print(f'Kanit karesi sayisi : {len(evidence)}')\n"
        "print(f'Taranan ham kare    : {stats.total_frames_scanned}')\n"
        "print(f'Elenen kare         : {stats.eliminated_frame_count}')\n"
        "print(f'GPU tasarruf orani  : %{stats.eliminated_ratio_pct}')\n"
        "print(f'Sure                : {stats.elapsed_sec}s')"
    )
    code(
        "# Kanit karelerini gorsel olarak goster\n"
        "import base64\n"
        "from IPython.display import Image as IPyImage, display\n"
        "\n"
        "for ef in evidence:\n"
        "    _, b64 = ef.base64_image.split(',', 1)\n"
        "    print(f\"[{ef.timestamp_str}] change_score={ef.change_score:.4f} fallback={ef.is_fallback}\")\n"
        "    display(IPyImage(data=base64.b64decode(b64), width=240))"
    )

    md(
        "## 2) Olay Kumeleme + Temsili Kareler (pre / peak / post)\n"
        "Ardisik kanit kareleri **Olay Gruplari**na kumelenir. Her grup icin, zirve karenin oncesi "
        "ve sonrasindan da kare cikarilir; boylece VLM tek durgun kare yerine kisa bir **dizi** gorup "
        "olayin akisini (baslangic->gelisim->sonuc) muhakeme edebilir."
    )
    code(
        "from src.sampler.context.representative_frame_extractor import RepresentativeFrameExtractor\n"
        "\n"
        "clusters = sampler.cluster_events(evidence)\n"
        "extractor = RepresentativeFrameExtractor(\n"
        "    config.sampler.pre_peak_offset_sec, config.sampler.post_peak_offset_sec)\n"
        "for c in clusters:\n"
        "    c.representative_frames = extractor.extract(VIDEO_PATH, c.peak_frame)\n"
        "\n"
        "print(f'{len(clusters)} olay grubu')\n"
        "for c in clusters:\n"
        "    roles = [(rf.label, rf.timestamp_str) for rf in c.representative_frames]\n"
        "    print(f'  Olay #{c.event_id}: {roles}')"
    )

    md(
        "## 3) VLM — Gorsel Anlama\n"
        "Secilen kareler VLM'e gonderilir (Gemini veya mock). VLM iki sey uretir:\n"
        "- **Insan-okur** Turkce sahne gozlemi,\n"
        "- **Makine-okur** yapilandirilmis olaylar (`EVENTS_JSON`: tip/zaman/guven)."
    )
    code(
        "from src.vlm.factory import get_vlm_client\n"
        "\n"
        "vlm = get_vlm_client(config.vlm, use_mock=config.app.use_mock_vlm)\n"
        "vlm_response = vlm.describe_events(clusters, prompt='Sahnede riskli bir durum var mi degerlendir.')\n"
        "\n"
        "print('Model:', vlm_response.model_name)\n"
        "print('\\n--- VLM Gozlemi ---\\n')\n"
        "print(vlm_response.description)\n"
        "print('\\n--- Yapilandirilmis olaylar (EVENTS_JSON) ---')\n"
        "print(vlm_response.structured_events)"
    )

    md(
        "## 4) Olay Tespiti (EventEngine)\n"
        "VLM ciktisi tipli `DetectedEvent`lere cevrilir. **Once** VLM'in dogrudan urettigi yapilandirilmis "
        "olaylar kullanilir (model-tabanli); yoksa anahtar-kelime + olumsuzlama tespiti **yedek** olarak devreye girer."
    )
    code(
        "from src.event_analysis.event_engine import EventEngine\n"
        "from src.event_analysis.schemas import EventEngineInput\n"
        "\n"
        "engine_input = EventEngineInput.from_vlm_response(vlm_response, timestamp=clusters[-1].end_time)\n"
        "detected = EventEngine().detect(engine_input)\n"
        "for d in detected:\n"
        "    print(f'  {d.event_type:<24} guven={d.confidence:.2f}  keywords={d.matched_keywords}')"
    )

    md(
        "## 5) Hibrit Bellek / RAG — Ilgili ISG Mevzuati\n"
        "Gozlemle ilgili mevzuat maddeleri anlamsal arama (Embedding + FAISS) ile getirilir. "
        "`USE_FAKE_RAG=True` iken hafif sahte mevzuat kullanilir (agir model indirilmez)."
    )
    code(
        "if USE_FAKE_RAG:\n"
        "    from dataclasses import dataclass\n"
        "    @dataclass\n"
        "    class _Doc:\n"
        "        text: str\n"
        "        score: float = 1.0\n"
        "    class _FakeRAG:\n"
        "        def seed_default_regulations(self): pass\n"
        "        def query(self, q, top_k=None):\n"
        "            return [\n"
        "                _Doc('ISG Yonetmeligi Madde 24: KKD (baret/yelek) zorunludur.'),\n"
        "                _Doc('Operasyonel Kural OK-07: Forklift trafiginde yaya gecitleri acik tutulmalidir.'),\n"
        "            ][:(top_k or 2)]\n"
        "    rag_service = _FakeRAG()\n"
        "else:\n"
        "    from src.memory.embedding_rag_service import EmbeddingRAGService\n"
        "    rag_service = EmbeddingRAGService(config.memory.embedding, config.memory.faiss)\n"
        "    rag_service.seed_default_regulations()\n"
        "\n"
        "for doc in rag_service.query('KKD eksikligi ve forklift yaya yakinligi', top_k=2):\n"
        "    print(' -', doc.text)"
    )

    md(
        "## 6) LangGraph Ajani — Muhakeme ve Karar\n"
        "Ajan; gozlemi, mevzuati ve olay sinyallerini alir, gerekirse araclarini (sql / retriever / timeline / "
        "verification) cagirir ve **sartname-uyumlu JSON** bir karar uretir: risk skoru, seviye, ozet ve aksiyon onerileri."
    )
    code(
        "from src.agent.langgraph_agent import SafirAgent\n"
        "\n"
        "agent = SafirAgent(\n"
        "    llm_config=config.llm, agent_config=config.agent,\n"
        "    event_store=None, rag_service=rag_service, use_mock_llm=config.app.use_mock_llm)\n"
        "\n"
        "context = (\n"
        "    f'## Guncel Gozlem\\n{vlm_response.description}\\n\\n'\n"
        "    '## Kullanici Istemi\\nSahnede riskli bir durum var mi degerlendir.'\n"
        ")\n"
        "decision = agent.run(context)\n"
        "\n"
        "print('Risk skoru :', decision.risk_score, f'({decision.risk_level})')\n"
        "print('Ozet       :', decision.summary)\n"
        "print('Aksiyonlar :', decision.actions)"
    )

    md(
        "## 7) Otomatik Eskalasyon (Human-on-the-Loop)\n"
        "Bloke edici bir operator onayi YOKTUR: sistem risk skoruna gore aksiyon kademesini kendisi secer. "
        "Yuksek/kritik riskte saha alarmi **otomatik** tetiklenir; operator sonradan denetler."
    )
    code(
        "from src.decision.escalation import EscalationPolicy\n"
        "\n"
        "policy = EscalationPolicy(config.escalation)\n"
        "esc = policy.evaluate(\n"
        "    risk_score=decision.risk_score, risk_level=decision.risk_level,\n"
        "    recommended_action=decision.recommended_action, summary=decision.summary)\n"
        "\n"
        "print('Kademe          :', esc.tier.value)\n"
        "print('Otomatik alarm  :', esc.auto_dispatched)\n"
        "print('alert_id        :', esc.alert_id)\n"
        "print('Gerekce         :', esc.reason)"
    )

    md(
        "## 8) Uctan Uca Entegre Kosu — Nihai Rapor\n"
        "Yukaridaki tum asamalar `SafirPipeline.run()` icinde birlesir. Tek cagriyla nihai "
        "**sartname-uyumlu JSON** raporu uretilir (ozet, olaylar, risk, aksiyonlar)."
    )
    code(
        "import json\n"
        "import src.main as safir_main\n"
        "\n"
        "if USE_FAKE_RAG:\n"
        "    safir_main.EmbeddingRAGService = lambda *a, **k: rag_service  # agir modeli indirme\n"
        "\n"
        "pipeline = safir_main.SafirPipeline(config)\n"
        "report = pipeline.run(VIDEO_PATH, 'Sahnede riskli bir durum var mi degerlendir.')\n"
        "\n"
        "print('risk         :', report.risk_score, f'({report.risk_level})')\n"
        "print('eskalasyon   :', report.escalation_tier, '| otomatik:', report.auto_dispatched)\n"
        "print('tespit tipler:', report.detected_event_types)\n"
        "print('\\n--- SARTNAME UYUMLU JSON ---')\n"
        "print(json.dumps(report.to_sartname_json(), ensure_ascii=False, indent=2))"
    )

    md(
        "---\n"
        "### Ozet\n"
        "Bu defter, SAFIR'in **CPU suzgec -> VLM -> olay tespiti -> RAG -> ajan -> otomatik eskalasyon -> rapor** "
        "akisini adim adim gosterdi. Gercek Gemini ciktisi icin `USE_MOCK=False` (ve `GEMINI_API_KEY`), tam offline "
        "demo icin `USE_MOCK=True` yeterlidir. Operator paneli (Streamlit) ayni backend'i kullanir: "
        "`streamlit run src/ui/dashboard.py`."
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="USE_MOCK=True olarak uret.")
    parser.add_argument("--out", default=str(ROOT / "notebooks" / "SAFIR_walkthrough.ipynb"))
    args = parser.parse_args()

    nb = build(use_mock_default=args.mock)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    print("Yazildi:", out)


if __name__ == "__main__":
    main()
