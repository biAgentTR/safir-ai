"""SAFIR adim-adim Jupyter walkthrough notebook'unu uretir (nbformat ile).

Kullanim:
    python scripts/build_notebook.py                 # notebooks/SAFIR_walkthrough.ipynb (Gemini)
    python scripts/build_notebook.py --mock          # USE_MOCK=True (offline dogrulama)
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
        "**Saha Analiz ve Farkindalik Icin Yapay Zeka Destekli Karar Sistemi** pipeline'ini asama asama "
        "calistirir; her adimin girdi/ciktisi ekranda gorunur. Video **Gemini** ile analiz edilir.\n"
        "\n"
        "Akis: **CPU Sampler → Temsili Kareler → VLM → Olay Tespiti → RAG → Ajan → Otomatik Eskalasyon → Rapor**"
    )

    # 0 - setup
    md(
        "## 0) Kurulum ve Ayarlar\n"
        "- `USE_MOCK=False` → `config.yaml`'daki backend (**Gemini**; `GEMINI_API_KEY` gerekli).\n"
        "- `USE_FAKE_RAG=True` → agir embedding modelini (bge-m3 ~2GB) indirmez; hafif mevzuat kullanir (demo hizli)."
    )
    code(
        "import sys, os\n"
        "from pathlib import Path\n"
        "\n"
        "_here = Path.cwd()\n"
        "PROJECT_ROOT = _here if (_here / 'src').exists() else _here.parent\n"
        "assert (PROJECT_ROOT / 'src').exists(), 'Proje koku (safir-ai/) bulunamadi.'\n"
        "sys.path.insert(0, str(PROJECT_ROOT))\n"
        "os.chdir(PROJECT_ROOT)\n"
        "\n"
        "# ---- AYARLAR ----\n"
        f"USE_MOCK = {use_mock_default}       # False: Gemini | True: offline\n"
        "USE_FAKE_RAG = True    # True: bge-m3 indirmez (hizli) | False: gercek FAISS RAG\n"
        "\n"
        "from src.utils.config_loader import load_config\n"
        "config = load_config()\n"
        "if USE_MOCK:\n"
        "    config = config.model_copy(update={'app': config.app.model_copy(\n"
        "        update={'use_mock_vlm': True, 'use_mock_llm': True})})\n"
        "print('VLM backend :', config.vlm.active_model, '(', config.vlm.models[config.vlm.active_model].model_name, ')')\n"
        "print('LLM backend :', config.llm.active_model)\n"
        "print('MOCK        :', USE_MOCK, '| FAKE_RAG:', USE_FAKE_RAG)\n"
        "if not USE_MOCK and not os.environ.get('GEMINI_API_KEY'):\n"
        "    print('\\n[UYARI] GEMINI_API_KEY tanimli degil. Terminalde:  set GEMINI_API_KEY=...  (veya notebook baslatmadan once export/$env).')"
    )

    # video input widget
    md(
        "## 📥 Video Girisi\n"
        "Asagidaki **kutudan bir video sec** (surukle-birak veya tikla) ya da bir dosya yolu yaz. "
        "Hicbiri verilmezse kucuk bir **sentetik** video otomatik uretilir."
    )
    code(
        "import ipywidgets as widgets\n"
        "from IPython.display import display\n"
        "\n"
        "uploader = widgets.FileUpload(accept='video/*', multiple=False, description='📹 Video sec')\n"
        "path_box = widgets.Text(value='', placeholder='veya yol: data/ornek.mp4', description='Yol:')\n"
        "display(widgets.VBox([uploader, path_box]))\n"
        "print('Video secip ALT hucreyi calistir. (Sentetik istersen ikisini de bos birak.)')"
    )
    code(
        "import tempfile, cv2, numpy as np\n"
        "\n"
        "def _resolve_video():\n"
        "    # 1) Yuklenen dosya (ipywidgets 8: tuple; 7: dict)\n"
        "    val = uploader.value\n"
        "    if val:\n"
        "        item = (list(val.values())[0] if isinstance(val, dict) else val[0])\n"
        "        name = item.get('name') or (item.get('metadata', {}) or {}).get('name') or 'uploaded.mp4'\n"
        "        content = item['content']\n"
        "        Path('data').mkdir(exist_ok=True)\n"
        "        p = f'data/{name}'\n"
        "        Path(p).write_bytes(bytes(content))\n"
        "        return p\n"
        "    # 2) Elle girilen yol\n"
        "    if path_box.value and Path(path_box.value).exists():\n"
        "        return path_box.value\n"
        "    # 3) Sentetik fallback\n"
        "    tmp = tempfile.mkdtemp(); p = str(Path(tmp) / 'synthetic.mp4')\n"
        "    frames = [np.full((120, 160, 3), 30, np.uint8) for _ in range(60)]\n"
        "    for i in range(20, 38):\n"
        "        cv2.rectangle(frames[i], (20, 20), (140, 100), (210, 210, 210), -1)\n"
        "    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (160, 120))\n"
        "    for f in frames: w.write(f)\n"
        "    w.release()\n"
        "    print('(Sentetik video uretildi.)')\n"
        "    return p\n"
        "\n"
        "VIDEO_PATH = _resolve_video()\n"
        "print('Kullanilan video:', VIDEO_PATH)"
    )

    # 1 - sampler
    md(
        "## 1) Adaptive Frame Sampler (CPU)\n"
        "Video CPU'da taranir; yalnizca **kanit kareleri** secilir (gurultu tabani dusulmus degisim). "
        "VLM'e sadece bunlar gider — GPU tasarrufu buradan."
    )
    code(
        "from src.sampler.adaptive_sampler import sampler_from_config\n"
        "\n"
        "sampler = sampler_from_config(config.sampler)\n"
        "evidence = sampler.process_video(VIDEO_PATH, sample_fps=config.sampler.sample_fps)\n"
        "s = sampler.last_run_stats\n"
        "print(f'Kanit karesi     : {len(evidence)}')\n"
        "print(f'Taranan ham kare : {s.total_frames_scanned}')\n"
        "print(f'Elenen kare      : {s.eliminated_frame_count}  (GPU tasarrufu %{s.eliminated_ratio_pct})')\n"
        "print(f'Sure             : {s.elapsed_sec}s')"
    )

    # 2 - clusters + representative frames SHOWN
    md(
        "## 2) Olay Kumeleme + VLM'e Gidecek Kareler (pre / peak / post)\n"
        "Kanit kareleri **Olay Gruplari**na kumelenir; her grup icin zirvenin oncesi/sonrasi da eklenir. "
        "**Asagida VLM'e GONDERILECEK kareler tam olarak gorunur** — model bunlari bir dizi olarak yorumlayacak."
    )
    code(
        "import base64\n"
        "from IPython.display import Image as IPyImage, display\n"
        "\n"
        "# Olay kumeleme sirasinda FrameSelector otomatik olarak calisir; VLM'e\n"
        "# gidecek kareler ile arsivlenecek kareler AYNI kaynaktan (c.representative_frames) gelir.\n"
        "clusters = sampler.cluster_events(evidence)\n"
        "\n"
        "print(f'{len(clusters)} olay grubu\\n')\n"
        "for c in clusters:\n"
        "    print(f'=== Olay #{c.event_id} — VLM\\'e gidecek {len(c.representative_frames)} kare ===')\n"
        "    imgs = []\n"
        "    for rf in c.representative_frames:\n"
        "        _, b64 = rf.base64_image.split(',', 1)\n"
        "        imgs.append(IPyImage(data=base64.b64decode(b64), width=200))\n"
        "        print(f'  • {rf.label:<11} @ {rf.timestamp_str}')\n"
        "    display(widgets.HBox([widgets.Image(value=i.data, format='jpeg', width=200) for i in imgs]))"
    )

    # RAG service (moved up; rule engine + agent need it)
    md(
        "## 3) Hibrit Bellek / RAG servisi\n"
        "Mevzuat aramasini yapan servis. `USE_FAKE_RAG=True` iken hafif sahte mevzuat kullanilir (agir model indirilmez)."
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
        "print('RAG servisi hazir:', type(rag_service).__name__)"
    )

    # 4 - VLM
    md(
        "## 4) VLM — Gorsel Anlama (Gemini)\n"
        "Yukaridaki kareler VLM'e gonderilir. Iki cikti: **insan-okur** Turkce gozlem ve **makine-okur** "
        "yapilandirilmis olaylar (`EVENTS_JSON`: tip/zaman/guven)."
    )
    code(
        "from src.vlm.factory import get_vlm_client\n"
        "\n"
        "vlm = get_vlm_client(config.vlm, use_mock=config.app.use_mock_vlm)\n"
        "vlm_response = vlm.describe_events(clusters, prompt='Sahnede riskli bir durum var mi degerlendir.')\n"
        "\n"
        "print('Model:', vlm_response.model_name, f'({vlm_response.latency_ms:.0f} ms)')\n"
        "print('\\n----- VLM Gozlemi -----\\n')\n"
        "print(vlm_response.description)\n"
        "print('\\n----- Yapilandirilmis olaylar (EVENTS_JSON) -----')\n"
        "for e in vlm_response.structured_events:\n"
        "    print('  ', e)"
    )

    # 5 - event analysis full detail
    md(
        "## 5) Olay Analizi — Neye Karar Verildi?\n"
        "VLM ciktisi tipli olaylara cevrilir (**once** model-tabanli EVENTS_JSON, yoksa anahtar-kelime yedegi), "
        "sonra zamansal iliskilendirme ve **kural motoru** ile tetiklenen ISG kurallari bulunur."
    )
    code(
        "from src.event_analysis.event_engine import EventEngine\n"
        "from src.event_analysis.temporal_reasoner import TemporalReasoner, DEFAULT_RELATION_WINDOW_SEC\n"
        "from src.event_analysis.rule_engine import RuleEngine\n"
        "from src.event_analysis.schemas import EventEngineInput\n"
        "from src.agent.tools import RetrieverTool\n"
        "\n"
        "engine_input = EventEngineInput.from_vlm_response(vlm_response, timestamp=clusters[-1].end_time)\n"
        "detected = EventEngine().detect(engine_input)\n"
        "print('1) Tespit edilen olaylar (DetectedEvent):')\n"
        "for d in detected:\n"
        "    print(f'   - {d.event_type:<24} guven={d.confidence:.2f}  keywords={d.matched_keywords}')\n"
        "\n"
        "temporal = TemporalReasoner(relation_window_sec=DEFAULT_RELATION_WINDOW_SEC).reason(detected)\n"
        "print('\\n2) Zamansal olaylar (TemporalEvent):')\n"
        "for t in temporal:\n"
        "    print(f'   - {t.event_type:<24} tekrar={t.occurrence_count} sure={t.duration:.1f}s')\n"
        "\n"
        "rules = RuleEngine(retriever=RetrieverTool(rag_service)).evaluate(temporal)\n"
        "print('\\n3) Tetiklenen ISG kurallari (RuleMatch):')\n"
        "for r in rules:\n"
        "    print(f'   - [{r.rule_id}] ({r.severity}) {r.rule_description}')"
    )

    # 6 - agent
    md(
        "## 6) LangGraph Ajani — Muhakeme ve Karar\n"
        "Ajan; gozlemi, mevzuati ve olay sinyallerini alir, gerekirse araclarini (sql / retriever / timeline / "
        "verification) cagirir ve **sartname-uyumlu JSON** karar uretir: risk skoru, seviye, ozet, aksiyonlar."
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
        "print('Aksiyonlar :')\n"
        "for a in decision.actions:\n"
        "    print('   -', a)"
    )

    # 7 - escalation
    md(
        "## 7) Otomatik Eskalasyon (Human-on-the-Loop)\n"
        "Bloke edici operator onayi YOK: sistem risk'e gore kademeyi kendisi secer. Yuksek/kritikte saha alarmi "
        "**otomatik** tetiklenir; operator sonradan denetler."
    )
    code(
        "from src.decision.escalation import EscalationPolicy\n"
        "\n"
        "esc = EscalationPolicy(config.escalation).evaluate(\n"
        "    risk_score=decision.risk_score, risk_level=decision.risk_level,\n"
        "    recommended_action=decision.recommended_action, summary=decision.summary)\n"
        "print('Kademe         :', esc.tier.value)\n"
        "print('Otomatik alarm :', esc.auto_dispatched)\n"
        "print('alert_id       :', esc.alert_id)\n"
        "print('Gerekce        :', esc.reason)"
    )

    # 8 - full pipeline
    md(
        "## 8) Uctan Uca Entegre Kosu — Nihai Rapor\n"
        "Tum asamalar `SafirPipeline.run()` icinde birlesir; tek cagriyla nihai **sartname-uyumlu JSON** uretilir."
    )
    code(
        "import json\n"
        "import src.main as safir_main\n"
        "\n"
        "if USE_FAKE_RAG:\n"
        "    safir_main.EmbeddingRAGService = lambda *a, **k: rag_service\n"
        "\n"
        "pipeline = safir_main.SafirPipeline(config)\n"
        "report = pipeline.run(VIDEO_PATH, 'Sahnede riskli bir durum var mi degerlendir.')\n"
        "\n"
        "print('risk          :', report.risk_score, f'({report.risk_level})')\n"
        "print('eskalasyon    :', report.escalation_tier, '| otomatik:', report.auto_dispatched)\n"
        "print('tespit tipler :', report.detected_event_types)\n"
        "print('\\n===== SARTNAME UYUMLU JSON =====')\n"
        "print(json.dumps(report.to_sartname_json(), ensure_ascii=False, indent=2))"
    )

    md(
        "---\n"
        "Bu defter SAFIR'in tam akisini adim adim gosterdi. Operator paneli ayni backend'i kullanir: "
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
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "notebooks" / "SAFIR_walkthrough.ipynb"))
    args = parser.parse_args()
    nb = build(use_mock_default=args.mock)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    print("Yazildi:", out)


if __name__ == "__main__":
    main()
