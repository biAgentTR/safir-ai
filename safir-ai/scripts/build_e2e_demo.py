"""SAFIR uctan-uca demo notebook'unu uretir: notebooks/safir_end_to_end_demo.ipynb

Gercek pipeline akisini (VIDEO -> Sampler -> Motion Region -> Cluster/Tracking ->
VLM -> Event/Temporal -> Decision -> Report) tek bir baglantili akista, her
asamada INPUT / PROCESSING / OUTPUT gorunur sekilde calistirir. Fake cikti yok.

Kullanim:
    python scripts/build_e2e_demo.py            # Gemini (USE_MOCK=False)
    python scripts/build_e2e_demo.py --mock     # offline dogrulama
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
        "# 🛡️ SAFIR — Uctan Uca (End-to-End) Demo\n"
        "\n"
        "Bir video verildiginde tum sistem tek bir **baglantili** akista calisir; her asamanin "
        "**gercek** ciktisi gorunur:\n"
        "\n"
        "```\n"
        "VIDEO -> VideoLoader -> FrameSampler -> Motion Region -> Cluster (tracking) ->\n"
        "         VLM -> Event/Temporal Analysis -> Decision/Risk -> Report\n"
        "```\n"
        "\n"
        "> **Mimari notu:** Bu sistem **VLM-oncelikli**dir. Klasik bir YOLO detektoru / ByteTrack "
        "takipcisi YERINE, CPU tabanli **Adaptive Frame Sampler** kullanilir: hareket bolgesi "
        "(`motion_bbox`) cikarir, kareleri zaman+IoU ile kumeler (tracking muadili) ve yalnizca "
        "**kanit karelerini** VLM'e gonderir. Nesne/tehlike algisini **VLM** (Gemini) semantik olarak yapar. "
        "Her asamada girdi/isleyen bilesen/cikti ayrica yazilir."
    )

    # 1
    md(
        "## 1) Environment & Configuration\n"
        "Importlar, proje koku, konfigurasyon, cihaz ve model bilgileri. Eksik bir bagimlilik varsa "
        "**acik** bir hata mesaji uretilir (sessizce yarida kalmaz)."
    )
    code(
        "import sys, os, platform\n"
        "from pathlib import Path\n"
        "\n"
        "_here = Path.cwd()\n"
        "PROJECT_ROOT = _here if (_here / 'src').exists() else _here.parent\n"
        "assert (PROJECT_ROOT / 'src').exists(), 'Proje koku (safir-ai/) bulunamadi.'\n"
        "sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)\n"
        "\n"
        "# ---- AYARLAR ----\n"
        f"USE_MOCK = {use_mock_default}     # False: Gemini (config.yaml) | True: offline\n"
        "USE_FAKE_RAG = True   # True: bge-m3 (~2GB) indirmez | False: gercek FAISS RAG\n"
        "\n"
        "# Bagimlilik kontrolu (acik hata)\n"
        "_missing = []\n"
        "for _m in ['cv2', 'numpy', 'langgraph', 'langchain_openai', 'httpx', 'yaml']:\n"
        "    try: __import__(_m)\n"
        "    except Exception as e: _missing.append(f'{_m} ({e})')\n"
        "if _missing:\n"
        "    raise ImportError('Eksik bagimliliklar: ' + ', '.join(_missing) +\n"
        "                      '  ->  pip install -r requirements-gemini.txt')\n"
        "\n"
        "from src.utils.config_loader import load_config\n"
        "config = load_config()\n"
        "if USE_MOCK:\n"
        "    config = config.model_copy(update={'app': config.app.model_copy(\n"
        "        update={'use_mock_vlm': True, 'use_mock_llm': True})})\n"
        "\n"
        "print('Python       :', platform.python_version())\n"
        "print('Device (cfg) :', config.system.device)\n"
        "print('VLM backend  :', config.vlm.active_model, '->', config.vlm.models[config.vlm.active_model].model_name)\n"
        "print('LLM backend  :', config.llm.active_model, '->', config.llm.models[config.llm.active_model].model_name)\n"
        "print('MOCK         :', USE_MOCK, '| FAKE_RAG:', USE_FAKE_RAG)\n"
        "if not USE_MOCK and not os.environ.get('GEMINI_API_KEY'):\n"
        "    print('\\n[UYARI] GEMINI_API_KEY tanimli degil — Gemini cagrilari basarisiz olur.')"
    )

    # 2
    md(
        "## 2) Input Video\n"
        "Videoyu **kutudan sec** (surukle-birak) ya da yol yaz; hicbiri yoksa sentetik uretilir. "
        "Ardindan cozunurluk / FPS / sure / kare sayisi ve ornek kareler gosterilir."
    )
    code(
        "import ipywidgets as widgets\n"
        "from IPython.display import display\n"
        "uploader = widgets.FileUpload(accept='video/*', multiple=False, description='📹 Video sec')\n"
        "path_box = widgets.Text(value='', placeholder='veya yol: data/ornek.mp4', description='Yol:')\n"
        "display(widgets.VBox([uploader, path_box]))\n"
        "print('Video secip ALTTAKI hucreyi calistir.')"
    )
    code(
        "import tempfile, cv2, numpy as np\n"
        "\n"
        "def _resolve_video():\n"
        "    val = uploader.value\n"
        "    if val:\n"
        "        item = (list(val.values())[0] if isinstance(val, dict) else val[0])\n"
        "        name = item.get('name') or (item.get('metadata', {}) or {}).get('name') or 'uploaded.mp4'\n"
        "        Path('data').mkdir(exist_ok=True); p = f'data/{name}'\n"
        "        Path(p).write_bytes(bytes(item['content'])); return p\n"
        "    if path_box.value and Path(path_box.value).exists():\n"
        "        return path_box.value\n"
        "    tmp = tempfile.mkdtemp(); p = str(Path(tmp) / 'synthetic.mp4')\n"
        "    frames = [np.full((240, 320, 3), 30, np.uint8) for _ in range(75)]\n"
        "    for i in range(25, 50):\n"
        "        cv2.rectangle(frames[i], (40, 60), (260, 190), (200, 200, 200), -1)  # 'devrilen' cisim\n"
        "    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (320, 240))\n"
        "    for f in frames: w.write(f)\n"
        "    w.release(); print('(Sentetik video uretildi.)'); return p\n"
        "\n"
        "VIDEO_PATH = _resolve_video()\n"
        "\n"
        "# --- Metadata ---\n"
        "cap = cv2.VideoCapture(VIDEO_PATH)\n"
        "fps = cap.get(cv2.CAP_PROP_FPS) or 0\n"
        "n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
        "w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n"
        "duration = n_frames / fps if fps else 0\n"
        "print('Video      :', VIDEO_PATH)\n"
        "print(f'Cozunurluk : {w}x{h}')\n"
        "print(f'FPS        : {fps:.1f}')\n"
        "print(f'Sure       : {duration:.1f} s')\n"
        "print(f'Kare sayisi: {n_frames}')\n"
        "\n"
        "# --- Ornek kareler ---\n"
        "from IPython.display import Image as IPyImage\n"
        "_imgs = []\n"
        "for idx in [0, n_frames // 2, max(0, n_frames - 1)]:\n"
        "    cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = cap.read()\n"
        "    if ok:\n"
        "        _ok, _buf = cv2.imencode('.jpg', fr)\n"
        "        _imgs.append(widgets.Image(value=_buf.tobytes(), format='jpeg', width=200))\n"
        "cap.release()\n"
        "print('\\nOrnek kareler (bas / orta / son):')\n"
        "display(widgets.HBox(_imgs))"
    )

    # 3
    md(
        "## 3) Frame Sampling  \n"
        "**INPUT:** ham video · **PROCESSING:** `AdaptiveFrameSampler` (CPU, hareket/degisim) · "
        "**OUTPUT:** yalnizca kanit kareleri.  \n"
        "Bu katman, agir YOLO/ByteTrack yerine gecer: GPU'yu VLM'e ayirir."
    )
    code(
        "from src.sampler.adaptive_sampler import sampler_from_config\n"
        "\n"
        "print('INPUT      : ', VIDEO_PATH)\n"
        "print('PROCESSING : AdaptiveFrameSampler (CPU motion/degisim suzgeci)')\n"
        "sampler = sampler_from_config(config.sampler)\n"
        "evidence = sampler.process_video(VIDEO_PATH, sample_fps=config.sampler.sample_fps)\n"
        "s = sampler.last_run_stats\n"
        "ratio_sampled = (100.0 * s.sampled_frames_evaluated / s.total_frames_scanned) if s.total_frames_scanned else 0\n"
        "print('OUTPUT     :')\n"
        "print(f'   Original frames : {s.total_frames_scanned}')\n"
        "print(f'   Sampled frames  : {s.sampled_frames_evaluated}  (sampling ratio %{ratio_sampled:.1f})')\n"
        "print(f'   Evidence frames : {s.evidence_frame_count}  (elenen %{s.eliminated_ratio_pct} -> VLM tasarrufu)')\n"
        "print(f'   Sure            : {s.elapsed_sec}s')"
    )
    code(
        "import base64\n"
        "def _show(image_bytes, width=200):\n"
        "    return widgets.Image(value=image_bytes, format='jpeg', width=width)\n"
        "print('Suzulen kanit kareleri:')\n"
        "row = []\n"
        "for ef in evidence:\n"
        "    row.append(_show(ef.image_bytes))\n"
        "display(widgets.HBox(row))\n"
        "for ef in evidence:\n"
        "    print(f'  [{ef.timestamp_str}] change_score={ef.change_score:.4f} motion_bbox={ef.motion_bbox}')"
    )

    # 4
    md(
        "## 4) Motion Region Perception (dusuk seviye algi)  \n"
        "**INPUT:** kanit kareleri · **PROCESSING:** OpenCV hareket maskesi -> sinirlayici kutu "
        "(`_bbox_from_mask`) · **OUTPUT:** hareketin oldugu bolgenin kutusu.  \n"
        "Not: Sinif etiketli (person/forklift) **semantik** algi VLM'de (Bolum 6) yapilir; burada "
        "CPU'nun buldugu **hareket bolgesi** cizilir."
    )
    code(
        "def _draw_bbox(image_bytes, bbox):\n"
        "    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)\n"
        "    if bbox:\n"
        "        x0, y0, x1, y1 = bbox\n"
        "        cv2.rectangle(arr, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)\n"
        "    ok, buf = cv2.imencode('.jpg', arr)\n"
        "    return buf.tobytes()\n"
        "\n"
        "boxed = [ef for ef in evidence if ef.motion_bbox]\n"
        "print(f'OUTPUT: {len(boxed)}/{len(evidence)} karede hareket bolgesi (motion region) bulundu.')\n"
        "display(widgets.HBox([_show(_draw_bbox(ef.image_bytes, ef.motion_bbox)) for ef in evidence]))\n"
        "for ef in evidence:\n"
        "    print(f'  [{ef.timestamp_str}] motion_region = {ef.motion_bbox}')"
    )

    # 5
    md(
        "## 5) Event Clustering & Tracking (zamansal-mekansal sureklilik)  \n"
        "**INPUT:** kanit kareleri + `motion_bbox` · **PROCESSING:** `cluster_events` — kareleri "
        "zaman araligi ve **bbox IoU** ile birlestirir (ByteTrack muadili sureklilik) · "
        "**OUTPUT:** Olay Gruplari + her grup icin **VLM'e gidecek** temsili kareler (pre/peak/post)."
    )
    code(
        "from src.sampler.context.representative_frame_extractor import RepresentativeFrameExtractor\n"
        "\n"
        "clusters = sampler.cluster_events(evidence)\n"
        "extractor = RepresentativeFrameExtractor(config.sampler.pre_peak_offset_sec, config.sampler.post_peak_offset_sec)\n"
        "for c in clusters:\n"
        "    c.representative_frames = extractor.extract(VIDEO_PATH, c.peak_frame)\n"
        "\n"
        "print(f'OUTPUT: {len(evidence)} kanit karesi -> {len(clusters)} Olay Grubu (surekli olaylar birlestirildi)\\n')\n"
        "for c in clusters:\n"
        "    print(f'=== Olay #{c.event_id}  ({c.start_time:.1f}s - {c.end_time:.1f}s) — VLM\\'e {len(c.representative_frames)} kare ===')\n"
        "    for rf in c.representative_frames:\n"
        "        print(f'   • {rf.label:<11} @ {rf.timestamp_str}')\n"
        "    display(widgets.HBox([_show(base64.b64decode(rf.base64_image.split(',',1)[1])) for rf in c.representative_frames]))"
    )

    # RAG service (needed by rule engine + agent)
    md("### RAG servisi (mevzuat) — kural motoru ve ajan kullanir")
    code(
        "if USE_FAKE_RAG:\n"
        "    from dataclasses import dataclass\n"
        "    @dataclass\n"
        "    class _Doc:\n"
        "        text: str; score: float = 1.0\n"
        "    class _FakeRAG:\n"
        "        def seed_default_regulations(self): pass\n"
        "        def query(self, q, top_k=None):\n"
        "            return [_Doc('ISG Yonetmeligi Madde 24: KKD (baret/yelek) zorunludur.'),\n"
        "                    _Doc('Operasyonel Kural OK-07: Forklift trafiginde yaya gecitleri acik tutulmalidir.')][:(top_k or 2)]\n"
        "    rag_service = _FakeRAG()\n"
        "else:\n"
        "    from src.memory.embedding_rag_service import EmbeddingRAGService\n"
        "    rag_service = EmbeddingRAGService(config.memory.embedding, config.memory.faiss)\n"
        "    rag_service.seed_default_regulations()\n"
        "print('RAG:', type(rag_service).__name__)"
    )

    # 6
    md(
        "## 6) VLM Analysis (semantik algi — Gemini)  \n"
        "**INPUT:** Olay Grubunun pre/peak/post kareleri + prompt · **PROCESSING:** VLM (Gemini) · "
        "**OUTPUT:** Turkce gozlem + yapilandirilmis olaylar (`EVENTS_JSON`).  \n"
        "VLM'e ne gonderildigi (prompt + kareler) ve modelin **ham** yaniti gosterilir."
    )
    code(
        "from src.vlm.factory import get_vlm_client\n"
        "from src.prompts import VLM_OBSERVER_SYSTEM_PROMPT\n"
        "\n"
        "user_prompt = 'Sahnede riskli bir durum var mi degerlendir.'\n"
        "print('INPUT (VLM):')\n"
        "print(f'   • {len(clusters)} olay grubu, toplam {sum(len(c.representative_frames) for c in clusters)} kare')\n"
        "print(f'   • Kullanici istemi: {user_prompt!r}')\n"
        "print('   • Sistem istemi (ilk 160 krk):', VLM_OBSERVER_SYSTEM_PROMPT[:160], '...')\n"
        "print('\\nPROCESSING : VLM =', config.vlm.models[config.vlm.active_model].model_name)\n"
        "\n"
        "vlm = get_vlm_client(config.vlm, use_mock=config.app.use_mock_vlm)\n"
        "vlm_response = vlm.describe_events(clusters, prompt=user_prompt)\n"
        "\n"
        "print('\\nOUTPUT (VLM ham yaniti):\\n')\n"
        "print(vlm_response.description)\n"
        "print('\\nYapilandirilmis olaylar (EVENTS_JSON):')\n"
        "for e in vlm_response.structured_events: print('  ', e)"
    )

    # 7
    md(
        "## 7) Event / Temporal Analysis  \n"
        "**INPUT:** VLM ciktisi · **PROCESSING:** `EventEngine` (tipli olay) -> `TemporalReasoner` "
        "(zaman ekseninde sureklilik/tekrar) -> `RuleEngine` (ISG mevzuati) · "
        "**OUTPUT:** olaylar + tetiklenen kurallar (zaman damgali)."
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
        "temporal = TemporalReasoner(relation_window_sec=DEFAULT_RELATION_WINDOW_SEC).reason(detected)\n"
        "rules = RuleEngine(retriever=RetrieverTool(rag_service)).evaluate(temporal)\n"
        "\n"
        "print('OUTPUT — Tespit edilen olaylar:')\n"
        "for d in detected:\n"
        "    print(f'   {{\"event\": \"{d.event_type}\", \"time\": {d.timestamp:.1f}, \"confidence\": {d.confidence:.2f}}}')\n"
        "print('\\nOUTPUT — Zamansal olaylar (sureklilik):')\n"
        "for t in temporal:\n"
        "    print(f'   {t.event_type:<22} tekrar={t.occurrence_count} sure={t.duration:.1f}s')\n"
        "print('\\nOUTPUT — Tetiklenen ISG kurallari:')\n"
        "for r in rules:\n"
        "    print(f'   [{r.rule_id}] ({r.severity}) {r.rule_description}')"
    )

    # 8
    md(
        "## 8) Decision / Risk Evaluation  \n"
        "**INPUT:** VLM gozlemi + mevzuat + olay sinyalleri · **PROCESSING:** `SafirAgent` (LangGraph, "
        "araclar: sql/retriever/timeline/verification) · **OUTPUT:** risk skoru/seviyesi, ozet, aksiyonlar."
    )
    code(
        "from src.agent.langgraph_agent import SafirAgent\n"
        "\n"
        "agent = SafirAgent(llm_config=config.llm, agent_config=config.agent,\n"
        "                   event_store=None, rag_service=rag_service, use_mock_llm=config.app.use_mock_llm)\n"
        "context = (f'## Guncel Gozlem\\n{vlm_response.description}\\n\\n## Kullanici Istemi\\n{user_prompt}')\n"
        "decision = agent.run(context)\n"
        "\n"
        "print('OUTPUT:')\n"
        "print('   Risk Level :', decision.risk_level.upper(), f'(skor {decision.risk_score}/100)')\n"
        "print('   Ozet       :', decision.summary)\n"
        "print('   Onerilen aksiyonlar:')\n"
        "for a in decision.actions: print('      -', a)\n"
        "\n"
        "from src.decision.escalation import EscalationPolicy\n"
        "esc = EscalationPolicy(config.escalation).evaluate(\n"
        "    risk_score=decision.risk_score, risk_level=decision.risk_level,\n"
        "    recommended_action=decision.recommended_action, summary=decision.summary)\n"
        "print('\\n   Otomatik eskalasyon:', esc.tier.value, '| alarm otomatik tetiklendi:', esc.auto_dispatched)"
    )

    # 9
    md(
        "## 9) Final Report  \n"
        "Tum asamalar `SafirPipeline.run()` icinde **baglantili** calisir; tek cagriyla nihai "
        "**sartname-uyumlu JSON** ve insan-okur ozet uretilir."
    )
    code(
        "import json\n"
        "import src.main as safir_main\n"
        "if USE_FAKE_RAG:\n"
        "    safir_main.EmbeddingRAGService = lambda *a, **k: rag_service\n"
        "\n"
        "pipeline = safir_main.SafirPipeline(config)\n"
        "report = pipeline.run(VIDEO_PATH, user_prompt)\n"
        "\n"
        "print('===== JSON (sartname uyumlu) =====')\n"
        "print(json.dumps(report.to_sartname_json(), ensure_ascii=False, indent=2))\n"
        "\n"
        "print('\\n===== Insan-okur ozet =====')\n"
        "print('Video Summary     :', report.summary or report.natural_language_summary)\n"
        "print('Detected Events   :', report.detected_event_types)\n"
        "print('Critical Moments  :', [f\"{e.timestamp:.1f}s: {e.description[:50]}\" for e in report.timeline])\n"
        "print('Risk Level        :', report.risk_level.upper(), f'({report.risk_score}/100)')\n"
        "print('Recommended Actions:')\n"
        "for a in (report.actions or [report.recommended_action]): print('   -', a)"
    )

    # 10
    md("## 10) End-to-End Summary")
    code(
        "ok = report.event_id is not None\n"
        "print('END-TO-END TEST RESULT')\n"
        "print('=' * 40)\n"
        "print(f'Video            : {VIDEO_PATH}')\n"
        "print(f'Duration         : {duration:.1f} s')\n"
        "print(f'Frames           : {n_frames}')\n"
        "print(f'Sampled Frames   : {s.sampled_frames_evaluated}')\n"
        "print(f'Evidence Frames  : {s.evidence_frame_count}')\n"
        "print(f'Event Groups     : {len(clusters)}')\n"
        "print(f'Detected Events  : {report.detected_event_types}')\n"
        "print(f'VLM Model        : {report.vlm_model}')\n"
        "print(f'LLM Model        : {report.llm_model}')\n"
        "print(f'Risk Level       : {report.risk_level.upper()} ({report.risk_score}/100)')\n"
        "print(f'Escalation       : {report.escalation_tier} (auto={report.auto_dispatched})')\n"
        "print(f'Final Report     : {\"YES\" if ok else \"NO\"}')\n"
        "print('=' * 40)\n"
        "print('STATUS:', 'WORKING' if ok else 'BLOCKED')"
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
    parser.add_argument("--out", default=str(ROOT / "notebooks" / "safir_end_to_end_demo.ipynb"))
    args = parser.parse_args()
    nb = build(use_mock_default=args.mock)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    print("Yazildi:", out)


if __name__ == "__main__":
    main()
