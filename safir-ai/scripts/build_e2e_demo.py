"""SAFIR uctan-uca demo notebook'unu uretir: notebooks/safir_end_to_end_demo.ipynb

FAITHFUL orchestration + observability:
- Notebook production implementasyonunu KOPYALAMAZ.
- Her seyi tek `run()` arkasina SAKLAMAZ.
- Bir `SafirPipeline` orneginin GERCEK asama metotlarini (stage_sample, stage_vlm,
  stage_events, stage_context, stage_decide, stage_escalate, build_report) tek tek
  cagirir ve her cagrinin gercek ciktisini gosterir.
- Varsayilan: GERCEK video (data/ dropdown) + GERCEK Gemini + GERCEK RAG. Mock/fake
  yalnizca ayri, acikca isaretli gelistirici bayraklaridir.

Kullanim:
    python scripts/build_e2e_demo.py            # gercek demo (Gemini + gercek RAG)
    python scripts/build_e2e_demo.py --dev      # DEV_MODE=True (mock+fake, offline dogrulama)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(dev_default: bool) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s))

    md(
        "# 🛡️ SAFIR — Uctan Uca (End-to-End) Teknik Demonstrasyon\n"
        "\n"
        "Bu defter, SAFIR'in **gercek calisan sisteminin teknik aynasidir**. Production `SafirPipeline`'in "
        "**gercek asama metotlarini tek tek cagirir** ve her cagrinin **gercek** ciktisini gosterir.\n"
        "\n"
        "> Notebook production mantigini **kopyalamaz** ve her seyi tek `run()` arkasina **saklamaz** — "
        "orchestration + observability katmanidir. VLM olarak **Gemini** kullanilir.\n"
        "\n"
        "### Gercek pipeline (koddan cikarilmis) ve cagrilan metotlar\n"
        "```\n"
        "VIDEO\n"
        "  ↓  SafirPipeline.stage_sample()   -> AdaptiveFrameSampler.process_video + cluster_events + RepresentativeFrameExtractor\n"
        "  ↓  SafirPipeline.stage_vlm()      -> GeminiVLM.describe_events            → GEMINI\n"
        "  ↓  SafirPipeline.stage_events()   -> EventEngine.detect + TemporalReasoner.reason + RuleEngine.evaluate\n"
        "  ↓  SafirPipeline.stage_context()  -> ContextBuilder.build (SQLite + FAISS RAG)\n"
        "  ↓  SafirPipeline.stage_decide()   -> SafirAgent.run (LangGraph)\n"
        "  ↓  SafirPipeline.stage_escalate() -> EscalationPolicy.evaluate\n"
        "  ↓  SafirPipeline.build_report()   -> SafirReport (+to_sartname_json)\n"
        "```\n"
        "Ayni metotlar `SafirPipeline.run()` tarafindan da ayni sirayla cagrilir (tek gercek kaynak)."
    )

    # 1 - Environment
    md(
        "## 1) Environment & Configuration\n"
        "Proje koku, konfigurasyon, cihaz, model bilgileri. Eksik bagimlilik varsa **acik** hata verilir.\n"
        "\n"
        "> `DEV_MODE` yalnizca gelistirici dogrulamasi icindir (mock VLM/LLM + sahte RAG, offline). "
        "**Mentor demosu icin `DEV_MODE=False`** (gercek Gemini + gercek RAG)."
    )
    code(
        "import sys, os, platform\n"
        "from pathlib import Path\n"
        "_here = Path.cwd()\n"
        "PROJECT_ROOT = _here if (_here / 'src').exists() else _here.parent\n"
        "assert (PROJECT_ROOT / 'src').exists(), 'Proje koku (safir-ai/) bulunamadi.'\n"
        "sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)\n"
        "\n"
        "# ---- MOD ----\n"
        f"DEV_MODE = {dev_default}   # False: GERCEK (Gemini + gercek RAG) | True: offline dogrulama (mock+fake)\n"
        "\n"
        "_missing = []\n"
        "for _m in ['cv2', 'numpy', 'langgraph', 'langchain_openai', 'httpx', 'yaml', 'ipywidgets']:\n"
        "    try: __import__(_m)\n"
        "    except Exception as e: _missing.append(f'{_m} ({e})')\n"
        "if _missing:\n"
        "    raise ImportError('Eksik bagimlilik: ' + ', '.join(_missing) + '  ->  pip install -r requirements-gemini.txt')\n"
        "\n"
        "from src.utils.config_loader import load_config\n"
        "config = load_config()\n"
        "if DEV_MODE:\n"
        "    config = config.model_copy(update={'app': config.app.model_copy(update={'use_mock_vlm': True, 'use_mock_llm': True})})\n"
        "\n"
        "print('Python      :', platform.python_version())\n"
        "print('Device (cfg):', config.system.device)\n"
        "print('VLM backend :', config.vlm.active_model, '->', config.vlm.models[config.vlm.active_model].model_name)\n"
        "print('LLM backend :', config.llm.active_model, '->', config.llm.models[config.llm.active_model].model_name)\n"
        "print('DEV_MODE    :', DEV_MODE, '(False = gercek Gemini + gercek RAG)')\n"
        "if not DEV_MODE and not os.environ.get('GEMINI_API_KEY'):\n"
        "    raise EnvironmentError('GEMINI_API_KEY tanimli degil. Gercek demo icin sart. (Secret: ortam degiskeni; koda yazma.)')"
    )

    # 2 - Input video
    md(
        "## 2) Input Video (gercek video)\n"
        "Videoyu `data/` klasorune koy ve **acilir listeden sec** (en guvenilir). Sonra metadata + ornek kareler gorunur.\n"
        "> Sentetik video yalnizca hicbir video secilmediginde, acikca isaretli bir **fallback**tir; mentor demosunda gercek video kullanin."
    )
    code(
        "import ipywidgets as widgets\n"
        "from IPython.display import display\n"
        "Path('data').mkdir(exist_ok=True)\n"
        "_exts = ('*.mp4', '*.avi', '*.mkv', '*.mov')\n"
        "_vids = sorted({str(p) for e in _exts for p in Path('data').glob(e)})\n"
        "video_dd = widgets.Dropdown(options=['(sentetik-fallback)'] + _vids, value=(_vids[0] if _vids else '(sentetik-fallback)'),\n"
        "                            description='data/ video:', style={'description_width': 'initial'}, layout=widgets.Layout(width='560px'))\n"
        "path_box = widgets.Text(value='', placeholder='veya tam yol', description='Yol:', layout=widgets.Layout(width='560px'))\n"
        "display(widgets.VBox([video_dd, path_box]))\n"
        "print(f'data/ altinda {len(_vids)} video bulundu. Sec -> ALT hucreyi calistir.')\n"
        "print('(data/ye yeni video eklediysen bu hucreyi TEKRAR calistir.)')"
    )
    code(
        "import tempfile, cv2, numpy as np\n"
        "def _resolve_video():\n"
        "    if video_dd.value and video_dd.value != '(sentetik-fallback)' and Path(video_dd.value).exists():\n"
        "        print('Kaynak: dropdown'); return video_dd.value\n"
        "    if path_box.value and Path(path_box.value).exists():\n"
        "        print('Kaynak: yol kutusu'); return path_box.value\n"
        "    tmp = tempfile.mkdtemp(); p = str(Path(tmp) / 'synthetic.mp4')\n"
        "    frames = [np.full((240, 320, 3), 30, np.uint8) for _ in range(75)]\n"
        "    for i in range(25, 50): cv2.rectangle(frames[i], (40, 60), (260, 190), (200, 200, 200), -1)\n"
        "    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (320, 240))\n"
        "    for f in frames: w.write(f)\n"
        "    w.release(); print('[!] Gercek video secilmedi -> SENTETIK fallback (mentor demosunda gercek video sec).'); return p\n"
        "\n"
        "VIDEO_PATH = _resolve_video()\n"
        "cap = cv2.VideoCapture(VIDEO_PATH)\n"
        "FPS = cap.get(cv2.CAP_PROP_FPS) or 0; N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
        "W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n"
        "DURATION = N_FRAMES / FPS if FPS else 0\n"
        "print(f'Video       : {VIDEO_PATH}')\n"
        "print(f'Cozunurluk  : {W}x{H}\\nFPS         : {FPS:.1f}\\nSure        : {DURATION:.1f} s\\nKare sayisi : {N_FRAMES}')\n"
        "row = []\n"
        "for idx in [0, N_FRAMES // 2, max(0, N_FRAMES - 1)]:\n"
        "    cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = cap.read()\n"
        "    if ok: _ok, _b = cv2.imencode('.jpg', fr); row.append(widgets.Image(value=_b.tobytes(), format='jpeg', width=180))\n"
        "cap.release(); print('Ornek kareler (bas/orta/son):'); display(widgets.HBox(row))"
    )

    # 3 - build pipeline (real construction)
    md(
        "## 3) Production Pipeline'i Kur (gercek __init__ wiring)\n"
        "`SafirPipeline(config)` tum gercek bilesenleri (VLM/Gemini, EventEngine, TemporalReasoner, RuleEngine, "
        "ContextBuilder, SafirAgent, EscalationPolicy, EventStore, RAG) production'daki gibi kurar. Asagida bu "
        "orneginin **gercek asama metotlarini** tek tek cagiracagiz."
    )
    code(
        "if DEV_MODE:\n"
        "    # Sadece DEV: agir bge-m3 (~2GB) indirmemek icin RAG'i hafif sahteyle degistir.\n"
        "    from dataclasses import dataclass\n"
        "    import src.main as safir_main\n"
        "    @dataclass\n"
        "    class _Doc:\n"
        "        text: str; score: float = 1.0\n"
        "    class _FakeRAG:\n"
        "        def seed_default_regulations(self): pass\n"
        "        def query(self, q, top_k=None):\n"
        "            return [_Doc('ISG Yonetmeligi Madde 24: KKD zorunludur.'), _Doc('Operasyonel Kural OK-07: yaya gecidi.')][:(top_k or 2)]\n"
        "    safir_main.EmbeddingRAGService = lambda *a, **k: _FakeRAG()\n"
        "else:\n"
        "    import src.main as safir_main  # gercek EmbeddingRAGService (bge-m3) — ilk kosuda ~2GB inebilir\n"
        "\n"
        "USER_PROMPT = 'Sahnede riskli bir durum var mi degerlendir.'\n"
        "pipeline = safir_main.SafirPipeline(config)\n"
        "print('SafirPipeline hazir. VLM =', type(pipeline._vlm).__name__, '| Agent LLM =', pipeline._agent.model_name)"
    )
    code(
        "# Gorsel yardimcilari\n"
        "import base64\n"
        "def _img(b, width=180): return widgets.Image(value=b, format='jpeg', width=width)\n"
        "def _b64(u): return base64.b64decode(u.split(',', 1)[1])\n"
        "def _draw_bbox(image_bytes, bbox):\n"
        "    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)\n"
        "    if bbox:\n"
        "        x0, y0, x1, y1 = bbox; cv2.rectangle(arr, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)\n"
        "    _ok, buf = cv2.imencode('.jpg', arr); return buf.tobytes()"
    )

    # 4 - stage_sample
    md(
        "## 4) `pipeline.stage_sample()` → Frame Sampling & Motion Region\n"
        "**INPUT:** video · **PROCESSING:** gercek `AdaptiveFrameSampler` (CPU) + `RepresentativeFrameExtractor` · "
        "**OUTPUT:** kanit kareleri + `motion_bbox` + Olay Gruplari."
    )
    code(
        "sampler, evidence, clusters = pipeline.stage_sample(VIDEO_PATH)\n"
        "st = sampler.last_run_stats\n"
        "ratio = (100.0 * st.sampled_frames_evaluated / st.total_frames_scanned) if st.total_frames_scanned else 0\n"
        "print(f'Original frames : {st.total_frames_scanned}')\n"
        "print(f'Sampled frames  : {st.sampled_frames_evaluated}  (%{ratio:.1f})')\n"
        "print(f'Evidence frames : {st.evidence_frame_count}  (elenen %{st.eliminated_ratio_pct})')\n"
        "print(f'Olay Gruplari   : {len(clusters)}')\n"
        "print('\\nKanit kareleri (kirmizi kutu = motion_bbox):')\n"
        "display(widgets.HBox([_img(_draw_bbox(f.image_bytes, f.motion_bbox)) for f in evidence]))\n"
        "print('\\nHer Olay Grubu icin VLM\\'e gidecek pre/peak/post kareler:')\n"
        "for c in clusters:\n"
        "    print(f\"  Olay #{c.event_id} ({c.start_time:.1f}-{c.end_time:.1f}s): {[rf.label for rf in c.representative_frames]}\")\n"
        "    display(widgets.HBox([_img(_b64(rf.base64_image)) for rf in c.representative_frames]))"
    )

    # 5 - stage_vlm
    md(
        "## 5) `pipeline.stage_vlm()` → VLM (Gemini) Girdi & Ham Yanit\n"
        "**INPUT (Gemini'ye giden):** pre/peak/post kareler + prompt · **PROCESSING:** gercek `GeminiVLM.describe_events` · "
        "**OUTPUT:** modelin **ham** yaniti + `EVENTS_JSON`."
    )
    code(
        "from src.prompts import VLM_OBSERVER_SYSTEM_PROMPT\n"
        "print('=== GEMINI INPUT ===')\n"
        "print('Kullanici istemi:', USER_PROMPT)\n"
        "print('Sistem istemi (ilk 200 krk):', VLM_OBSERVER_SYSTEM_PROMPT[:200], '...')\n"
        "print('Gonderilen kare sayisi:', sum(len(c.representative_frames) for c in clusters))\n"
        "\n"
        "vlm_response = pipeline.stage_vlm(clusters, USER_PROMPT)   # <-- gercek Gemini cagrisi\n"
        "\n"
        "print('\\n=== GEMINI RAW OUTPUT (model:', vlm_response.model_name, ') ===\\n')\n"
        "print(vlm_response.description)\n"
        "print('\\n=== Parse edilmis EVENTS_JSON ===')\n"
        "for e in vlm_response.structured_events: print('  ', e)\n"
        "if vlm_response.description.startswith('[HATA]'):\n"
        "    print('\\n[!] Gemini cagrisi basarisiz -> pipeline dayanikliligi devrede (degraded).')"
    )

    # 6 - stage_events
    md(
        "## 6) `pipeline.stage_events()` → Event / Temporal Analysis\n"
        "**PROCESSING:** gercek `EventEngine.detect` → `TemporalReasoner.reason` → `RuleEngine.evaluate` · "
        "**OUTPUT:** tipli olaylar, zamansal sureklilik, tetiklenen ISG kurallari."
    )
    code(
        "detected, temporal, rules, latest_ts = pipeline.stage_events(vlm_response, clusters)\n"
        "print('Tespit edilen olaylar (DetectedEvent):')\n"
        "for d in detected:\n"
        "    print(f'   {{\"event\": \"{d.event_type}\", \"time\": {d.timestamp:.1f}, \"confidence\": {d.confidence:.2f}}}')\n"
        "print('\\nZamansal olaylar (TemporalEvent):')\n"
        "for t in temporal: print(f'   {t.event_type:<22} tekrar={t.occurrence_count} sure={t.duration:.1f}s')\n"
        "print('\\nTetiklenen ISG kurallari (RuleMatch):')\n"
        "for r in rules: print(f'   [{r.rule_id}] ({r.severity}) {r.rule_description}')"
    )

    # 7 - stage_context
    md(
        "## 7) `pipeline.stage_context()` → Ajan Baglami (RAG dahil)\n"
        "**PROCESSING:** gercek `ContextBuilder.build` (SQLite + FAISS RAG) + kural ozeti · "
        "**OUTPUT:** ajana gidecek istem blogu + ilgili mevzuat."
    )
    code(
        "prompt_block, context = pipeline.stage_context(vlm_response, USER_PROMPT, latest_ts, rules)\n"
        "print('Ilgili mevzuat (RAG):')\n"
        "for r in context.relevant_regulations: print('   -', r)\n"
        "print('\\nAjana giden istem blogu (ilk 600 krk):\\n')\n"
        "print(prompt_block[:600], '...')"
    )

    # 8 - stage_decide + stage_escalate
    md(
        "## 8) `pipeline.stage_decide()` + `stage_escalate()` → Karar / Risk\n"
        "**PROCESSING:** gercek `SafirAgent.run` (LangGraph) → `EscalationPolicy.evaluate` · "
        "**OUTPUT:** risk seviyesi, ozet, aksiyonlar, otomatik eskalasyon."
    )
    code(
        "decision = pipeline.stage_decide(prompt_block)     # <-- gercek ajan (Gemini LLM)\n"
        "escalation = pipeline.stage_escalate(decision, vlm_response)\n"
        "print('Risk Level :', decision.risk_level.upper(), f'(skor {decision.risk_score}/100)')\n"
        "print('Ozet       :', decision.summary)\n"
        "print('Aksiyonlar :')\n"
        "for a in decision.actions: print('   -', a)\n"
        "print('\\nOtomatik eskalasyon:', escalation.tier.value, '| alarm otomatik tetiklendi:', escalation.auto_dispatched)\n"
        "print('Gerekce            :', escalation.reason)"
    )

    # 9 - build_report
    md(
        "## 9) `pipeline.build_report()` → Final Report\n"
        "**PROCESSING:** gercek olay kaydi (EventBuilder/History/Store) + `SafirReport` · "
        "**OUTPUT:** sartname-uyumlu JSON + insan-okur ozet."
    )
    code(
        "import json\n"
        "report = pipeline.build_report(\n"
        "    video_source=VIDEO_PATH, sampler=sampler, clusters=clusters, vlm_response=vlm_response,\n"
        "    context=context, decision=decision, escalation=escalation,\n"
        "    temporal_events=temporal, rule_matches=rules, latest_timestamp=latest_ts)\n"
        "\n"
        "print('===== JSON (sartname uyumlu) =====')\n"
        "print(json.dumps(report.to_sartname_json(), ensure_ascii=False, indent=2))\n"
        "print('\\n===== Insan-okur ozet =====')\n"
        "print('Video Summary      :', report.summary or report.natural_language_summary)\n"
        "print('Detected Events    :', report.detected_event_types)\n"
        "print('Critical Moments   :', [f'{e.timestamp:.1f}s: {e.description[:48]}' for e in report.timeline])\n"
        "print('Risk Level         :', report.risk_level.upper(), f'({report.risk_score}/100)')\n"
        "print('Recommended Actions:')\n"
        "for a in (report.actions or [report.recommended_action]): print('   -', a)"
    )

    # 10 - summary
    md("## 10) SAFIR END-TO-END TEST")
    code(
        "gemini_ok = not vlm_response.description.startswith('[HATA]')\n"
        "checks = [\n"
        "    ('Video Input',      Path(VIDEO_PATH).exists()),\n"
        "    ('Frame Sampling',   st.evidence_frame_count > 0),\n"
        "    ('Clustering/Track', len(clusters) > 0),\n"
        "    ('Gemini Inference', gemini_ok),\n"
        "    ('Output Parsing',   isinstance(vlm_response.structured_events, list)),\n"
        "    ('Event Analysis',   len(detected) > 0),\n"
        "    ('Risk/Decision',    decision is not None),\n"
        "    ('Escalation',       escalation is not None),\n"
        "    ('Structured JSON',  bool(report.to_sartname_json())),\n"
        "    ('Final Report',     report.event_id is not None),\n"
        "]\n"
        "print('=' * 42); print('        SAFIR END-TO-END TEST'); print('=' * 42)\n"
        "for name, ok in checks:\n"
        "    mark = '✓' if ok else '✗'\n"
        "    print(f'  {name:<22} {mark}')\n"
        "print('=' * 42)\n"
        "overall = all(ok for _, ok in checks)\n"
        "print('STATUS:', 'PASS' if overall else 'BLOCKED')\n"
        "if not overall:\n"
        "    failed = [n for n, ok in checks if not ok][0]\n"
        "    print('\\nFailed stage :', failed)\n"
        "    if failed == 'Gemini Inference':\n"
        "        print('Error        :', vlm_response.description)\n"
        "        print('Root cause   : Gemini cagrisi basarisiz (kota/model/anahtar).')\n"
        "        print('Cozum        : GEMINI_API_KEY + config.yaml model_name (kotali model).')"
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="DEV_MODE=True (mock+fake, offline).")
    parser.add_argument("--out", default=str(ROOT / "notebooks" / "safir_end_to_end_demo.ipynb"))
    args = parser.parse_args()
    nb = build(dev_default=args.dev)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    print("Yazildi:", out)


if __name__ == "__main__":
    main()
