"""SAFIR Operator Paneli - ince giris noktasi (Streamlit).

Panelin tum mantigi, OOP bilesenlere ayrilmis olarak `src/ui/` paketindedir:
- `api_client.py`     : `SafirApiClient` (tum HTTP cagrilari)
- `theme.py`          : sabitler, CSS temasi, risk rozeti, sesli uyari
- `report_export.py`  : `ReportExporter` (HTML/PDF)
- `components/`       : SidebarView, InputPanel, LiveProgressView, ReportView
                        ve alt paneller (KPI, kanit galerisi, ajan/RAG,
                        olay cizelgesi + otomatik eskalasyon, JSON rapor)
- `app.py`            : `SafirDashboardApp` orkestratoru

Bu dosya yalnizca uygulamayi baslatir; boylece Docker/`streamlit run` hedefi
(`src/ui/dashboard.py`) degismeden kalir.

Calistirma:
    streamlit run src/ui/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run src/ui/dashboard.py` betigi, betik klasorunu (src/ui) path'e
# ekler ama proje kokunu (safir-ai) EKLEMEZ; bu yuzden `import src.ui.*`
# basarisiz olur. Proje kokunu (bu dosyanin 2 ust dizini) sys.path'e ekleyip
# hem streamlit hem dogrudan calistirmada import'un calismasini garanti ederiz.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ui.app import SafirDashboardApp  # noqa: E402


def main() -> None:
    """Streamlit operator panelinin ana giris noktasi."""
    SafirDashboardApp().run()


# `streamlit run src/ui/dashboard.py` betigi `__name__ == "__main__"` ile
# calistirir; bu guard hem streamlit hem dogrudan calistirma icin yeterlidir.
if __name__ == "__main__":
    main()
