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

from src.ui.app import SafirDashboardApp


def main() -> None:
    """Streamlit operator panelinin ana giris noktasi."""
    SafirDashboardApp().run()


# `streamlit run src/ui/dashboard.py` betigi `__name__ == "__main__"` ile
# calistirir; bu guard hem streamlit hem dogrudan calistirma icin yeterlidir.
if __name__ == "__main__":
    main()
