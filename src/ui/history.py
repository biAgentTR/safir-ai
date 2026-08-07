# -*- coding: utf-8 -*-
"""
SAFİR — Geçmiş Olaylar Bölümü (Esra Ersoy, UI / Modül 4)

Backend'in GET /events ucundan geçmiş analiz olaylarını çeker ve operatöre
tablo halinde sunar. Dashboard'a iki satırla bağlanır:

    from history import render_history_section      # import bloğuna
    render_history_section(API_BASE)                # dosyanın en altına

Tasarım notları:
- Panel kuralına uyar: veriye ASLA doğrudan (DB/dosya) dokunmaz, yalnızca
  SAFIR API'sine HTTP ile konuşur (bkz. requirements-dashboard açıklaması).
- Alan adlarında esnektir: backend id/event_id, created_at/timestamp gibi
  farklı adlar dönse de kırılmaz.
- Backend kapalıysa çökmez; sakin bir uyarı gösterir.
"""

import httpx
import pandas as pd
import streamlit as st

RISK_ROZET = {"kritik": "🔴", "yuksek": "🟠", "orta": "🟡", "dusuk": "🟢"}


def _olaylari_getir(api_base: str, limit: int):
    """GET /events çağrısı. (liste, hata) döndürür — hata yoksa None."""
    try:
        r = httpx.get(f"{api_base}/events", params={"limit": limit}, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Backend düz liste de dönebilir, {"events": [...]} sarmalı da
        if isinstance(data, dict):
            data = data.get("events") or data.get("items") or []
        return data, None
    except Exception as e:
        return [], str(e)


def _satira_cevir(ev: dict) -> dict:
    """Bir olay kaydını tabloda gösterilecek satıra çevirir (alan adlarına esnek)."""
    seviye = str(ev.get("risk_level") or ev.get("level") or "").lower()
    return {
        "ID": ev.get("id", ev.get("event_id", "-")),
        "Zaman": str(ev.get("created_at") or ev.get("timestamp") or "-"),
        "Risk": f"{RISK_ROZET.get(seviye, '⚪')} {seviye or '-'}",
        "Skor": ev.get("risk_score", ev.get("score", "-")),
        "Özet": (str(ev.get("description") or ev.get("summary") or ev.get("aciklama") or ""))[:120],
        "Operatör Geri Bildirimi": ev.get("feedback") or ev.get("operator_feedback") or "-",
    }


def render_history_section(api_base: str = "http://localhost:8000") -> None:
    """Geçmiş Olaylar bölümünü çizer. dashboard.py'nin en altından çağrılır."""
    st.divider()
    st.header("📜 Geçmiş Olaylar (SQL)")
    st.caption("Önceki analizlerde kaydedilen olaylar — kaynak: backend `/events` (SQLite: data/safir_events.db)")

    ust1, ust2, _ = st.columns([1, 1, 2])
    with ust1:
        limit = st.number_input("Kayıt sayısı", min_value=5, max_value=200, value=50, step=5)
    with ust2:
        st.write("")  # buton hizalama
        yenile = st.button("🔄 Yenile", key="gecmis_yenile")

    olaylar, hata = _olaylari_getir(api_base, int(limit))

    if hata:
        st.warning(f"Geçmiş olaylar alınamadı — backend çalışıyor mu? Ayrıntı: {hata}")
        return
    if not olaylar:
        st.info("Henüz kayıtlı olay yok. Bir video analiz ettiğinizde olaylar burada listelenecek.")
        return

    # Risk seviyesine göre filtre
    seviyeler = ["hepsi"] + sorted({str(e.get("risk_level") or e.get("level") or "").lower()
                                    for e in olaylar if (e.get("risk_level") or e.get("level"))})
    secili = st.selectbox("Risk seviyesi filtresi", seviyeler, index=0, key="gecmis_filtre")
    if secili != "hepsi":
        olaylar = [e for e in olaylar
                   if str(e.get("risk_level") or e.get("level") or "").lower() == secili]

    st.metric("Listelenen olay", len(olaylar))
    df = pd.DataFrame([_satira_cevir(e) for e in olaylar])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Tek olayın ham JSON detayı (listeden — ekstra API çağrısı gerekmez)
    with st.expander("🔍 Olay detayı (ham JSON)"):
        idler = [str(e.get("id", e.get("event_id"))) for e in olaylar
                 if e.get("id") is not None or e.get("event_id") is not None]
        if idler:
            secili_id = st.selectbox("Olay ID", idler, key="gecmis_detay_id")
            secilen = next((e for e in olaylar
                            if str(e.get("id", e.get("event_id"))) == secili_id), None)
            if secilen:
                st.json(secilen)
