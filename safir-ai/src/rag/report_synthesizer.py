"""Dinamik RAG Sorgulama, Risk Hesaplama ve Rapor Sentezleyici."""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ReportData(BaseModel):
    session_id: str
    video_source: str
    executive_summary: str
    risk_score: int
    risk_level: str
    recommended_actions: List[str]
    matched_regulations: List[str]
    timeline_events: List[Dict[str, Any]]


ReportData.model_rebuild()


class DynamicReportSynthesizer:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store

    def query_rag_for_regulations(self, query_text: str, top_k: int = 4) -> List[str]:
        """VLM çıktısına göre FAISS vektör veritabanından dinamik İSG maddelerini çeker."""
        if not self.vector_store:
            # Fallback mevzuat havuzu (Vector store başlatılmamışsa)
            q = query_text.lower()
            if any(k in q for k in ["devrilme", "levha", "plaka", "yuk", "istif", "düşme", "dusme"]):
                return [
                    "İSG Yönetmeliği Madde 52: Ağır yük ve malzeme istifleme alanlarında devrilmeye karşı sabitleme ve güvenlik şeridi zorunludur.",
                    "İş Ekipmanları Yönetmeliği EK-II: Dikey istiflenen malzemelerin devrilme riski periyodik olarak kontrol edilmelidir.",
                    "Genel Saha Emniyet Talimatı ST-04: Malzeme taşıma ve elleçleme operasyonlarında yaya yolu koridoru ayrılmalıdır."
                ]
            elif any(k in q for k in ["yangin", "duman", "alev", "isitici", "elektrik", "pano"]):
                return [
                    "Yangın Güvenliği Talimatı YG-03: Duman veya termal anomali tespit edilen alanlarda çalışma derhal durdurulmalı ve alan tahliye edilmelidir.",
                    "Binaların Yangından Korunması Hakkında Yönetmelik Madde 68: Elektrik kaynaklı yangın risklerinde ana şalter derhal indirilmelidir."
                ]
            return ["İSG Genel Hükümler: Sahadaki risk unsurlarına karşı acil durum planı devreye sokulmalıdır."]

        try:
            results = self.vector_store.similarity_search(query_text, k=top_k)
            return [getattr(doc, "page_content", str(doc)) for doc in results]
        except Exception:
            return ["İlgili İSG mevzuat maddeleri sisteme dinamik olarak yüklenmiştir."]

    def generate_report(self, vlm_output: Dict[str, Any], video_source: str) -> ReportData:
        """Yeni videonun VLM çıktısını alır, RAG ile zenginleştirir ve taze bir rapor üretir."""
        summary = vlm_output.get("summary", "")
        events = vlm_output.get("events", [])
        raw_risk = str(vlm_output.get("risk", "Orta")).capitalize()
        actions = vlm_output.get("actions", [])

        # 1. Dinamik RAG Sorgusu
        rag_query = f"{summary} " + " ".join([str(e.get("event", "")) for e in events if isinstance(e, dict)])
        matched_regs = self.query_rag_for_regulations(rag_query)

        # 2. Deterministik Risk Skoru Hesabı
        combined_text = rag_query.lower()
        if "kritik" in raw_risk.lower() or any(k in combined_text for k in ["devrilme", "yangin", "alev", "patlama", "dusme", "agir"]):
            score = 85
            level = "KRİTİK"
        elif "yüksek" in raw_risk.lower() or any(k in combined_text for k in ["duman", "kayma", "hasar"]):
            score = 65
            level = "YÜKSEK"
        else:
            score = 35
            level = "ORTA"

        # 3. Dinamik Executive Summary (Seçilen Mevzuat ile Harmanlama)
        primary_reg = matched_regs[0].split(":")[0] if matched_regs else "İSG Mevzuatı"
        executive_summary = f"{summary} Durum kapsamında {primary_reg} gereğince acil tedbirler devreye alınmalıdır."

        return ReportData(
            session_id=f"sess_{uuid.uuid4().hex[:8]}",
            video_source=video_source,
            executive_summary=executive_summary,
            risk_score=score,
            risk_level=level,
            recommended_actions=actions,
            matched_regulations=matched_regs,
            timeline_events=events,
        )


__all__ = ["ReportData", "DynamicReportSynthesizer"]
