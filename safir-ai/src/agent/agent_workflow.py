"""Modul 4 - LangGraph Agent Workflow: `SafirAgent`'in kararli genel-erisim noktasi.

Gercek LangGraph durum makinesi (dugumler, kenarlar, risk skorlama, mock/gercek
LLM secimi) `src/agent/langgraph_agent.py` icinde tanimlidir ve test edilmis
haliyle degistirilmeden korunur. Bu modul, Modul 4 spesifikasyonundaki
`src/agent/agent_workflow.py` dosya sozlesmesini saglayan ince bir yeniden
disa-aktarim (re-export) katmanidir; boylece hem `from src.agent.langgraph_agent
import SafirAgent` hem `from src.agent.agent_workflow import SafirAgent` calisir.
"""

from __future__ import annotations

from src.agent.langgraph_agent import AgentDecision, SafirAgent

__all__ = ["AgentDecision", "SafirAgent"]


if __name__ == "__main__":
    # Modul 4'un bagimsiz calistirilabilirlik testi (GPU/vLLM gerekmeden):
    #   python -m src.agent.agent_workflow
    # Sahte VLM (Modul 2'nin MockVLMClient'i) ve sahte RAG verisiyle (sabit
    # ornek ISG mevzuati) zenginlestirilmis bir baglam kurup SafirAgent'in
    # LangGraph durum makinesini mock LLM ile calistirir; sonucu tam bir
    # `SafirReport` JSON'u olarak konsola basar.
    import datetime
    import logging

    from src.rag.embedding_rag_service import DEFAULT_ISG_REGULATIONS
    from src.sampler.schema import EvidenceFrame
    from src.schemas.report import SafirReport
    from src.utils.config_loader import load_config
    from src.vlm.vlm_client import MockVLMClient

    logging.basicConfig(level=logging.INFO)

    demo_config = load_config()

    # --- Sahte VLM verisi (Modul 2) ---
    demo_prompt = "Sahnede riskli bir durum var mi degerlendir."
    demo_evidence_frame = EvidenceFrame(
        evidence_id="ev0",
        frame_id=0,
        timestamp_sec=12.0,
        timestamp_str="00:12",
        change_score=0.42,
        image_bytes=b"",
        base64_image="data:image/jpeg;base64,AA==",
        image_shape=(1, 1, 3),
    )
    mock_vlm = MockVLMClient()
    vlm_response = mock_vlm.analyze_evidence([demo_evidence_frame], prompt=demo_prompt)

    # --- Sahte RAG verisi (Modul 3'un varsayilan ISG mevzuat seti) ---
    mock_regulations = DEFAULT_ISG_REGULATIONS[:2]

    # --- Modul 4: LangGraph Ajani (mock LLM) ---
    demo_agent = SafirAgent(
        llm_config=demo_config.llm,
        agent_config=demo_config.agent,
        event_store=None,
        rag_service=None,
        use_mock_llm=True,
    )

    demo_context = (
        f"## Guncel Gozlem (t={demo_evidence_frame.timestamp_sec:.1f}s)\n{vlm_response.description}\n\n"
        f"## Kullanici Istemi\n{demo_prompt}\n\n"
        "## Ilgili Operasyonel Mevzuat\n" + "\n".join(f"- {r}" for r in mock_regulations)
    )
    demo_decision = demo_agent.run(demo_context)

    demo_report = SafirReport(
        video_source="mock://demo",
        generated_at=datetime.datetime.utcnow().isoformat() + "Z",
        natural_language_summary=vlm_response.description,
        risk_score=demo_decision.risk_score,
        risk_level=demo_decision.risk_level,
        recommended_action=demo_decision.recommended_action,
        relevant_regulations=mock_regulations,
        vlm_model=vlm_response.model_name,
        llm_model=demo_agent.model_name,
    )

    print(demo_report.model_dump_json(indent=2))
