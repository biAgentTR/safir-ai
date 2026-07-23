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
    # SafirAgent'i mock LLM ile ve arac bagimliliklari olmadan (mock veriye
    # duserek) calistirip uretilen SafirReport karar alanlarini yazdirir.
    import logging

    from src.utils.config_loader import load_config

    logging.basicConfig(level=logging.INFO)

    demo_config = load_config()
    demo_agent = SafirAgent(
        llm_config=demo_config.llm,
        agent_config=demo_config.agent,
        event_store=None,
        rag_service=None,
        use_mock_llm=True,
    )

    demo_context = (
        "## Guncel Gozlem (t=12.0s)\n"
        "Sahada bir personel korumasiz alanda hareket ediyor; baret takmamis.\n\n"
        "## Kullanici Istemi\n"
        "Sahnede riskli bir durum var mi degerlendir.\n"
    )
    demo_decision = demo_agent.run(demo_context)

    print(f"risk_score={demo_decision.risk_score}")
    print(f"risk_level={demo_decision.risk_level}")
    print(f"recommended_action={demo_decision.recommended_action}")
