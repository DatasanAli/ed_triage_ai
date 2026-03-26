"""
src/agents
==========
LangGraph triage explainability agent.

Primary exports:
    triage_graph  — compiled LangGraph, call .invoke() with patient + prediction
    TriageState   — TypedDict schema (for type hints in calling code)

Quick start:
    from src.agents import triage_graph

    result = triage_graph.invoke(
        {
            "patient": {
                "age": 68, "gender": "Female", "chief_complaint": "CHEST PAIN",
                "heart_rate": 110, "systolic_bp": 135, "diastolic_bp": 85,
                "resp_rate": 18, "temperature": 98.6, "spo2": 99,
            },
            "prediction": {
                "predicted_class": 2,
                "probabilities": [0.12, 0.78, 0.10],
            },
        },
        config={"configurable": {"thread_id": "visit-abc123"}},
    )
    report = result["final_report"]
    print(report["reconciled_level"])    # "L2-Emergent" (more cautious of model vs LLM)
    print(report["clinical_rationale"]) # LLM independent reasoning
    print(report["flags"])              # escalation signals and uncertainty notes
"""

from .graph import triage_graph
from .state import TriageState

__all__ = ["triage_graph", "TriageState"]
