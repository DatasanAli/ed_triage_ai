"""
graph.py
========
Assembles and compiles the triage explainability LangGraph.

Graph topology:
  START
    ├──→ predict_node   (parallel)  ─┐
    └──→ retrieve_node  (parallel)  ─┤
                                     ↓
                               analyze_node   [LLM — independent triage reasoning]
                                     ↓
                             synthesize_node  [deterministic]
                                     ↓
                                    END

predict_node and retrieve_node fan out from START and run concurrently.
analyze_node fans in from both — LangGraph waits for both upstream nodes
to complete before executing it. No explicit synchronization code needed.

Usage:
    from src.agents.graph import triage_graph

    result = triage_graph.invoke(
        {
            "patient":    patient_dict,
            "prediction": {"predicted_class": 1, "probabilities": [0.12, 0.78, 0.10]},
        },
        config={"configurable": {"thread_id": "visit-abc123"}},
    )
    report = result["final_report"]
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import TriageState
from .nodes import (
    predict_node,
    retrieve_node,
    analyze_node,
    synthesize_node,
)


def build_triage_graph():
    """
    Build and compile the triage explainability StateGraph.

    Returns a compiled LangGraph graph ready for .invoke() / .stream().

    Checkpointer: InMemorySaver (dev/demo only).

    thread_id in configurable maps to the checkpointer's persistence key.
    Reuse the same thread_id to stream intermediate state across calls.
    """
    builder = StateGraph(TriageState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("predict_node",    predict_node)
    builder.add_node("retrieve_node",   retrieve_node)
    builder.add_node("analyze_node",    analyze_node)
    builder.add_node("synthesize_node", synthesize_node)

    # ── Parallel fan-out from START ───────────────────────────────────────────
    builder.add_edge(START, "predict_node")
    builder.add_edge(START, "retrieve_node")

    # ── Fan-in to analyze_node ────────────────────────────────────────────────
    # LangGraph automatically gates analyze_node until BOTH predecessors have
    # written their outputs into state. No explicit synchronization needed.
    builder.add_edge("predict_node",  "analyze_node")
    builder.add_edge("retrieve_node", "analyze_node")

    # ── Sequential chain after fan-in ─────────────────────────────────────────
    builder.add_edge("analyze_node",    "synthesize_node")
    builder.add_edge("synthesize_node", END)

    # ── Compile with in-memory checkpointer ───────────────────────────────────
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Module-level singleton — compiled once at import time.
# All callers share this instance; each invocation is isolated by thread_id.
triage_graph = build_triage_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Quick offline test — run directly to validate graph topology
# (no AWS / Pinecone / Claude credentials needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # ── Mock the two network-dependent nodes ─────────────────────────────────
    def mock_retrieve(state):
        """Stubbed retrieve_node — returns hardcoded cases, no Pinecone call."""
        return {
            "similar_cases": [
                {
                    "case_id": "stay_28451",
                    "score": 0.89,
                    "metadata": {
                        "triage_level": 2,
                        "chief_complaint": "CHEST PAIN",
                        "icd_title": "NSTEMI",
                        "disposition": "ADMITTED",
                        "patient_info": "Gender: Female, Age: 65",
                    },
                }
            ],
            "cases_text": (
                "SIMILAR CASE 1 (similarity: 0.89):\n"
                "  Patient: Gender: Female, Age: 65 | ESI: 2 | Chief Complaint: CHEST PAIN\n"
                "  Diagnosis: NSTEMI\n"
                "  Outcome: ADMITTED\n"
            ),
            "retrieval_ms": 0.0,
            "errors": [],
        }

    def mock_analyze(state):
        """Stubbed analyze_node — returns fixed text, no Claude call."""
        return {
            "clinical_analysis":   (
                "This patient presents with tachycardia (110 bpm) and chest pain, "
                "consistent with ESI Level 2. Historical cases show similar female "
                "patients with chest pain were predominantly admitted with cardiac diagnoses."
            ),
            "rag_escalation_flag": False,
            "rag_majority_label":  "L2-Emergent",
            "llm_esi":             2,
            "llm_agreement":       True,
            "reconciled_label":    "L2-Emergent",
            "reconciled_class":    1,
            "errors":              [],
        }

    # Build graph with mocked nodes
    test_builder = StateGraph(TriageState)
    test_builder.add_node("predict_node",    predict_node)   # real — no network
    test_builder.add_node("retrieve_node",   mock_retrieve)
    test_builder.add_node("analyze_node",    mock_analyze)
    test_builder.add_node("synthesize_node", synthesize_node)  # real — deterministic

    test_builder.add_edge(START, "predict_node")
    test_builder.add_edge(START, "retrieve_node")
    test_builder.add_edge("predict_node",  "analyze_node")
    test_builder.add_edge("retrieve_node", "analyze_node")
    test_builder.add_edge("analyze_node",    "synthesize_node")
    test_builder.add_edge("synthesize_node", END)

    test_graph = test_builder.compile(checkpointer=InMemorySaver())

    # ── Test input ────────────────────────────────────────────────────────────
    test_patient = {
        "age": 68, "gender": "Female", "chief_complaint": "CHEST PAIN",
        "heart_rate": 110, "systolic_bp": 135, "diastolic_bp": 85,
        "resp_rate": 18, "temperature": 98.6, "spo2": 99,
    }
    test_prediction = {"predicted_class": 1, "probabilities": [0.12, 0.78, 0.10]}

    result = test_graph.invoke(
        {"patient": test_patient, "prediction": test_prediction},
        config={"configurable": {"thread_id": "offline-test-001"}},
    )

    report = result["final_report"]

    # ── Assertions ────────────────────────────────────────────────────────────
    assert report["triage_level"]     == "L2-Emergent",  f"Got {report['triage_level']}"
    assert report["triage_class"]     == 1,              f"Got {report['triage_class']}"
    assert report["reconciled_level"] == "L2-Emergent",  f"Got {report['reconciled_level']}"
    assert report["confidence_pct"]   == 78,             f"Got {report['confidence_pct']}"
    assert report["uncertainty_flag"] is False,          "Should not be flagged at 78%"
    assert report["cases_retrieved"]  == 1,              "Expected 1 mock case"
    assert report["clinical_rationale"],                 "clinical_rationale is empty"
    assert report["generated_at"],                       "Missing timestamp"
    assert report["model_version"]    == "model_v1"

    print("All assertions passed.\n")
    print(json.dumps(report, indent=2))
