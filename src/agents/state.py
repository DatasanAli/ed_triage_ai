"""
state.py
========
TriageState — the shared TypedDict passed between all LangGraph nodes.

Every field is Optional (except the two inputs) so nodes can be tested
in isolation without needing the full pipeline to have run first.
All node outputs write to separate keys to avoid reducer conflicts.
"""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class TriageState(TypedDict):
    # ── Caller-supplied inputs ────────────────────────────────────────────────
    # Must be populated before graph.invoke()

    patient: dict
    """
    Raw patient dict.

    Required keys:
      age (int), gender (str), chief_complaint (str),
      heart_rate (float), systolic_bp (float), diastolic_bp (float),
      resp_rate (float), temperature (float), spo2 (float)

    Optional keys (used by model_runner for richer inference):
      pain (float)             — pain score 0-10; omit or None -> imputed to median
      arrival_transport (str)  — "WALK IN" | "AMBULANCE" | "HELICOPTER" | "UNKNOWN"
      hpi (str)                — history of present illness free text
    """

    prediction: dict
    """
    Model output. Accepted shapes:
      {"predicted_class": int, "probabilities": [float, float, float]}
      {"probabilities": [float, float, float]}   <- predicted_class inferred
      [float, float, float]                      <- raw probs list
    Normalised by predict_node via ModelPrediction.normalize().
    """

    # ── predict_node output ───────────────────────────────────────────────────

    model_output: Optional[dict]
    """
    Annotated prediction dict produced by predict_node.
    Shape:
      {
        "predicted_class":  int,          # 0=L1-Critical, 1=L2-Emergent, 2=L3-Urgent
        "predicted_label":  str,
        "probabilities":    [f, f, f],
        "confidence":       float,        # max(probabilities)
        "uncertainty_flag": bool,         # True if confidence below per-class threshold
        "prob_breakdown":   {str: str},   # {"L1-Critical": "61%", ...}
      }
    """

    # ── retrieve_node output ──────────────────────────────────────────────────

    similar_cases: Optional[list]
    """
    Raw list[dict] from EDTriageRAG.retrieve_cases().
    Each dict: {case_id, score, metadata}.
    """

    cases_text: Optional[str]
    """
    Human-readable case summary from EDTriageRAG.format_cases_for_prompt().
    Fed directly into the ANALYZE_HUMAN prompt template.
    """

    retrieval_ms: Optional[float]
    """Pinecone query latency in milliseconds (observability)."""

    # ── analyze_node output (LLM call #1) ────────────────────────────────────

    clinical_analysis: Optional[str]
    """
    3-5 sentence narrative from LLM explaining:
      1. Which vitals/symptoms drive the triage level
      2. How the presentation compares to historical cases
      3. Any ambiguity or flags
    Surfaced in final_report as clinical_rationale.
    """

    # ── predict_node supplementary outputs ───────────────────────────────────

    shap_features: Optional[list]
    """
    Top SHAP features from the model endpoint response (top_features field).
    Each element: {"feature": str, "shap": float, "direction": "toward/away from <label>"}.
    None when prediction was supplied as raw probs/class without SHAP (e.g. ModelRunner path).
    Passed to analyze_node to surface internal model evidence in the LLM prompt.
    """

    safety_flag: Optional[bool]
    """
    Endpoint-level safety flag: True when the model's predicted class conflicts with
    clinical early-warning scores (e.g. NEWS2 ≥ 7 while predicting L3).
    Surfaced in the analyze_node prompt as a pre-computed red flag.
    """

    safety_reason: Optional[str]
    """Human-readable explanation of safety_flag (e.g. "NEWS2=7 conflicts with L3 prediction")."""

    # ── analyze_node escalation outputs ──────────────────────────────────────

    rag_escalation_flag: Optional[bool]
    """
    True when the RAG majority vote is MORE urgent than the model's predicted class.
    Computed deterministically in analyze_node from the top-5 retrieved cases.
    When True: synthesize_node surfaces a RAG ESCALATION flag in final_report
    and reconciliation takes the more urgent level.
    """

    rag_majority_label: Optional[str]
    """
    The triage label ("L1-Critical" | "L2-Emergent" | "L3-Urgent") corresponding
    to the majority vote across the top-5 RAG cases.
    Surfaced in final_report alongside the escalation flag.
    """

    llm_esi: Optional[int]
    """
    The LLM's independent ESI recommendation (1, 2, or 3) parsed from analyze_node output.
    Derived by asking Claude to reason over patient + SHAP + RAG BEFORE seeing the model label.
    None if the LLM response could not be parsed.
    """

    llm_agreement: Optional[bool]
    """
    True  — LLM recommendation matches model's predicted class.
    False — LLM recommends a different (typically more urgent) level.
    None  — could not be determined (parse failure or llm_esi is None).
    """

    reconciled_label: Optional[str]
    """
    The effective triage label used for nursing protocol selection.
    Always the MORE cautious of model vs LLM: min(model_class, llm_class).
    Falls back to model's predicted_label when llm_esi is None.
    """

    reconciled_class: Optional[int]
    """0-based class index corresponding to reconciled_label (0=L1, 1=L2, 2=L3)."""

    # ── synthesize_node output ────────────────────────────────────────────────

    final_report: Optional[dict]
    """
    Structured nurse-facing output assembled by synthesize_node.
    See synthesize_node docstring for full shape.
    """

    # ── Shared error accumulator ──────────────────────────────────────────────

    errors: Annotated[list, operator.add]
    """
    Non-fatal warnings accumulated across nodes (e.g. low RAG similarity,
    low model confidence). Surfaced as "flags" in final_report.
    Graph does not halt on errors — they are advisory only.

    Uses operator.add as a reducer so that parallel nodes (predict_node and
    retrieve_node) can both append to this list in the same execution step
    without a LangGraph InvalidUpdateError.
    """
