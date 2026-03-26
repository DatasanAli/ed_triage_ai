"""
nodes.py
========
All four LangGraph node functions for the triage explainability graph.

Node execution order:
  predict_node  ─┐
                 ├─→  analyze_node  →  synthesize_node
  retrieve_node ─┘

predict_node and retrieve_node run in parallel (fan-out from START).
analyze_node fans-in from both — LangGraph waits for both to complete.
"""

import sys
import os
import time
from collections import Counter
from datetime import datetime, timezone

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage

# Allow imports from src/ when running this module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from retreival.retrieval import EDTriageRAG

from .state import TriageState
from .inference import ModelPrediction
from .prompts import (
    ANALYZE_SYSTEM, ANALYZE_HUMAN, UNCERTAINTY_NOTE,
)

# ── Class index → label (0-based, matches ModelPrediction.LABEL_MAP) ─────────
_LABEL_MAP = {0: "L1-Critical", 1: "L2-Emergent", 2: "L3-Urgent"}

# ── Shared LLM client (AWS Bedrock) ───────────────────────────────────────────
# Auth via IAM role or AWS_PROFILE env var — no API key needed.
# credentials_profile_name is only passed when AWS_PROFILE is set (local dev);
# omitted in SageMaker/Lambda where the instance role provides credentials.
_bedrock_kwargs: dict = dict(
    model       = "us.anthropic.claude-sonnet-4-6",
    region_name = os.getenv("AWS_REGION", "us-east-1"),
    temperature = 0.1,
    max_tokens  = 800,
)
_aws_profile = os.getenv("AWS_PROFILE")
if _aws_profile:
    _bedrock_kwargs["credentials_profile_name"] = _aws_profile

_llm = ChatBedrockConverse(**_bedrock_kwargs)

# ── RAG singleton ─────────────────────────────────────────────────────────────
# EDTriageRAG makes two AWS calls on first use (_init_clients).
# The module-level singleton ensures that happens once per process,
# not once per graph invocation.
_rag_instance: EDTriageRAG | None = None


def _get_rag() -> EDTriageRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = EDTriageRAG()
    return _rag_instance


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_shap_block(shap_features: list, predicted_label: str) -> str:
    """
    Format top SHAP features highlighting internal consistency.

    Features whose direction contains "away from" are contradicting the prediction
    — this is the key signal we want the LLM to reason about.
    """
    if not shap_features:
        return "Feature importance data not available for this prediction."

    supporting    = [f for f in shap_features if "away" not in f.get("direction", "")]
    contradicting = [f for f in shap_features if "away"     in f.get("direction", "")]

    lines = []
    if contradicting:
        lines.append(
            f" {len(contradicting)}/5 top features push AGAINST {predicted_label}:"
        )
        for f in contradicting:
            lines.append(
                f"     {f['feature']:<26} SHAP={f['shap']:+.3f}  ({f['direction']})"
            )
    if supporting:
        lines.append(
            f"✓   {len(supporting)}/5 top features support {predicted_label}:"
        )
        for f in supporting:
            lines.append(
                f"     {f['feature']:<26} SHAP={f['shap']:+.3f}  ({f['direction']})"
            )
    return "\n".join(lines)


def _format_safety_block(safety_flag: bool | None, safety_reason: str | None) -> str:
    """Return a bolded safety warning line, or empty string when flag is not set."""
    if not safety_flag:
        return ""
    reason = safety_reason or "Predicted class conflicts with clinical early-warning scores."
    return f" ENDPOINT SAFETY FLAG: {reason}"


def _parse_analyze_response(response_text: str, model_class: int) -> dict:
    """
    Extract structured fields from the analyze_node LLM response.

    Expected format:
        REASONING: [prose]
        RECOMMENDED ESI: [1/2/3] — [justification]
        AGREEMENT: [AGREE/DISAGREE] — [explanation]

    Falls back gracefully when the LLM deviates from the format.
    """
    clinical_analysis = response_text   # fallback: full text
    llm_esi           = None
    llm_agreement     = None

    lines = response_text.split("\n")

    # ── Extract REASONING block ───────────────────────────────────────────────
    reasoning_lines: list[str] = []
    in_reasoning = False
    for line in lines:
        stripped = line.strip()
        upper    = stripped.upper()
        if upper.startswith("REASONING:"):
            in_reasoning = True
            rest = stripped[len("REASONING:"):].strip()
            if rest:
                reasoning_lines.append(rest)
        elif upper.startswith("RECOMMENDED ESI:") or upper.startswith("AGREEMENT:"):
            in_reasoning = False
        elif in_reasoning and stripped:
            reasoning_lines.append(stripped)
    if reasoning_lines:
        clinical_analysis = " ".join(reasoning_lines)

    # ── Extract RECOMMENDED ESI ───────────────────────────────────────────────
    for line in lines:
        clean = line.strip().lstrip("*# ")
        if "RECOMMENDED ESI:" in clean.upper():
            try:
                after = clean.upper().split("RECOMMENDED ESI:")[1].strip()
                candidate = int(after[0])
                if candidate in (1, 2, 3):
                    llm_esi = candidate
            except (ValueError, IndexError):
                pass

    # ── Extract AGREEMENT ─────────────────────────────────────────────────────
    for line in lines:
        clean = line.strip().lstrip("*# ")
        if "AGREEMENT:" in clean.upper():
            rest = clean.upper().split("AGREEMENT:")[1].strip()
            if rest.startswith("AGREE") and not rest.startswith("DISAGREE"):
                llm_agreement = True
            elif rest.startswith("DISAGREE"):
                llm_agreement = False

    # Derive agreement from ESI comparison if not explicitly parsed
    if llm_esi is not None and llm_agreement is None:
        llm_agreement = ((llm_esi - 1) == model_class)

    return {
        "clinical_analysis": clinical_analysis,
        "llm_esi":           llm_esi,
        "llm_agreement":     llm_agreement,
    }


def _reconcile(model_class: int, llm_esi: int | None) -> tuple[int, str]:
    """
    Return (reconciled_class, reconciled_label).

    Always takes the more cautious (lower class index = more urgent) of model vs LLM.
    Falls back to model_class when llm_esi is None.
    """
    if llm_esi is None:
        return model_class, _LABEL_MAP[model_class]
    llm_class      = llm_esi - 1   # ESI 1→class 0, ESI 2→class 1, ESI 3→class 2
    reconciled     = min(model_class, llm_class)
    return reconciled, _LABEL_MAP[reconciled]


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: predict_node
# ─────────────────────────────────────────────────────────────────────────────

def predict_node(state: TriageState) -> dict:
    """
    Normalize and annotate the arch4 model prediction.

    Reads : state["prediction"] (optional), state["patient"]
    Writes: state["model_output"], state["shap_features"],
            state["safety_flag"], state["safety_reason"]

    Two modes:
      1. Prediction pre-supplied — state["prediction"] is a dict/list ->
         normalize it directly via ModelPrediction.normalize() (fast, no I/O).
      2. Prediction absent — state["prediction"] is None ->
         ModelRunner.get().predict() runs the full model pipeline,
         loading artifacts from S3 on the first call (cold start ~30-60 s).

    When the full SageMaker endpoint response is supplied as prediction, this node
    also extracts top_features (SHAP), safety_flag, and safety_reason so that
    analyze_node can surface them in the LLM prompt. It also handles the endpoint's
    dict-format probabilities, converting them to the list format ModelPrediction expects.

    If either path fails, writes an error to state["errors"] and returns a
    safe fallback (L3-Urgent, uncertainty flagged) so the graph continues.
    """
    new_errors    = []
    shap_features = None
    safety_flag   = None
    safety_reason = None

    try:
        raw_prediction = state.get("prediction")

        if raw_prediction is None:
            # Run the model — loads S3 artifacts on first call (singleton)
            from .model_runner import ModelRunner
            raw_prediction = ModelRunner.get().predict(state["patient"])

        # Extract supplementary fields from a full endpoint response dict
        if isinstance(raw_prediction, dict):
            shap_features = raw_prediction.get("top_features")
            safety_flag   = raw_prediction.get("safety_flag")
            safety_reason = raw_prediction.get("safety_reason")

            # Endpoint returns probabilities keyed by label; convert to list
            # before passing to ModelPrediction.normalize() which expects a list.
            probs = raw_prediction.get("probabilities")
            if isinstance(probs, dict):
                raw_prediction = {
                    **raw_prediction,
                    "probabilities": [
                        probs.get("L1-Critical", 0.0),
                        probs.get("L2-Emergent", 0.0),
                        probs.get("L3-Urgent/LessUrgent", probs.get("L3-Urgent", 0.0)),
                    ],
                }

        model_output = ModelPrediction.normalize(raw_prediction)

    except Exception as e:
        new_errors.append(f"predict_node: failed — {e}")
        # Safe fallback: treat as L3-Urgent with max uncertainty
        model_output = {
            "predicted_class":  2,
            "predicted_label":  "L3-Urgent",
            "probabilities":    [0.0, 0.0, 1.0],
            "confidence":       1.0,
            "uncertainty_flag": True,
            "prob_breakdown":   {"L1-Critical": "0%", "L2-Emergent": "0%", "L3-Urgent": "100%"},
        }

    return {
        "model_output":   model_output,
        "shap_features":  shap_features,
        "safety_flag":    safety_flag,
        "safety_reason":  safety_reason,
        "errors":         new_errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: retrieve_node
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_node(state: TriageState) -> dict:
    """
    Retrieve top-5 similar historical cases from Pinecone via EDTriageRAG.

    Reads : state["patient"]
    Writes: state["similar_cases"], state["cases_text"], state["retrieval_ms"]

    Uses the module-level _rag_instance singleton to avoid re-initializing
    AWS (Secrets Manager) and Pinecone clients on every invocation.

    Non-fatal error handling:
      - 0 cases returned   -> warning added to errors, cases_text set to fallback
      - Low similarity     -> warning added (nurse should weight cases cautiously)
      - Any exception      -> error logged, fallback text used, graph continues
    """
    new_errors = []

    try:
        rag = _get_rag()
        cases, elapsed_ms = rag.retrieve_cases(state["patient"], top_k=5)

        if not cases:
            new_errors.append("retrieve_node: no similar cases found in Pinecone index")
            cases_text = "No similar historical cases found in the database."
        else:
            # Warn if best match is below the clinical similarity threshold
            best_score = cases[0]["score"]
            if best_score < 0.65:
                new_errors.append(
                    f"retrieve_node: best similarity score is {best_score:.3f} "
                    f"(< 0.65) — historical cases may not be closely matched"
                )
            cases_text = rag.format_cases_for_prompt(cases)

    except Exception as e:
        new_errors.append(f"retrieve_node: RAG retrieval failed — {e}")
        cases      = []
        cases_text = "Historical case retrieval unavailable."
        elapsed_ms = 0.0

    return {
        "similar_cases": cases,
        "cases_text":    cases_text,
        "retrieval_ms":  elapsed_ms,
        "errors":        new_errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: analyze_node  (LLM call #1)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_node(state: TriageState) -> dict:
    """
    Produce an independent clinical assessment with structured agreement signal.

    Reads : state["patient"], state["model_output"], state["cases_text"],
            state["similar_cases"], state["shap_features"],
            state["safety_flag"], state["safety_reason"]
    Writes: state["clinical_analysis"], state["rag_escalation_flag"],
            state["rag_majority_label"], state["llm_esi"],
            state["llm_agreement"], state["reconciled_label"],
            state["reconciled_class"]

    Fan-in node — LangGraph guarantees both predict_node and retrieve_node
    have completed before this runs.

    Key design: the LLM is asked to reason INDEPENDENTLY first (over SHAP evidence
    and RAG cases), then compare to the model. This prevents anchoring bias where
    the LLM would simply justify the model's answer rather than scrutinise it.
    SHAP features whose direction is "away from <predicted_label>" are surfaced
    as internal contradictions — the single most reliable rule-free escalation signal.

    LLM: claude-sonnet-4-6, temperature=0.1, max_tokens=700
    """
    new_errors = []
    patient = state["patient"]
    mo      = state["model_output"] or {}

    model_class      = mo.get("predicted_class", 2)
    predicted_label  = mo.get("predicted_label", "L3-Urgent")
    uncertainty_note = UNCERTAINTY_NOTE if mo.get("uncertainty_flag") else ""

    # ── Format SHAP and safety blocks for prompt ──────────────────────────────
    shap_block   = _format_shap_block(state.get("shap_features"), predicted_label)
    safety_block = _format_safety_block(state.get("safety_flag"), state.get("safety_reason"))

    # ── Pre-compute RAG vs model comparison ──────────────────────────────────
    # ESI levels in Pinecone are 1-based (1=Critical, 2=Emergent, 3=Urgent);
    # model classes are 0-based (0=L1, 1=L2, 2=L3) — offset by 1.
    raw_cases        = state.get("similar_cases") or []
    agree_count      = 0
    comparison_lines = []
    rag_cls_list     = []

    for i, c in enumerate(raw_cases[:5], start=1):
        m       = c.get("metadata", {})
        esi     = m.get("triage_level")
        rag_cls = (int(esi) - 1) if esi is not None else None
        agrees  = (rag_cls == model_class) if rag_cls is not None else False
        if agrees:
            agree_count += 1
        if rag_cls is not None:
            rag_cls_list.append(rag_cls)

        hr_d   = f"{m['heart_rate'] - patient['heart_rate']:+.0f}"  if m.get("heart_rate") else "N/A"
        sbp_d  = f"{m['sbp']        - patient['systolic_bp']:+.0f}" if m.get("sbp")        else "N/A"
        spo2_d = f"{m['spo2']       - patient['spo2']:+.0f}"         if m.get("spo2")       else "N/A"

        comparison_lines.append(
            f"  Case {i} (sim={c.get('score', 0):.2f}, ESI {esi}, dx={m.get('icd_title', '?')}, "
            f"outcome={m.get('disposition', '?')}, "
            f"{'AGREES' if agrees else 'DISAGREES with model'}): "
            f"HR {m.get('heart_rate', '?')} bpm (Δ{hr_d}), "
            f"BP {m.get('sbp', '?')}/{m.get('dbp', '?')} mmHg (SBP Δ{sbp_d}), "
            f"SpO2 {m.get('spo2', '?')}% (Δ{spo2_d}), "
            f"CC: {m.get('chief_complaint', '?')}"
        )

    n_cases = len(raw_cases[:5]) or 1
    rag_comparison_block = (
        f"Model predicted: {predicted_label} | "
        f"RAG cases agreeing: {agree_count}/{len(raw_cases[:5])} "
        f"({agree_count / n_cases:.0%})\n"
        + "\n".join(comparison_lines)
    )

    # ── RAG escalation: majority vote (kept for backward compat + transparency) ─
    if rag_cls_list:
        majority_class      = Counter(rag_cls_list).most_common(1)[0][0]
        rag_escalation_flag = majority_class < model_class
        rag_majority_label  = _LABEL_MAP[majority_class]
    else:
        rag_escalation_flag = False
        rag_majority_label  = predicted_label

    # ── Build and send prompt ─────────────────────────────────────────────────
    prompt_text = ANALYZE_HUMAN.format(
        chief_complaint      = patient.get("chief_complaint", "not recorded"),
        age                  = patient.get("age", "unknown"),
        gender               = patient.get("gender", "unknown"),
        heart_rate           = patient.get("heart_rate", "?"),
        systolic_bp          = patient.get("systolic_bp", "?"),
        diastolic_bp         = patient.get("diastolic_bp", "?"),
        resp_rate            = patient.get("resp_rate", "?"),
        temperature          = patient.get("temperature", "?"),
        spo2                 = patient.get("spo2", "?"),
        shap_block           = shap_block,
        safety_block         = safety_block,
        predicted_label      = predicted_label,
        confidence           = mo["confidence"],
        prob_l1              = mo["prob_breakdown"]["L1-Critical"],
        prob_l2              = mo["prob_breakdown"]["L2-Emergent"],
        prob_l3              = mo["prob_breakdown"]["L3-Urgent"],
        uncertainty_note     = uncertainty_note,
        cases_text           = state.get("cases_text") or "No historical cases available.",
        rag_comparison_block = rag_comparison_block,
    )

    try:
        response = _llm.invoke(
            [
                SystemMessage(content=ANALYZE_SYSTEM),
                HumanMessage(content=prompt_text),
            ]
        )
        raw_response = response.content.strip()
    except Exception as e:
        new_errors.append(f"analyze_node: LLM call failed — {e}")
        raw_response = (
            f"REASONING: Clinical analysis unavailable. "
            f"Model predicted {predicted_label} with {mo['confidence']:.0%} confidence."
        )

    # ── Parse structured fields from LLM response ─────────────────────────────
    parsed = _parse_analyze_response(raw_response, model_class)

    # ── Reconcile model vs LLM (always take the more cautious level) ──────────
    reconciled_class, reconciled_label = _reconcile(model_class, parsed["llm_esi"])

    return {
        "clinical_analysis":   parsed["clinical_analysis"],
        "llm_esi":             parsed["llm_esi"],
        "llm_agreement":       parsed["llm_agreement"],
        "reconciled_label":    reconciled_label,
        "reconciled_class":    reconciled_class,
        "rag_escalation_flag": rag_escalation_flag,
        "rag_majority_label":  rag_majority_label,
        "errors":              new_errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: synthesize_node  (deterministic — no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_elapsed(ms: float) -> str:
    """Human-readable elapsed time for display (e.g. '900 ms', '1.2 s')."""
    if ms < 1000:
        return f"{int(round(ms))} ms"
    return f"{ms / 1000:.1f} s"


def synthesize_node(state: TriageState) -> dict:
    """
    Assemble all node outputs into the structured triage recommendation report.

    Reads : all state fields
    Writes: state["final_report"]

    No LLM call. Deterministic format assembly only.
    This is intentionally the only node that touches final_report — all
    natural language content is generated upstream; synthesis just structures it.

    Output shape of final_report:
      {
        "triage_level":       str,    # model's raw prediction: "L1-Critical" / "L2-Emergent" / "L3-Urgent"
        "triage_class":       int,    # 0 / 1 / 2
        "reconciled_level":   str,    # effective recommendation: more cautious of model vs LLM
        "reconciled_class":   int,
        "llm_esi":            int,    # LLM's independent ESI (1/2/3)
        "llm_agreement":      bool,   # True if LLM agrees with model
        "confidence_pct":     int,    # 0-100
        "uncertainty_flag":   bool,
        "probabilities":      dict,   # {"L1-Critical": 0.xx, ...}
        "patient_summary":    dict,   # chief_complaint, age, gender, vitals
        "clinical_rationale": str,    # independent LLM reasoning from analyze_node
        "similar_cases":      list,   # top RAG cases with vitals + outcome
        "cases_retrieved":    int,
        "retrieval_ms":       float,
        "flags":              list,   # escalation signals + uncertainty notes
        "generated_at":       str,    # ISO 8601 UTC
        "model_version":      str,
      }
    """
    mo      = state.get("model_output") or {}
    patient = state.get("patient") or {}
    errors  = list(state.get("errors") or [])

    # ── Clean similar_cases to the relevant fields ────────────────────────────
    raw_cases = state.get("similar_cases") or []
    clean_cases = []
    for c in raw_cases:
        m = c.get("metadata", {})
        raw_level = m.get("triage_level")
        clean_cases.append({
            "case_id":         c.get("case_id", "unknown"),
            "similarity":      round(c.get("score", 0.0), 2),
            # Cast to int so Streamlit dataframes show "2" not "2.0"
            "triage_level":    int(raw_level) if raw_level is not None else None,
            "patient_info":    m.get("patient_info"),
            "chief_complaint": m.get("chief_complaint"),
            "heart_rate":      m.get("heart_rate"),
            "sbp":             m.get("sbp"),
            "dbp":             m.get("dbp"),
            "resp_rate":       m.get("resp_rate"),
            "temp":            m.get("temp"),
            "spo2":            m.get("spo2"),
            "diagnosis":       m.get("icd_title") or m.get("primary_diagnosis"),
            "outcome":         m.get("disposition"),
        })

    # ── Build flags list ──────────────────────────────────────────────────────
    flags = list(errors)  # surface all accumulated errors as flags

    if mo.get("uncertainty_flag"):
        flags.insert(
            0,
            f"Model confidence is {mo.get('confidence', 0):.0%} "
            f"— below threshold for {mo.get('predicted_label', 'this level')}. "
            f"Apply additional clinical judgment."
        )

    # LLM disagreement flag — shown first so it's the most prominent
    if state.get("llm_agreement") is False:
        llm_esi      = state.get("llm_esi")
        llm_label    = _LABEL_MAP.get((llm_esi - 1) if llm_esi else 2, "unknown")
        reconciled   = state.get("reconciled_label") or mo.get("predicted_label", "unknown")
        flags.insert(
            0,
            f"LLM DISAGREEMENT — independent clinical reasoning recommended {llm_label} "
            f"(model predicted {mo.get('predicted_label', 'unknown')}). "
            f"Nursing actions reflect {reconciled} protocol."
        )

    if state.get("rag_escalation_flag"):
        rag_label = state.get("rag_majority_label", "higher level")
        flags.insert(
            0,
            f"RAG ESCALATION — majority of similar historical cases were triaged {rag_label} "
            f"(more urgent than model's {mo.get('predicted_label', 'assignment')}). "
        )

    # ── Assemble final_report ─────────────────────────────────────────────────
    probs = mo.get("probabilities", [0.0, 0.0, 0.0])
    final_report = {
        "triage_level":       mo.get("predicted_label", "unknown"),
        "triage_class":       mo.get("predicted_class"),
        "reconciled_level":   state.get("reconciled_label") or mo.get("predicted_label", "unknown"),
        "reconciled_class":   state.get("reconciled_class"),
        "llm_esi":            state.get("llm_esi"),
        "llm_agreement":      state.get("llm_agreement"),
        "confidence_pct":     int(round(mo.get("confidence", 0) * 100)),
        "uncertainty_flag":   mo.get("uncertainty_flag", False),
        "probabilities": {
            "L1-Critical": round(probs[0], 4),
            "L2-Emergent": round(probs[1], 4),
            "L3-Urgent":   round(probs[2], 4),
        },
        "patient_summary": {
            "chief_complaint": patient.get("chief_complaint"),
            "age":             patient.get("age"),
            "gender":          patient.get("gender"),
            "vitals": {
                "heart_rate":   patient.get("heart_rate"),
                "bp":           f"{patient.get('systolic_bp', '?')}/{patient.get('diastolic_bp', '?')}",
                "resp_rate":    patient.get("resp_rate"),
                "temperature":  patient.get("temperature"),
                "spo2":         patient.get("spo2"),
            },
        },
        "clinical_rationale": state.get("clinical_analysis") or "Not available.",
        "shap_features":      state.get("shap_features") or [],
        "safety_flag":        state.get("safety_flag"),
        "safety_reason":      state.get("safety_reason"),
        "similar_cases":      clean_cases,
        "cases_retrieved":    len(clean_cases),
        "retrieval_ms":       int(round(state.get("retrieval_ms") or 0.0)),
        "retrieval_time_display": _fmt_elapsed(state.get("retrieval_ms") or 0.0),
        "flags":              flags,
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "model_version":      "model_v1",
    }

    return {"final_report": final_report, "errors": errors}
