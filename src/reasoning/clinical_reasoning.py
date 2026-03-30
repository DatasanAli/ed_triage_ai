"""
clinical_reasoning.py
=====================
LLM-based clinical reasoning for ED triage decision support.

Architecture:
  This module provides an INDEPENDENT second opinion on triage acuity.
  It does NOT explain the arch4_v1 model — it reasons from first principles
  using the patient presentation and similar historical cases retrieved via RAG.

  Two-signal system:
    Signal 1: arch4_v1 (discriminative ML model, fast, validated on 8,383 cases)
    Signal 2: LLM reasoning (generative, grounded in RAG evidence, interpretable)

  When signals agree  → high confidence triage decision
  When signals disagree → flag for closer clinical review

Usage:
    from reasoning.clinical_reasoning import ClinicalReasoner

    reasoner = ClinicalReasoner()
    result = reasoner.reason(
        patient=patient_dict,
        model_prediction={"esi_level": 2, "confidence": 0.78, "probabilities": [0.05, 0.78, 0.17]},
        shap_features=[{"feature": "news2_score", "value": 8, "shap": 0.42}, ...],
        retrieved_cases=cases,   # from EDTriageRAG.retrieve_cases()
    )

    print(result["reasoning"])        # clinical narrative
    print(result["llm_esi"])          # LLM's independent ESI recommendation
    print(result["agreement"])        # True/False
    print(result["confidence_note"])  # human-readable confidence statement
"""

import json
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")  # None in SageMaker — uses instance role

# Claude claude-sonnet-4-6 via Bedrock
MODEL_ID = "us.anthropic.claude-sonnet-4-6"

ESI_DESCRIPTIONS = {
    1: "ESI Level 1 — Immediate (life-saving intervention required)",
    2: "ESI Level 2 — Emergent (high-risk, should not wait)",
    3: "ESI Level 3 — Urgent (stable but requires multiple resources)",
    4: "ESI Level 4 — Less Urgent (one resource needed)",
    5: "ESI Level 5 — Non-Urgent (no resources needed)",
}


class ClinicalReasoner:
    """
    Independent LLM-based triage reasoning grounded in RAG evidence.

    Lazy initialization — AWS client only created on first use.
    """

    def __init__(self):
        self._client = None

    def _init_client(self):
        if self._client is not None:
            return
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        self._client = session.client("bedrock-runtime")

    # -------------------------------------------------------------------------
    # Prompt builder
    # -------------------------------------------------------------------------

    def _build_prompt(
        self,
        patient: dict,
        model_prediction: dict,
        shap_features: list[dict],
        retrieved_cases: list[dict],
    ) -> str:
        """
        Build the clinical reasoning prompt.

        The prompt asks Claude to act as an experienced triage clinician
        reviewing the case independently — not to explain a black-box model.
        The model prediction is provided for comparison at the end, not as
        the premise of the reasoning.
        """

        # ── Patient presentation ──────────────────────────────────────────────
        vitals_parts = []
        if patient.get("heart_rate"):
            vitals_parts.append(f"HR {patient['heart_rate']:.0f} bpm")
        if patient.get("systolic_bp") and patient.get("diastolic_bp"):
            vitals_parts.append(f"BP {patient['systolic_bp']:.0f}/{patient['diastolic_bp']:.0f} mmHg")
        if patient.get("resp_rate"):
            vitals_parts.append(f"RR {patient['resp_rate']:.0f}")
        if patient.get("spo2"):
            vitals_parts.append(f"SpO2 {patient['spo2']:.0f}%")
        if patient.get("temperature"):
            vitals_parts.append(f"Temp {patient['temperature']:.1f}F")
        vitals_str = ", ".join(vitals_parts) if vitals_parts else "Not recorded"

        patient_str = []
        if patient.get("age"):
            patient_str.append(f"Age: {patient['age']}")
        if patient.get("gender"):
            patient_str.append(f"Gender: {patient['gender']}")
        if patient.get("arrival_transport"):
            patient_str.append(f"Arrival: {patient['arrival_transport']}")
        if patient.get("pain"):
            patient_str.append(f"Pain: {patient['pain']}/10")

        # ── SHAP features ─────────────────────────────────────────────────────
        shap_str = ""
        if shap_features:
            top = sorted(shap_features, key=lambda x: abs(x["shap"]), reverse=True)[:5]
            shap_lines = []
            for f in top:
                direction = "increased" if f["shap"] > 0 else "decreased"
                shap_lines.append(
                    f"  - {f['feature']} = {f['value']} "
                    f"({direction} acuity prediction, impact={abs(f['shap']):.3f})"
                )
            shap_str = "Key clinical features identified by automated analysis:\n" + "\n".join(shap_lines)

        # ── Retrieved cases ───────────────────────────────────────────────────
        cases_str = ""
        if retrieved_cases:
            lines = []
            for i, case in enumerate(retrieved_cases, 1):
                m = case["metadata"]
                lines.append(f"Case {i} (similarity: {case['score']}):")
                lines.append(
                    f"  Patient: {m.get('patient_info', 'unknown')} | "
                    f"ESI assigned: {int(m.get('triage_level', '?'))} | "
                    f"CC: {m.get('chief_complaint', 'unknown')}"
                )
                vitals = []
                if m.get("heart_rate"): vitals.append(f"HR {m['heart_rate']:.0f}")
                if m.get("sbp") and m.get("dbp"): vitals.append(f"BP {m['sbp']:.0f}/{m['dbp']:.0f}")
                if m.get("spo2"): vitals.append(f"SpO2 {m['spo2']:.0f}%")
                if vitals:
                    lines.append(f"  Vitals: {', '.join(vitals)}")
                if m.get("hpi"):
                    lines.append(f"  History: {m['hpi'][:250]}...")
                lines.append(f"  Diagnosis: {m.get('icd_title', 'unknown')}")
                lines.append(f"  Outcome: {m.get('disposition', 'unknown')}")
                lines.append("")
            cases_str = "\n".join(lines)

        # ── Model prediction ──────────────────────────────────────────────────
        esi_level = model_prediction.get("esi_level", "?")
        confidence = model_prediction.get("confidence", 0)
        probs = model_prediction.get("probabilities", [])
        probs_str = ""
        if probs:
            labels = ["L1-Critical", "L2-Emergent", "L3-Urgent"]
            probs_str = ", ".join(f"{l}: {p:.0%}" for l, p in zip(labels, probs))

        # ── Full prompt ───────────────────────────────────────────────────────
        prompt = f"""You are an experienced emergency department triage nurse with 15 years of clinical experience. You are reviewing a new patient presentation and providing an independent triage assessment.

## Current Patient Presentation

Chief Complaint: {patient.get('chief_complaint', 'Not recorded')}
{', '.join(patient_str)}
Vitals: {vitals_str}
{f"HPI: {patient['hpi']}" if patient.get('hpi') else ""}

{shap_str}

## Similar Historical Cases from ED Records

The following cases from our ED records had similar presentations:

{cases_str}

## Automated Model Reference

An automated ML model (trained on 8,383 ED cases) predicted:
ESI Level {esi_level} (confidence: {confidence:.0%})
{f"Probability breakdown: {probs_str}" if probs_str else ""}

## Your Task

Produce TWO outputs, clearly separated by the exact headers below.

---SHORT---
One to two sentences maximum. State the recommended ESI level and the single most important clinical reason. If any retrieved case had a dangerous diagnosis, worse outcome, or higher acuity despite a similar presentation, add one ⚠️ warning sentence flagging the specific risk (e.g. "⚠️ One similar case had acute limb ischemia — examine peripheral pulses."). If the retrieved cases are consistent with the prediction, no warning is needed. Do not explain, do not list differentials.
Format the ESI line exactly as: "RECOMMENDED ESI: [number] — [one reason]"

---LONG---
Provide a full independent assessment with four sections:

1. **Clinical Assessment** (2-3 sentences): Most likely concern, urgency drivers.

2. **Evidence from Similar Cases** (2-3 sentences): Review critically — do not assume retrieved cases confirm the current acuity. Flag any "wolf in sheep's clothing" patterns: retrieved cases with higher acuity, dangerous diagnoses, or worse outcomes despite similar chief complaint. Note red flag findings (pulseless extremity, AMS, sepsis, etc.) that could apply here.

3. **Triage Recommendation**: Your independent ESI level with justification. If you disagree with the model, explain why and flag for clinical review.
   Format exactly as: "RECOMMENDED ESI: [number] — [justification]"

4. **Confidence**: HIGH, MODERATE, or LOW — with a brief explanation.
"""
        return prompt

    # -------------------------------------------------------------------------
    # LLM call
    # -------------------------------------------------------------------------

    def _call_claude(self, prompt: str) -> str:
        self._init_client()

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.1,   # low temperature — clinical reasoning, not creative
            "messages": [
                {"role": "user", "content": prompt}
            ],
        })

        response = self._client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        return json.loads(response["body"].read())["content"][0]["text"]

    # -------------------------------------------------------------------------
    # Parse LLM response
    # -------------------------------------------------------------------------

    def _split_sections(self, response_text: str) -> tuple[str, str]:
        """Split the response into short and long sections."""
        short = ""
        long  = ""
        if "---SHORT---" in response_text and "---LONG---" in response_text:
            parts = response_text.split("---SHORT---", 1)[1]
            short_part, long_part = parts.split("---LONG---", 1)
            short = short_part.strip()
            long  = long_part.strip()
        else:
            # Fallback: treat entire response as long
            long = response_text.strip()
        return short, long

    def _parse_response(self, response_text: str, model_esi: int) -> dict:
        """
        Extract structured fields from the LLM response.
        Falls back gracefully if parsing fails.
        """
        short, long = self._split_sections(response_text)

        llm_esi    = None
        confidence = None

        # Parse from the long section preferentially, fall back to full text
        parse_target = long if long else response_text
        for line in parse_target.split("\n"):
            stripped = line.strip()
            clean = stripped.lstrip("*# ")
            if "RECOMMENDED ESI:" in clean.upper():
                try:
                    part = clean.upper().split("RECOMMENDED ESI:")[1].strip()
                    llm_esi = int(part.split()[0].strip("—-– *"))
                except (ValueError, IndexError):
                    pass
            lower = stripped.lower()
            if "confidence" in lower:
                upper = stripped.upper()
                if "HIGH" in upper:
                    confidence = "HIGH"
                elif "MODERATE" in upper:
                    confidence = "MODERATE"
                elif "LOW" in upper:
                    confidence = "LOW"

        # Determine agreement
        agreement = (llm_esi == model_esi) if llm_esi else None

        # Build confidence note
        if agreement is True:
            confidence_note = f"Model and clinical reasoning agree on ESI {model_esi}."
        elif agreement is False:
            confidence_note = (
                f"⚠️ Disagreement: Model predicted ESI {model_esi}, "
                f"clinical reasoning recommends ESI {llm_esi}. "
                f"Flag for clinical review."
            )
        else:
            confidence_note = "Could not parse LLM ESI recommendation."

        return {
            "llm_esi":         llm_esi,
            "agreement":       agreement,
            "confidence":      confidence,
            "confidence_note": confidence_note,
            "reasoning_short": short,
            "reasoning_long":  long,
        }

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def reason(
        self,
        patient: dict,
        model_prediction: dict,
        retrieved_cases: list[dict],
        shap_features: list[dict] = None,
    ) -> dict:
        """
        Generate independent clinical reasoning for a triage case.

        Args:
            patient:          Dict with patient data (same format as EDTriageRAG)
            model_prediction: Dict with keys: esi_level, confidence, probabilities
            retrieved_cases:  Output of EDTriageRAG.retrieve_cases()
            shap_features:    List of dicts with keys: feature, value, shap
                              (optional — from LightGBM SHAP analysis)

        Returns:
            Dict with:
              reasoning       — full LLM clinical narrative
              llm_esi         — LLM's recommended ESI level (int)
              model_esi       — model's predicted ESI level (int)
              agreement       — True/False/None
              confidence      — HIGH/MODERATE/LOW
              confidence_note — human-readable agreement statement
              prompt          — the prompt sent to Claude (for debugging)
        """
        prompt = self._build_prompt(
            patient=patient,
            model_prediction=model_prediction,
            shap_features=shap_features or [],
            retrieved_cases=retrieved_cases,
        )

        raw_response = self._call_claude(prompt)
        parsed       = self._parse_response(raw_response, model_prediction.get("esi_level"))

        return {
            "model_esi":  model_prediction.get("esi_level"),
            "prompt":     prompt,
            "raw":        raw_response,
            **parsed,
        }


# -----------------------------------------------------------------------------
# Quick test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from retreival.retrieval import EDTriageRAG

    rag      = EDTriageRAG()
    reasoner = ClinicalReasoner()

    # ESI 2 test case — 80yo female, lethargy/SOB, low BP, ambulance
    patient = {
        "age": 80, "gender": "Female", "race": "WHITE",
        "chief_complaint": "LETHARGY/shortness of breath",
        "heart_rate": 95, "systolic_bp": 99, "diastolic_bp": 47,
        "resp_rate": 28, "temperature": 100.2, "spo2": 100,
        "arrival_transport": "AMBULANCE",
    }

    model_prediction = {
        "esi_level":     2,
        "confidence":    0.78,
        "probabilities": [0.05, 0.78, 0.17],
    }

    # Optional SHAP — simulated for test
    shap_features = [
        {"feature": "news2_score",  "value": 7,    "shap":  0.42},
        {"feature": "sbp",          "value": 99,   "shap":  0.31},
        {"feature": "resp_rate",    "value": 28,   "shap":  0.28},
        {"feature": "age",          "value": 80,   "shap":  0.19},
        {"feature": "mews_score",   "value": 3,    "shap":  0.15},
    ]

    print("Retrieving similar cases...")
    cases, elapsed_ms = rag.retrieve_cases(patient, top_k=3, exclude_id="stay_32822973")
    print(f"Retrieved {len(cases)} cases in {elapsed_ms:.0f}ms")
    print()

    print("Generating clinical reasoning...")
    result = reasoner.reason(
        patient=patient,
        model_prediction=model_prediction,
        retrieved_cases=cases,
        shap_features=shap_features,
    )

    print("─" * 60)
    print("NURSE VIEW (short)")
    print("─" * 60)
    print(result["reasoning_short"])
    print()
    print("─" * 60)
    print("PHYSICIAN VIEW (long)")
    print("─" * 60)
    print(result["reasoning_long"])
    print()
    print("─" * 60)
    print(f"Model ESI:  {result['model_esi']}")
    print(f"LLM ESI:    {result['llm_esi']}")
    print(f"Agreement:  {result['agreement']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Note:       {result['confidence_note']}")
