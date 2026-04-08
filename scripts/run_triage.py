"""
run_triage.py
=============
Live end-to-end runner for the ED Triage AI agent graph.

Flow:
  1. Call edtriage-live SageMaker endpoint  →  probabilities + SHAP + safety_flag
  2. Pass full endpoint response into triage_graph.invoke()
  3. predict_node extracts SHAP + normalises probs
  4. retrieve_node pulls similar RAG cases from Pinecone  (runs in parallel)
  5. analyze_node: LLM reasons independently over model + SHAP + RAG
  6. synthesize_node assembles final_report

Prerequisites:
  - AWS_PROFILE=ed-triage (local) OR IAM instance role (SageMaker/EC2)
  - edtriage-live endpoint running
  - Pinecone index populated
  - pip install -r requirements.txt

Usage:
  AWS_PROFILE=ed-triage PYTHONPATH=src python run_triage.py
"""

import json
import sys
import os

import boto3

# Allow `from src.agents...` imports when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.agents.graph import triage_graph

# ── AWS session ───────────────────────────────────────────────────────────────
# Uses AWS_PROFILE env var for local dev; falls back to instance role otherwise.
_profile = os.getenv("AWS_PROFILE")
_region  = os.getenv("AWS_REGION", "us-east-1")
_session = boto3.Session(profile_name=_profile, region_name=_region)

ENDPOINT_NAME = "edtriage-live"


def _build_triage_text(chief_complaint: str, hpi: str = "") -> str:
    """
    Replicate the CC-emphasized text format used during arch4 training.
    Must match inference.py in the SageMaker container exactly.
    """
    cc  = " ".join(str(chief_complaint or "").split()[:24])
    hpi = " ".join(str(hpi or "").split()[:160])
    parts = []
    if cc:  parts.append(f"Chief complaint: {cc}.")
    if cc:  parts.append(f"Presenting with {cc}.")   # CC repeated for emphasis
    if hpi: parts.append(f"History: {hpi}.")
    return " ".join(parts)


def call_endpoint(patient: dict) -> dict:
    """
    Invoke the edtriage-live SageMaker endpoint and return the full response dict.

    Maps agent patient field names → endpoint field names (sbp, dbp, temp_f).
    Returns the raw endpoint JSON including probabilities, top_features, safety_flag.
    """
    payload = {
        "triage_text":       _build_triage_text(
                                 patient.get("chief_complaint", ""),
                                 patient.get("hpi", ""),
                             ),
        "heart_rate":        patient.get("heart_rate"),
        "sbp":               patient.get("systolic_bp"),
        "dbp":               patient.get("diastolic_bp"),
        "resp_rate":         patient.get("resp_rate"),
        "spo2":              patient.get("spo2"),
        "temp_f":            patient.get("temperature"),
        "age":               patient.get("age"),
        "arrival_transport": patient.get("arrival_transport", "UNKNOWN"),
    }
    # Pain is optional — omit entirely if not provided so the endpoint sets pain_missing=1
    if patient.get("pain") is not None:
        payload["pain"] = patient["pain"]

    runtime  = _session.client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName = ENDPOINT_NAME,
        ContentType  = "application/json",
        Accept       = "application/json",
        Body         = json.dumps(payload),
    )
    return json.loads(response["Body"].read())


# ── Patient input ─────────────────────────────────────────────────────────────
# Edit these fields to test different presentations.
patient = {
    "age":             57,
    "gender":          "Female",
    "chief_complaint": "FEVERS CHILLS",
    "heart_rate":      77,
    "systolic_bp":     174,
    "diastolic_bp":    66,
    "resp_rate":       14,
    "temperature":     98.0,
    "spo2":            98,
    # Optional fields — uncomment to include
    # "pain":             5,
    # "arrival_transport": "WALK IN",
    # "hpi":              "Patient reports fever and chills for 2 days with burning on urination.",
}

# ── Call SageMaker endpoint ───────────────────────────────────────────────────
print(f"Calling {ENDPOINT_NAME} endpoint...")
prediction = call_endpoint(patient)
print(f"Model: {prediction.get('predicted_label')}  "
      f"(confidence: {max(prediction.get('probabilities', {}).values(), default=0):.0%})")
if prediction.get("safety_flag"):
    print(f"  SAFETY FLAG: {prediction.get('safety_reason')}")
print()

# ── Run graph ─────────────────────────────────────────────────────────────────
# prediction is the full endpoint response — predict_node extracts SHAP from it.
print("Running triage agent graph...")
print(f"Patient: {patient['age']}yo {patient['gender']} | {patient['chief_complaint']}\n")

result = triage_graph.invoke(
    {"patient": patient, "prediction": prediction},
    config={"configurable": {"thread_id": "visit-001"}},
)

report = result["final_report"]

# ── Print structured report ───────────────────────────────────────────────────
print("=" * 60)
print(f"MODEL PREDICTION : {report['triage_level']}  ({report['confidence_pct']}% confidence)")
print(f"RECONCILED LEVEL : {report['reconciled_level']}"
      + (f"  [escalated from {report['triage_level']}]"
         if report['reconciled_level'] != report['triage_level'] else ""))
if report["uncertainty_flag"]:
    print("  *** LOW CONFIDENCE — apply additional clinical judgment ***")
print()
print("PROBABILITIES:")
for label, prob in report["probabilities"].items():
    print(f"  {label}: {prob:.0%}")
print()
print(f"LLM INDEPENDENT ESI : {report.get('llm_esi')}  "
      f"({'agrees' if report.get('llm_agreement') else 'DISAGREES'} with model)")
print()
print("CLINICAL RATIONALE:")
print(report["clinical_rationale"])
print()
print(f"SIMILAR CASES RETRIEVED: {report['cases_retrieved']}  ({report['retrieval_time_display']})")
for case in report["similar_cases"]:
    print(f"  [{case['similarity']:.2f}] ESI-{case['triage_level']} | "
          f"{case['chief_complaint']} → {case['diagnosis']} | {case['outcome']}")
if report["flags"]:
    print()
    print("FLAGS:")
    for flag in report["flags"]:
        print(f"  ! {flag}")
print("=" * 60)
print()
print("Full JSON report:")
print(json.dumps(report, indent=2))
