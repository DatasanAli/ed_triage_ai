# Clinical Reasoning Layer — Design & Status

## Overview

The clinical reasoning layer provides an independent, interpretable second opinion on ED triage acuity. It is the final synthesis step in a three-part explainability pipeline:

```
Patient Presentation
        │
        ├──► ML Model (arch4_v1)          → ESI prediction + confidence + probabilities
        │
        ├──► RAG Retrieval (Pinecone)      → Top-k similar historical cases
        │
        └──► Clinical Reasoning (Claude)   → Independent ESI recommendation + narrative
                        │
                        └──► Agreement Signal → high confidence OR flag for review
```

---

## Design Philosophy: Independent Second Clinician

The LLM is **not** asked to explain the ML model. It cannot — arch4_v1 is a BioClinicalBERT + LightGBM fusion whose internals are opaque to an LLM. Framing it as an explainer would produce hallucinated rationales that sound plausible but are disconnected from what the model actually learned.

Instead, Claude is framed as an experienced triage nurse reviewing the case from first principles:

- It sees the patient presentation (chief complaint, vitals, demographics, HPI if available)
- It sees similar historical cases retrieved via RAG
- It produces an **independent** ESI recommendation
- The ML model prediction is shown at the end, for comparison only

**Two-signal system:**
| Signals | Interpretation |
|---|---|
| ML model and LLM agree | High confidence triage decision |
| ML model and LLM disagree | Flag for closer clinical review |

---

## What Works Today

### Input
| Field | Source | Status |
|---|---|---|
| Chief complaint | Patient dict | ✅ |
| Vitals (HR, BP, RR, SpO2, Temp) | Patient dict | ✅ |
| Demographics (age, gender, arrival) | Patient dict | ✅ |
| HPI narrative | Patient dict (optional) | ✅ |
| Retrieved similar cases (top-k) | Pinecone RAG | ✅ |
| ML model ESI level | arch4_v1 prediction dict | ✅ |
| ML model confidence + probabilities | arch4_v1 prediction dict | ✅ |
| SHAP feature attributions | LightGBM SHAP | ⚠️ Not yet wired |

### Output
| Field | Description | Status |
|---|---|---|
| `reasoning` | Full clinical narrative from Claude | ✅ |
| `llm_esi` | LLM's independent ESI recommendation (1–5) | ✅ |
| `model_esi` | ML model's predicted ESI | ✅ |
| `agreement` | True / False / None | ✅ |
| `confidence` | HIGH / MODERATE / LOW | ✅ |
| `confidence_note` | Human-readable agreement statement | ✅ |
| `prompt` | Full prompt sent to Claude (for debugging) | ✅ |

---

## What Is Missing

### 1. SHAP Feature Attributions

The prompt has a placeholder for SHAP values from the LightGBM branch of arch4_v1. When provided, they appear in the prompt as:

```
Key clinical features identified by automated analysis:
  - news2_score = 7 (increased acuity prediction, impact=0.42)
  - sbp = 99 (increased acuity prediction, impact=0.31)
  - resp_rate = 28 (increased acuity prediction, impact=0.28)
```

**What needs to happen:**
- Add a SHAP computation cell to `notebooks/arch4_training_v1.ipynb` (or a standalone inference script)
- At inference time, run `shap.TreeExplainer` on the LightGBM model for the patient's structured features
- Pass the top-5 SHAP values as `shap_features` to `ClinicalReasoner.reason()`

Currently `shap_features` defaults to `[]` and the section is omitted from the prompt. The system works without it — SHAP just adds one more grounding signal for the LLM.

### 2. Live arch4_v1 Model Prediction

The system currently requires a pre-computed `model_prediction` dict:

```python
model_prediction = {
    "esi_level":     2,
    "confidence":    0.78,
    "probabilities": [0.05, 0.78, 0.17],  # [L1-Critical, L2-Emergent, L3-Urgent]
}
```

**What needs to happen:**
- Load the saved arch4_v1 model weights from S3
- Build a `TriagePredictor` inference class that takes a patient dict and returns this structure
- Wire `TriagePredictor` → `EDTriageRAG.retrieve_cases()` → `ClinicalReasoner.reason()` into a single `explain()` call

A complete single-call interface would look like:

```python
result = explain(patient)
# Returns: ESI prediction + retrieved cases + clinical reasoning narrative + agreement signal
```

---

## Prompt Design

The prompt instructs Claude to structure its response in four sections:

1. **Clinical Assessment** — most likely concern, urgency drivers
2. **Evidence from Similar Cases** — critical review of retrieved cases, not just confirmation.
   Specifically asks Claude to flag:
   - Retrieved cases with higher acuity or dangerous diagnoses despite similar chief complaint ("wolf in sheep's clothing")
   - Patterns of admission or escalation in retrieved cases
   - Red flag findings in retrieved case histories (pulseless extremity, AMS, etc.)
3. **Triage Recommendation** — `RECOMMENDED ESI: [1-5] — [justification]` (parsed programmatically)
4. **Confidence** — HIGH / MODERATE / LOW with rationale

The ML model prediction is shown after the task instructions so it cannot anchor Claude's reasoning before it forms an independent view.

---

## Infrastructure

| Component | Details |
|---|---|
| Model | `us.anthropic.claude-sonnet-4-6` via Amazon Bedrock |
| Temperature | 0.1 (clinical reasoning, not creative) |
| Max tokens | 1,024 |
| AWS credentials | `ed-triage` named profile or SageMaker IAM role |
| Lazy initialization | AWS client only created on first `reason()` call |

---

## Observed Behavior (Test Cases)

### ESI 2 — 80yo female, lethargy/SOB, BP 99/47, ambulance

- **Model**: ESI 2 (78% confidence)
- **LLM**: ESI 2 — agreed, with note that ESI 1 threshold is low and should be reassessed within minutes
- **Agreement**: True
- **Confidence**: HIGH
- **Notable**: Claude correctly identified the case as borderline ESI 1/2, flagged septic shock / acute decompensation differential, and noted that NEWS2=7 carries ICU-level deterioration risk. Retrieved case with BP 71/40 and cardiac tamponade arrest were surfaced as escalation risk signals.

---

## Usage

```python
from retreival.retrieval import EDTriageRAG
from reasoning.clinical_reasoning import ClinicalReasoner

rag      = EDTriageRAG()
reasoner = ClinicalReasoner()

patient = {
    "age": 80, "gender": "Female",
    "chief_complaint": "LETHARGY/shortness of breath",
    "heart_rate": 95, "systolic_bp": 99, "diastolic_bp": 47,
    "resp_rate": 28, "temperature": 100.2, "spo2": 100,
    "arrival_transport": "AMBULANCE",
}

model_prediction = {
    "esi_level": 2, "confidence": 0.78, "probabilities": [0.05, 0.78, 0.17]
}

cases, _ = rag.retrieve_cases(patient, top_k=3)

result = reasoner.reason(
    patient=patient,
    model_prediction=model_prediction,
    retrieved_cases=cases,
    shap_features=None,   # optional — wire in LightGBM SHAP when available
)

print(result["reasoning"])        # full clinical narrative
print(result["llm_esi"])          # LLM's ESI recommendation
print(result["agreement"])        # True / False
print(result["confidence_note"])  # human-readable summary
```
