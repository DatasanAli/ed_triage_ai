# Architecture 4 — User Walkthrough

> Plain-English explanation of how the full ED Triage AI system works, using a single patient example.

---

## Sample Patient

| Field | Value |
|-------|-------|
| Age | 68 |
| Gender | Female |
| Chief complaint | "Fever, confusion, low blood pressure" |
| Heart rate | 118 bpm |
| Systolic BP | 88 mmHg |
| Diastolic BP | 52 mmHg |
| Respiratory rate | 24 breaths/min |
| Temperature | 101.8°F |
| SpO2 | 93% |
| Pain | 6/10 |
| Arrival | Ambulance |

---

## The Four-Stage Pipeline

```
Patient Input
     │
     ▼
[1] SageMaker Endpoint (edtriage-live)
     │
     ├──────────────────────────────┐
     ▼                              ▼
[2a] predict_node              [2b] retrieve_node   ← parallel
     │                              │
     └──────────────────────────────┘
                    ▼
             [3] analyze_node
                    │
                    ▼
            [4] synthesize_node
                    │
                    ▼
              Final Report
```

---

## Stage 1 — ML Model (BioClinicalBERT + LightGBM)

The patient data is sent to the `edtriage-live` SageMaker endpoint, which runs the arch4 model.

**Two specialists look at the same patient:**

**BioClinicalBERT** reads the text:
```
"Chief complaint: Fever, confusion, low blood pressure.
 Presenting with Fever, confusion, low blood pressure."
```
It produces a 768-number vector encoding the clinical meaning of those words. "Confusion" + "low blood pressure" + "fever" together push this vector strongly toward high acuity.

**LightGBM** reads the structured numbers:
```
[heart_rate=118, sbp=88, dbp=52, resp_rate=24,
 temp_f=101.8, spo2=93, age=68, pain=6,
 arrival_transport=AMBULANCE, ...]
```
LightGBM processes these as decision trees — splitting on thresholds like "sbp < 90?" and "spo2 < 94?" — and combines hundreds of trees into a probability estimate.

**Both signals are fused** into a final prediction:

```
Predicted     : L2-Emergent
Confidence    : 78%
Probabilities : L1-Critical 15% | L2-Emergent 78% | L3-Urgent 7%
```

The endpoint also returns **SHAP feature attributions** — which features pushed the prediction toward or away from the predicted class — and a **safety flag** if vitals are physiologically extreme.

For this patient: `safety_flag = True` — SBP 88 mmHg and SpO2 93% are flagged as extreme.

---

## Stage 2 — Parallel: predict_node + retrieve_node

These two nodes run simultaneously (LangGraph fan-out from START).

### predict_node

Normalizes the raw SageMaker response into a clean `model_output` dict:

```python
{
    "predicted_class": 1,          # 0=L1-Critical, 1=L2-Emergent, 2=L3-Urgent
    "predicted_label": "L2-Emergent",
    "confidence_pct": 78,
    "probabilities": {"L1-Critical": 0.15, "L2-Emergent": 0.78, "L3-Urgent": 0.07},
    "top_features": [
        {"feature": "sbp", "shap": -0.42, "direction": "toward L1-Critical"},
        {"feature": "spo2", "shap": -0.38, "direction": "toward L1-Critical"},
        {"feature": "heart_rate", "shap": -0.31, "direction": "toward L1-Critical"},
        {"feature": "resp_rate", "shap": -0.24, "direction": "toward L1-Critical"},
        {"feature": "age", "shap": -0.18, "direction": "toward L1-Critical"},
    ],
    "safety_flag": True,
    "safety_reason": "SBP 88 mmHg and SpO2 93% are physiologically extreme"
}
```

Note: the SHAP features are pushing **away from** the predicted class (L2-Emergent) and toward L1-Critical. This is a contradiction — the model's own internal evidence says the patient may be more critical than the prediction suggests.

### retrieve_node

Simultaneously, the RAG system queries Pinecone for the 5 most similar historical cases:

1. Build query text from patient demographics + CC + vitals
2. Embed with Bedrock Titan (`amazon.titan-embed-text-v2:0`)
3. Cosine similarity search against `ed-triage-cases` index (~8,383 MIMIC-IV-Ext cases)

**Retrieved cases:**
```
[0.91] ESI-1 | Fever, confusion, hypotension → Septic shock | ICU admitted
[0.88] ESI-1 | Altered mental status, low BP → Sepsis       | ICU admitted
[0.85] ESI-2 | Fever, tachycardia, hypotension → UTI/Sepsis | Admitted
[0.82] ESI-1 | Confusion, fever, tachycardia → Bacteremia   | ICU admitted
[0.79] ESI-2 | Fever, shortness of breath → Pneumonia       | Admitted
```

4 out of 5 similar cases were ESI-1 (Critical). This is a strong RAG signal that the patient may be more critical than the model's L2 prediction.

---

## Stage 3 — analyze_node (Claude via AWS Bedrock)

`analyze_node` fans in from both `predict_node` and `retrieve_node`, then calls Claude Sonnet (`us.anthropic.claude-sonnet-4-6`) with a structured prompt containing:

1. **Patient vitals and demographics** — the raw clinical picture
2. **SHAP block** — which features drove the model prediction and whether they contradict it
3. **RAG cases** — the 5 similar historical cases with ESI, diagnosis, outcome
4. **ML prediction** — shown last, as a "reference only" — to prevent anchoring bias

The LLM reasons **independently** before seeing the model's recommendation. It then states its own ESI and whether it agrees.

**LLM response for this patient:**

```
REASONING:
68F presenting with fever (101.8°F), hypotension (SBP 88 mmHg), tachycardia (118 bpm),
tachypnea (RR 24), and altered mental status meets sepsis criteria (qSOFA ≥ 2). SpO2 93%
indicates early respiratory compromise. The SHAP evidence shows all top features pushing
toward L1-Critical, contradicting the L2 prediction. 4/5 similar historical cases were
triaged ESI-1 and admitted to the ICU.

RECOMMENDED ESI: 1 — Hypotension + AMS + fever in an elderly patient arriving by
ambulance requires immediate resuscitation; waiting as ESI-2 is not appropriate.

AGREEMENT: DISAGREE — The patient's hemodynamic instability and altered mentation indicate
a higher acuity than L2-Emergent; this presentation is consistent with septic shock.
```

---

## Stage 4 — synthesize_node (Reconciliation)

`synthesize_node` compares the two signals and always takes the **more urgent** (safety-first):

```
ML model  →  L2-Emergent  (class 1)
LLM       →  L1-Critical  (class 0)

reconciled_class = min(1, 0) = 0  →  L1-Critical
```

It also generates **escalation flags** based on:
- SHAP contradiction (features pushing away from predicted class)
- RAG majority disagreement (most similar cases had a different ESI)
- Safety flag from the model
- LLM disagreement with the model

**Final report for this patient:**
```
Reconciled level : L1-Critical  [escalated from L2-Emergent]
Model prediction : L2-Emergent (78% confidence)
LLM independent  : ESI-1  [DISAGREES]
Agreement        : No

Flags:
  ! SHAP contradiction: top features push toward L1-Critical
  ! RAG majority: 4/5 similar cases were ESI-1
  ! Safety flag: SBP 88 mmHg and SpO2 93% are physiologically extreme
  ! LLM escalation: independent assessment is more urgent than model prediction
```

---

## Why This Design

| Design choice | Reason |
|---------------|--------|
| **LLM reasons before seeing ML prediction** | Prevents anchoring bias — the LLM forms an independent opinion first |
| **Safety-first reconciliation** | When signals disagree, always take the more urgent — the cost of under-triaging is higher than over-triaging |
| **SHAP contradiction flagging** | If the model's own features push against its prediction, that's a red flag even when the prediction looks confident |
| **RAG majority signal** | If 4/5 similar historical cases had a different ESI, that's clinically meaningful evidence |
| **No hospital protocols** | Each hospital applies its own protocols — the system recommends a triage level, not nursing actions |

---

## What the Clinician Sees

In the Streamlit UI, the clinician receives:

- **Reconciled triage level** — the final recommendation with escalation note if overridden
- **Clinical rationale** — the LLM's 2–4 sentence independent reasoning with specific vital citations
- **SHAP features** — top 5 model drivers with direction (supporting or contradicting the prediction)
- **Similar cases** — 5 historical cases with ESI, chief complaint, diagnosis, outcome, and similarity score
- **Flags** — any escalation signals (SHAP contradiction, RAG disagreement, safety flag, LLM disagreement)
- **Confidence** — model confidence percentage with low-confidence warning if below threshold

---

*Written April 2026 — Capstone Project: ED Triage AI, Group 7, AAI590*
