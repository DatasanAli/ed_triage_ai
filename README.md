# ED Triage AI

Clinical decision support system for emergency department triage. Combines a deployed ML model, retrieval-augmented generation (RAG) from historical ED cases, and an LLM-based reasoning agent to produce transparent, evidence-grounded triage recommendations.

---

## Problem Statement

ED triage requires rapid, accurate patient prioritization — but ML models alone are black boxes. Clinicians need to know *why* a patient is high-acuity, not just that they are. This system provides a second opinion with supporting evidence: similar historical cases, SHAP-attributed features, and independent clinical reasoning from an LLM.

---

## System Architecture

The system is a four-stage pipeline triggered by a single patient input:

```
Patient Input
     │
     ▼
[1] SageMaker Endpoint (edtriage-live)
     │  BioClinicalBERT + LightGBM fusion model
     │  Outputs: predicted class, probabilities, SHAP features, safety flag
     │
     ├──────────────────────────┐
     ▼                          ▼
[2a] predict_node           [2b] retrieve_node   ← parallel
      Normalizes prediction       Embeds patient via Bedrock Titan
      Extracts SHAP features      Queries Pinecone for top-k similar cases
     │                          │
     └──────────────────────────┘
     ▼
[3] analyze_node
     │  Claude (via AWS Bedrock) reasons independently:
     │    - Patient vitals, CC, HPI
     │    - SHAP feature attributions (which features pushed acuity up/down)
     │    - Top similar historical cases (diagnosis, outcome, similarity score)
     │    - Model prediction (reference only — LLM reasons first)
     │  Outputs: independent ESI recommendation + clinical rationale
     │
     ▼
[4] synthesize_node
     │  Compares ML prediction vs LLM recommendation
     │  Reconciles: always take the more urgent (lower ESI) of the two
     │  Generates escalation flags
     │
     ▼
Final Report
```

### Reconciliation Logic

The system uses a **two-signal model**:
- **Signal 1**: arch4 ML model (discriminative, fast, validated on 8,383 cases)
- **Signal 2**: LLM reasoning (generative, grounded in RAG evidence, interpretable)

When signals agree → high confidence. When they disagree → flag for clinical review.

The reconciled triage level always takes the **more urgent** of the two signals — the system errs on the side of safety.

---

## ML Model — arch4

**Architecture**: BioClinicalBERT + LightGBM fusion

- BioClinicalBERT encodes chief complaint and HPI narrative (clinical text embeddings)
- LightGBM processes structured features: vitals, demographics, clinical scores (NEWS2, qSOFA, MEWS)
- Both signals are fused for final ESI classification

**Training data**: ~8,383 de-identified ED encounters

**Output classes**:
| Class | ESI Level | Acuity |
|-------|-----------|--------|
| 0 | ESI 1 | Critical — immediate life-saving intervention |
| 1 | ESI 2 | Emergent — high-risk, should not wait |
| 2 | ESI 3 | Urgent — stable but requires multiple resources |

**Additional outputs**: class probabilities, SHAP feature attributions per prediction, safety flag for physiologically extreme presentations

---

## RAG Pipeline

Historical ED cases are indexed in Pinecone using Bedrock Titan embeddings. For each new patient, the system:

1. Builds a structured query text (CC + vitals + demographics)
2. Embeds it with Amazon Titan (`amazon.titan-embed-text-v2:0`)
3. Retrieves top-k most similar cases from Pinecone (`ed-triage-cases` index)
4. Returns each case with: similarity score, chief complaint, ESI assigned, vitals, diagnosis, disposition

The same text format used at index time is used at query time — ensuring semantic alignment between stored and query vectors.

---

## LangGraph Agent

The agent is implemented in `src/agents/` using LangGraph. Key design principles:

- **No hospital protocols**: The system does not generate nursing action items or ESI timing targets. Each hospital has its own protocols. The agent's job is to reason independently and recommend a triage level with evidence.
- **LLM reasons before seeing prediction**: The `analyze_node` prompt presents patient data, SHAP features, and RAG cases first. The ML model prediction is shown at the end as a "reference" — this prevents anchoring bias.
- **Independent recommendation**: The LLM produces its own ESI recommendation (SHORT: 1-2 sentences; LONG: full clinical assessment). The `synthesize_node` then compares and reconciles.

**State schema** (`TriageState`): patient, prediction, shap_features, safety_flag, retrieved_cases, clinical_analysis, final_report

**Final report fields**:
- `reconciled_level` — the triage recommendation (e.g. "L2-Emergent")
- `reconciled_class` — integer class index
- `model_level` / `llm_esi` — raw signals before reconciliation
- `llm_agreement` — whether model and LLM agreed
- `clinical_rationale` — LLM's full independent reasoning (short + long)
- `shap_features` — top features driving the model prediction, with direction
- `similar_cases` — top-k RAG cases with metadata
- `flags` — escalation signals (SHAP contradiction, RAG majority disagreement, safety flag, LLM disagreement)
- `safety_flag` / `safety_reason` — extreme vitals detected by model

---

## SageMaker Pipeline

The ML model is trained, evaluated, and deployed via an automated SageMaker Pipeline. The pipeline follows a champion/challenger pattern — new models only deploy if they beat the registered champion on macro-F1.

Pipeline steps: `Preprocess → Train → Evaluate → CheckChampion → Register → Deploy`

The deployed endpoint (`edtriage-live`) serves arch4 inference with SHAP output enabled.

See [sagemaker/README.md](sagemaker/README.md) for full pipeline documentation, deployment instructions, and endpoint testing.

---

## Repository Structure

```
ed_triage_ai/
├── requirements.txt               # Project dependencies
│
├── src/
│   ├── agents/                    # LangGraph triage agent
│   │   ├── graph.py               # Compiled graph (4 nodes)
│   │   ├── nodes.py               # Node implementations
│   │   ├── state.py               # TriageState TypedDict
│   │   ├── prompts.py             # LLM prompts (ANALYZE_SYSTEM, ANALYZE_HUMAN)
│   │   └── __init__.py            # Exports: triage_graph, TriageState
│   ├── backend/                   # FastAPI service
│   │   ├── main.py                # /health and /predict routes
│   │   ├── schemas.py             # TriageRequest + TriageResponse
│   │   ├── config.py              # pydantic-settings (env vars)
│   │   └── sagemaker_service.py   # run_triage_inference — pipeline entry point
│   ├── frontend/                  # Streamlit UI
│   │   └── app.py                 # Intake form + results page
│   ├── retreival/
│   │   └── retrieval.py           # EDTriageRAG — Pinecone retrieval via Titan embeddings
│   ├── reasoning/
│   │   └── clinical_reasoning.py  # Standalone ClinicalReasoner (independent LLM reasoning)
│   └── embeddings/                # Pinecone index build tools
│
├── sagemaker/
│   ├── pipeline/                  # Pipeline DAG and runner
│   ├── steps/                     # Preprocess, train, evaluate, deploy
│   └── models/arch4/              # BioClinicalBERT + LightGBM implementation
│
├── notebooks/                     # arch4 training, EDA, data cleaning, feature engineering,
│                                  # comprehensive pipeline test
├── scripts/                       # run_triage.py (CLI runner), eval_e2e_pipeline.py
├── experimental/                  # Archived architecture explorations (arch1/2/5/6, GatorTron, Llama)
└── docs/                          # orchestration.md, arch4 walkthrough, RAG design
```

---

## Running the System

### CLI (local or SageMaker Studio)

```bash
export AWS_PROFILE=ed-triage   # local only — SageMaker uses instance role
export PYTHONPATH=.

python scripts/run_triage.py
```

### Backend + Frontend

```bash
# Terminal 1 — Backend
uvicorn src.backend.main:app --reload --port 8000

# Terminal 2 — Frontend
streamlit run src/frontend/app.py
```

> Use `src/backend/requirements.txt` for local installs — the root `requirements.txt` includes heavy SageMaker ML packages.

Or open `notebooks/comprehensive_test.ipynb` for a full evaluation across 10 clinical scenarios (covering ESI 1–3, edge cases, SHAP contradictions, and RAG majority disagreement).

---

## Dependencies

- **AWS Bedrock** — Claude (`us.anthropic.claude-sonnet-4-6`) for LLM reasoning; Titan for embeddings
- **Pinecone** — vector store for historical case retrieval
- **LangGraph** — agent graph orchestration
- **LightGBM + BioClinicalBERT** — arch4 model (served via SageMaker endpoint)
- **SHAP** — feature attribution (computed at inference time by the endpoint)

Pinecone API key is stored in AWS Secrets Manager (`prod/pinecone/api_key`), not in `.env`.

---

## Design Philosophy

- **Support, not replace**: The system provides a second opinion with evidence. Clinical judgment always takes precedence.
- **Hospital-agnostic**: No nursing action items, no ESI timing targets. Each hospital applies its own protocols to the triage recommendation.
- **Transparency**: Every recommendation includes the features that drove it, similar historical cases, and the LLM's independent reasoning — so clinicians can agree, disagree, or escalate.
- **Safety-first reconciliation**: When model and LLM disagree, the system takes the more urgent recommendation.

---

## Dataset

~9,146 de-identified emergency department encounters with:
- Chief complaints and HPI narratives
- Vital signs (HR, BP, RR, SpO2, temperature)
- Demographics (age, gender, race)
- Arrival transport
- Ground-truth ESI labels (1–3 for this system)
- Clinical scores derived during feature engineering: NEWS2, qSOFA, MEWS

---

## References

- Gilboy, N., et al. (2011). Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care
- Levin, S., et al. (2018). Machine-learning-based electronic triage more accurately classifies patients with respect to clinical outcomes
- Raita, Y., et al. (2019). Emergency department triage prediction of clinical outcomes using machine learning models
- Cutillo, C. M., et al. (2024). Machine intelligence in healthcare — perspectives on trustworthiness, explainability, usability, and transparency

---

**Note**: This system is a capstone research project. Clinical deployment requires appropriate validation, regulatory approval, and integration with hospital EHR workflows.
