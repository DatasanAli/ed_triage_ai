# ORCHESTRATION — ED Triage AI

> **Last updated:** 2026-04-02
> **Status:** Fully integrated. Streamlit frontend → FastAPI backend → LangGraph agentic pipeline (SageMaker + Pinecone RAG + Bedrock LLM).

---

## System Overview

**TriagePulse** is an Emergency Department triage assistant. The system consists of:

1. **Streamlit Frontend** (`frontend/app.py`) — Collects patient demographics, triage notes, and vitals. Displays enriched prediction results including triage priority, clinical reasoning, key SHAP factors, similar historical cases, and a technical details card.
2. **FastAPI Backend** (`backend/`) — Receives requests from the frontend, runs the full inference pipeline via `run_triage_inference`, and returns an enriched `TriageResponse`.
3. **LangGraph Agentic Pipeline** (`src/agents/`) — Orchestrates the full inference pipeline: SageMaker prediction + Pinecone RAG retrieval (parallel) → LLM clinical analysis → reconciliation/synthesis.
4. **SageMaker Endpoint** (`edtriage-live`) — Hosts the arch4 model (BioClinicalBERT + LightGBM fusion).
5. **Pinecone RAG** (`src/retreival/`) — Retrieves top-5 similar historical cases using Bedrock Titan embeddings against the `ed-triage-cases` index.
6. **Bedrock LLM** — Claude Sonnet (`claude-sonnet-4-6`) provides independent ESI assessment and clinical rationale.

### Current Data Flow

```
┌─────────────┐    POST /predict    ┌─────────────┐
│  Streamlit   │ ─────────────────► │   FastAPI    │
│  Frontend    │ ◄───────────────── │   Backend    │
│  :8501       │   TriageResponse   │  :8000       │
└─────────────┘                     └──────┬───────┘
                                           │ run_triage_inference()
                                           ▼
                                  ┌─────────────────────────────┐
                                  │   LangGraph triage_graph     │
                                  │                              │
                                  │  predict_node ──┐            │
                                  │  (SageMaker)    ├──► analyze_node (Claude)
                                  │  retrieve_node ─┘        │   │
                                  │  (Pinecone RAG)           ▼   │
                                  │                    synthesize_node
                                  │                    (reconciliation)
                                  └─────────────────────────────┘
```

- `predict_node` and `retrieve_node` run in **parallel** (fan-out from START).
- `analyze_node` fans in both results and calls Claude via Bedrock.
- `synthesize_node` deterministically reconciles model vs LLM recommendation (always takes the more cautious of the two).
- **State management:** `st.session_state` on the frontend. No database. No backend file storage.
- **Patient history:** Stored in `st.session_state` for the duration of the browser session only.

---

## Directory Structure

```text
ed_triage_ai/
├── ORCHESTRATION.md              # THIS FILE — single source of truth
├── Makefile                      # Endpoint lifecycle commands (deploy / delete)
├── backend/                      # FastAPI service
│   ├── main.py                   # /health and /predict routes
│   ├── schemas.py                # TriageRequest + TriageResponse (enriched)
│   ├── config.py                 # pydantic-settings (env vars)
│   └── sagemaker_service.py      # run_triage_inference — pipeline entry point
├── frontend/                     # Streamlit UI
│   └── app.py                    # Intake form + results page
├── src/agents/                   # LangGraph graph + nodes
│   ├── graph.py                  # triage_graph definition
│   ├── nodes.py                  # predict_node, retrieve_node, analyze_node, synthesize_node
│   ├── state.py                  # TriageState TypedDict
│   └── prompts.py                # LLM prompt templates
├── src/retreival/                # Pinecone RAG
│   └── retrieval.py              # EDTriageRAG — embed + search + format
└── sagemaker/                    # ML training pipeline (DO NOT MODIFY)
```

---

## API Contract

### `GET /health`

**Response:**
```json
{"status": "ok"}
```

---

### `POST /predict`

Accepts triage data from the frontend, runs the full agentic pipeline, returns an enriched prediction.

#### Request Body (`TriageRequest`)

| Field               | Type            | Required | Default      | Notes                                           |
|---------------------|-----------------|----------|--------------|-------------------------------------------------|
| `model`             | `str`           | No       | `"arch4"`    | Which model architecture to invoke              |
| `triage_notes`      | `str`           | **Yes**  | —            | Free-text clinical notes from the UI            |
| `age`               | `int \| None`   | No       | `None`       | Patient age in years                            |
| `sex`               | `str \| None`   | No       | `None`       | Female, Male, or Other                          |
| `heart_rate`        | `int \| None`   | No       | `None`       | BPM                                             |
| `resp_rate`         | `int \| None`   | No       | `None`       | Breaths per minute                              |
| `sbp`               | `int \| None`   | No       | `None`       | Systolic blood pressure (mmHg)                  |
| `dbp`               | `int \| None`   | No       | `None`       | Diastolic blood pressure (mmHg)                 |
| `spo2`              | `int \| None`   | No       | `None`       | Oxygen saturation (%)                           |
| `temp_f`            | `float \| None` | No       | `None`       | Temperature in Fahrenheit                       |
| `pain`              | `int \| None`   | No       | `None`       | Pain scale 0–10                                 |
| `arrival_transport` | `str`           | No       | `"Walk In"`  | One of: Walk In, Ambulance, Helicopter, Unknown |

**Example request:**
```json
{
  "triage_notes": "68F presenting with fever, confusion, and low blood pressure for 6 hours.",
  "age": 68,
  "sex": "Female",
  "heart_rate": 118,
  "resp_rate": 24,
  "sbp": 88,
  "dbp": 52,
  "spo2": 93,
  "temp_f": 101.8,
  "pain": 6,
  "arrival_transport": "Ambulance"
}
```

#### Response Body (`TriageResponse`)

| Field                | Type               | Notes                                                        |
|----------------------|--------------------|--------------------------------------------------------------|
| `predicted_class`    | `int`              | 0=L1-Critical, 1=L2-Emergent, 2=L3-Urgent/LessUrgent        |
| `predicted_label`    | `str`              | `"L1-Critical"`, `"L2-Emergent"`, `"L3-Urgent/LessUrgent"`  |
| `probabilities`      | `dict[str, float]` | Confidence per class                                         |
| `top_features`       | `list[dict]`       | Top 5 SHAP drivers: `{feature, shap, direction}`             |
| `safety_flag`        | `bool`             | True if clinical scores conflict with prediction             |
| `safety_reason`      | `str \| None`      | Human-readable explanation if flagged                        |
| `model_used`         | `str`              | Which model produced this result                             |
| `reconciled_label`   | `str \| None`      | More cautious of model vs LLM recommendation                 |
| `reconciled_class`   | `int \| None`      | 0-based class index for reconciled_label                     |
| `llm_esi`            | `int \| None`      | LLM independent ESI recommendation (1/2/3)                   |
| `llm_agreement`      | `bool \| None`     | True if LLM agrees with model prediction                     |
| `clinical_rationale` | `str \| None`      | LLM clinical reasoning narrative                             |
| `similar_cases`      | `list \| None`     | Top-5 RAG-retrieved historical cases with vitals + outcome   |
| `flags`              | `list[str] \| None`| Escalation signals and uncertainty notes                     |
| `confidence_pct`     | `int \| None`      | Model confidence 0–100                                       |
| `uncertainty_flag`   | `bool \| None`     | True if confidence is below threshold                        |

---

## Backend (`backend/`)

`run_triage_inference(request: TriageRequest)` in `sagemaker_service.py` is the pipeline entry point called by `POST /predict`. It:

1. Transforms the request into a SageMaker payload via `transform_request()`
2. Calls the SageMaker endpoint via `invoke_endpoint()`
3. Maps the request to a `patient` dict via `_request_to_patient()`
4. Invokes the LangGraph `triage_graph` with `{patient, prediction}` as input
5. Shapes the `final_report` from the graph into the `TriageResponse` contract

### Transformation helpers

- `transform_request(request) -> dict` — converts `TriageRequest` to SageMaker payload (renames `triage_notes` → `triage_text`, uppercases `arrival_transport`, drops Nones)
- `transform_response(response, model_used) -> dict` — converts raw endpoint response to base `TriageResponse` fields
- `_request_to_patient(request) -> dict` — maps `TriageRequest` to the patient dict shape the LangGraph graph expects

---

## LangGraph Pipeline (`src/agents/`)

### Nodes

| Node | Input | Output | Notes |
|------|-------|--------|-------|
| `predict_node` | `patient`, `prediction` (raw SageMaker) | `model_output` | Normalises SageMaker response into a clean `model_output` dict |
| `retrieve_node` | `patient` | `similar_cases`, `cases_text` | Embeds patient via Bedrock Titan, searches Pinecone `ed-triage-cases` top-5 |
| `analyze_node` | `model_output`, `similar_cases`, `patient` | `llm_esi`, `llm_agreement`, `clinical_rationale` | Calls Claude Sonnet via Bedrock with structured prompt |
| `synthesize_node` | all prior state | `final_report` | Deterministic reconciliation: `reconciled = min(model_class, llm_class)` |

`predict_node` and `retrieve_node` run in parallel; `analyze_node` fans in both.

### Reconciliation logic

The reconciled label is always the **more cautious** of model vs LLM:
```python
reconciled_class = min(model_class, llm_class)  # lower index = higher severity
```

---

## Configuration (`backend/config.py`)

| Env Var                          | Field                     | Default           |
|----------------------------------|---------------------------|-------------------|
| `TRIAGE_SAGEMAKER_ENDPOINT_NAME` | `sagemaker_endpoint_name` | `"edtriage-live"` |
| `TRIAGE_AWS_REGION`              | `aws_region`              | `"us-east-1"`     |
| `TRIAGE_AWS_PROFILE`             | `aws_profile`             | `"ed-triage"`     |
| `TRIAGE_USE_MOCK`                | `use_mock`                | `False`           |
| `TRIAGE_DEFAULT_MODEL`           | `default_model`           | `"arch4"`         |

---

## How to Run

### Endpoint lifecycle

```bash
make deploy-endpoint   # Download champion model from S3, repack, deploy edtriage-live (~5-10 min)
make delete-endpoint   # Tear down edtriage-live to stop incurring compute costs
```

Run these from the `ed_triage_ai/` directory. Deploy before starting the backend; delete when done testing.

### Backend + Frontend

```bash
# Terminal 1 — Backend (from ed_triage_ai/)
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend (from ed_triage_ai/)
streamlit run frontend/app.py
```

> **Note:** The root `requirements.txt` is for SageMaker Studio and includes heavy ML packages. Do not `pip install -r requirements.txt` locally — use `backend/requirements.txt` instead.

---

## Task List

### Completed

- [x] Create FastAPI backend (`main.py`, `schemas.py`, `config.py`, `sagemaker_service.py`)
- [x] Build Streamlit frontend (`frontend/app.py`)
- [x] Deploy `edtriage-live` SageMaker endpoint via `repack_and_deploy.py`
- [x] Integrate backend with live endpoint (`use_mock=False` by default)
- [x] Refactor `sagemaker_service.py`: extract `invoke_endpoint`, implement `run_triage_inference` as pipeline entry point
- [x] Add `Makefile` with `deploy-endpoint` and `delete-endpoint` targets
- [x] Implement LangGraph agentic pipeline (`src/agents/graph.py`, `nodes.py`, `state.py`, `prompts.py`)
- [x] Implement Pinecone RAG retrieval (`src/retreival/retrieval.py`) via Bedrock Titan embeddings
- [x] Wire LangGraph pipeline into `run_triage_inference` — backend now calls graph instead of SageMaker directly
- [x] Extend `TriageResponse` with enriched fields (reconciled label, llm_esi, clinical_rationale, similar_cases, flags, confidence_pct)
- [x] Update frontend to display enriched results: clinical reasoning, key SHAP factors, similar cases with vitals/demographics, technical details card
- [x] Add `sex` field to intake form and patient dict for LLM/RAG context
- [x] Fix AWS auth: use `boto3.Session(profile_name=aws_profile)` with static IAM credentials (`ed-triage` profile)
- [x] Verify Pinecone index (`ed-triage-cases`) populated and reachable
- [x] Verify Bedrock access (Claude Sonnet + Titan embed) from backend runtime

---

## Conflict Protocol

If an implementer cannot complete a task because this spec is ambiguous, incomplete, or incorrect, stop and report back with a clear description of the issue. The user will relay this to the Planner for a spec update.
