# ORCHESTRATION — ED Triage AI

> **Last updated:** 2026-04-01
> **Status:** Backend + frontend integrated with live SageMaker endpoint. LangGraph + RAG orchestration pending.

---

## System Overview

**TriagePulse** is an Emergency Department triage assistant. The system currently consists of:

1. **Streamlit Frontend** (`frontend/app.py`) — Collects patient triage notes and vitals, sends them to the backend, and displays the prediction results.
2. **FastAPI Backend** (`backend/`) — Receives requests from the frontend, calls the orchestration service, and returns the result to the frontend.
3. **SageMaker Endpoint** (`edtriage-live`) — Hosts the trained arch4 model (BioClinicalBERT + LightGBM fusion). Deployed and live.

### Current Data Flow

```
┌─────────────┐    POST /predict    ┌─────────────┐   InvokeEndpoint   ┌────────────┐
│  Streamlit   │ ─────────────────► │   FastAPI    │ ─────────────────► │ SageMaker  │
│  Frontend    │ ◄───────────────── │   Backend    │ ◄───────────────── │  edtriage  │
│  :8501       │   TriageResponse   │  :8000       │   raw JSON         │   -live    │
└─────────────┘                     └─────────────┘                     └────────────┘
```

### Target Data Flow (after LangGraph integration)

```
┌─────────────┐    POST /predict    ┌─────────────┐
│  Streamlit   │ ─────────────────► │   FastAPI    │
│  Frontend    │ ◄───────────────── │   Backend    │
│  :8501       │   TriageResponse   │  :8000       │
└─────────────┘                     └──────┬───────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │  orchestration/  │  ◄─── NEW SERVICE
                                  │                  │
                                  │  Owns the full   │ ──► SageMaker (edtriage-live)
                                  │  inference       │ ──► LangGraph + RAG (Pinecone)
                                  │  pipeline        │ ──► Claude via Bedrock
                                  └─────────────────┘
```

- **State management:** `st.session_state` on the frontend. No database. No backend file storage.
- **Patient history:** Stored in `st.session_state` for the duration of the browser session only.

---

## Directory Structure

```text
ed_triage_ai/
├── ORCHESTRATION.md              # THIS FILE — single source of truth
├── Makefile                      # Endpoint lifecycle commands (deploy / delete)
├── backend/                      # FastAPI service
├── orchestration/                # Inference orchestration service (PENDING — to be created)
├── frontend/                     # Streamlit UI
├── src/agents/                   # LangGraph graph + nodes (to be consumed by orchestration/)
├── src/retreival/                # Pinecone RAG (to be consumed by orchestration/)
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

Accepts triage data from the frontend, runs the inference pipeline, returns a prediction.

#### Request Body (`TriageRequest`)

| Field               | Type            | Required | Default      | Notes                                           |
|---------------------|-----------------|----------|--------------|-------------------------------------------------|
| `model`             | `str`           | No       | `"arch4"`    | Which model architecture to invoke              |
| `triage_notes`      | `str`           | **Yes**  | —            | Free-text clinical notes from the UI            |
| `age`               | `int \| None`   | No       | `None`       | Patient age in years                            |
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
  "triage_notes": "54M presenting with acute chest pain radiating to left arm. Diaphoretic.",
  "age": 54,
  "heart_rate": 132,
  "resp_rate": 22,
  "sbp": 88,
  "dbp": 52,
  "spo2": 84,
  "temp_f": 98.6,
  "pain": 8,
  "arrival_transport": "Ambulance"
}
```

#### Response Body (`TriageResponse`)

| Field             | Type               | Notes                                                        |
|-------------------|--------------------|--------------------------------------------------------------|
| `predicted_class` | `int`              | 0=L1-Critical, 1=L2-Emergent, 2=L3-Urgent/LessUrgent        |
| `predicted_label` | `str`              | `"L1-Critical"`, `"L2-Emergent"`, `"L3-Urgent/LessUrgent"`  |
| `probabilities`   | `dict[str, float]` | Confidence per class                                         |
| `top_features`    | `list[dict]`       | Top 5 SHAP drivers: `{feature, shap, direction}`             |
| `safety_flag`     | `bool`             | True if clinical scores conflict with prediction             |
| `safety_reason`   | `str \| None`      | Human-readable explanation if flagged                        |
| `model_used`      | `str`              | Which model produced this result                             |

> **Note:** The response schema will be extended when LangGraph orchestration is added (see below).

---

## Backend (`backend/`)

### Integration point for orchestration

`backend/sagemaker_service.py` exposes `run_triage_inference(request: TriageRequest)` — the function called by `POST /predict`. Currently it calls the SageMaker endpoint directly.

**When the orchestration service is ready, `run_triage_inference` is the only function that needs to change.** It will delegate to `orchestration/` instead of calling the endpoint directly.

### Transformation helpers (remain in `backend/`)

- `transform_request(request) -> dict` — converts `TriageRequest` to SageMaker payload format
- `transform_response(response, model_used) -> dict` — converts raw endpoint response to `TriageResponse`

---

## Orchestration Service (`orchestration/`) — Pending

The `orchestration/` directory should be created as a sibling to `backend/`. It will own the full inference pipeline end-to-end: calling the SageMaker endpoint, running the LangGraph graph (RAG + LLM reasoning), and returning an enriched result.

`invoke_endpoint` (currently in `backend/sagemaker_service.py`) should move here, since endpoint invocation is part of the orchestration pipeline, not the HTTP layer.

The backend calls into this service; the backend does not need to know about SageMaker, LangGraph, Pinecone, or Bedrock directly.

### External dependencies

| Service | Purpose |
|---------|---------|
| SageMaker (`edtriage-live`) | ML model inference |
| AWS Bedrock (`claude-sonnet-4-6`) | LLM clinical reasoning |
| AWS Bedrock (Titan embed) | Patient embedding for RAG |
| Pinecone (`ed-triage-cases`) | Historical case vector search — API key in AWS Secrets Manager: `prod/pinecone/api_key` |

### Enriched response fields (to be added to `TriageResponse`)

When orchestration is complete, the response will include additional fields sourced from LangGraph:
- Reconciled triage level (more cautious of model vs LLM)
- LLM's independent ESI assessment and agreement flag
- Clinical rationale from the LLM
- Similar historical cases from Pinecone
- Escalation flags

`schemas.py` will need to be extended accordingly.

---

## Configuration (`backend/config.py`)

| Env Var                          | Field                     | Default           |
|----------------------------------|---------------------------|-------------------|
| `TRIAGE_SAGEMAKER_ENDPOINT_NAME` | `sagemaker_endpoint_name` | `"edtriage-live"` |
| `TRIAGE_AWS_REGION`              | `aws_region`              | `"us-east-1"`     |
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
- [x] Refactor `sagemaker_service.py`: extract `invoke_endpoint` as a reusable function, rename orchestration entry point to `run_triage_inference`
- [x] Add `Makefile` with `deploy-endpoint` and `delete-endpoint` targets

### Pending — LangGraph + RAG Integration
- [ ] Create `orchestration/` service directory as a sibling to `backend/`
- [ ] Move `invoke_endpoint` from `backend/sagemaker_service.py` into `orchestration/`
- [ ] Implement the orchestration service using `src/agents/` (LangGraph) and `src/retreival/` (Pinecone RAG)
- [ ] Update `run_triage_inference` in `backend/sagemaker_service.py` to delegate to `orchestration/`
- [ ] Extend `TriageResponse` in `schemas.py` to surface enriched fields
- [ ] Update `frontend/app.py` to display enriched fields (rationale, similar cases, flags)
- [ ] Verify Pinecone index (`ed-triage-cases`) is populated and reachable
- [ ] Verify Bedrock access (Claude + Titan embed) from the backend runtime environment

---

## Conflict Protocol

If an implementer cannot complete a task because this spec is ambiguous, incomplete, or incorrect, stop and report back with a clear description of the issue. The user will relay this to the Planner for a spec update.
