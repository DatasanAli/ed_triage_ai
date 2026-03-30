# ORCHESTRATION — ED Triage AI

> **Last updated:** 2026-03-29  
> **Status:** Initial setup — backend stub + frontend integration

---

## System Overview

**TriagePulse** is an Emergency Department triage assistant. The system consists of:

1. **Streamlit Frontend** (`frontend/app.py`) — Collects patient triage notes and optional vitals, sends them to the backend, and displays the prediction results.
2. **FastAPI Backend** (`backend/`) — Receives requests from the frontend, transforms them into the format the ML model expects, calls the SageMaker inference endpoint (or returns mock data), transforms the response back, and returns it to the frontend.
3. **SageMaker Endpoint** (external) — Hosts the trained arch4 model (BioClinicalBERT + LightGBM fusion). **Not deployed yet** — the backend must stub this with mock data for now.

### Data Flow

```
┌─────────────┐    POST /predict    ┌─────────────┐   InvokeEndpoint   ┌────────────┐
│  Streamlit   │ ─────────────────► │   FastAPI    │ ─────────────────► │ SageMaker  │
│  Frontend    │ ◄───────────────── │   Backend    │ ◄───────────────── │ Endpoint   │
│  :8501       │   TriageResponse   │  :8000       │   raw JSON         │ (stubbed)  │
└─────────────┘                     └─────────────┘                     └────────────┘
```

- **State management:** `st.session_state` on the frontend. No database. No backend file storage.
- **Patient history:** Stored in `st.session_state` for the duration of the browser session only.

---

## Directory Tree

```text
ed_triage_ai/
├── ORCHESTRATION.md              # THIS FILE — single source of truth
├── backend/
│   ├── __init__.py               # empty, makes it a Python package
│   ├── main.py                   # FastAPI app + endpoints + CORS
│   ├── schemas.py                # Pydantic request/response models
│   ├── config.py                 # Environment-aware settings (BaseSettings)
│   └── sagemaker_service.py      # Transform layers + SageMaker client (stubbed)
├── frontend/
│   └── app.py                    # Existing Streamlit app (to be updated)
├── sagemaker/                    # Existing — ML pipeline code (DO NOT MODIFY)
├── src/                          # Existing — notebooks/RAG code (DO NOT MODIFY)
├── requirements.txt              # Root deps (to be updated)
└── .env                          # Local overrides (gitignored)
```

---

## API Contract

### `GET /health`

Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

---

### `POST /predict`

Accepts triage data from the frontend, calls the model, returns a prediction.

#### Request Body (`TriageRequest`)

| Field               | Type            | Required | Default      | Notes                                     |
|---------------------|-----------------|----------|--------------|-------------------------------------------|
| `model`             | `str`           | No       | `"arch4"`    | Which model architecture to invoke        |
| `triage_notes`      | `str`           | **Yes**  | —            | Free-text clinical notes from the UI      |
| `age`               | `int \| None`   | No       | `None`       | Patient age in years                      |
| `heart_rate`        | `int \| None`   | No       | `None`       | BPM                                       |
| `resp_rate`         | `int \| None`   | No       | `None`       | Breaths per minute                        |
| `sbp`               | `int \| None`   | No       | `None`       | Systolic blood pressure (mmHg)            |
| `dbp`               | `int \| None`   | No       | `None`       | Diastolic blood pressure (mmHg)           |
| `spo2`              | `int \| None`   | No       | `None`       | Oxygen saturation (%)                     |
| `temp_f`            | `float \| None` | No       | `None`       | Temperature in Fahrenheit                 |
| `pain`              | `int \| None`   | No       | `None`       | Pain scale 0–10                           |
| `arrival_transport` | `str`           | No       | `"Walk In"`  | One of: Walk In, Ambulance, Helicopter, Unknown |

**Example request:**
```json
{
  "model": "arch4",
  "triage_notes": "54M presenting with acute chest pain radiating to left arm. Diaphoretic, shortness of breath.",
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

| Field              | Type               | Notes                                          |
|--------------------|--------------------|-------------------------------------------------|
| `predicted_class`  | `int`              | 0, 1, or 2                                     |
| `predicted_label`  | `str`              | `"L1-Critical"`, `"L2-Emergent"`, or `"L3-Urgent/LessUrgent"` |
| `probabilities`    | `dict[str, float]` | Confidence per class                            |
| `top_features`     | `list[dict]`       | Top 5 SHAP drivers: `{feature, shap, direction}` |
| `safety_flag`      | `bool`             | True if clinical scores conflict with prediction |
| `safety_reason`    | `str \| None`      | Human-readable explanation if flagged           |
| `model_used`       | `str`              | Which model produced this result                |

**Example response:**
```json
{
  "predicted_class": 0,
  "predicted_label": "L1-Critical",
  "probabilities": {
    "L1-Critical": 0.942,
    "L2-Emergent": 0.051,
    "L3-Urgent/LessUrgent": 0.007
  },
  "top_features": [
    {"feature": "spo2", "shap": -0.3124, "direction": "toward L1-Critical"},
    {"feature": "heart_rate", "shap": 0.2187, "direction": "toward L1-Critical"},
    {"feature": "shock_index", "shap": 0.1843, "direction": "toward L1-Critical"},
    {"feature": "news2_score", "shap": 0.1521, "direction": "toward L1-Critical"},
    {"feature": "sbp", "shap": -0.1104, "direction": "toward L1-Critical"}
  ],
  "safety_flag": false,
  "safety_reason": null,
  "model_used": "arch4"
}
```

---

## Transformation Layers (Backend)

The backend has two transformation functions in `sagemaker_service.py`:

### `transform_request(request: TriageRequest) -> dict`
Converts frontend payload → SageMaker payload:
- Rename `triage_notes` → `triage_text`
- Uppercase `arrival_transport` (e.g., `"Ambulance"` → `"AMBULANCE"`)
- Remove the `model` field
- Exclude any field whose value is `None`

### `transform_response(sagemaker_response: dict, model_used: str) -> dict`
Converts SageMaker response → frontend response:
- Keep: `predicted_class`, `predicted_label`, `probabilities`, `top_features`, `safety_flag`, `safety_reason`
- Add: `model_used`
- Drop: `lgbm_shap`

---

## Frontend Data Contract

The Streamlit frontend needs to:
1. Collect the triage form fields listed in `TriageRequest` above.
2. On submit, POST to `{BACKEND_URL}/predict` (default: `http://localhost:8000/predict`).
3. On success, store the response in `st.session_state` and navigate to the results page.
4. Display the results page using data from `TriageResponse` (not hardcoded mocks).
5. Maintain a list of past triages in `st.session_state.triage_history` for the "Recent Triage" sidebar item.

The `BACKEND_URL` should be configurable (hardcoded to `http://localhost:8000` for now).

---

## Configuration (`backend/config.py`)

| Env Var                          | Field                    | Default            |
|----------------------------------|--------------------------|--------------------|
| `TRIAGE_SAGEMAKER_ENDPOINT_NAME` | `sagemaker_endpoint_name` | `"edtriage-live"` |
| `TRIAGE_AWS_REGION`              | `aws_region`             | `"us-east-1"`      |
| `TRIAGE_USE_MOCK`                | `use_mock`               | `True`             |
| `TRIAGE_DEFAULT_MODEL`           | `default_model`          | `"arch4"`          |

---

## Mock Data (used when `use_mock=True`)

```python
MOCK_SAGEMAKER_RESPONSE = {
    "predicted_class": 0,
    "predicted_label": "L1-Critical",
    "probabilities": {
        "L1-Critical": 0.942,
        "L2-Emergent": 0.051,
        "L3-Urgent/LessUrgent": 0.007,
    },
    "top_features": [
        {"feature": "spo2", "shap": -0.3124, "direction": "toward L1-Critical"},
        {"feature": "heart_rate", "shap": 0.2187, "direction": "toward L1-Critical"},
        {"feature": "shock_index", "shap": 0.1843, "direction": "toward L1-Critical"},
        {"feature": "news2_score", "shap": 0.1521, "direction": "toward L1-Critical"},
        {"feature": "sbp", "shap": -0.1104, "direction": "toward L1-Critical"},
    ],
    "safety_flag": False,
    "safety_reason": None,
    "lgbm_shap": {},
}
```

---

## How to Run

A `.venv` virtual environment exists at the project root. Always activate it before running anything.

```bash
# Activate the venv (run once per terminal session)
source .venv/bin/activate

# Install backend dependencies (first time only, or after changes)
pip install -r backend/requirements.txt

# Terminal 1 — Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
streamlit run frontend/app.py
```

> **Note:** The root `requirements.txt` is designed for SageMaker Studio and includes heavy ML packages. Do not `pip install -r requirements.txt` locally — use `backend/requirements.txt` instead. Streamlit is already installed in the system environment where the user is running it.

---

## Active Task List

### Backend Agent
- [ ] Create `backend/__init__.py` (empty file)
- [ ] Create `backend/config.py` with `BaseSettings`
- [ ] Create `backend/schemas.py` with `TriageRequest` and `TriageResponse`
- [ ] Create `backend/sagemaker_service.py` with `transform_request`, `transform_response`, `invoke_endpoint` (stubbed)
- [ ] Create `backend/main.py` with `POST /predict`, `GET /health`, and CORS middleware
- [ ] Update root `requirements.txt` to add `fastapi`, `uvicorn[standard]`, `pydantic-settings`
- [ ] Verify: `curl http://localhost:8000/health` returns `{"status":"ok"}`
- [ ] Verify: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"triage_notes":"chest pain"}'` returns a valid `TriageResponse`
- [ ] Verify: Swagger UI at `http://localhost:8000/docs` renders correctly

### Frontend Agent
- [ ] Update `frontend/app.py` to POST to `http://localhost:8000/predict` on form submit
- [ ] Update the results page to render data from the backend response (not hardcoded mocks)
- [ ] Use `st.session_state.triage_history` to store past triages
- [ ] Wire up the "Recent Triage" sidebar nav to display history from session state
- [ ] Verify: Full flow works end-to-end (intake → submit → results)

---

## Conflict Protocol

If an implementer agent cannot complete a task because the spec in this file is ambiguous, incomplete, or incorrect, they should **stop and report back** with a clear description of the issue. The user will relay this to the Planner for a spec update.
