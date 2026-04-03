"""
FastAPI application for the ED Triage backend.

Endpoints:
- GET  /health  — Health check
- POST /predict — Accept triage data, return prediction
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import TriageRequest, TriageResponse
from backend.sagemaker_service import run_triage_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TriagePulse API",
    description="Emergency Department triage prediction service",
    version="0.1.0",
)

# CORS — allow the Streamlit frontend (port 8501) and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/predict", response_model=TriageResponse)
def predict(request: TriageRequest):
    """
    Accept triage data from the frontend, run (or mock) the SageMaker
    inference, and return a structured prediction response.
    """
    try:
        result = run_triage_inference(request)
        return TriageResponse(**result)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
