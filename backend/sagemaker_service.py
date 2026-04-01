"""
SageMaker interaction layer.

Contains:
- transform_request:  Frontend payload  → SageMaker payload
- transform_response: SageMaker payload → Frontend payload
- invoke_endpoint:    Calls the SageMaker endpoint (or returns mock data)
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import boto3

from backend.config import settings
from backend.schemas import TriageRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock data — used when settings.use_mock is True
# ---------------------------------------------------------------------------
MOCK_SAGEMAKER_RESPONSE: dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def transform_request(request: TriageRequest) -> dict[str, Any]:
    """
    Convert a frontend TriageRequest into the dict payload SageMaker expects.

    Rules (per ORCHESTRATION.md):
    - Rename ``triage_notes`` → ``triage_text``
    - Uppercase ``arrival_transport``
    - Remove the ``model`` field
    - Exclude any field whose value is ``None``
    """
    payload: dict[str, Any] = {}

    # Rename triage_notes → triage_text
    payload["triage_text"] = request.triage_notes

    # Uppercase arrival_transport
    payload["arrival_transport"] = request.arrival_transport.upper()

    # Include remaining optional fields, dropping Nones
    optional_fields = [
        "age", "heart_rate", "resp_rate", "sbp", "dbp", "spo2", "temp_f", "pain",
    ]
    for field_name in optional_fields:
        value = getattr(request, field_name)
        if value is not None:
            payload[field_name] = value

    return payload


def transform_response(sagemaker_response: dict[str, Any], model_used: str) -> dict[str, Any]:
    """
    Convert a raw SageMaker response dict into the frontend-facing response dict.

    Rules (per ORCHESTRATION.md):
    - Keep: predicted_class, predicted_label, probabilities, top_features,
            safety_flag, safety_reason
    - Add:  model_used
    - Drop: lgbm_shap
    """
    return {
        "predicted_class": sagemaker_response["predicted_class"],
        "predicted_label": sagemaker_response["predicted_label"],
        "probabilities": sagemaker_response["probabilities"],
        "top_features": sagemaker_response["top_features"],
        "safety_flag": sagemaker_response["safety_flag"],
        "safety_reason": sagemaker_response.get("safety_reason"),
        "model_used": model_used,
    }


# ---------------------------------------------------------------------------
# Endpoint invocation
# ---------------------------------------------------------------------------

def invoke_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Call the SageMaker endpoint with a pre-built payload dict and return the
    raw response.

    NOTE: This function is temporary scaffolding. The target state is for
    invoke_endpoint to move into the orchestration/ service, which will own the
    full inference pipeline (SageMaker + LangGraph + RAG). run_triage_inference
    will then delegate to orchestration/ rather than calling this directly.
    """
    if settings.use_mock:
        logger.info("Using mock SageMaker response (TRIAGE_USE_MOCK=True)")
        return copy.deepcopy(MOCK_SAGEMAKER_RESPONSE)

    logger.info("Invoking SageMaker endpoint: %s", settings.sagemaker_endpoint_name)
    runtime = boto3.client("sagemaker-runtime", region_name=settings.aws_region)
    sm_result = runtime.invoke_endpoint(
        EndpointName=settings.sagemaker_endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(sm_result["Body"].read().decode("utf-8"))


def run_triage_inference(request: TriageRequest) -> dict[str, Any]:
    """
    Inference orchestration entry point for the /predict route.

    Transforms the frontend request, calls the SageMaker endpoint, and returns
    a response dict ready for the frontend. Future: this will delegate to the
    orchestration/ service instead of calling invoke_endpoint directly.
    """
    model_used = request.model or settings.default_model
    sm_payload = transform_request(request)

    logger.info("Request payload (SageMaker format): %s", json.dumps(sm_payload, indent=2))
    raw_response = invoke_endpoint(sm_payload)
    result = transform_response(raw_response, model_used)
    logger.info("Response payload (frontend format): %s", json.dumps(result, indent=2))
    return result
