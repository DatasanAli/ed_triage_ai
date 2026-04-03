"""
Pydantic request/response models for the /predict endpoint.

Mirrors the API contract defined in ORCHESTRATION.md exactly.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    """Incoming triage assessment request from the Streamlit frontend."""

    model: str = Field(default="arch4", description="Model architecture to invoke")
    triage_notes: str = Field(..., description="Free-text clinical notes from the UI")
    age: Optional[int] = Field(default=None, description="Patient age in years")
    heart_rate: Optional[int] = Field(default=None, description="BPM")
    resp_rate: Optional[int] = Field(default=None, description="Breaths per minute")
    sbp: Optional[int] = Field(default=None, description="Systolic blood pressure (mmHg)")
    dbp: Optional[int] = Field(default=None, description="Diastolic blood pressure (mmHg)")
    spo2: Optional[int] = Field(default=None, description="Oxygen saturation (%)")
    temp_f: Optional[float] = Field(default=None, description="Temperature in Fahrenheit")
    pain: Optional[int] = Field(default=None, description="Pain scale 0-10")
    arrival_transport: str = Field(
        default="Walk In",
        description="One of: Walk In, Ambulance, Helicopter, Unknown",
    )


class TopFeature(BaseModel):
    """A single SHAP-driven feature explanation."""

    feature: str
    shap: float
    direction: str


class TriageResponse(BaseModel):
    """Prediction result returned to the Streamlit frontend."""

    predicted_class: int = Field(description="0, 1, or 2")
    predicted_label: str = Field(
        description='One of: "L1-Critical", "L2-Emergent", "L3-Urgent/LessUrgent"'
    )
    probabilities: dict[str, float] = Field(description="Confidence per class")
    top_features: list[TopFeature] = Field(description="Top 5 SHAP drivers")
    safety_flag: bool = Field(description="True if clinical scores conflict with prediction")
    safety_reason: Optional[str] = Field(
        default=None, description="Human-readable explanation if flagged"
    )
    model_used: str = Field(description="Which model produced this result")

    # Enriched fields from the agentic pipeline (LangGraph + RAG + LLM)
    reconciled_label: Optional[str] = Field(default=None, description="More cautious of model vs LLM recommendation")
    reconciled_class: Optional[int] = Field(default=None, description="0-based class index for reconciled_label")
    llm_esi: Optional[int] = Field(default=None, description="LLM independent ESI recommendation (1/2/3)")
    llm_agreement: Optional[bool] = Field(default=None, description="True if LLM agrees with model prediction")
    clinical_rationale: Optional[str] = Field(default=None, description="LLM clinical reasoning narrative")
    similar_cases: Optional[list] = Field(default=None, description="Top RAG-retrieved historical cases")
    flags: Optional[list[str]] = Field(default=None, description="Escalation signals and uncertainty notes")
    confidence_pct: Optional[int] = Field(default=None, description="Model confidence 0-100")
    uncertainty_flag: Optional[bool] = Field(default=None, description="True if confidence is below threshold")
