"""
TriagePulse — Emergency Department Triage Assistant (Streamlit UI).

Single-file Streamlit app with two views:
  1. Intake form — collects triage notes + optional vitals
  2. Results    — displays triage prediction returned by the FastAPI backend

Run with:  streamlit run frontend/app.py
"""

import streamlit as st
import requests
from datetime import datetime, timezone

# ── Backend ───────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"

TRANSPORT_OPTIONS = ["Walk In", "Ambulance", "Helicopter", "Unknown"]


# ── Helpers ───────────────────────────────────────────────────────────────────

# Map backend feature names to friendly display names
FEATURE_DISPLAY_NAMES = {
    "spo2": "SpO2 (Oxygen Saturation)",
    "heart_rate": "Heart Rate",
    "shock_index": "Shock Index",
    "news2_score": "NEWS2 Score",
    "sbp": "Systolic BP",
    "dbp": "Diastolic BP",
    "resp_rate": "Respiratory Rate",
    "age": "Age",
    "temp_f": "Temperature (°F)",
    "pain": "Pain Score",
    "arrival_transport": "Arrival Transport",
}


def shap_features_to_drivers(top_features: list) -> list:
    """Convert TriageResponse top_features (SHAP) to driver dicts for the UI."""
    drivers = []
    for feat in top_features:
        name = feat.get("feature", "")
        shap_val = feat.get("shap", 0.0)
        direction = feat.get("direction", "")
        display_name = FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " ").title())
        is_critical = abs(shap_val) >= 0.15  # treat high-impact SHAP as critical
        icon = "&#9888;" if is_critical else "&#9829;"
        detail = f"SHAP: {shap_val:+.4f} — {direction}"
        drivers.append({
            "title": display_name,
            "detail": detail,
            "icon": icon,
            "critical": is_critical,
        })
    return drivers


def label_to_recommendations(predicted_label: str) -> list:
    """Return protocol recommendations based on predicted triage label."""
    if "L1" in predicted_label or "Critical" in predicted_label:
        return [
            "Activate Rapid Response / Resuscitation Team immediately.",
            "Prepare point-of-care ultrasound (POCUS) at bedside.",
            "Order immediate ABG, Lactate, and CBC/CMP panels.",
        ]
    elif "L2" in predicted_label or "Emergent" in predicted_label:
        return [
            "Assign to monitored ED bed within 15 minutes.",
            "Obtain 12-lead ECG and point-of-care labs.",
            "Notify attending physician for prompt evaluation.",
        ]
    else:
        return [
            "Place in waiting area with standard monitoring.",
            "Reassess vitals every 30 minutes.",
            "Initiate symptom-directed workup as needed.",
        ]


# ── CSS injection ─────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Public+Sans:wght@300;400;500;600;700&display=swap');

    /* ── Design system tokens ──────────────────────────────────── */
    :root {
        --primary: #00478d;
        --primary-container: #005eb8;
        --on-primary: #ffffff;
        --primary-fixed: #d6e3ff;
        --secondary: #4a6178;
        --tertiary: #940010;
        --tertiary-container: #bb1b21;
        --tertiary-fixed: #ffdad6;
        --on-surface: #171c22;
        --on-surface-variant: #424752;
        --surface: #f8f9ff;
        --surface-container-low: #f0f4fd;
        --surface-container: #eaeef7;
        --surface-container-high: #e4e8f1;
        --surface-container-highest: #dee3eb;
        --surface-container-lowest: #ffffff;
        --outline-variant: #c2c6d4;
        --error-container: #ffdad6;
        --on-error-container: #93000a;
    }

    /* ── Global overrides ──────────────────────────────────────── */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--surface) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ───────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        width: 100%;
    }
    [data-testid="stSidebar"] .block-container {
        padding: 0 !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }

    .sidebar-logo {
        display: flex; align-items: center; gap: 12px;
        padding: 24px 24px 20px; margin-bottom: 8px;
    }
    .sidebar-logo-icon {
        width: 40px; height: 40px; border-radius: 8px;
        background: var(--primary); display: flex;
        align-items: center; justify-content: center;
        color: white; font-size: 20px; flex-shrink: 0;
    }
    .sidebar-logo h1 {
        font-size: 18px; font-weight: 700; color: #0f172a;
        margin: 0; line-height: 1.1;
    }
    .sidebar-logo p {
        font-family: 'Public Sans', sans-serif;
        font-size: 10px; text-transform: uppercase;
        letter-spacing: 2px; color: #64748b;
        margin: 2px 0 0; font-weight: 700;
    }

    .nav-item {
        display: flex; align-items: center; gap: 12px;
        padding: 12px 24px; color: #475569;
        font-size: 14px; font-weight: 500;
        text-decoration: none; transition: all 0.15s;
        border-right: 4px solid transparent;
        cursor: default;
    }
    .nav-item:hover { background: #f1f5f9; color: #0f172a; }
    .nav-item.active {
        background: #eff6ff; color: #1d4ed8;
        border-right-color: #1d4ed8; font-weight: 600;
    }
    .nav-icon { font-size: 18px; width: 24px; text-align: center; }
    .nav-divider {
        border: none; border-top: 1px solid #e2e8f0;
        margin: 16px 0;
    }
    .nav-footer { margin-top: auto; }

    /* ── History item in sidebar ───────────────────────────────── */
    .history-item {
        padding: 10px 24px;
        border-bottom: 1px solid #f1f5f9;
        cursor: default;
    }
    .history-label {
        font-family: 'Public Sans', sans-serif;
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.5px;
        color: var(--tertiary); margin-bottom: 2px;
    }
    .history-label.l2 { color: var(--primary); }
    .history-label.l3 { color: var(--secondary); }
    .history-notes {
        font-size: 11px; color: #475569;
        white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis; max-width: 200px;
    }
    .history-time {
        font-size: 10px; color: #94a3b8; margin-top: 2px;
    }

    /* ── Top header bar ────────────────────────────────────────── */
    .top-header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 16px 0; margin-bottom: 16px;
        border-bottom: 1px solid #eaeef7;
    }
    .top-header-left {
        display: flex; align-items: center; gap: 16px;
    }
    .brand-name {
        font-size: 20px; font-weight: 900; color: #1d4ed8;
        letter-spacing: -0.5px;
    }
    .header-divider {
        width: 1px; height: 24px; background: #e2e8f0;
    }
    .page-title {
        font-size: 16px; font-weight: 700; color: var(--on-surface);
    }
    .top-header-right {
        display: flex; align-items: center; gap: 16px;
    }
    .search-box {
        background: #f1f5f9; border: none; border-radius: 20px;
        padding: 8px 16px 8px 36px; font-size: 13px; color: #64748b;
        width: 220px; position: relative;
    }
    .header-icon {
        width: 36px; height: 36px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        color: #64748b; font-size: 18px;
    }
    .avatar {
        width: 32px; height: 32px; border-radius: 50%;
        background: #cbd5e1; display: flex; align-items: center;
        justify-content: center; font-size: 14px; color: #475569;
        border: 2px solid #e2e8f0;
    }

    /* ── Cards ──────────────────────────────────────────────────── */
    .card {
        background: var(--surface-container-lowest);
        border-radius: 12px; padding: 32px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .card-vitals {
        background: rgba(222,227,235,0.5);
        backdrop-filter: blur(12px);
        border-radius: 12px; padding: 24px;
        border: 1px solid rgba(194,198,212,0.2);
    }

    /* ── Section labels ────────────────────────────────────────── */
    .section-label {
        font-family: 'Public Sans', sans-serif;
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2px;
        margin-bottom: 4px;
    }
    .section-label.primary { color: var(--primary); }
    .section-label.secondary { color: var(--secondary); }
    .section-title {
        font-size: 22px; font-weight: 800;
        color: var(--on-surface); margin: 4px 0 16px;
    }

    /* ── Badges ─────────────────────────────────────────────────── */
    .badge-row { display: flex; gap: 8px; }
    .badge {
        font-family: 'Public Sans', sans-serif;
        font-size: 9px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.5px;
        padding: 4px 10px; border-radius: 20px;
        background: var(--surface-container-highest);
        color: var(--on-surface-variant);
    }

    /* ── Vitals input labels ───────────────────────────────────── */
    .vital-label {
        font-family: 'Public Sans', sans-serif;
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.5px;
        color: var(--on-surface-variant);
        margin-bottom: 4px; padding-left: 4px;
    }

    /* ── Pain slider labels ────────────────────────────────────── */
    .pain-header {
        display: flex; justify-content: space-between;
        align-items: center; margin-bottom: 4px; padding: 0 4px;
    }
    .pain-label {
        font-family: 'Public Sans', sans-serif;
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; color: var(--on-surface-variant);
    }
    .pain-value {
        font-size: 14px; font-weight: 900; color: var(--primary);
    }
    .pain-range {
        display: flex; justify-content: space-between;
        padding: 0 4px; margin-top: -8px;
    }
    .pain-range span {
        font-family: 'Public Sans', sans-serif;
        font-size: 8px; font-weight: 700;
        color: #94a3b8; text-transform: uppercase;
    }

    /* ── System monitoring indicator ───────────────────────────── */
    .monitor-card {
        margin-top: 24px; padding: 16px;
        background: rgba(214,227,255,0.3);
        border-radius: 8px;
        border: 1px solid rgba(169,199,255,0.5);
        display: flex; align-items: center; gap: 12px;
    }
    .pulse-ring {
        position: relative; width: 32px; height: 32px;
        display: flex; align-items: center; justify-content: center;
    }
    .pulse-ring::before {
        content: ''; position: absolute; inset: 0;
        border-radius: 50%; background: var(--primary-container);
        animation: pulse-anim 2s ease-in-out infinite; opacity: 0.2;
    }
    @keyframes pulse-anim {
        0%, 100% { transform: scale(1); opacity: 0.2; }
        50% { transform: scale(1.4); opacity: 0; }
    }
    .pulse-icon { font-size: 18px; color: var(--primary); z-index: 1; }
    .monitor-label {
        font-family: 'Public Sans', sans-serif;
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.5px;
        color: #001b3d;
    }
    .monitor-status { font-size: 12px; color: #00468c; }

    /* ── Tip card ──────────────────────────────────────────────── */
    .tip-card {
        display: flex; gap: 16px; align-items: center;
        background: var(--surface-container-lowest);
        border-radius: 12px; padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-top: 16px;
    }
    .tip-icon-box {
        width: 44px; height: 44px; border-radius: 8px;
        background: #cee5ff; display: flex;
        align-items: center; justify-content: center;
        font-size: 20px; color: #021d31; flex-shrink: 0;
    }
    .tip-title { font-size: 12px; font-weight: 700; color: var(--on-surface); }
    .tip-text {
        font-size: 11px; color: var(--on-surface-variant); margin-top: 2px;
    }

    /* ── Submit button styling ─────────────────────────────────── */
    .submit-area {
        display: flex; justify-content: center; padding: 16px 0 8px;
    }

    /* style the primary stButton */
    [data-testid="stButton"] > button[kind="primary"] {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        padding: 16px 32px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 14px rgba(0,71,141,0.2) !important;
        transition: all 0.15s !important;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        background: var(--primary-container) !important;
    }
    [data-testid="stButton"] > button[kind="secondary"],
    [data-testid="stButton"] > button:not([kind="primary"]) {
        background: var(--surface-container-highest) !important;
        color: var(--on-surface) !important;
        border: none !important;
        padding: 16px 32px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }

    /* ── Input field overrides ─────────────────────────────────── */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #f8fafc !important;
        border: 1px solid #dde3ed !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        color: var(--on-surface) !important;
        font-family: 'Inter', sans-serif !important;
        padding: 10px 14px !important;
    }
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(0,71,141,0.08) !important;
        background: #fff !important;
    }
    [data-testid="stTextArea"] textarea {
        font-size: 14px !important;
        font-weight: 400 !important;
        line-height: 1.7 !important;
        resize: none !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #f8fafc !important;
        border: 1px solid #dde3ed !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        color: var(--on-surface) !important;
    }

    /* hide default streamlit labels — we use custom HTML labels */
    [data-testid="stNumberInput"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label {
        display: none !important;
    }

    /* slider accent */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background: var(--primary) !important;
    }
    [data-testid="stSlider"] label { display: none !important; }
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {
        display: none !important;
    }

    /* ── Intake form card ──────────────────────────────────────── */
    .intake-card {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        padding: 32px 36px;
        margin-bottom: 24px;
    }
    .intake-section-title {
        font-size: 11px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.5px;
        color: var(--on-surface-variant);
        margin-bottom: 16px; margin-top: 24px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f1f5f9;
    }
    .intake-section-title:first-child { margin-top: 0; }
    .field-label {
        font-size: 12px; font-weight: 600;
        color: #475569; margin-bottom: 4px;
        letter-spacing: 0.2px;
    }

    /* ── Results page styles ───────────────────────────────────── */
    .breadcrumb {
        font-family: 'Public Sans', sans-serif;
        font-size: 11px; font-weight: 600;
        letter-spacing: 1px; color: var(--on-surface-variant);
        margin-bottom: 4px;
    }
    .breadcrumb .active { color: var(--primary); font-weight: 700; }
    .results-title {
        font-size: 28px; font-weight: 800;
        color: var(--on-surface); letter-spacing: -0.5px;
        margin-bottom: 24px;
    }
    .assigned-to {
        display: flex; align-items: center; gap: 12px;
        background: var(--surface-container-low);
        padding: 8px 16px; border-radius: 12px; float: right;
    }
    .assigned-label {
        font-family: 'Public Sans', sans-serif;
        font-size: 10px; text-transform: uppercase;
        color: var(--on-surface-variant); text-align: right;
    }
    .assigned-name {
        font-size: 14px; font-weight: 600;
        color: var(--on-surface); text-align: right;
    }

    /* Priority card */
    .priority-card {
        background: var(--surface-container-lowest);
        border-radius: 12px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border-left: 6px solid var(--tertiary);
        display: flex; margin-bottom: 24px;
    }
    .priority-content { padding: 32px; flex: 1; }
    .priority-badge {
        font-family: 'Public Sans', sans-serif;
        font-size: 11px; font-weight: 700; color: var(--tertiary);
        background: var(--tertiary-fixed);
        padding: 4px 10px; border-radius: 4px;
        display: inline-block; margin-bottom: 16px;
    }
    .priority-level {
        font-size: 52px; font-weight: 900;
        color: var(--tertiary); letter-spacing: -2px;
        line-height: 1; margin-bottom: 12px;
    }
    .priority-desc {
        color: var(--on-surface-variant);
        line-height: 1.6; max-width: 400px; font-size: 14px;
    }
    .target-zone {
        background: var(--tertiary-container);
        color: #ffceca; padding: 32px;
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        text-align: center; min-width: 200px;
    }
    .target-icon { font-size: 40px; margin-bottom: 8px; }
    .target-label {
        font-family: 'Public Sans', sans-serif;
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2px;
        opacity: 0.8; margin-bottom: 4px;
    }
    .target-name { font-size: 22px; font-weight: 700; }

    /* Confidence bars */
    .confidence-section {
        background: var(--surface-container-low);
        padding: 24px; border-radius: 12px;
        border: 1px solid rgba(194,198,212,0.1);
    }
    .confidence-title {
        font-family: 'Public Sans', sans-serif;
        font-size: 12px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.5px;
        color: var(--on-surface-variant); margin-bottom: 20px;
    }
    .conf-row { margin-bottom: 20px; }
    .conf-row:last-child { margin-bottom: 0; }
    .conf-labels {
        display: flex; justify-content: space-between;
        font-family: 'Public Sans', sans-serif;
        font-size: 12px; margin-bottom: 6px;
    }
    .conf-name { font-weight: 600; }
    .conf-name.critical { color: var(--tertiary); font-weight: 700; }
    .conf-name.muted { color: var(--on-surface-variant); }
    .conf-pct { font-weight: 700; color: var(--on-surface); }
    .conf-pct.muted { color: var(--on-surface-variant); font-weight: 500; }
    .conf-track {
        width: 100%; border-radius: 999px;
        background: var(--surface-container-highest); overflow: hidden;
    }
    .conf-track.lg { height: 12px; }
    .conf-track.sm { height: 8px; }
    .conf-fill { height: 100%; border-radius: 999px; }
    .conf-fill.critical { background: var(--tertiary); }
    .conf-fill.muted { background: var(--primary-container); opacity: 0.4; }
    .conf-fill.faint { background: var(--primary-container); opacity: 0.2; }

    /* Clinical drivers */
    .drivers-section {
        background: var(--surface-container-lowest);
        padding: 24px; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .driver-item {
        display: flex; align-items: flex-start; gap: 12px;
        padding: 12px; border-radius: 8px; margin-bottom: 8px;
    }
    .driver-item:last-child { margin-bottom: 0; }
    .driver-item.critical-bg { background: rgba(255,218,214,0.3); }
    .driver-item.normal-bg { background: var(--surface-container); }
    .driver-icon {
        font-size: 14px; margin-top: 2px; flex-shrink: 0;
    }
    .driver-icon.critical { color: var(--tertiary); }
    .driver-icon.primary { color: var(--primary); }
    .driver-title { font-size: 12px; font-weight: 700; }
    .driver-title.critical { color: var(--on-error-container); }
    .driver-title.normal { color: var(--on-surface); }
    .driver-detail {
        font-size: 11px; color: var(--on-surface-variant); margin-top: 2px;
    }

    /* Patient profile card */
    .patient-card {
        background: rgba(228,232,241,0.5);
        backdrop-filter: blur(12px);
        padding: 24px; border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.4);
        margin-bottom: 20px;
    }
    .patient-header {
        display: flex; align-items: center; gap: 16px;
        margin-bottom: 20px;
    }
    .patient-avatar {
        width: 48px; height: 48px; border-radius: 50%;
        background: var(--surface-container-highest);
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; color: var(--on-surface-variant);
    }
    .patient-name {
        font-size: 18px; font-weight: 700; color: var(--on-surface);
    }
    .patient-meta {
        font-family: 'Public Sans', sans-serif;
        font-size: 12px; color: var(--on-surface-variant);
    }
    .vitals-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .vital-display {
        padding: 16px; background: var(--surface-container-lowest);
        border-radius: 8px; border: 1px solid rgba(194,198,212,0.1);
    }
    .vital-display-label {
        font-family: 'Public Sans', sans-serif;
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; color: var(--on-surface-variant);
        margin-bottom: 4px;
    }
    .vital-display-value {
        font-size: 24px; font-weight: 900; color: var(--on-surface);
        display: inline;
    }
    .vital-display-unit {
        font-size: 12px; color: var(--on-surface-variant);
        margin-left: 4px;
    }
    .vital-heart-icon {
        color: var(--tertiary); font-size: 14px;
        margin-left: 4px; display: inline;
    }

    /* Recommendations */
    .rec-section {
        background: var(--surface-container-lowest);
        padding: 24px; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    .rec-title {
        font-family: 'Public Sans', sans-serif;
        font-size: 12px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.5px;
        color: var(--on-surface-variant); margin-bottom: 16px;
    }
    .rec-item {
        display: flex; align-items: flex-start; gap: 16px;
        margin-bottom: 16px;
    }
    .rec-item:last-child { margin-bottom: 0; }
    .rec-check {
        width: 24px; height: 24px; border-radius: 4px;
        background: var(--primary-fixed);
        display: flex; align-items: center; justify-content: center;
        color: var(--primary); font-size: 14px; font-weight: 700;
        flex-shrink: 0;
    }
    .rec-text {
        font-size: 14px; font-weight: 500;
        color: var(--on-surface); line-height: 1.4;
    }

    /* Reconciled banner */
    .reconciled-banner {
        background: #fffbeb; border: 1px solid #f59e0b;
        border-left: 6px solid #f59e0b;
        border-radius: 8px; padding: 16px 20px;
        margin-bottom: 20px;
        display: flex; align-items: flex-start; gap: 12px;
    }
    .reconciled-icon { font-size: 20px; flex-shrink: 0; margin-top: 1px; }
    .reconciled-title {
        font-size: 13px; font-weight: 700; color: #92400e; margin-bottom: 4px;
    }
    .reconciled-body { font-size: 13px; color: #78350f; line-height: 1.5; }

    /* Flags */
    .flags-section {
        margin-bottom: 20px;
    }
    .flag-item {
        display: flex; align-items: flex-start; gap: 10px;
        background: rgba(255,218,214,0.25);
        border: 1px solid rgba(187,27,33,0.2);
        border-radius: 8px; padding: 12px 16px;
        margin-bottom: 8px; font-size: 13px;
        color: var(--on-error-container); line-height: 1.5;
    }
    .flag-icon { flex-shrink: 0; font-size: 14px; margin-top: 1px; }

    /* Clinical rationale */
    .rationale-section {
        background: var(--surface-container-lowest);
        border-radius: 12px; padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    .rationale-text {
        font-size: 14px; color: var(--on-surface);
        line-height: 1.75; white-space: pre-wrap;
    }

    /* Similar cases table */
    .cases-section {
        background: var(--surface-container-lowest);
        border-radius: 12px; padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    .cases-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .cases-table th {
        font-family: 'Public Sans', sans-serif;
        font-size: 10px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; color: var(--on-surface-variant);
        padding: 8px 10px; border-bottom: 1px solid var(--outline-variant);
        text-align: left;
    }
    .cases-table td {
        padding: 10px 10px; color: var(--on-surface);
        border-bottom: 1px solid var(--surface-container);
        vertical-align: top;
    }
    .cases-table tr:last-child td { border-bottom: none; }
    .esi-badge {
        display: inline-block; font-weight: 700; font-size: 11px;
        padding: 2px 8px; border-radius: 4px;
    }
    .esi-1 { background: var(--tertiary-fixed); color: var(--tertiary); }
    .esi-2 { background: var(--primary-fixed); color: var(--primary); }
    .esi-3 { background: var(--surface-container-highest); color: var(--secondary); }
    .sim-score { font-weight: 600; color: var(--primary); }

    /* Footer status bar */
    .footer-bar {
        display: flex; justify-content: space-between;
        align-items: center; margin-top: 32px;
        padding-top: 20px;
        border-top: 1px solid rgba(194,198,212,0.3);
        font-family: 'Public Sans', sans-serif;
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2px;
        color: var(--on-surface-variant);
    }
    .footer-left { display: flex; align-items: center; gap: 16px; }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #22c55e; display: inline-block;
        margin-right: 6px;
    }

    /* Hide number input steppers */
    [data-testid="stNumberInput"] button { display: none !important; }
    [data-testid="stNumberInput"] > div { gap: 0 !important; }

    /* Hide text area label */
    [data-testid="stTextArea"] label { display: none !important; }

    /* ── Hide sidebar entirely ──────────────────────────────────── */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* ── Fix unclickable buttons below HTML markdown blocks ─────── */
    /* Raise Streamlit buttons above any HTML overlay divs */
    [data-testid="stButton"] {
        position: relative !important;
        z-index: 10 !important;
    }

    /* ── Sidebar nav buttons — match .nav-item look ─────────────── */
    [data-testid="stSidebar"] [data-testid="stButton"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button,
    [data-testid="stSidebar"] [data-testid="stButton"] > button:not([kind="primary"]) {
        background: transparent !important;
        color: #475569 !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        box-shadow: none !important;
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100% !important;
        border-right: 4px solid transparent !important;
        transition: all 0.15s !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        background: #f1f5f9 !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"].nav-active > button {
        background: #eff6ff !important;
        color: #1d4ed8 !important;
        border-right-color: #1d4ed8 !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        active_page = st.session_state.get("page", "intake")
        results_cls = "active" if active_page == "results" else ""

        # Logo (static HTML)
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">&#10010;</div>
            <div>
                <h1>ED Central</h1>
                <p>Station 4</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Patient Intake — real button for navigation ────────
        if active_page == "intake":
            st.markdown('<div class="nav-item active"><span class="nav-icon">&#128100;</span> Patient Intake</div>', unsafe_allow_html=True)
        else:
            if st.button("\U0001f464  Patient Intake", key="nav_intake_btn", use_container_width=True):
                for key in ["triage_notes", "age", "heart_rate", "resp_rate",
                            "sbp", "dbp", "spo2", "temp_f", "pain",
                            "arrival_transport", "form_data", "triage_result", "is_loading"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.page = "intake"
                st.rerun()

        # ── Triage History nav label ───────────────────────────
        st.markdown(f'<div class="nav-item {results_cls}"><span class="nav-icon">&#128336;</span> Triage History</div>', unsafe_allow_html=True)

        # Push footer to bottom with spacer
        st.markdown("<div style='flex:1; min-height:200px;'></div>",
                    unsafe_allow_html=True)

        st.markdown("""
        <hr class="nav-divider">
        <div class="nav-item">
            <span class="nav-icon">&#10067;</span> Help
        </div>
        <div class="nav-item">
            <span class="nav-icon">&#10145;</span> Logout
        </div>
        """, unsafe_allow_html=True)


# ── Page 1: Intake form ──────────────────────────────────────────────────────

def render_intake_page():
    import json as _json
    import threading, time as _time

    is_loading = st.session_state.get("is_loading", False)

    st.markdown("""
    <div class="top-header">
        <div class="top-header-left">
            <span class="brand-name">TriagePulse</span>
            <div class="header-divider"></div>
            <span class="page-title">New Patient Entry</span>
        </div>
        <div class="top-header-right">
            <span class="header-icon">&#128276;</span>
            <span class="header-icon">&#9881;</span>
            <div class="avatar">&#128100;</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Collapsed summary shown while loading ─────────────────────
    if is_loading:
        fd = st.session_state.get("form_data", {})
        first = fd.get("first_name", "").strip()
        last  = fd.get("last_name", "").strip()
        name  = f"{first} {last}".strip() or "Patient"
        age   = fd.get("age")
        notes = fd.get("triage_notes", "")[:120]
        age_str = f", {age}y" if age else ""
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #dde3ed;border-radius:12px;
                    padding:16px 20px;margin-bottom:16px;
                    display:flex;align-items:center;gap:16px;">
            <div style="font-size:28px;">&#128100;</div>
            <div>
                <div style="font-size:14px;font-weight:700;color:#0f172a;">{name}{age_str}</div>
                <div style="font-size:12px;color:#64748b;margin-top:2px;">{notes}{"..." if len(fd.get("triage_notes","")) > 120 else ""}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Patient Info ─────────────────────────────────────────
        st.markdown('<div class="intake-section-title">Patient Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.markdown('<div class="field-label">First Name</div>', unsafe_allow_html=True)
            st.text_input("First Name", key="first_name", placeholder="Jane", label_visibility="collapsed")
        with c2:
            st.markdown('<div class="field-label">Last Name</div>', unsafe_allow_html=True)
            st.text_input("Last Name", key="last_name", placeholder="Smith", label_visibility="collapsed")
        with c3:
            st.markdown('<div class="field-label">Age</div>', unsafe_allow_html=True)
            st.number_input("Age", min_value=0, max_value=120, value=None, key="age",
                            placeholder="—", label_visibility="collapsed")

        # ── Triage Notes ─────────────────────────────────────────
        st.markdown('<div class="intake-section-title">Triage Notes</div>', unsafe_allow_html=True)
        st.text_area(
            "Triage Notes", height=160, key="triage_notes",
            placeholder="Chief complaint, symptoms, relevant history...",
            label_visibility="collapsed",
        )

        # ── Vitals ───────────────────────────────────────────────
        st.markdown('<div class="intake-section-title">Vitals</div>', unsafe_allow_html=True)
        v1, v2, v3, v4, v5, v6 = st.columns(6)
        with v1:
            st.markdown('<div class="field-label">Heart Rate</div>', unsafe_allow_html=True)
            st.number_input("HR", min_value=20, max_value=250, value=None, key="heart_rate",
                            placeholder="BPM", label_visibility="collapsed")
        with v2:
            st.markdown('<div class="field-label">Resp Rate</div>', unsafe_allow_html=True)
            st.number_input("RR", min_value=4, max_value=60, value=None, key="resp_rate",
                            placeholder="br/m", label_visibility="collapsed")
        with v3:
            st.markdown('<div class="field-label">SBP</div>', unsafe_allow_html=True)
            st.number_input("SBP", min_value=40, max_value=300, value=None, key="sbp",
                            placeholder="mmHg", label_visibility="collapsed")
        with v4:
            st.markdown('<div class="field-label">DBP</div>', unsafe_allow_html=True)
            st.number_input("DBP", min_value=10, max_value=200, value=None, key="dbp",
                            placeholder="mmHg", label_visibility="collapsed")
        with v5:
            st.markdown('<div class="field-label">SpO2</div>', unsafe_allow_html=True)
            st.number_input("SpO2", min_value=50, max_value=100, value=None, key="spo2",
                            placeholder="%", label_visibility="collapsed")
        with v6:
            st.markdown('<div class="field-label">Temp (°F)</div>', unsafe_allow_html=True)
            st.number_input("Temp", min_value=85.0, max_value=115.0, value=None, key="temp_f",
                            placeholder="°F", step=0.1, label_visibility="collapsed")

        # ── Pain + Transport ─────────────────────────────────────
        st.markdown('<div class="intake-section-title">Assessment</div>', unsafe_allow_html=True)
        p_col, t_col = st.columns([3, 1])
        with p_col:
            pain_val = st.session_state.get("pain", 4)
            st.markdown(f'<div class="field-label">Pain Level &nbsp;<span style="color:var(--primary);font-weight:700;">{pain_val} / 10</span></div>', unsafe_allow_html=True)
            st.slider("Pain", min_value=0, max_value=10, value=4, key="pain", label_visibility="collapsed")
        with t_col:
            st.markdown('<div class="field-label">Arrival Transport</div>', unsafe_allow_html=True)
            st.selectbox("Transport", TRANSPORT_OPTIONS, key="arrival_transport", label_visibility="collapsed")

    # ── Submit button (disabled while loading) ───────────────────
    _l, center, _r = st.columns([2, 3, 2])
    with center:
        clicked = st.button(
            "Analyzing..." if is_loading else "Run Triage Assessment",
            type="primary",
            use_container_width=True,
            key="submit_btn",
            disabled=is_loading,
        )

        # Status steps shown directly below the button
        status_box = st.empty()

    if clicked and not is_loading:
        form_data = {
            "first_name": st.session_state.get("first_name", ""),
            "last_name": st.session_state.get("last_name", ""),
            "triage_notes": st.session_state.get("triage_notes", ""),
            "age": st.session_state.get("age"),
            "heart_rate": st.session_state.get("heart_rate"),
            "resp_rate": st.session_state.get("resp_rate"),
            "sbp": st.session_state.get("sbp"),
            "dbp": st.session_state.get("dbp"),
            "spo2": st.session_state.get("spo2"),
            "temp_f": st.session_state.get("temp_f"),
            "pain": st.session_state.get("pain", 4),
            "arrival_transport": st.session_state.get("arrival_transport", "Walk In"),
        }
        st.session_state.form_data = form_data
        st.session_state.is_loading = True
        st.rerun()

    if is_loading:
        _backend_keys = {"triage_notes", "age", "heart_rate", "resp_rate",
                         "sbp", "dbp", "spo2", "temp_f", "pain", "arrival_transport"}
        form_data = st.session_state.get("form_data", {})
        payload = {k: v for k, v in form_data.items()
                   if k in _backend_keys and v is not None}

        result_holder = {}

        def _do_request():
            try:
                r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=120)
                r.raise_for_status()
                result_holder["data"] = r.json()
            except Exception as e:
                result_holder["error"] = e

        thread = threading.Thread(target=_do_request)
        thread.start()

        steps = [
            ("🔬", "Running ML model inference..."),
            ("🔍", "Retrieving similar historical cases..."),
            ("🧠", "LLM clinical analysis in progress..."),
            ("📋", "Synthesizing triage recommendation..."),
        ]
        step_idx = 0
        while thread.is_alive():
            icon, msg = steps[min(step_idx, len(steps) - 1)]
            status_box.markdown(f"""
            <div style="display:flex;align-items:center;gap:14px;padding:16px 20px;
                        background:#f0f6ff;border-radius:10px;border:1px solid #c7d9f5;
                        margin-top:12px;">
                <div style="font-size:22px">{icon}</div>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#1d4ed8;">{msg}</div>
                    <div style="font-size:11px;color:#64748b;margin-top:2px;">
                        Step {min(step_idx+1, len(steps))} of {len(steps)}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            _time.sleep(4)
            step_idx += 1

        thread.join()
        status_box.empty()
        st.session_state.is_loading = False

        if "error" in result_holder:
            exc = result_holder["error"]
            if isinstance(exc, requests.exceptions.ConnectionError):
                st.error(f"Cannot reach the backend at {BACKEND_URL}. "
                         "Make sure `uvicorn backend.main:app --reload --port 8000` is running.")
            elif isinstance(exc, requests.exceptions.HTTPError):
                st.error(f"Backend returned an error: {exc.response.status_code} — {exc.response.text}")
            else:
                st.error(f"Unexpected error: {exc}")
            st.stop()

        triage_result = result_holder["data"]
        print(f"[TriagePulse] ← response: {_json.dumps(triage_result, indent=2)}")

        st.session_state.triage_result = triage_result
        history_entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "triage_notes": form_data.get("triage_notes", ""),
            "result": triage_result,
        }
        st.session_state.triage_history.insert(0, history_entry)
        if len(st.session_state.triage_history) > 20:
            st.session_state.triage_history = st.session_state.triage_history[:20]

        st.session_state.page = "results"
        st.rerun()


# ── Page 2: Results ──────────────────────────────────────────────────────────

def render_results_page():
    # Pull real data from session state — no mocks
    triage_result = st.session_state.get("triage_result", {})
    form_data = st.session_state.get("form_data", {})

    if not triage_result:
        st.error("No triage result found. Please complete the intake form first.")
        if st.button("Go to Intake", key="go_intake_btn"):
            st.session_state.page = "intake"
            st.rerun()
        return

    prob = triage_result.get("probabilities", {
        "L1-Critical": 0.0, "L2-Emergent": 0.0, "L3-Urgent/LessUrgent": 0.0
    })
    predicted_label = triage_result.get("predicted_label", "Unknown")
    model_used = triage_result.get("model_used", "arch4")
    safety_flag = triage_result.get("safety_flag", False)
    safety_reason = triage_result.get("safety_reason", None)
    top_features = triage_result.get("top_features", [])

    # Agentic pipeline enriched fields
    reconciled_label = triage_result.get("reconciled_label")
    llm_agreement = triage_result.get("llm_agreement")
    llm_esi = triage_result.get("llm_esi")
    clinical_rationale = triage_result.get("clinical_rationale")
    similar_cases = triage_result.get("similar_cases") or []
    flags = triage_result.get("flags") or []

    # Derive UI data from real backend response
    drivers = shap_features_to_drivers(top_features)

    # Parse label
    label_parts = predicted_label.split("-", 1)
    level_code = label_parts[0] if label_parts else predicted_label
    level_desc = label_parts[1].upper() if len(label_parts) > 1 else ""

    # ── Collapsed patient header ──────────────────────────────────
    first_name = form_data.get("first_name", "").strip()
    last_name  = form_data.get("last_name", "").strip()
    patient_name = f"{first_name} {last_name}".strip() or "Patient"
    age_val = form_data.get("age")
    notes_preview = form_data.get("triage_notes", "")[:120]
    age_str = f", {age_val}y" if age_val else ""
    st.markdown(f"""
    <div style="background:#f8fafc;border:1px solid #dde3ed;border-radius:12px;
                padding:16px 20px;margin-bottom:20px;
                display:flex;align-items:center;gap:16px;">
        <div style="font-size:28px;">&#128100;</div>
        <div style="flex:1;">
            <div style="font-size:14px;font-weight:700;color:#0f172a;">{patient_name}{age_str}</div>
            <div style="font-size:12px;color:#64748b;margin-top:2px;">{notes_preview}{"..." if len(form_data.get("triage_notes","")) > 120 else ""}</div>
        </div>
        <div style="font-size:11px;color:#94a3b8;">{datetime.now(timezone.utc).strftime("%H:%M UTC")}</div>
    </div>
    """, unsafe_allow_html=True)

    # Safety flag banner
    if safety_flag and safety_reason:
        st.warning(f"\u26a0\ufe0f **Safety Alert:** {safety_reason}")

    # Reconciled label banner
    if llm_agreement is False and reconciled_label and reconciled_label != predicted_label:
        esi_num = llm_esi or "?"
        st.markdown(f"""
        <div class="reconciled-banner">
            <div class="reconciled-icon">&#9888;</div>
            <div>
                <div class="reconciled-title">LLM CLINICAL OVERRIDE — Effective Recommendation: {reconciled_label}</div>
                <div class="reconciled-body">
                    The model predicted <strong>{predicted_label}</strong>, but independent LLM reasoning
                    recommended ESI {esi_num} (<strong>{reconciled_label}</strong>).
                    The more cautious level is used for protocol selection.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Escalation flags
    if flags:
        flag_items = "".join(
            f'<div class="flag-item"><span class="flag-icon">&#9873;</span>{f}</div>'
            for f in flags
        )
        st.markdown(f'<div class="flags-section">{flag_items}</div>', unsafe_allow_html=True)

    # ── Priority card (smaller, full width) ───────────────────────
    st.markdown(f"""
    <div style="background:#fff;border-radius:12px;border-left:5px solid var(--tertiary);
                box-shadow:0 1px 3px rgba(0,0,0,0.06);
                display:flex;align-items:center;gap:0;margin-bottom:20px;overflow:hidden;">
        <div style="padding:20px 28px;flex:1;">
            <div style="font-family:'Public Sans',sans-serif;font-size:10px;font-weight:700;
                        color:var(--tertiary);background:var(--tertiary-fixed);
                        padding:3px 8px;border-radius:4px;display:inline-block;margin-bottom:10px;">
                PREDICTED PRIORITY
            </div>
            <div style="font-size:36px;font-weight:900;color:var(--tertiary);
                        letter-spacing:-1px;line-height:1;margin-bottom:6px;">
                {level_code} — {level_desc}
            </div>
            <div style="font-size:13px;color:var(--on-surface-variant);">
                Model: <strong>{model_used}</strong> &nbsp;·&nbsp; Classified as <strong>{predicted_label}</strong>
            </div>
        </div>
        <div style="background:var(--tertiary-container);color:#ffceca;
                    padding:20px 32px;text-align:center;min-width:160px;align-self:stretch;
                    display:flex;flex-direction:column;justify-content:center;">
            <div style="font-size:32px;margin-bottom:4px;">&#9888;</div>
            <div style="font-family:'Public Sans',sans-serif;font-size:9px;font-weight:700;
                        text-transform:uppercase;letter-spacing:2px;opacity:0.8;margin-bottom:2px;">
                Predicted Class
            </div>
            <div style="font-size:16px;font-weight:700;">{predicted_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence + Drivers ──────────────────────────────────────
    c_left, c_right = st.columns(2)

    with c_left:
        l1_pct = prob.get("L1-Critical", 0.0) * 100
        l2_pct = prob.get("L2-Emergent", 0.0) * 100
        l3_pct = prob.get("L3-Urgent/LessUrgent", 0.0) * 100
        st.markdown(f"""
        <div class="confidence-section">
            <div class="confidence-title">Confidence Breakdown</div>
            <div class="conf-row">
                <div class="conf-labels">
                    <span class="conf-name critical">Level 1 (Critical)</span>
                    <span class="conf-pct">{l1_pct:.1f}%</span>
                </div>
                <div class="conf-track lg">
                    <div class="conf-fill critical" style="width:{l1_pct}%"></div>
                </div>
            </div>
            <div class="conf-row">
                <div class="conf-labels">
                    <span class="conf-name muted">Level 2 (Emergent)</span>
                    <span class="conf-pct muted">{l2_pct:.1f}%</span>
                </div>
                <div class="conf-track sm">
                    <div class="conf-fill muted" style="width:{l2_pct}%"></div>
                </div>
            </div>
            <div class="conf-row">
                <div class="conf-labels">
                    <span class="conf-name muted">Level 3 (Urgent)</span>
                    <span class="conf-pct muted">{l3_pct:.1f}%</span>
                </div>
                <div class="conf-track sm">
                    <div class="conf-fill faint" style="width:max(1%,{l3_pct}%)"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        drivers_items = []
        for d in drivers:
            bg_cls = "critical-bg" if d["critical"] else "normal-bg"
            icon_cls = "critical" if d["critical"] else "primary"
            title_cls = "critical" if d["critical"] else "normal"
            drivers_items.append(
                f'<div class="driver-item {bg_cls}">'
                f'<span class="driver-icon {icon_cls}">{d["icon"]}</span>'
                f'<div>'
                f'<div class="driver-title {title_cls}">{d["title"]}</div>'
                f'<div class="driver-detail">{d["detail"]}</div>'
                f'</div></div>'
            )
        drivers_html = "".join(drivers_items) if drivers_items else (
            "<p style='color:var(--on-surface-variant);font-size:13px;'>No SHAP feature data available.</p>"
        )
        st.markdown(
            f'<div class="drivers-section">'
            f'<div class="confidence-title">Clinical Drivers</div>'
            f'{drivers_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Action button ─────────────────────────────────────────────
    if st.button("Triage Another Patient", use_container_width=True, key="restart_btn"):
        for key in ["triage_notes", "age", "heart_rate", "resp_rate",
                    "sbp", "dbp", "spo2", "temp_f", "pain",
                    "arrival_transport", "form_data", "triage_result", "is_loading"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.page = "intake"
        st.rerun()

    # Clinical Rationale
    if clinical_rationale:
        st.markdown("""
        <div class="rationale-section">
            <div class="confidence-title">Clinical Rationale (LLM Analysis)</div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="rationale-text">{clinical_rationale}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Similar Historical Cases
    if similar_cases:
        rows = ""
        for c in similar_cases:
            esi = c.get("triage_level", "?")
            esi_cls = f"esi-{esi}" if esi in (1, 2, 3) else "esi-3"
            hr = f"{c['heart_rate']:.0f}" if c.get("heart_rate") else "—"
            sbp = c.get("sbp"); dbp = c.get("dbp")
            bp = f"{sbp:.0f}/{dbp:.0f}" if sbp and dbp else "—"
            spo2 = f"{c['spo2']:.0f}%" if c.get("spo2") else "—"
            rows += f"""
            <tr>
                <td><span class="sim-score">{c.get('similarity', 0):.2f}</span></td>
                <td><span class="esi-badge {esi_cls}">ESI {esi}</span></td>
                <td>{c.get('chief_complaint') or '—'}</td>
                <td>{hr}</td>
                <td>{bp}</td>
                <td>{spo2}</td>
                <td>{c.get('diagnosis') or '—'}</td>
                <td>{c.get('outcome') or '—'}</td>
            </tr>"""
        st.markdown(f"""
        <div class="cases-section">
            <div class="confidence-title">Similar Historical Cases (RAG)</div>
            <table class="cases-table">
                <thead>
                    <tr>
                        <th>Similarity</th>
                        <th>ESI</th>
                        <th>Chief Complaint</th>
                        <th>HR</th>
                        <th>BP</th>
                        <th>SpO2</th>
                        <th>Diagnosis</th>
                        <th>Outcome</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Footer — show actual model_used from backend response
    now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    st.markdown(f"""
    <div class="footer-bar">
        <div class="footer-left">
            <span><span class="status-dot"></span> AI Engine Online</span>
            <span>Model: {model_used}</span>
        </div>
        <span>Last Sync: {now_utc}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="TriagePulse — ED Triage Assistant",
        page_icon="&#10010;",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "page" not in st.session_state:
        st.session_state.page = "intake"
    if "triage_history" not in st.session_state:
        st.session_state.triage_history = []

    inject_css()

    if st.session_state.page == "intake":
        render_intake_page()
    else:
        render_results_page()


if __name__ == "__main__":
    main()
