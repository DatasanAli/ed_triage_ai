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
        color: #171c22 !important;
    }

    /* Ensure all native Streamlit text elements are dark */
    .stApp p:not(button p):not([data-testid="stButton"] p),
    .stApp h1, .stApp h2, .stApp h3, .stApp label {
        color: #171c22;
    }

    /* Restore white text on primary buttons */
    [data-testid="stButton"] > button[kind="primary"] p,
    [data-testid="stButton"] > button[kind="primary"] span {
        color: white !important;
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

    /* ── Loading overlay ───────────────────────────────────────── */
    .triage-overlay {
        position: fixed;
        inset: 0;
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(3px);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .triage-overlay-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 40px 48px;
        width: 420px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
        text-align: center;
    }
    .overlay-title {
        font-size: 17px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .overlay-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 28px;
    }
    .overlay-step {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
        text-align: left;
    }
    .overlay-step:last-child { border-bottom: none; }
    .overlay-step-icon { font-size: 18px; width: 28px; text-align: center; }
    .overlay-step-text { font-size: 13px; flex: 1; }
    .step-done   { color: #94a3b8; }
    .step-done .overlay-step-text { text-decoration: line-through; }
    .step-active { color: #1d4ed8; font-weight: 600; }
    .step-pending { color: #cbd5e1; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
    .step-active .overlay-step-icon { animation: pulse 1.2s ease-in-out infinite; }
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
        st.markdown(f"👤 **{name}{age_str}**")
        st.caption(notes + ("..." if len(fd.get("triage_notes", "")) > 120 else ""))
    else:
        # ── Dev shortcuts ─────────────────────────────────────────
        TEST_PATIENTS = {
            "1 · STEMI Suspect (L1)": {
                "first_name": "James", "last_name": "Patterson", "age": 68, "sex": "Male",
                "triage_notes": "Crushing chest pain radiating to left arm with diaphoresis and shortness of breath. Patient diaphoretic, pale, and in acute distress.",
                "heart_rate": 120, "resp_rate": 24, "sbp": 85, "dbp": 55, "spo2": 91, "temp_f": 98.2, "pain": 9,
                "arrival_transport": "Ambulance",
            },
            "2 · Under-triage Trap (L1/L2)": {
                "first_name": "Robert", "last_name": "Chen", "age": 55, "sex": "Male",
                "triage_notes": "Chest pain.",
                "heart_rate": 110, "resp_rate": 22, "sbp": 90, "dbp": 60, "spo2": 94, "temp_f": 98.6, "pain": 8,
                "arrival_transport": "Ambulance",
            },
            "3 · Ankle Sprain (L3)": {
                "first_name": "Tyler", "last_name": "Brooks", "age": 28, "sex": "Male",
                "triage_notes": "Twisted ankle playing basketball. Mild swelling, able to partially bear weight.",
                "heart_rate": 72, "resp_rate": 14, "sbp": 120, "dbp": 78, "spo2": 99, "temp_f": 98.4, "pain": 4,
                "arrival_transport": "Walk In",
            },
            "4 · Appendicitis Suspect (L2)": {
                "first_name": "Maria", "last_name": "Santos", "age": 35, "sex": "Female",
                "triage_notes": "Worsening right lower quadrant pain over 12 hours with nausea, vomiting, and low-grade fever.",
                "heart_rate": 98, "resp_rate": 18, "sbp": 115, "dbp": 72, "spo2": 98, "temp_f": 100.8, "pain": 7,
                "arrival_transport": "Unknown",
            },
            "5 · Sepsis Suspect (L1) — Dorothy": {
                "first_name": "Dorothy", "last_name": "Williams", "age": 72, "sex": "Female",
                "triage_notes": "Fever, confusion, and low blood pressure for the past 6 hours. History of recurrent UTIs. Possible urosepsis.",
                "heart_rate": 118, "resp_rate": 26, "sbp": 88, "dbp": 52, "spo2": 92, "temp_f": 103.1, "pain": 5,
                "arrival_transport": "Ambulance",
            },
            "6 · Acute Stroke (L1)": {
                "first_name": "Michael", "last_name": "Torres", "age": 61, "sex": "Male",
                "triage_notes": "Sudden onset left-sided weakness and facial droop. Unable to speak clearly. Onset 40 minutes ago. FAST exam positive.",
                "heart_rate": 88, "resp_rate": 16, "sbp": 178, "dbp": 105, "spo2": 96, "temp_f": 98.4, "pain": 2,
                "arrival_transport": "Ambulance",
            },
            "7 · Pediatric Fever (L2)": {
                "first_name": "Liam", "last_name": "Johnson", "age": 3, "sex": "Male",
                "triage_notes": "High fever and irritability in a 3-year-old. Not eating since yesterday. No rash or neck stiffness.",
                "heart_rate": 130, "resp_rate": 28, "sbp": 95, "dbp": 60, "spo2": 97, "temp_f": 103.2, "pain": 6,
                "arrival_transport": "Unknown",
            },
            "8 · MVA Poly-trauma (L1)": {
                "first_name": "David", "last_name": "Kim", "age": 44, "sex": "Male",
                "triage_notes": "High-speed motor vehicle collision. Severe chest and abdominal pain. Seatbelt sign present. GCS 14.",
                "heart_rate": 125, "resp_rate": 20, "sbp": 94, "dbp": 64, "spo2": 96, "temp_f": 98.0, "pain": 9,
                "arrival_transport": "Ambulance",
            },
            "9 · Suicidal Ideation (L2)": {
                "first_name": "Emma", "last_name": "Reeves", "age": 30, "sex": "Female",
                "triage_notes": "Active suicidal thoughts with hopelessness and severe depression. No specific plan but expressing intent.",
                "heart_rate": 82, "resp_rate": 14, "sbp": 118, "dbp": 75, "spo2": 99, "temp_f": 98.6, "pain": 0,
                "arrival_transport": "Walk In",
            },
            "10 · Sparse Text Stress Test": {
                "first_name": "John", "last_name": "Doe", "age": 50, "sex": "Male",
                "triage_notes": "Pain.",
                "heart_rate": 104, "resp_rate": 19, "sbp": 135, "dbp": 88, "spo2": 96, "temp_f": 99.1, "pain": 5,
                "arrival_transport": "Walk In",
            },
        }

        tc_col, btn_col = st.columns([3, 1])
        with tc_col:
            selected = st.selectbox("🧪 Load test patient", ["— select —"] + list(TEST_PATIENTS.keys()),
                                    key="test_patient_select", label_visibility="collapsed")
        with btn_col:
            if st.button("Load", key="fill_test", disabled=(selected == "— select —")):
                p = TEST_PATIENTS[selected]
                for k, v in p.items():
                    st.session_state[k] = v
                del st.session_state["test_patient_select"]
                st.rerun()

        # ── Patient Info ─────────────────────────────────────────
        st.markdown('<div class="intake-section-title">Patient Information</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
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
        with c4:
            st.markdown('<div class="field-label">Sex</div>', unsafe_allow_html=True)
            st.selectbox("Sex", ["—", "Female", "Male", "Other"], key="sex", label_visibility="collapsed")

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
        p_col, t_col = st.columns([1, 3])
        with p_col:
            st.markdown('<div class="field-label">Pain (0–10)</div>', unsafe_allow_html=True)
            st.number_input("Pain", min_value=0, max_value=10, value=None, key="pain",
                            placeholder="0–10", label_visibility="collapsed")
        with t_col:
            st.markdown('<div class="field-label">Arrival Transport</div>', unsafe_allow_html=True)
            st.selectbox("Transport", TRANSPORT_OPTIONS, key="arrival_transport", label_visibility="collapsed")

    # ── Submit button (hidden while loading via CSS overlay) ─────
    if not is_loading:
        _l, center, _r = st.columns([2, 3, 2])
        with center:
            clicked = st.button(
                "Run Triage Assessment",
                type="primary",
                use_container_width=True,
                key="submit_btn",
            )
    else:
        clicked = False

    overlay_slot = st.empty()

    if clicked and not is_loading:
        _sex_raw = st.session_state.get("sex", "—")
        form_data = {
            "first_name": st.session_state.get("first_name", ""),
            "last_name": st.session_state.get("last_name", ""),
            "age": st.session_state.get("age"),
            "sex": _sex_raw if _sex_raw != "—" else None,
            "triage_notes": st.session_state.get("triage_notes", ""),
            "heart_rate": st.session_state.get("heart_rate"),
            "resp_rate": st.session_state.get("resp_rate"),
            "sbp": st.session_state.get("sbp"),
            "dbp": st.session_state.get("dbp"),
            "spo2": st.session_state.get("spo2"),
            "temp_f": st.session_state.get("temp_f"),
            "pain": st.session_state.get("pain"),
            "arrival_transport": st.session_state.get("arrival_transport", "Walk In"),
        }
        st.session_state.form_data = form_data
        st.session_state.is_loading = True
        st.rerun()

    if is_loading:
        _backend_keys = {"triage_notes", "age", "sex", "heart_rate", "resp_rate",
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
            ("🔬", "Running ML model inference"),
            ("🔍", "Retrieving similar historical cases"),
            ("🧠", "LLM clinical analysis"),
            ("📋", "Synthesizing triage recommendation"),
        ]

        def _render_overlay(current_idx):
            rows = ""
            for i, (icon, msg) in enumerate(steps):
                if i < current_idx:
                    rows += f'<div class="overlay-step step-done"><span class="overlay-step-icon">✓</span><span class="overlay-step-text">{msg}</span></div>'
                elif i == current_idx:
                    rows += f'<div class="overlay-step step-active"><span class="overlay-step-icon">{icon}</span><span class="overlay-step-text">{msg}…</span></div>'
                else:
                    rows += f'<div class="overlay-step step-pending"><span class="overlay-step-icon">{icon}</span><span class="overlay-step-text">{msg}</span></div>'
            overlay_slot.markdown(f"""
            <div class="triage-overlay">
                <div class="triage-overlay-card">
                    <div class="overlay-title">Analyzing Patient</div>
                    <div class="overlay-subtitle">Running triage assessment pipeline…</div>
                    {rows}
                </div>
            </div>
            """, unsafe_allow_html=True)

        step_idx = 0
        while thread.is_alive():
            _render_overlay(min(step_idx, len(steps) - 1))
            _time.sleep(4)
            step_idx += 1

        thread.join()
        overlay_slot.empty()
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
    st.markdown("""
    <div class="top-header">
        <div class="top-header-left">
            <span class="brand-name">TriagePulse</span>
            <div class="header-divider"></div>
            <span class="page-title">Triage Assessment</span>
        </div>
        <div class="top-header-right">
            <span class="header-icon">&#128276;</span>
            <span class="header-icon">&#9881;</span>
            <div class="avatar">&#128100;</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pull real data from session state — no mocks
    triage_result = st.session_state.get("triage_result", {})
    form_data = st.session_state.get("form_data", {})

    if not triage_result:
        st.error("No triage result found. Please complete the intake form first.")
        if st.button("Go to Intake", key="go_intake_btn"):
            st.session_state.page = "intake"
            st.rerun()
        return

    predicted_label = triage_result.get("predicted_label", "Unknown")
    model_used      = triage_result.get("model_used", "arch4")
    safety_flag     = triage_result.get("safety_flag", False)
    safety_reason   = triage_result.get("safety_reason", None)
    top_features    = triage_result.get("top_features", [])

    reconciled_label   = triage_result.get("reconciled_label") or predicted_label
    llm_agreement      = triage_result.get("llm_agreement")
    clinical_rationale = triage_result.get("clinical_rationale")
    similar_cases      = triage_result.get("similar_cases") or []
    was_upgraded       = llm_agreement is False and reconciled_label != predicted_label

    # Nurse-friendly label/colour mapping
    LEVEL_META = {
        "L1-Critical": {
            "color": "#b91c1c", "bg": "#fef2f2", "border": "#fca5a5",
            "badge_bg": "#fee2e2", "badge_color": "#991b1b",
            "title": "ESI 1 — Immediate / Critical",
            "action": "Bring to resuscitation bay immediately. Do not leave unattended.",
            "icon": "🚨",
        },
        "L2-Emergent": {
            "color": "#c2410c", "bg": "#fff7ed", "border": "#fdba74",
            "badge_bg": "#ffedd5", "badge_color": "#9a3412",
            "title": "ESI 2 — Emergent",
            "action": "Assign to monitored bed within 15 minutes. Notify attending.",
            "icon": "⚠️",
        },
        "L3-Urgent": {
            "color": "#a16207", "bg": "#fefce8", "border": "#fde047",
            "badge_bg": "#fef9c3", "badge_color": "#854d0e",
            "title": "ESI 3 — Urgent / Less Urgent",
            "action": "Place in waiting area. Reassess vitals every 30 minutes.",
            "icon": "🔔",
        },
    }
    # Normalise key
    rec_key = reconciled_label.replace("/LessUrgent", "").replace("LessUrgent", "").strip()
    meta = LEVEL_META.get(rec_key, LEVEL_META["L3-Urgent"])

    # ── Patient header + new patient button ──────────────────────
    first_name   = form_data.get("first_name", "").strip()
    last_name    = form_data.get("last_name", "").strip()
    patient_name = f"{first_name} {last_name}".strip() or "Patient"
    age_val      = form_data.get("age")
    notes_preview = form_data.get("triage_notes", "")[:120]
    age_str      = f", {age_val}y" if age_val else ""

    # Build vitals strip for header
    VITAL_DEFS = [
        ("HR",   "heart_rate", "bpm",  lambda v: v > 100 or v < 50),
        ("RR",   "resp_rate",  "br/m", lambda v: v > 20 or v < 10),
        ("SBP",  "sbp",        "mmHg", lambda v: v < 90 or v > 160),
        ("SpO₂", "spo2",       "%",    lambda v: v < 95),
        ("Temp", "temp_f",     "°F",   lambda v: v > 100.4 or v < 96),
        ("Pain", "pain",       "/10",  lambda v: v >= 7),
    ]
    vitals_html = ""
    for label, key, unit, is_abnormal in VITAL_DEFS:
        val = form_data.get(key)
        if val is not None:
            abnormal = is_abnormal(val)
            color  = "#b91c1c" if abnormal else "#475569"
            weight = "700" if abnormal else "500"
            vitals_html += (
                f'<span style="margin-right:16px;font-size:12px;color:{color};font-weight:{weight};">'
                f'<span style="font-size:10px;color:#94a3b8;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-right:3px;">{label}</span>{val} {unit}</span>'
            )

    hdr_col, btn_col = st.columns([10, 2])
    with hdr_col:
        st.write(f"👤 **{patient_name}{age_str}**")
        st.caption(notes_preview + ("..." if len(form_data.get("triage_notes", "")) > 120 else ""))
        if vitals_html:
            st.markdown(f'<div style="margin-top:4px;">{vitals_html}</div>', unsafe_allow_html=True)
            st.empty()
    with btn_col:
        if st.button("Triage Next Patient", type="primary", use_container_width=True, key="new_patient_btn"):
            for key in ["triage_notes", "age", "heart_rate", "resp_rate",
                        "sbp", "dbp", "spo2", "temp_f", "pain",
                        "arrival_transport", "form_data", "triage_result", "is_loading"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.page = "intake"
            st.rerun()

    # ── Triage decision card ──────────────────────────────────────
    upgraded_html = f"""<div style="margin-top:10px;font-size:13px;color:{meta['color']};
        background:{meta['badge_bg']};border-radius:8px;padding:10px 14px;">
        ↑ <strong>Upgraded from initial assessment</strong> — clinical review flagged higher urgency
    </div>""" if was_upgraded else ""

    safety_html = f"""<div style="margin-top:10px;font-size:13px;color:#7c2d12;
        background:#fff7ed;border:1px solid #fdba74;border-radius:8px;padding:10px 14px;">
        ⚠️ <strong>Safety Alert:</strong> {safety_reason}
    </div>""" if (safety_flag and safety_reason) else ""

    # ── Build key factor chips ────────────────────────────────────
    chips_html = ""
    if top_features:
        for f in top_features:
            name = FEATURE_DISPLAY_NAMES.get(f.get("feature", ""), f.get("feature", "").replace("_", " ").title())
            is_against = "away" in f.get("direction", "")
            chip_bg     = "#fee2e2" if is_against else "#dcfce7"
            chip_color  = "#b91c1c" if is_against else "#15803d"
            chip_border = "#fca5a5" if is_against else "#86efac"
            arrow       = "↑" if is_against else "↓"
            chips_html += f"""<span style="display:inline-flex;align-items:center;gap:4px;
                background:{chip_bg};color:{chip_color};border:1px solid {chip_border};
                border-radius:20px;padding:4px 12px;font-size:12px;font-weight:700;
                margin:3px 4px 3px 0;white-space:nowrap;">
                {arrow} {name}
            </span>"""
        chips_html = f'<div style="display:flex;flex-wrap:wrap;margin-top:12px;">{chips_html}</div>'

    st.markdown(f"""
    <div style="background:{meta['bg']};border:1px solid {meta['border']};
                border-left:6px solid {meta['color']};border-radius:12px;
                padding:24px 28px;margin:16px 0;">
        <div style="display:flex;align-items:flex-start;gap:16px;">
            <div style="font-size:36px;line-height:1;">{meta['icon']}</div>
            <div style="flex:1;">
                <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                            letter-spacing:1.5px;color:{meta['color']};margin-bottom:4px;">
                    Triage Priority
                </div>
                <div style="font-size:28px;font-weight:900;color:{meta['color']};
                            letter-spacing:-0.5px;line-height:1.1;margin-bottom:8px;">
                    {meta['title']}
                </div>
                <div style="font-size:14px;font-weight:600;color:{meta['color']};">
                    {meta['action']}
                </div>
                {chips_html}
            </div>
        </div>
    </div>
    {upgraded_html}
    {safety_html}
    <div id="results-anchor"></div>
    """, unsafe_allow_html=True)
    st.empty()

    # ── Clinical reasoning + Similar cases side by side ───────────
    st.divider()
    left_col, right_col = st.columns(2)

    with left_col:
        if clinical_rationale:
            st.subheader("Clinical Reasoning")
            st.write(clinical_rationale)

    with right_col:
        st.subheader("Similar Past Cases")
        if not similar_cases:
            st.caption("No similar cases retrieved.")
        else:
            URGENCY_LABEL = {1: "Critical", 2: "Emergent", 3: "Urgent"}
            URGENCY_COLOR = {1: "#b91c1c", 2: "#c2410c", 3: "#a16207"}
            URGENCY_BG    = {1: "#fee2e2", 2: "#ffedd5", 3: "#fef9c3"}
            rows = ""
            for c in similar_cases:
                esi           = c.get("triage_level")
                urgency_label = URGENCY_LABEL.get(esi, "Unknown")
                urg_color     = URGENCY_COLOR.get(esi, "#64748b")
                urg_bg        = URGENCY_BG.get(esi, "#f1f5f9")
                outcome       = (c.get("outcome") or "—").title()
                complaint     = (c.get("chief_complaint") or "—").title()
                diagnosis     = (c.get("diagnosis") or "—").title()
                similarity    = c.get("similarity")
                sim_str       = f" · {int(similarity * 100)}% match" if similarity is not None else ""
                vitals_parts  = []
                if c.get("heart_rate"): vitals_parts.append(f"HR {c['heart_rate']}")
                if c.get("sbp"):        vitals_parts.append(f"SBP {c['sbp']}")
                if c.get("spo2"):       vitals_parts.append(f"SpO₂ {c['spo2']}%")
                vitals_str = "  ·  ".join(vitals_parts)
                patient_info = (c.get("patient_info") or "").replace("Gender: ", "").replace("Race: ", "").replace("Age: ", "")
                rows += f"""<tr style="border-bottom:1px solid #f1f5f9;">
                    <td style="padding:8px 4px;">
                        <div style="font-weight:600;font-size:13px;color:#0f172a;">{complaint}</div>
                        <div style="font-size:11px;color:#475569;margin-top:2px;">{patient_info}</div>
                        <div style="font-size:11px;color:#64748b;margin-top:2px;">{diagnosis}{sim_str}</div>
                        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{vitals_str}</div>
                    </td>
                    <td style="padding:8px 4px;white-space:nowrap;">
                        <span style="background:{urg_bg};color:{urg_color};font-size:11px;
                                     font-weight:700;padding:2px 8px;border-radius:4px;">ESI {esi} — {urgency_label}</span>
                        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">{outcome}</div>
                    </td>
                </tr>"""
            st.markdown(f'<table style="width:100%;border-collapse:collapse;">{rows}</table>', unsafe_allow_html=True)
            st.empty()

    # ── Technical info card ───────────────────────────────────────────────────
    LLM_ESI_LABEL_MAP = {
        1: "ESI 1 — Critical",
        2: "ESI 2 — Emergent",
        3: "ESI 3 — Urgent/Less Urgent",
    }

    model_label    = triage_result.get("predicted_label", "—")
    model_conf     = triage_result.get("confidence_pct")
    model_conf_str = f"{model_conf}% confidence" if model_conf is not None else ""
    llm_esi        = triage_result.get("llm_esi")
    llm_label      = LLM_ESI_LABEL_MAP.get(llm_esi, "—")
    reconciled     = triage_result.get("reconciled_label") or "—"
    llm_agreement  = triage_result.get("llm_agreement")
    agreement_str  = "Agreement" if llm_agreement else "Override"

    st.divider()
    st.subheader("Technical Details")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.caption("arch4 Model")
        st.write(f"**{model_label}**")
        if model_conf_str:
            st.caption(model_conf_str)
    with t2:
        st.caption("LLM Independent")
        st.write(f"**{llm_label}**")
        st.caption(agreement_str)
    with t3:
        st.caption("Reconciled Result")
        st.write(f"**{reconciled}**")


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
