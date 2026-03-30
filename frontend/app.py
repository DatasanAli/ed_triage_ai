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
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] {
        background: var(--surface-container-lowest) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        color: var(--on-surface) !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stTextArea"] textarea {
        background: var(--surface-container-low) !important;
        font-weight: 400 !important;
        font-size: 16px !important;
        line-height: 1.7 !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        box-shadow: 0 0 0 2px var(--primary-fixed) !important;
    }
    [data-testid="stNumberInput"] input:focus {
        box-shadow: 0 0 0 2px var(--primary-fixed) !important;
    }

    /* hide default streamlit labels — we use custom HTML labels */
    [data-testid="stNumberInput"] label,
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

    /* ── Streamlit number input hide steppers for cleaner look ── */
    [data-testid="stNumberInput"] button {
        display: none !important;
    }

    /* Make text area label hidden since we use custom */
    [data-testid="stTextArea"] label { display: none !important; }
    </style>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        active_page = st.session_state.get("page", "intake")
        intake_cls = "active" if active_page == "intake" else ""
        results_cls = "active" if active_page == "results" else ""

        st.markdown(f"""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">&#10010;</div>
            <div>
                <h1>ED Central</h1>
                <p>Station 4</p>
            </div>
        </div>
        <div class="nav-item {intake_cls}">
            <span class="nav-icon">&#128100;</span> Patient Intake
        </div>
        <div class="nav-item {results_cls}">
            <span class="nav-icon">&#128336;</span> Recent Triage
        </div>
        """, unsafe_allow_html=True)

        # ── Recent Triage history ──────────────────────────────
        history = st.session_state.get("triage_history", [])
        if history:
            for entry in history[:5]:  # show last 5
                result = entry.get("result", {})
                label = result.get("predicted_label", "Unknown")
                notes_preview = entry.get("triage_notes", "")[:60]
                timestamp = entry.get("timestamp", "")
                # pick label colour class
                if "L1" in label:
                    lbl_cls = ""
                elif "L2" in label:
                    lbl_cls = "l2"
                else:
                    lbl_cls = "l3"
                st.markdown(f"""
                <div class="history-item">
                    <div class="history-label {lbl_cls}">{label}</div>
                    <div class="history-notes">{notes_preview}{"…" if len(entry.get("triage_notes","")) > 60 else ""}</div>
                    <div class="history-time">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding: 12px 24px; font-size: 12px; color: #94a3b8;">
                No triage history yet.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="nav-item">
            <span class="nav-icon">&#128202;</span> Analytics
        </div>
        """, unsafe_allow_html=True)

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
    # Top header bar
    st.markdown("""
    <div class="top-header">
        <div class="top-header-left">
            <span class="brand-name">TriagePulse</span>
            <div class="header-divider"></div>
            <span class="page-title">New Patient Entry</span>
        </div>
        <div class="top-header-right">
            <span class="search-box">&#128269; &nbsp;Search Patients...</span>
            <span class="header-icon">&#128276;</span>
            <span class="header-icon">&#9881;</span>
            <div class="avatar">&#128100;</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Two-column layout
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        # Primary Documentation card
        st.markdown("""
        <div class="card" style="min-height: 580px; display: flex; flex-direction: column;">
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px;">
                <div>
                    <div class="section-label primary">Primary Documentation</div>
                    <div class="section-title" style="margin-bottom: 0;">Triage Notes</div>
                </div>
                <div class="badge-row">
                    <span class="badge">Voice Enabled</span>
                    <span class="badge">Autosave Active</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Overlay the text area on top of the card using negative margin
        st.markdown("<div style='margin-top:-520px; padding: 0 32px 32px;'>",
                    unsafe_allow_html=True)
        st.text_area(
            "Triage Notes",
            height=450,
            key="triage_notes",
            placeholder="Enter chief complaint, detailed symptoms, and medical history here...",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        # Vitals panel
        st.markdown("""
        <div class="section-label secondary">Diagnostic Layer</div>
        <div class="section-title">Optional Vitals</div>
        """, unsafe_allow_html=True)

        # Age — full width
        st.markdown('<div class="vital-label">Age</div>', unsafe_allow_html=True)
        st.number_input("Age", min_value=0, max_value=120, value=None,
                        key="age", placeholder="Years", label_visibility="collapsed")

        # Heart Rate | RR
        v1, v2 = st.columns(2)
        with v1:
            st.markdown('<div class="vital-label">Heart Rate (BPM)</div>',
                        unsafe_allow_html=True)
            st.number_input("Heart Rate", min_value=20, max_value=250,
                            value=None, key="heart_rate", placeholder="72",
                            label_visibility="collapsed")
        with v2:
            st.markdown('<div class="vital-label">RR (Breaths/m)</div>',
                        unsafe_allow_html=True)
            st.number_input("RR", min_value=4, max_value=60, value=None,
                            key="resp_rate", placeholder="16",
                            label_visibility="collapsed")

        # SBP | DBP
        v3, v4 = st.columns(2)
        with v3:
            st.markdown('<div class="vital-label">SBP (mmHg)</div>',
                        unsafe_allow_html=True)
            st.number_input("SBP", min_value=40, max_value=300, value=None,
                            key="sbp", placeholder="120",
                            label_visibility="collapsed")
        with v4:
            st.markdown('<div class="vital-label">DBP (mmHg)</div>',
                        unsafe_allow_html=True)
            st.number_input("DBP", min_value=10, max_value=200, value=None,
                            key="dbp", placeholder="80",
                            label_visibility="collapsed")

        # SpO2 | Temp
        v5, v6 = st.columns(2)
        with v5:
            st.markdown('<div class="vital-label">SpO2 (%)</div>',
                        unsafe_allow_html=True)
            st.number_input("SpO2", min_value=50, max_value=100, value=None,
                            key="spo2", placeholder="98",
                            label_visibility="collapsed")
        with v6:
            st.markdown('<div class="vital-label">Temp (F)</div>',
                        unsafe_allow_html=True)
            st.number_input("Temp", min_value=85.0, max_value=115.0,
                            value=None, key="temp_f", placeholder="98.6",
                            step=0.1, label_visibility="collapsed")

        # Pain slider
        pain_val = st.session_state.get("pain", 4)
        st.markdown(f"""
        <div class="pain-header">
            <span class="pain-label">Pain Level (0-10)</span>
            <span class="pain-value">{pain_val} / 10</span>
        </div>
        """, unsafe_allow_html=True)
        st.slider("Pain", min_value=0, max_value=10, value=4,
                  key="pain", label_visibility="collapsed")
        st.markdown("""
        <div class="pain-range">
            <span>None</span>
            <span>Severe</span>
        </div>
        """, unsafe_allow_html=True)

        # Arrival transport
        st.markdown('<div class="vital-label" style="margin-top:12px;">Arrival Transport</div>',
                    unsafe_allow_html=True)
        st.selectbox("Transport", TRANSPORT_OPTIONS, key="arrival_transport",
                     label_visibility="collapsed")

        # System monitoring indicator
        st.markdown("""
        <div class="monitor-card">
            <div class="pulse-ring">
                <span class="pulse-icon">&#9764;</span>
            </div>
            <div>
                <div class="monitor-label">System Monitoring</div>
                <div class="monitor-status">Ready for diagnostic stream</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tip card
        st.markdown("""
        <div class="tip-card">
            <div class="tip-icon-box">&#128161;</div>
            <div>
                <div class="tip-title">Did you know?</div>
                <div class="tip-text">Type 'HX' to quickly import patient's known clinical history.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Submit button — centered
    _l, center, _r = st.columns([1, 2, 1])
    with center:
        if st.button("&#10004;  Complete Triage Assessment", type="primary",
                     use_container_width=True, key="submit_btn"):
            form_data = {
                "triage_notes": st.session_state.get("triage_notes", ""),
                "age": st.session_state.get("age"),
                "heart_rate": st.session_state.get("heart_rate"),
                "resp_rate": st.session_state.get("resp_rate"),
                "sbp": st.session_state.get("sbp"),
                "dbp": st.session_state.get("dbp"),
                "spo2": st.session_state.get("spo2"),
                "temp_f": st.session_state.get("temp_f"),
                "pain": st.session_state.get("pain", 4),
                "arrival_transport": st.session_state.get(
                    "arrival_transport", "Walk In"),
            }
            st.session_state.form_data = form_data

            # POST to /predict
            with st.spinner("Analyzing triage data..."):
                try:
                    # Filter out None values before sending
                    payload = {k: v for k, v in form_data.items() if v is not None}
                    import json as _json
                    print(f"[TriagePulse] → POST {BACKEND_URL}/predict")
                    print(f"[TriagePulse]   payload: {_json.dumps(payload, indent=2)}")
                    resp = requests.post(
                        f"{BACKEND_URL}/predict",
                        json=payload,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    triage_result = resp.json()
                    print(f"[TriagePulse] ← {resp.status_code} response:")
                    print(f"[TriagePulse]   {_json.dumps(triage_result, indent=2)}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Cannot reach the backend at {BACKEND_URL}. "
                        "Make sure `uvicorn backend.main:app --reload --port 8000` is running."
                    )
                    st.stop()
                except requests.exceptions.HTTPError as exc:
                    st.error(
                        f"Backend returned an error: "
                        f"{exc.response.status_code} — {exc.response.text}"
                    )
                    st.stop()
                except Exception as exc:
                    st.error(f"Unexpected error calling backend: {exc}")
                    st.stop()

            # Store result in session state
            st.session_state.triage_result = triage_result

            # Append to history (keep last 20)
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

    # Derive UI data from real backend response
    drivers = shap_features_to_drivers(top_features)
    recommendations = label_to_recommendations(predicted_label)

    # Patient vitals from submitted form_data
    hr_val = form_data.get("heart_rate")
    sbp_val = form_data.get("sbp")
    dbp_val = form_data.get("dbp")
    age_val = form_data.get("age")
    hr_display = str(hr_val) if hr_val is not None else "—"
    bp_display = (
        f"{sbp_val}/{dbp_val}"
        if sbp_val is not None and dbp_val is not None
        else "—"
    )
    age_display = f"{age_val}Y" if age_val is not None else ""

    # Parse label for display (e.g. "L1-Critical" → level_code="L1", level_desc="CRITICAL")
    label_parts = predicted_label.split("-", 1)
    level_code = label_parts[0] if label_parts else predicted_label
    level_desc = label_parts[1].upper() if len(label_parts) > 1 else ""

    # Breadcrumb + header
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px;">
        <div>
            <div class="breadcrumb">
                TRIAGE WORKFLOW &nbsp;&#8250;&nbsp; PATIENT ASSESSMENT &nbsp;&#8250;&nbsp;
                <span class="active">ANALYSIS RESULT</span>
            </div>
            <div class="results-title">Diagnostic Output</div>
        </div>
        <div class="assigned-to">
            <div class="avatar" style="width:36px;height:36px;">&#128105;&#8205;&#9877;</div>
            <div>
                <div class="assigned-label">Assigned To</div>
                <div class="assigned-name">Dr. Aris Thorne</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Safety flag banner
    if safety_flag and safety_reason:
        st.warning(f"\u26a0\ufe0f **Safety Alert:** {safety_reason}")

    left_col, right_col = st.columns([7, 5], gap="large")

    with left_col:
        # Priority card — derived from predicted_label
        st.markdown(f"""
        <div class="priority-card">
            <div class="priority-content">
                <div class="priority-badge">PREDICTED PRIORITY</div>
                <div class="priority-level">{level_code} -<br>{level_desc}</div>
                <div class="priority-desc">
                    Model: <strong>{model_used}</strong>. Based on submitted vitals and
                    triage notes, the system classified this patient as
                    <strong>{predicted_label}</strong>.
                </div>
            </div>
            <div class="target-zone">
                <div class="target-icon">&#9888;</div>
                <div class="target-label">Predicted Class</div>
                <div class="target-name">{predicted_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence + Drivers side by side
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
                "<p style='color:var(--on-surface-variant);font-size:13px;'>"
                "No SHAP feature data available.</p>"
            )
            st.markdown(
                f'<div class="drivers-section">'
                f'<div class="confidence-title">Clinical Drivers</div>'
                f'{drivers_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

    with right_col:
        # Patient mini-profile derived from submitted form_data
        st.markdown(f"""
        <div class="patient-card">
            <div class="patient-header">
                <div class="patient-avatar">&#128100;</div>
                <div>
                    <div class="patient-name">Current Patient</div>
                    <div class="patient-meta">{age_display} &bull; Submitted {datetime.now(timezone.utc).strftime("%H:%M UTC")}</div>
                </div>
            </div>
            <div class="vitals-grid">
                <div class="vital-display">
                    <div class="vital-display-label">Heart Rate</div>
                    <span class="vital-display-value">{hr_display}</span>
                    <span class="vital-display-unit">BPM</span>
                    <span class="vital-heart-icon">&#9829;</span>
                </div>
                <div class="vital-display">
                    <div class="vital-display-label">BP (Sys/Dia)</div>
                    <span class="vital-display-value">{bp_display}</span>
                    <span class="vital-display-unit">mmHg</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Protocol recommendations — derived from predicted_label
        rec_items = []
        for r in recommendations:
            rec_items.append(
                f'<div class="rec-item">'
                f'<div class="rec-check">\u2713</div>'
                f'<div class="rec-text">{r}</div>'
                f'</div>'
            )
        recs_html = "".join(rec_items)
        st.markdown(
            f'<div class="rec-section">'
            f'<div class="rec-title">Protocol Recommendations</div>'
            f'{recs_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Action buttons
        if st.button("Confirm and Record Triage", type="primary",
                     use_container_width=True, key="confirm_btn"):
            pass  # History already appended on submit

        if st.button("Triage Another Patient", use_container_width=True,
                     key="restart_btn"):
            # Clear form/result data but keep triage_history
            for key in ["triage_notes", "age", "heart_rate", "resp_rate",
                        "sbp", "dbp", "spo2", "temp_f", "pain",
                        "arrival_transport", "form_data", "triage_result"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.page = "intake"
            st.rerun()

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
    render_sidebar()

    if st.session_state.page == "intake":
        render_intake_page()
    else:
        render_results_page()


if __name__ == "__main__":
    main()
