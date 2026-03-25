"""
Preprocessing script for SageMaker ProcessingStep.

Reads consolidated_dataset_features.csv, selects base columns, builds
triage_text (CC_2x + HPI), creates 3-class target, and writes stratified
train/val/test splits.

SageMaker contract:
  Input:  /opt/ml/processing/input/features/consolidated_dataset_features.csv
  Output: /opt/ml/processing/output/train/train.csv
          /opt/ml/processing/output/validation/val.csv
          /opt/ml/processing/output/test/test.csv
"""

import os
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths (env-var overrides for local/JupyterLab execution) ──────────────────
INPUT_DIR    = os.environ.get("PREPROCESS_INPUT_DIR",  "/opt/ml/processing/input/features")
OUTPUT_TRAIN = os.environ.get("PREPROCESS_OUTPUT_TRAIN", "/opt/ml/processing/output/train")
OUTPUT_VAL   = os.environ.get("PREPROCESS_OUTPUT_VAL",   "/opt/ml/processing/output/validation")
OUTPUT_TEST  = os.environ.get("PREPROCESS_OUTPUT_TEST",  "/opt/ml/processing/output/test")

SEED = 42

# Columns carried forward — raw inputs to transform_structured() in training
BASE_COLUMNS = [
    "stay_id", "triage",
    "chiefcomplaint", "HPI", "arrival_transport",
    "pain", "pain_missing", "age",
    "temp_f", "heart_rate", "resp_rate", "spo2", "sbp", "dbp",
]


# ── Text construction (matches arch4_training_v1.ipynb) ───────────────────────

def clip_words(text, max_words):
    text = "" if pd.isna(text) else str(text).replace("\n", " ").strip()
    return " ".join(text.split()[:max_words]) if text else ""


def build_triage_text(row):
    """CC-emphasized text: CC repeated twice + HPI. PMH dropped."""
    cc  = clip_words(row["chiefcomplaint"], 24)
    hpi = clip_words(row["HPI"], 160)
    parts = []
    if cc:  parts.append(f"Chief complaint: {cc}.")
    if cc:  parts.append(f"Presenting with {cc}.")  # CC emphasis
    if hpi: parts.append(f"History: {hpi}.")
    return " ".join(parts)


# ── 3-class target (matches arch4_training_v1.ipynb) ──────────────────────────

TARGET_MAP = {1: 0, 2: 1, 3: 2, 4: 2}  # L4 merged into L3


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")
    input_path = csv_files[0]
    print(f"Reading {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded: {df.shape}")

    # ── Select & build features ───────────────────────────────────────────────
    df = df[BASE_COLUMNS].copy()
    df["triage_text"]   = df.apply(build_triage_text, axis=1)
    df["triage_3class"] = df["triage"].map(TARGET_MAP).astype(int)

    print(f"\n3-class distribution:")
    print(df["triage_3class"].value_counts().sort_index().to_string())

    # ── Columns to write (includes triage_text and triage_3class) ─────────────
    output_cols = BASE_COLUMNS + ["triage_text", "triage_3class"]

    X = df[output_cols]
    y = df["triage_3class"].values

    # ── Stratified 80/10/10 split ─────────────────────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED,
    )

    total = len(df)
    print(f"\nTrain: {len(X_train)} ({len(X_train)/total:.1%})")
    print(f"Val:   {len(X_val)}  ({len(X_val)/total:.1%})")
    print(f"Test:  {len(X_test)}  ({len(X_test)/total:.1%})")

    # ── Write splits ──────────────────────────────────────────────────────────
    for split_dir in [OUTPUT_TRAIN, OUTPUT_VAL, OUTPUT_TEST]:
        os.makedirs(split_dir, exist_ok=True)

    X_train.to_csv(os.path.join(OUTPUT_TRAIN, "train.csv"), index=False)
    X_val.to_csv(os.path.join(OUTPUT_VAL,     "val.csv"),   index=False)
    X_test.to_csv(os.path.join(OUTPUT_TEST,   "test.csv"),  index=False)

    print(f"\nWrote train.csv  → {OUTPUT_TRAIN}")
    print(f"Wrote val.csv    → {OUTPUT_VAL}")
    print(f"Wrote test.csv   → {OUTPUT_TEST}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
