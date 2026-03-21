"""
exp_utils.py — Shared utilities for arch4 text-encoder experiments.

Provides:
  - load_splits_from_s3()      : loads CSV splits + rebuilds LightGBM OOF probs
  - build_lgbm_baseline()      : 5-fold cross-fitting with identical params/seed as arch4_v1
  - FusionHead                 : lightweight MLP for embedding + LightGBM prob fusion
  - train_fusion_head()        : trains FusionHead on pre-extracted embeddings, returns reports
  - print_comparison_table()   : formats delta table vs LightGBM-only and BioClinicalBERT
"""

import io
import random
import time
import warnings

import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
SEED         = 42
S3_BUCKET    = "ed-triage-capstone-group7"
DATA_KEY     = "Data_Output/consolidated_dataset_features.csv"
SPLIT_PREFIX = "Data_Output/splits/arch4_v1/"
NUM_CLASSES  = 3
CLASS_NAMES  = ["L1-Critical", "L2-Emergent", "L3-Urgent/LessUrgent"]

# BioClinicalBERT hybrid baseline (arch4_v1 test set)
BIOCLINICALBERT_BASELINE = {
    "L1-Critical_f1":          0.57,
    "L2-Emergent_f1":          0.63,
    "L3-Urgent/LessUrgent_f1": 0.80,
    "accuracy":                0.72,
    "macro_f1":                0.67,
    "weighted_f1":             0.72,
}

RAW_VITALS  = ["temp_f", "heart_rate", "resp_rate", "spo2", "sbp", "dbp"]
CLIP_BOUNDS = {
    "temp_f":     (85.0, 115.0),
    "heart_rate": (20.0, 250.0),
    "resp_rate":  (4.0,  60.0),
    "spo2":       (50.0, 100.0),
    "sbp":        (40.0, 300.0),
    "dbp":        (10.0, 200.0),
}
TRANSPORT_MAP = {"WALK IN": 0, "UNKNOWN": 1, "AMBULANCE": 2, "HELICOPTER": 3}

STRUCTURED_FEATURES = [
    "heart_rate", "sbp", "dbp", "resp_rate", "spo2", "temp_f",
    "shock_index", "map", "pulse_pressure",
    "news2_score", "mews_score",
    "age", "transport_ordinal",
    "pain", "pain_missing",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# Text construction
# ═══════════════════════════════════════════════════════════════════════════════
def clip_words(text, max_words):
    text = "" if pd.isna(text) else str(text).replace("\n", " ").strip()
    return " ".join(text.split()[:max_words]) if text else ""


def build_triage_text(row):
    """CC-emphasized text: CC_2x + HPI (PMH dropped)."""
    cc  = clip_words(row["chiefcomplaint"], 24)
    hpi = clip_words(row["HPI"], 160)
    parts = []
    if cc:  parts.append(f"Chief complaint: {cc}.")
    if cc:  parts.append(f"Presenting with {cc}.")
    if hpi: parts.append(f"History: {hpi}.")
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Clinical early warning scores
# ═══════════════════════════════════════════════════════════════════════════════
def compute_news2(row):
    """NEWS2 aggregate score (Scale 1). Temp in deg-F. No AVPU — max 15."""
    score = 0
    rr = row["resp_rate"]
    if   rr <= 8:   score += 3
    elif rr <= 11:  score += 1
    elif rr <= 20:  score += 0
    elif rr <= 24:  score += 2
    else:           score += 3
    spo2 = row["spo2"]
    if   spo2 <= 91:  score += 3
    elif spo2 <= 93:  score += 2
    elif spo2 <= 95:  score += 1
    sbp = row["sbp"]
    if   sbp <= 90:   score += 3
    elif sbp <= 100:  score += 2
    elif sbp <= 110:  score += 1
    elif sbp <= 219:  score += 0
    else:             score += 3
    hr = row["heart_rate"]
    if   hr <= 40:   score += 3
    elif hr <= 50:   score += 1
    elif hr <= 90:   score += 0
    elif hr <= 110:  score += 1
    elif hr <= 130:  score += 2
    else:            score += 3
    temp = row["temp_f"]
    if   temp <= 95.0:    score += 3
    elif temp <= 96.8:    score += 1
    elif temp <= 100.4:   score += 0
    elif temp <= 102.2:   score += 1
    else:                 score += 2
    return score


def compute_mews(row):
    """MEWS aggregate score. No AVPU — max 12."""
    score = 0
    sbp = row["sbp"]
    if   sbp < 70:    score += 3
    elif sbp < 81:    score += 2
    elif sbp < 101:   score += 1
    elif sbp < 200:   score += 0
    else:             score += 2
    hr = row["heart_rate"]
    if   hr < 40:     score += 2
    elif hr < 51:     score += 1
    elif hr < 101:    score += 0
    elif hr < 111:    score += 1
    elif hr < 130:    score += 2
    else:             score += 3
    rr = row["resp_rate"]
    if   rr < 9:      score += 2
    elif rr < 15:     score += 0
    elif rr < 21:     score += 1
    elif rr < 30:     score += 2
    else:             score += 3
    temp = row["temp_f"]
    if   temp < 95.0:    score += 2
    elif temp <= 101.1:  score += 0
    else:                score += 2
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# Structured feature pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def fit_structured_stats(train_df):
    """Compute imputation medians from training rows only."""
    return {
        "vital_medians": {col: float(train_df[col].median()) for col in RAW_VITALS},
        "pain_median":   float(train_df["pain"].median()),
        "age_median":    float(train_df["age"].median()),
    }


def transform_structured(df_in, stats):
    """Build pruned 15 structured features using train-set imputation stats."""
    feat = df_in.copy()
    for col in RAW_VITALS:
        feat[col] = feat[col].fillna(stats["vital_medians"][col]).clip(*CLIP_BOUNDS[col])
    feat["pain"] = feat["pain"].fillna(stats["pain_median"]).clip(0.0, 10.0)
    feat["age"]  = feat["age"].fillna(stats["age_median"]).clip(18, 120)
    feat["pain_missing"]      = feat["pain_missing"].astype(int)
    feat["transport_ordinal"] = feat["arrival_transport"].map(TRANSPORT_MAP).fillna(1).astype(int)
    feat["shock_index"]    = (feat["heart_rate"] / feat["sbp"].replace(0, np.nan)).fillna(0.0)
    feat["map"]            = (feat["sbp"] + 2.0 * feat["dbp"]) / 3.0
    feat["pulse_pressure"] = feat["sbp"] - feat["dbp"]
    feat["news2_score"] = feat.apply(compute_news2, axis=1)
    feat["mews_score"]  = feat.apply(compute_mews, axis=1)
    return feat[STRUCTURED_FEATURES].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# S3 data loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_splits_from_s3():
    """
    Load train/val/test splits from S3, reconstruct triage_text if missing,
    compute structured features, return (frames, labels, structured arrays).

    Returns:
        dict with keys:
            X_train, X_val, X_test      — DataFrames (contain triage_text col)
            y_train, y_val, y_test      — numpy int arrays
            X_train_struct, X_val_struct, X_test_struct — float32 numpy arrays (15 features)
            structured_stats            — dict of imputation medians
    """
    s3 = boto3.client("s3", region_name="us-east-1")

    def _load(key):
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    print("Loading splits from S3...")
    X_train = _load(f"{SPLIT_PREFIX}train.csv")
    X_val   = _load(f"{SPLIT_PREFIX}val.csv")
    X_test  = _load(f"{SPLIT_PREFIX}test.csv")

    # Reconstruct triage_text if not present (safety fallback)
    for frame in [X_train, X_val, X_test]:
        if "triage_text" not in frame.columns:
            frame["triage_text"] = frame.apply(build_triage_text, axis=1)

    y_train = X_train["triage_3class"].values.astype(int)
    y_val   = X_val["triage_3class"].values.astype(int)
    y_test  = X_test["triage_3class"].values.astype(int)

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Structured features (fitted on train only)
    stats        = fit_structured_stats(X_train)
    X_train_st   = transform_structured(X_train, stats).values.astype(np.float32)
    X_val_st     = transform_structured(X_val,   stats).values.astype(np.float32)
    X_test_st    = transform_structured(X_test,  stats).values.astype(np.float32)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_struct": X_train_st, "X_val_struct": X_val_st, "X_test_struct": X_test_st,
        "structured_stats": stats,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LightGBM 5-fold cross-fitting (identical to arch4_v1)
# ═══════════════════════════════════════════════════════════════════════════════
def build_lgbm_baseline(X_train_np, y_train, X_val_np, y_val, X_test_np, y_test):
    """
    5-fold cross-fit LightGBM with identical params/seed as arch4_v1.

    Returns:
        lgbm_probs_train  — OOF probs (n_train, 3)
        lgbm_probs_val    — avg of 5-fold val probs
        lgbm_probs_test   — avg of 5-fold test probs
        lgbm_test_report  — classification_report dict for LightGBM-only on test
        lgbm_fold_models  — list of 5 trained LGBMClassifier
    """
    print("\nBuilding LightGBM baseline (5-fold cross-fitting)...")
    train_counts = {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))}
    sqrt_w = np.sqrt(np.array([len(y_train) / (3 * train_counts[c]) for c in range(NUM_CLASSES)]))
    sw_map = {c: float(sqrt_w[c]) for c in range(NUM_CLASSES)}

    lgbm_params = dict(
        objective="multiclass", num_class=NUM_CLASSES,
        n_estimators=500, max_depth=-1, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.5, reg_lambda=2.0,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )

    oof_probs        = np.zeros((len(X_train_np), NUM_CLASSES), dtype=np.float32)
    val_probs_folds  = []
    test_probs_folds = []
    fold_models      = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_idx, hold_idx) in enumerate(skf.split(X_train_np, y_train), start=1):
        X_fit,  y_fit  = X_train_np[fit_idx],  y_train[fit_idx]
        X_hold, y_hold = X_train_np[hold_idx], y_train[hold_idx]
        sw_fit = np.array([sw_map[int(l)] for l in y_fit], dtype=np.float32)

        model = LGBMClassifier(**lgbm_params)
        model.fit(
            X_fit, y_fit, sample_weight=sw_fit,
            eval_set=[(X_hold, y_hold)],
            callbacks=[early_stopping(30, verbose=False)],
        )

        oof_probs[hold_idx] = model.predict_proba(X_hold)
        val_probs_folds.append(model.predict_proba(X_val_np))
        test_probs_folds.append(model.predict_proba(X_test_np))
        fold_models.append(model)

        hold_f1 = f1_score(y_hold, oof_probs[hold_idx].argmax(1), average="macro")
        print(f"  Fold {fold} | best_iter={model.best_iteration_} | holdout macro-F1={hold_f1:.4f}")

    lgbm_probs_train = oof_probs
    lgbm_probs_val   = np.mean(np.stack(val_probs_folds),  axis=0)
    lgbm_probs_test  = np.mean(np.stack(test_probs_folds), axis=0)

    lgbm_preds_test = lgbm_probs_test.argmax(axis=1)
    lgbm_test_report = classification_report(
        y_test, lgbm_preds_test,
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    lgbm_test_f1 = f1_score(y_test, lgbm_preds_test, average="macro")

    print(f"\nLightGBM-only test macro-F1: {lgbm_test_f1:.4f}")
    print(classification_report(y_test, lgbm_preds_test, target_names=CLASS_NAMES, zero_division=0))

    return lgbm_probs_train, lgbm_probs_val, lgbm_probs_test, lgbm_test_report, fold_models


# ═══════════════════════════════════════════════════════════════════════════════
# Class weights (same sqrt-dampened scheme as arch4_v1)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_class_weights(y_train, device=None):
    """Sqrt-dampened inverse frequency class weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts = {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))}
    raw = torch.tensor(
        [len(y_train) / (NUM_CLASSES * counts[c]) for c in range(NUM_CLASSES)],
        dtype=torch.float32,
    )
    return torch.sqrt(raw).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# FusionHead — lightweight MLP for pre-extracted embeddings + LightGBM probs
# ═══════════════════════════════════════════════════════════════════════════════
class EmbeddingFusionDataset(Dataset):
    """Dataset for pre-extracted embeddings + LightGBM probs."""

    def __init__(self, embeddings, lgbm_probs, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.lgbm_probs = torch.tensor(lgbm_probs, dtype=torch.float32)
        self.labels     = torch.tensor(labels,      dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.lgbm_probs[idx], self.labels[idx]


class FusionHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear (same arch as arch4_v1 head)."""

    def __init__(self, embedding_dim, tree_dim=3, num_classes=3,
                 hidden_dim=96, dropout=0.55):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim + tree_dim),
            nn.Linear(embedding_dim + tree_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings, tree_probs):
        fused = torch.cat([embeddings, tree_probs], dim=1)
        return self.head(fused)


def train_fusion_head(
    train_embs, train_lgbm, y_train,
    val_embs,   val_lgbm,   y_val,
    test_embs,  test_lgbm,  y_test,
    embedding_dim,
    hidden_dim=96,
    dropout=0.55,
    lr=1e-3,
    weight_decay=0.01,
    epochs=60,
    patience=8,
    batch_size=128,
    device=None,
):
    """
    Train FusionHead on pre-extracted embeddings + LightGBM probs.

    Returns:
        test_report  — classification_report dict on test set
        val_report   — classification_report dict on val set
        best_val_f1  — best validation macro-F1
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)

    train_ds = EmbeddingFusionDataset(train_embs, train_lgbm, y_train)
    val_ds   = EmbeddingFusionDataset(val_embs,   val_lgbm,   y_val)
    test_ds  = EmbeddingFusionDataset(test_embs,  test_lgbm,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model     = FusionHead(embedding_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    class_wts = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_wts)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_f1    = -1.0
    bad_epochs     = 0
    best_state     = None

    print(f"\nTraining FusionHead (emb_dim={embedding_dim}, hidden={hidden_dim}, dropout={dropout})")
    print(f"  lr={lr}, epochs={epochs}, patience={patience}, batch_size={batch_size}")

    for epoch in range(1, epochs + 1):
        # — Train —
        model.train()
        for emb, lgbm, lab in train_loader:
            emb, lgbm, lab = emb.to(device), lgbm.to(device), lab.to(device)
            logits = model(emb, lgbm)
            loss   = criterion(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # — Validate —
        model.eval()
        val_preds = []
        with torch.no_grad():
            for emb, lgbm, lab in val_loader:
                emb, lgbm = emb.to(device), lgbm.to(device)
                logits = model(emb, lgbm)
                val_preds.extend(logits.argmax(1).cpu().numpy())
        val_f1 = f1_score(y_val, val_preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            bad_epochs  = 0
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == 1 or bad_epochs == 0:
            tag = "<- best" if bad_epochs == 0 else f"patience {bad_epochs}/{patience}"
            print(f"  Epoch {epoch:02d} | val_f1={val_f1:.4f} | {tag}")

        if bad_epochs >= patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    # — Evaluate with best model —
    model.load_state_dict(best_state)
    model.eval()

    def _predict(loader):
        preds = []
        with torch.no_grad():
            for emb, lgbm, _ in loader:
                emb, lgbm = emb.to(device), lgbm.to(device)
                logits = model(emb, lgbm)
                preds.extend(logits.argmax(1).cpu().numpy())
        return np.array(preds)

    val_preds  = _predict(val_loader)
    test_preds = _predict(test_loader)

    val_report = classification_report(
        y_val, val_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    test_report = classification_report(
        y_test, test_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )

    print(f"\nBest val macro-F1: {best_val_f1:.4f}")
    print(f"Test macro-F1:     {test_report['macro avg']['f1-score']:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES, zero_division=0))

    return test_report, val_report, best_val_f1


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ═══════════════════════════════════════════════════════════════════════════════
def print_comparison_table(model_name, param_count, test_report, lgbm_test_report,
                           wall_clock_secs=None):
    """
    Print formatted delta table comparing candidate model vs LightGBM-only
    and BioClinicalBERT hybrid baselines.
    """
    bcb = BIOCLINICALBERT_BASELINE

    # Extract metrics from classification_report dicts
    def _extract(rpt):
        return {
            "L1-Critical_f1":          rpt["L1-Critical"]["f1-score"],
            "L2-Emergent_f1":          rpt["L2-Emergent"]["f1-score"],
            "L3-Urgent/LessUrgent_f1": rpt["L3-Urgent/LessUrgent"]["f1-score"],
            "accuracy":                rpt["accuracy"],
            "macro_f1":                rpt["macro avg"]["f1-score"],
            "weighted_f1":             rpt["weighted avg"]["f1-score"],
        }

    cand = _extract(test_report)
    lgbm = _extract(lgbm_test_report)

    rows = [
        ("L1-Critical F1",  "L1-Critical_f1"),
        ("L2-Emergent F1",  "L2-Emergent_f1"),
        ("L3-Urgent/LU F1", "L3-Urgent/LessUrgent_f1"),
        ("Accuracy",        "accuracy"),
        ("Macro-F1",        "macro_f1"),
        ("Weighted-F1",     "weighted_f1"),
    ]

    print()
    print("=" * 90)
    print(f"  {model_name} ({param_count}) — Comparison Table")
    if wall_clock_secs is not None:
        mins = wall_clock_secs / 60
        print(f"  Wall-clock time: {mins:.1f} min ({wall_clock_secs:.0f}s)")
    print("=" * 90)
    header = f"{'Metric':<22} {'LightGBM':>10} {'Candidate':>10} {'BioClBERT':>10} {'D vs LGBM':>10} {'D vs BCB':>10}"
    print(header)
    print("-" * 90)

    for label, key in rows:
        l_val = lgbm[key]
        c_val = cand[key]
        b_val = bcb[key]
        d_lgbm = c_val - l_val
        d_bcb  = c_val - b_val
        print(
            f"{label:<22} {l_val:>10.4f} {c_val:>10.4f} {b_val:>10.4f} "
            f"{d_lgbm:>+10.4f} {d_bcb:>+10.4f}"
        )

    print("-" * 90)

    # Check success criteria
    lift_macro = cand["macro_f1"] - lgbm["macro_f1"]
    lift_l1    = cand["L1-Critical_f1"] - lgbm["L1-Critical_f1"]
    lift_l2    = cand["L2-Emergent_f1"] - lgbm["L2-Emergent_f1"]

    print("\nSuccess criteria vs LightGBM-only:")
    print(f"  Macro-F1 lift > +0.10:      {lift_macro:+.4f}  {'PASS' if lift_macro > 0.10 else 'FAIL'}")
    print(f"  L1-Critical F1 lift > +0.06: {lift_l1:+.4f}  {'PASS' if lift_l1 > 0.06 else 'FAIL'}")
    print(f"  L2-Emergent F1 lift > +0.14: {lift_l2:+.4f}  {'PASS' if lift_l2 > 0.14 else 'FAIL'}")

    beats_bcb_macro = cand["macro_f1"] > bcb["macro_f1"]
    print(f"\n  Beats BioClinicalBERT on macro-F1: {'YES' if beats_bcb_macro else 'NO'} "
          f"({cand['macro_f1']:.4f} vs {bcb['macro_f1']:.4f})")
    print("=" * 90)
"""
exp_utils.py — Shared utilities for arch4 text-encoder experiments.

Provides:
  - load_splits_from_s3()      : loads CSV splits + rebuilds LightGBM OOF probs
  - build_lgbm_baseline()      : 5-fold cross-fitting with identical params/seed as arch4_v1
  - FusionHead                 : lightweight MLP for embedding + LightGBM prob fusion
  - train_fusion_head()        : trains FusionHead on pre-extracted embeddings, returns reports
  - print_comparison_table()   : formats delta table vs LightGBM-only and BioClinicalBERT
"""

import io
import random
import time
import warnings

import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
SEED         = 42
S3_BUCKET    = "ed-triage-capstone-group7"
DATA_KEY     = "Data_Output/consolidated_dataset_features.csv"
SPLIT_PREFIX = "Data_Output/splits/arch4_v1/"
NUM_CLASSES  = 3
CLASS_NAMES  = ["L1-Critical", "L2-Emergent", "L3-Urgent/LessUrgent"]

# BioClinicalBERT hybrid baseline (arch4_v1 test set)
BIOCLINICALBERT_BASELINE = {
    "L1-Critical_f1":          0.57,
    "L2-Emergent_f1":          0.63,
    "L3-Urgent/LessUrgent_f1": 0.80,
    "accuracy":                0.72,
    "macro_f1":                0.67,
    "weighted_f1":             0.72,
}

RAW_VITALS  = ["temp_f", "heart_rate", "resp_rate", "spo2", "sbp", "dbp"]
CLIP_BOUNDS = {
    "temp_f":     (85.0, 115.0),
    "heart_rate": (20.0, 250.0),
    "resp_rate":  (4.0,  60.0),
    "spo2":       (50.0, 100.0),
    "sbp":        (40.0, 300.0),
    "dbp":        (10.0, 200.0),
}
TRANSPORT_MAP = {"WALK IN": 0, "UNKNOWN": 1, "AMBULANCE": 2, "HELICOPTER": 3}

STRUCTURED_FEATURES = [
    "heart_rate", "sbp", "dbp", "resp_rate", "spo2", "temp_f",
    "shock_index", "map", "pulse_pressure",
    "news2_score", "mews_score",
    "age", "transport_ordinal",
    "pain", "pain_missing",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# Text construction
# ═══════════════════════════════════════════════════════════════════════════════
def clip_words(text, max_words):
    text = "" if pd.isna(text) else str(text).replace("\n", " ").strip()
    return " ".join(text.split()[:max_words]) if text else ""


def build_triage_text(row):
    """CC-emphasized text: CC_2x + HPI (PMH dropped)."""
    cc  = clip_words(row["chiefcomplaint"], 24)
    hpi = clip_words(row["HPI"], 160)
    parts = []
    if cc:  parts.append(f"Chief complaint: {cc}.")
    if cc:  parts.append(f"Presenting with {cc}.")
    if hpi: parts.append(f"History: {hpi}.")
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Clinical early warning scores
# ═══════════════════════════════════════════════════════════════════════════════
def compute_news2(row):
    """NEWS2 aggregate score (Scale 1). Temp in deg-F. No AVPU — max 15."""
    score = 0
    rr = row["resp_rate"]
    if   rr <= 8:   score += 3
    elif rr <= 11:  score += 1
    elif rr <= 20:  score += 0
    elif rr <= 24:  score += 2
    else:           score += 3
    spo2 = row["spo2"]
    if   spo2 <= 91:  score += 3
    elif spo2 <= 93:  score += 2
    elif spo2 <= 95:  score += 1
    sbp = row["sbp"]
    if   sbp <= 90:   score += 3
    elif sbp <= 100:  score += 2
    elif sbp <= 110:  score += 1
    elif sbp <= 219:  score += 0
    else:             score += 3
    hr = row["heart_rate"]
    if   hr <= 40:   score += 3
    elif hr <= 50:   score += 1
    elif hr <= 90:   score += 0
    elif hr <= 110:  score += 1
    elif hr <= 130:  score += 2
    else:            score += 3
    temp = row["temp_f"]
    if   temp <= 95.0:    score += 3
    elif temp <= 96.8:    score += 1
    elif temp <= 100.4:   score += 0
    elif temp <= 102.2:   score += 1
    else:                 score += 2
    return score


def compute_mews(row):
    """MEWS aggregate score. No AVPU — max 12."""
    score = 0
    sbp = row["sbp"]
    if   sbp < 70:    score += 3
    elif sbp < 81:    score += 2
    elif sbp < 101:   score += 1
    elif sbp < 200:   score += 0
    else:             score += 2
    hr = row["heart_rate"]
    if   hr < 40:     score += 2
    elif hr < 51:     score += 1
    elif hr < 101:    score += 0
    elif hr < 111:    score += 1
    elif hr < 130:    score += 2
    else:             score += 3
    rr = row["resp_rate"]
    if   rr < 9:      score += 2
    elif rr < 15:     score += 0
    elif rr < 21:     score += 1
    elif rr < 30:     score += 2
    else:             score += 3
    temp = row["temp_f"]
    if   temp < 95.0:    score += 2
    elif temp <= 101.1:  score += 0
    else:                score += 2
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# Structured feature pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def fit_structured_stats(train_df):
    """Compute imputation medians from training rows only."""
    return {
        "vital_medians": {col: float(train_df[col].median()) for col in RAW_VITALS},
        "pain_median":   float(train_df["pain"].median()),
        "age_median":    float(train_df["age"].median()),
    }


def transform_structured(df_in, stats):
    """Build pruned 15 structured features using train-set imputation stats."""
    feat = df_in.copy()
    for col in RAW_VITALS:
        feat[col] = feat[col].fillna(stats["vital_medians"][col]).clip(*CLIP_BOUNDS[col])
    feat["pain"] = feat["pain"].fillna(stats["pain_median"]).clip(0.0, 10.0)
    feat["age"]  = feat["age"].fillna(stats["age_median"]).clip(18, 120)
    feat["pain_missing"]      = feat["pain_missing"].astype(int)
    feat["transport_ordinal"] = feat["arrival_transport"].map(TRANSPORT_MAP).fillna(1).astype(int)
    feat["shock_index"]    = (feat["heart_rate"] / feat["sbp"].replace(0, np.nan)).fillna(0.0)
    feat["map"]            = (feat["sbp"] + 2.0 * feat["dbp"]) / 3.0
    feat["pulse_pressure"] = feat["sbp"] - feat["dbp"]
    feat["news2_score"] = feat.apply(compute_news2, axis=1)
    feat["mews_score"]  = feat.apply(compute_mews, axis=1)
    return feat[STRUCTURED_FEATURES].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# S3 data loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_splits_from_s3():
    """
    Load train/val/test splits from S3, reconstruct triage_text if missing,
    compute structured features, return (frames, labels, structured arrays).

    Returns:
        dict with keys:
            X_train, X_val, X_test      — DataFrames (contain triage_text col)
            y_train, y_val, y_test      — numpy int arrays
            X_train_struct, X_val_struct, X_test_struct — float32 numpy arrays (15 features)
            structured_stats            — dict of imputation medians
    """
    s3 = boto3.client("s3", region_name="us-east-1")

    def _load(key):
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    print("Loading splits from S3...")
    X_train = _load(f"{SPLIT_PREFIX}train.csv")
    X_val   = _load(f"{SPLIT_PREFIX}val.csv")
    X_test  = _load(f"{SPLIT_PREFIX}test.csv")

    # Reconstruct triage_text if not present (safety fallback)
    for frame in [X_train, X_val, X_test]:
        if "triage_text" not in frame.columns:
            frame["triage_text"] = frame.apply(build_triage_text, axis=1)

    y_train = X_train["triage_3class"].values.astype(int)
    y_val   = X_val["triage_3class"].values.astype(int)
    y_test  = X_test["triage_3class"].values.astype(int)

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Structured features (fitted on train only)
    stats        = fit_structured_stats(X_train)
    X_train_st   = transform_structured(X_train, stats).values.astype(np.float32)
    X_val_st     = transform_structured(X_val,   stats).values.astype(np.float32)
    X_test_st    = transform_structured(X_test,  stats).values.astype(np.float32)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_struct": X_train_st, "X_val_struct": X_val_st, "X_test_struct": X_test_st,
        "structured_stats": stats,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LightGBM 5-fold cross-fitting (identical to arch4_v1)
# ═══════════════════════════════════════════════════════════════════════════════
def build_lgbm_baseline(X_train_np, y_train, X_val_np, y_val, X_test_np, y_test):
    """
    5-fold cross-fit LightGBM with identical params/seed as arch4_v1.

    Returns:
        lgbm_probs_train  — OOF probs (n_train, 3)
        lgbm_probs_val    — avg of 5-fold val probs
        lgbm_probs_test   — avg of 5-fold test probs
        lgbm_test_report  — classification_report dict for LightGBM-only on test
        lgbm_fold_models  — list of 5 trained LGBMClassifier
    """
    print("\nBuilding LightGBM baseline (5-fold cross-fitting)...")
    train_counts = {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))}
    sqrt_w = np.sqrt(np.array([len(y_train) / (3 * train_counts[c]) for c in range(NUM_CLASSES)]))
    sw_map = {c: float(sqrt_w[c]) for c in range(NUM_CLASSES)}

    lgbm_params = dict(
        objective="multiclass", num_class=NUM_CLASSES,
        n_estimators=500, max_depth=-1, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.5, reg_lambda=2.0,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )

    oof_probs        = np.zeros((len(X_train_np), NUM_CLASSES), dtype=np.float32)
    val_probs_folds  = []
    test_probs_folds = []
    fold_models      = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_idx, hold_idx) in enumerate(skf.split(X_train_np, y_train), start=1):
        X_fit,  y_fit  = X_train_np[fit_idx],  y_train[fit_idx]
        X_hold, y_hold = X_train_np[hold_idx], y_train[hold_idx]
        sw_fit = np.array([sw_map[int(l)] for l in y_fit], dtype=np.float32)

        model = LGBMClassifier(**lgbm_params)
        model.fit(
            X_fit, y_fit, sample_weight=sw_fit,
            eval_set=[(X_hold, y_hold)],
            callbacks=[early_stopping(30, verbose=False)],
        )

        oof_probs[hold_idx] = model.predict_proba(X_hold)
        val_probs_folds.append(model.predict_proba(X_val_np))
        test_probs_folds.append(model.predict_proba(X_test_np))
        fold_models.append(model)

        hold_f1 = f1_score(y_hold, oof_probs[hold_idx].argmax(1), average="macro")
        print(f"  Fold {fold} | best_iter={model.best_iteration_} | holdout macro-F1={hold_f1:.4f}")

    lgbm_probs_train = oof_probs
    lgbm_probs_val   = np.mean(np.stack(val_probs_folds),  axis=0)
    lgbm_probs_test  = np.mean(np.stack(test_probs_folds), axis=0)

    lgbm_preds_test = lgbm_probs_test.argmax(axis=1)
    lgbm_test_report = classification_report(
        y_test, lgbm_preds_test,
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    lgbm_test_f1 = f1_score(y_test, lgbm_preds_test, average="macro")

    print(f"\nLightGBM-only test macro-F1: {lgbm_test_f1:.4f}")
    print(classification_report(y_test, lgbm_preds_test, target_names=CLASS_NAMES, zero_division=0))

    return lgbm_probs_train, lgbm_probs_val, lgbm_probs_test, lgbm_test_report, fold_models


# ═══════════════════════════════════════════════════════════════════════════════
# Class weights (same sqrt-dampened scheme as arch4_v1)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_class_weights(y_train, device=None):
    """Sqrt-dampened inverse frequency class weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts = {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))}
    raw = torch.tensor(
        [len(y_train) / (NUM_CLASSES * counts[c]) for c in range(NUM_CLASSES)],
        dtype=torch.float32,
    )
    return torch.sqrt(raw).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# FusionHead — lightweight MLP for pre-extracted embeddings + LightGBM probs
# ═══════════════════════════════════════════════════════════════════════════════
class EmbeddingFusionDataset(Dataset):
    """Dataset for pre-extracted embeddings + LightGBM probs."""

    def __init__(self, embeddings, lgbm_probs, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.lgbm_probs = torch.tensor(lgbm_probs, dtype=torch.float32)
        self.labels     = torch.tensor(labels,      dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.lgbm_probs[idx], self.labels[idx]


class FusionHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear (same arch as arch4_v1 head)."""

    def __init__(self, embedding_dim, tree_dim=3, num_classes=3,
                 hidden_dim=96, dropout=0.55):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim + tree_dim),
            nn.Linear(embedding_dim + tree_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings, tree_probs):
        fused = torch.cat([embeddings, tree_probs], dim=1)
        return self.head(fused)


def train_fusion_head(
    train_embs, train_lgbm, y_train,
    val_embs,   val_lgbm,   y_val,
    test_embs,  test_lgbm,  y_test,
    embedding_dim,
    hidden_dim=96,
    dropout=0.55,
    lr=1e-3,
    weight_decay=0.01,
    epochs=60,
    patience=8,
    batch_size=128,
    device=None,
):
    """
    Train FusionHead on pre-extracted embeddings + LightGBM probs.

    Returns:
        test_report  — classification_report dict on test set
        val_report   — classification_report dict on val set
        best_val_f1  — best validation macro-F1
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)

    train_ds = EmbeddingFusionDataset(train_embs, train_lgbm, y_train)
    val_ds   = EmbeddingFusionDataset(val_embs,   val_lgbm,   y_val)
    test_ds  = EmbeddingFusionDataset(test_embs,  test_lgbm,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model     = FusionHead(embedding_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    class_wts = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_wts)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_f1    = -1.0
    bad_epochs     = 0
    best_state     = None

    print(f"\nTraining FusionHead (emb_dim={embedding_dim}, hidden={hidden_dim}, dropout={dropout})")
    print(f"  lr={lr}, epochs={epochs}, patience={patience}, batch_size={batch_size}")

    for epoch in range(1, epochs + 1):
        # — Train —
        model.train()
        for emb, lgbm, lab in train_loader:
            emb, lgbm, lab = emb.to(device), lgbm.to(device), lab.to(device)
            logits = model(emb, lgbm)
            loss   = criterion(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # — Validate —
        model.eval()
        val_preds = []
        with torch.no_grad():
            for emb, lgbm, lab in val_loader:
                emb, lgbm = emb.to(device), lgbm.to(device)
                logits = model(emb, lgbm)
                val_preds.extend(logits.argmax(1).cpu().numpy())
        val_f1 = f1_score(y_val, val_preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            bad_epochs  = 0
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == 1 or bad_epochs == 0:
            tag = "<- best" if bad_epochs == 0 else f"patience {bad_epochs}/{patience}"
            print(f"  Epoch {epoch:02d} | val_f1={val_f1:.4f} | {tag}")

        if bad_epochs >= patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    # — Evaluate with best model —
    model.load_state_dict(best_state)
    model.eval()

    def _predict(loader):
        preds = []
        with torch.no_grad():
            for emb, lgbm, _ in loader:
                emb, lgbm = emb.to(device), lgbm.to(device)
                logits = model(emb, lgbm)
                preds.extend(logits.argmax(1).cpu().numpy())
        return np.array(preds)

    val_preds  = _predict(val_loader)
    test_preds = _predict(test_loader)

    val_report = classification_report(
        y_val, val_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    test_report = classification_report(
        y_test, test_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )

    print(f"\nBest val macro-F1: {best_val_f1:.4f}")
    print(f"Test macro-F1:     {test_report['macro avg']['f1-score']:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES, zero_division=0))

    return test_report, val_report, best_val_f1


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ═══════════════════════════════════════════════════════════════════════════════
def print_comparison_table(model_name, param_count, test_report, lgbm_test_report,
                           wall_clock_secs=None):
    """
    Print formatted delta table comparing candidate model vs LightGBM-only
    and BioClinicalBERT hybrid baselines.
    """
    bcb = BIOCLINICALBERT_BASELINE

    # Extract metrics from classification_report dicts
    def _extract(rpt):
        return {
            "L1-Critical_f1":          rpt["L1-Critical"]["f1-score"],
            "L2-Emergent_f1":          rpt["L2-Emergent"]["f1-score"],
            "L3-Urgent/LessUrgent_f1": rpt["L3-Urgent/LessUrgent"]["f1-score"],
            "accuracy":                rpt["accuracy"],
            "macro_f1":                rpt["macro avg"]["f1-score"],
            "weighted_f1":             rpt["weighted avg"]["f1-score"],
        }

    cand = _extract(test_report)
    lgbm = _extract(lgbm_test_report)

    rows = [
        ("L1-Critical F1",  "L1-Critical_f1"),
        ("L2-Emergent F1",  "L2-Emergent_f1"),
        ("L3-Urgent/LU F1", "L3-Urgent/LessUrgent_f1"),
        ("Accuracy",        "accuracy"),
        ("Macro-F1",        "macro_f1"),
        ("Weighted-F1",     "weighted_f1"),
    ]

    print()
    print("=" * 90)
    print(f"  {model_name} ({param_count}) — Comparison Table")
    if wall_clock_secs is not None:
        mins = wall_clock_secs / 60
        print(f"  Wall-clock time: {mins:.1f} min ({wall_clock_secs:.0f}s)")
    print("=" * 90)
    header = f"{'Metric':<22} {'LightGBM':>10} {'Candidate':>10} {'BioClBERT':>10} {'D vs LGBM':>10} {'D vs BCB':>10}"
    print(header)
    print("-" * 90)

    for label, key in rows:
        l_val = lgbm[key]
        c_val = cand[key]
        b_val = bcb[key]
        d_lgbm = c_val - l_val
        d_bcb  = c_val - b_val
        print(
            f"{label:<22} {l_val:>10.4f} {c_val:>10.4f} {b_val:>10.4f} "
            f"{d_lgbm:>+10.4f} {d_bcb:>+10.4f}"
        )

    print("-" * 90)

    # Check success criteria
    lift_macro = cand["macro_f1"] - lgbm["macro_f1"]
    lift_l1    = cand["L1-Critical_f1"] - lgbm["L1-Critical_f1"]
    lift_l2    = cand["L2-Emergent_f1"] - lgbm["L2-Emergent_f1"]

    print("\nSuccess criteria vs LightGBM-only:")
    print(f"  Macro-F1 lift > +0.10:      {lift_macro:+.4f}  {'PASS' if lift_macro > 0.10 else 'FAIL'}")
    print(f"  L1-Critical F1 lift > +0.06: {lift_l1:+.4f}  {'PASS' if lift_l1 > 0.06 else 'FAIL'}")
    print(f"  L2-Emergent F1 lift > +0.14: {lift_l2:+.4f}  {'PASS' if lift_l2 > 0.14 else 'FAIL'}")

    beats_bcb_macro = cand["macro_f1"] > bcb["macro_f1"]
    print(f"\n  Beats BioClinicalBERT on macro-F1: {'YES' if beats_bcb_macro else 'NO'} "
          f"({cand['macro_f1']:.4f} vs {bcb['macro_f1']:.4f})")
    print("=" * 90)
