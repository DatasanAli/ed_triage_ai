"""
Arch4 training script for SageMaker TrainingStep.

BioClinicalBERT + LightGBM fusion (MeanPoolHybridModel).
Pruned 15 structured features, CC_2x+HPI text format.

SageMaker contract:
  SM_CHANNEL_TRAIN      → directory containing train.csv
  SM_CHANNEL_VALIDATION → directory containing val.csv
  SM_MODEL_DIR          → directory to write model artifacts
  Hyperparameters       → passed as CLI args (--epochs, --head-lr, etc.)

Artifacts written to SM_MODEL_DIR:
  best_model.pt           BioClinicalBERT + fusion head weights
  lgbm_fold{1..5}.joblib  LightGBM fold models
  lgbm_fold_summary.csv   Per-fold LightGBM metrics
  history.csv             Epoch-level training history
  config.json             Full run config + test metrics
"""

import argparse
import glob
import json
import math
import os
import random
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")

# ── SageMaker environment ──────────────────────────────────────────────────────
TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN",      "/opt/ml/input/data/train")
VAL_DIR   = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
MODEL_DIR = os.environ.get("SM_MODEL_DIR",           "/opt/ml/model")

# ── Constants ──────────────────────────────────────────────────────────────────
SEED       = 42
NUM_CLASSES = 3
BERT_MODEL  = "emilyalsentzer/Bio_ClinicalBERT"
CLASS_NAMES = ["L1-Critical", "L2-Emergent", "L3-Urgent/LessUrgent"]

RAW_VITALS = ["temp_f", "heart_rate", "resp_rate", "spo2", "sbp", "dbp"]
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
    "age",
    "transport_ordinal",
    "pain", "pain_missing",
]
assert len(STRUCTURED_FEATURES) == 15


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Structured feature pipeline ────────────────────────────────────────────────

def compute_news2(row):
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


def fit_structured_stats(train_df):
    return {
        "vital_medians": {col: float(train_df[col].median()) for col in RAW_VITALS},
        "pain_median":   float(train_df["pain"].median()),
        "age_median":    float(train_df["age"].median()),
    }


def transform_structured(df_in, stats):
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


# ── Dataset ────────────────────────────────────────────────────────────────────

class TriageDataset(Dataset):
    def __init__(self, frame, lgbm_probs, labels, tokenizer, max_len):
        # Pre-tokenize all texts upfront to avoid per-sample overhead during training
        texts = frame["triage_text"].tolist()
        print(f"  Pre-tokenizing {len(texts)} texts (max_len={max_len})...")
        enc = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.lgbm_probs     = torch.tensor(lgbm_probs, dtype=torch.float32)
        self.labels         = torch.tensor(labels, dtype=torch.long)
        print(f"  Done. Tokenized shape: {tuple(self.input_ids.shape)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "lgbm_probs":     self.lgbm_probs[idx],
            "label":          self.labels[idx],
        }


# ── Model ──────────────────────────────────────────────────────────────────────

class MeanPoolHybridModel(nn.Module):
    def __init__(self, bert_model_name, tree_dim=3, num_classes=3,
                 hidden_dim=96, dropout=0.55):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_dim  = self.bert.config.hidden_size  # 768
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(bert_dim + tree_dim),
            nn.Linear(bert_dim + tree_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def masked_mean_pool(self, last_hidden_state, attention_mask):
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, input_ids, attention_mask, tree_probs):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled   = self.masked_mean_pool(bert_out.last_hidden_state, attention_mask)
        fused    = torch.cat([pooled, tree_probs], dim=1)
        return self.fusion_head(fused)


def freeze_all_bert(model):
    for param in model.bert.parameters():
        param.requires_grad = False


def unfreeze_top_bert_layers(model, n_layers):
    freeze_all_bert(model)
    total = len(model.bert.encoder.layer)
    for layer in model.bert.encoder.layer[total - n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training helpers ───────────────────────────────────────────────────────────

def build_head_optimizer(model, head_lr):
    return AdamW([{"params": model.fusion_head.parameters(), "lr": head_lr, "weight_decay": 0.01}])


def build_finetune_optimizer(model, n_layers, bert_base_lr, fusion_lr, lr_decay):
    param_groups = [{"params": model.fusion_head.parameters(), "lr": fusion_lr, "weight_decay": 0.01}]
    layers       = model.bert.encoder.layer
    total        = len(layers)
    selected     = list(range(total - n_layers, total))
    for rank, layer_idx in enumerate(selected):
        scale = lr_decay ** (len(selected) - 1 - rank)
        param_groups.append({
            "params": layers[layer_idx].parameters(),
            "lr": bert_base_lr * scale,
            "weight_decay": 0.01,
        })
    return AdamW(param_groups)


def build_scheduler(optimizer, train_loader, num_epochs, accumulation_steps):
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps     = steps_per_epoch * num_epochs
    warmup_steps    = max(1, int(0.10 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return scheduler


def train_epoch(model, loader, optimizer, scheduler, criterion, device, accumulation_steps, scaler):
    model.train()
    total_loss = 0.0
    preds, labels = [], []
    optimizer.zero_grad()

    for step, batch in enumerate(loader, start=1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tree_probs     = batch["lgbm_probs"].to(device)
        target         = batch["label"].to(device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(input_ids, attention_mask, tree_probs)
            loss   = criterion(logits, target) / accumulation_steps

        scaler.scale(loss).backward()

        if step % accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        preds.extend(logits.detach().argmax(dim=1).cpu().numpy())
        labels.extend(target.detach().cpu().numpy())

    return {
        "loss":     total_loss / len(loader),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tree_probs     = batch["lgbm_probs"].to(device)
            target         = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(input_ids, attention_mask, tree_probs)
                loss   = criterion(logits, target)

            total_loss += loss.item()
            preds.extend(logits.detach().argmax(dim=1).cpu().numpy())
            labels.extend(target.detach().cpu().numpy())

    report = classification_report(labels, preds, labels=[0, 1, 2], output_dict=True, zero_division=0)
    return {
        "loss":        total_loss / len(loader),
        "macro_f1":    f1_score(labels, preds, average="macro"),
        "critical_f1": float(report["0"]["f1-score"]),
        "preds":       np.array(preds),
        "labels":      np.array(labels),
    }


def load_split(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {directory}")
    return pd.read_csv(files[0])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",            type=int,   default=20)
    parser.add_argument("--head-warmup-epochs", type=int,   default=1)
    parser.add_argument("--patience",          type=int,   default=3)
    parser.add_argument("--batch-size",        type=int,   default=16)
    parser.add_argument("--accumulation-steps", type=int,  default=2)
    parser.add_argument("--max-len",           type=int,   default=384)
    parser.add_argument("--head-lr",           type=float, default=2e-4)
    parser.add_argument("--bert-base-lr",      type=float, default=6e-6)
    parser.add_argument("--fusion-lr",         type=float, default=1.5e-4)
    parser.add_argument("--lr-decay",          type=float, default=0.80)
    parser.add_argument("--unfreeze-layers",   type=int,   default=3)
    parser.add_argument("--fusion-hidden-dim", type=int,   default=96)
    parser.add_argument("--fusion-dropout",    type=float, default=0.55)
    args = parser.parse_args()

    set_seed(SEED)

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load splits ───────────────────────────────────────────────────────────
    print(f"\nLoading train from {TRAIN_DIR}")
    print(f"Loading val   from {VAL_DIR}")
    train_df = load_split(TRAIN_DIR)
    val_df   = load_split(VAL_DIR)
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    y_train = train_df["triage_3class"].values
    y_val   = val_df["triage_3class"].values

    # ── Structured features ───────────────────────────────────────────────────
    structured_stats = fit_structured_stats(train_df)
    X_train_struct   = transform_structured(train_df, structured_stats)
    X_val_struct     = transform_structured(val_df,   structured_stats)

    X_train_np = X_train_struct.values.astype(np.float32)
    X_val_np   = X_val_struct.values.astype(np.float32)

    print(f"\nStructured features: {len(STRUCTURED_FEATURES)}")
    print(f"Structured stats: {json.dumps(structured_stats, indent=2)}")

    # ── Class weights ─────────────────────────────────────────────────────────
    train_counts = {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))}
    raw_weights  = torch.tensor(
        [len(y_train) / (NUM_CLASSES * train_counts[c]) for c in range(NUM_CLASSES)],
        dtype=torch.float32,
    )
    class_weights    = torch.sqrt(raw_weights)
    sqrt_w           = np.sqrt(np.array([len(y_train) / (3 * train_counts[c]) for c in range(NUM_CLASSES)]))
    sample_weight_map = {c: float(sqrt_w[c]) for c in range(NUM_CLASSES)}

    print(f"\nClass weights (sqrt-dampened): {[round(x, 4) for x in class_weights.tolist()]}")

    # ── LightGBM 5-fold cross-fitting ─────────────────────────────────────────
    print("\nFitting LightGBM (5-fold cross-fitting)...")
    lgbm_params = dict(
        objective="multiclass",
        num_class=NUM_CLASSES,
        n_estimators=500,
        max_depth=-1,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )

    oof_probs        = np.zeros((len(X_train_np), NUM_CLASSES), dtype=np.float32)
    val_probs_folds  = []
    lgbm_fold_models = []
    fold_rows        = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_idx, hold_idx) in enumerate(skf.split(X_train_np, y_train), start=1):
        X_fit,  y_fit  = X_train_np[fit_idx],  y_train[fit_idx]
        X_hold, y_hold = X_train_np[hold_idx], y_train[hold_idx]
        sw_fit = np.array([sample_weight_map[int(l)] for l in y_fit], dtype=np.float32)

        lgbm_model = LGBMClassifier(**lgbm_params)
        lgbm_model.fit(
            X_fit, y_fit,
            sample_weight=sw_fit,
            eval_set=[(X_hold, y_hold)],
            callbacks=[early_stopping(30, verbose=False)],
        )

        oof_probs[hold_idx] = lgbm_model.predict_proba(X_hold)
        val_probs_folds.append(lgbm_model.predict_proba(X_val_np))
        lgbm_fold_models.append(lgbm_model)

        hold_pred = oof_probs[hold_idx].argmax(axis=1)
        fold_rows.append({
            "fold":             fold,
            "best_iteration":   int(lgbm_model.best_iteration_),
            "holdout_macro_f1": float(f1_score(y_hold, hold_pred, average="macro")),
        })
        print(f"  Fold {fold} | best_iter={lgbm_model.best_iteration_} | "
              f"macro-F1={fold_rows[-1]['holdout_macro_f1']:.4f}")

    lgbm_probs_train = oof_probs
    lgbm_probs_val   = np.mean(np.stack(val_probs_folds, axis=0), axis=0)

    print(f"OOF train macro-F1: {f1_score(y_train, lgbm_probs_train.argmax(axis=1), average='macro'):.4f}")
    print(f"Val    macro-F1:    {f1_score(y_val, lgbm_probs_val.argmax(axis=1), average='macro'):.4f}")

    # ── Tokenizer + Datasets ──────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {BERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    train_dataset = TriageDataset(train_df, lgbm_probs_train, y_train, tokenizer, args.max_len)
    val_dataset   = TriageDataset(val_df,   lgbm_probs_val,   y_val,   tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)}")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MeanPoolHybridModel(
        BERT_MODEL,
        tree_dim=NUM_CLASSES,
        num_classes=NUM_CLASSES,
        hidden_dim=args.fusion_hidden_dim,
        dropout=args.fusion_dropout,
    ).to(device)
    model.bert.gradient_checkpointing_enable()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Two-phase training ────────────────────────────────────────────────────
    best_val_f1     = -1.0
    bad_epochs      = 0
    history         = []
    global_epoch    = 0
    best_model_path = os.path.join("/tmp", "best_arch4.pt")

    phase_plan = [
        ("head_warmup",                        args.head_warmup_epochs),
        (f"top{args.unfreeze_layers}_finetune", args.epochs),
    ]

    stop_training = False
    for phase_name, phase_epochs in phase_plan:
        if phase_name == "head_warmup":
            freeze_all_bert(model)
            optimizer = build_head_optimizer(model, args.head_lr)
        else:
            unfreeze_top_bert_layers(model, args.unfreeze_layers)
            optimizer = build_finetune_optimizer(
                model, args.unfreeze_layers, args.bert_base_lr, args.fusion_lr, args.lr_decay
            )
            bad_epochs = 0

        scheduler = build_scheduler(optimizer, train_loader, phase_epochs, args.accumulation_steps)
        print(f"\nPhase: {phase_name} | epochs={phase_epochs} | trainable={count_trainable(model):,}")

        for _ in range(phase_epochs):
            global_epoch += 1
            train_metrics = train_epoch(
                model, train_loader, optimizer, scheduler, criterion,
                device, args.accumulation_steps, scaler,
            )
            val_metrics = eval_epoch(model, val_loader, criterion, device)

            improved = val_metrics["macro_f1"] > best_val_f1
            if improved:
                best_val_f1 = val_metrics["macro_f1"]
                bad_epochs  = 0
                torch.save(model.state_dict(), best_model_path)
                status = "<- best"
            else:
                bad_epochs += 1
                status = f"patience {bad_epochs}/{args.patience}"

            history.append({
                "phase":           phase_name,
                "epoch":           global_epoch,
                "train_loss":      train_metrics["loss"],
                "val_loss":        val_metrics["loss"],
                "train_f1":        train_metrics["macro_f1"],
                "val_f1":          val_metrics["macro_f1"],
                "val_critical_f1": val_metrics["critical_f1"],
            })

            print(
                f"Epoch {global_epoch:02d} | "
                f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f} | "
                f"critical_f1={val_metrics['critical_f1']:.4f} | {status}"
            )

            if bad_epochs >= args.patience:
                print(f"Early stopping triggered in phase '{phase_name}'.")
                stop_training = True
                break

        if stop_training:
            break

    print(f"\nBest validation macro-F1: {best_val_f1:.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Best model weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
    print(f"Saved best_model.pt → {MODEL_DIR}")

    # LightGBM fold models
    for idx, fold_model in enumerate(lgbm_fold_models, start=1):
        path = os.path.join(MODEL_DIR, f"lgbm_fold{idx}.joblib")
        joblib.dump(fold_model, path)
        print(f"Saved lgbm_fold{idx}.joblib → {MODEL_DIR}")

    # Training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(MODEL_DIR, "history.csv"), index=False)

    # LightGBM fold summary
    pd.DataFrame(fold_rows).to_csv(os.path.join(MODEL_DIR, "lgbm_fold_summary.csv"), index=False)

    # Validation metrics on best model
    val_final = eval_epoch(model, val_loader, criterion, device)
    all_val_probs = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["lgbm_probs"].to(device),
                )
            all_val_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    val_probs_final = np.vstack(all_val_probs)
    val_roc_auc = roc_auc_score(y_val, val_probs_final, multi_class="ovr", average="macro")

    print(f"\nFinal val macro-F1:    {val_final['macro_f1']:.4f}")
    print(f"Final val critical-F1: {val_final['critical_f1']:.4f}")
    print(f"Final val ROC-AUC:     {val_roc_auc:.4f}")
    print("\nValidation classification report:")
    print(classification_report(y_val, val_final["preds"], target_names=CLASS_NAMES, zero_division=0))

    # Config
    config = {
        "architecture":        "arch4",
        "description":         "BioClinicalBERT + LightGBM fusion, pruned 15 features, CC_2x+HPI text",
        "bert_model":          BERT_MODEL,
        "structured_features": STRUCTURED_FEATURES,
        "structured_stats":    structured_stats,
        "transport_map":       TRANSPORT_MAP,
        "class_weights":       [float(x) for x in class_weights.tolist()],
        "hyperparameters":     vars(args),
        "val_metrics": {
            "best_val_macro_f1": round(best_val_f1, 4),
            "final_macro_f1":    round(val_final["macro_f1"], 4),
            "final_critical_f1": round(val_final["critical_f1"], 4),
            "roc_auc_ovr":       round(val_roc_auc, 4),
        },
    }
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy inference.py into model archive if it exists
    inference_src = os.path.join(os.path.dirname(__file__), "inference.py")
    if os.path.exists(inference_src):
        inference_dst = os.path.join(MODEL_DIR, "inference.py")
        shutil.copy2(inference_src, inference_dst)
        print(f"Copied inference → {inference_dst}")

    print("\nAll artifacts saved. Training complete.")


if __name__ == "__main__":
    main()
