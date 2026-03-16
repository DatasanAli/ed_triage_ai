"""
Mock training script for SageMaker TrainingStep (CPU smoke test).

Proves the SageMaker training contract works end to end using a trivial model.
Reads the same split format produced by preprocess.py. Uses bert-tiny instead
of BioClinicalBERT, skips LightGBM, trains 1 epoch on CPU.

SageMaker contract:
  SM_CHANNEL_TRAIN      → directory containing train.csv
  SM_CHANNEL_VALIDATION → directory containing val.csv
  SM_MODEL_DIR          → directory to write model artifacts
  SM_HP_*               → hyperparameters (optional overrides)
"""

import json
import os
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ── SageMaker environment ─────────────────────────────────────────────────────
TRAIN_DIR   = os.environ.get("SM_CHANNEL_TRAIN",      "/opt/ml/input/data/train")
VAL_DIR     = os.environ.get("SM_CHANNEL_VALIDATION",  "/opt/ml/input/data/validation")
MODEL_DIR   = os.environ.get("SM_MODEL_DIR",           "/opt/ml/model")
NUM_EPOCHS  = int(os.environ.get("SM_HP_EPOCHS", "1"))

SEED = 42
NUM_CLASSES = 3
MAX_LEN     = 128      # short for mock — bert-tiny has 128 hidden dim
BATCH_SIZE  = 32
LR          = 2e-4
BERT_MODEL  = "prajjwal1/bert-tiny"   # 4M params, 128-dim

DEVICE = torch.device("cpu")  # mock always runs on CPU

# ── Structured feature pipeline (mirrors arch4_training_v1.ipynb §5) ──────────

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
    "age", "transport_ordinal",
    "pain", "pain_missing",
]


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


# ── Dataset ───────────────────────────────────────────────────────────────────

class TriageMockDataset(Dataset):
    """Tokenized triage_text + label (no LightGBM probs for mock)."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          self.labels[idx],
        }


# ── Model (trivial: bert-tiny → mean pool → Linear) ──────────────────────────

class MockTriageModel(nn.Module):
    def __init__(self, bert_model_name, num_classes=NUM_CLASSES):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_dim  = self.bert.config.hidden_size  # 128 for bert-tiny
        self.head = nn.Linear(bert_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.head(pooled)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_split(directory, pattern="*.csv"):
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files in {directory}")
    return pd.read_csv(files[0])


def main():
    print(f"Device: {DEVICE}")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Val dir:   {VAL_DIR}")
    print(f"Model dir: {MODEL_DIR}")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df = load_split(TRAIN_DIR)
    val_df   = load_split(VAL_DIR)
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    y_train = train_df["triage_3class"].values
    y_val   = val_df["triage_3class"].values

    # ── Structured features (fit on train, transform both) ────────────────────
    stats = fit_structured_stats(train_df)
    print(f"Structured stats: {json.dumps(stats, indent=2)}")

    X_train_struct = transform_structured(train_df, stats)
    X_val_struct   = transform_structured(val_df, stats)
    print(f"Structured features shape: {X_train_struct.shape}")

    # ── Tokenize ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    train_dataset = TriageMockDataset(
        train_df["triage_text"].tolist(), y_train, tokenizer, MAX_LEN,
    )
    val_dataset = TriageMockDataset(
        val_df["triage_text"].tolist(), y_val, tokenizer, MAX_LEN,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MockTriageModel(BERT_MODEL).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # ── Train ─────────────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total   += labels.size(0)

        train_acc = correct / total
        avg_loss  = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | loss={avg_loss:.4f} | acc={train_acc:.4f}")

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)
            logits = model(input_ids, attention_mask)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total   += labels.size(0)
    print(f"Validation accuracy: {correct/total:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save structured stats alongside model for reproducibility
    stats_path = os.path.join(MODEL_DIR, "structured_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved model     → {model_path}")
    print(f"Saved stats     → {stats_path}")
    print("Mock training complete.")


if __name__ == "__main__":
    main()
