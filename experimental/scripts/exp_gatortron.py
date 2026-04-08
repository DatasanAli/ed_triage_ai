"""
exp_gatortron.py — GatorTron-base (345M) + LightGBM fusion experiment.

Model:  UFNLP/gatortron-base (345M params, pre-trained on 90B+ words of clinical text)
Method: Mean-pooled embeddings -> fuse with LightGBM probs -> fusion head.
        Same two-phase training as arch4_v1 (head warmup -> top-N layer fine-tune).

Run:    python exp_gatortron.py      (on SageMaker ml.g4dn.2xlarge)
"""

import math
import subprocess
import sys
import time

# ── Install dependencies ──────────────────────────────────────────────────────
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "boto3", "transformers", "lightgbm", "scikit-learn", "joblib",
    "torch", "numpy==1.26.4", "pandas", "sentencepiece",
])

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from exp_utils import (
    BIOCLINICALBERT_BASELINE, CLASS_NAMES, NUM_CLASSES, SEED,
    build_lgbm_baseline, compute_class_weights, load_splits_from_s3,
    print_comparison_table, set_seed,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_NAME         = "UFNLP/gatortron-base"
MAX_LEN            = 384
BATCH_SIZE         = 16
ACCUMULATION_STEPS = 2
HEAD_WARMUP_EPOCHS = 1
FINETUNE_EPOCHS    = 20
PATIENCE           = 3
HEAD_LR            = 2e-4
BERT_BASE_LR       = 6e-6
FUSION_LR          = 1.5e-4
LR_DECAY           = 0.80
UNFREEZE_LAYERS    = 3
FUSION_HIDDEN_DIM  = 96
FUSION_DROPOUT     = 0.55

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTOCAST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE       = torch.float16
scaler          = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════
class TriageDataset(Dataset):
    def __init__(self, texts, lgbm_probs, labels, tokenizer, max_len):
        self.texts      = texts
        self.lgbm_probs = torch.tensor(lgbm_probs, dtype=torch.float32)
        self.labels     = torch.tensor(labels, dtype=torch.long)
        self.tokenizer  = tokenizer
        self.max_len    = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "lgbm_probs":     self.lgbm_probs[idx],
            "label":          self.labels[idx],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Model — identical architecture to arch4_v1 MeanPoolHybridModel
# ═══════════════════════════════════════════════════════════════════════════════
class MeanPoolHybridModel(nn.Module):
    def __init__(self, model_name, tree_dim=3, num_classes=3,
                 hidden_dim=FUSION_HIDDEN_DIM, dropout=FUSION_DROPOUT):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim  = self.bert.config.hidden_size
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
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.masked_mean_pool(out.last_hidden_state, attention_mask)
        fused  = torch.cat([pooled, tree_probs], dim=1)
        return self.fusion_head(fused)


def _get_encoder_layers(model):
    """Get encoder layer list — handles both BertModel and MegatronBertModel."""
    bert = model.bert
    if hasattr(bert, "encoder") and hasattr(bert.encoder, "layer"):
        return bert.encoder.layer
    raise AttributeError("Cannot find encoder layers in model — check architecture")


def freeze_all_bert(model):
    for param in model.bert.parameters():
        param.requires_grad = False


def unfreeze_top_layers(model, n_layers):
    freeze_all_bert(model)
    layers = _get_encoder_layers(model)
    for layer in layers[len(layers) - n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ═══════════════════════════════════════════════════════════════════════════════
# Training & evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0
    preds, labels = [], []
    optimizer.zero_grad()

    for step, batch in enumerate(loader, start=1):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        tp   = batch["lgbm_probs"].to(DEVICE)
        tgt  = batch["label"].to(DEVICE)

        with torch.amp.autocast(device_type=AUTOCAST_DEVICE, dtype=AMP_DTYPE,
                                enabled=torch.cuda.is_available()):
            logits = model(ids, mask, tp)
            loss   = criterion(logits, tgt) / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if step % ACCUMULATION_STEPS == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUMULATION_STEPS
        preds.extend(logits.detach().argmax(1).cpu().numpy())
        labels.extend(tgt.detach().cpu().numpy())

    return {
        "loss":     total_loss / len(loader),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            tp   = batch["lgbm_probs"].to(DEVICE)
            tgt  = batch["label"].to(DEVICE)
            with torch.amp.autocast(device_type=AUTOCAST_DEVICE, dtype=AMP_DTYPE,
                                    enabled=torch.cuda.is_available()):
                logits = model(ids, mask, tp)
                loss   = criterion(logits, tgt)
            total_loss += loss.item()
            preds.extend(logits.detach().argmax(1).cpu().numpy())
            labels.extend(tgt.detach().cpu().numpy())

    report = classification_report(labels, preds, labels=[0, 1, 2], output_dict=True, zero_division=0)
    return {
        "loss":        total_loss / len(loader),
        "macro_f1":    f1_score(labels, preds, average="macro"),
        "critical_f1": float(report["0"]["f1-score"]),
        "preds":       np.array(preds),
        "labels":      np.array(labels),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    set_seed(SEED)
    wall_t0 = time.time()

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load data ──────────────────────────────────────────────────────────────
    data = load_splits_from_s3()
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    # ── LightGBM baseline ──────────────────────────────────────────────────────
    lgbm_probs_train, lgbm_probs_val, lgbm_probs_test, lgbm_test_report, _ = \
        build_lgbm_baseline(
            data["X_train_struct"], y_train,
            data["X_val_struct"],   y_val,
            data["X_test_struct"],  y_test,
        )

    # ── Tokenizer & dataloaders ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_texts = data["X_train"]["triage_text"].tolist()
    val_texts   = data["X_val"]["triage_text"].tolist()
    test_texts  = data["X_test"]["triage_text"].tolist()

    train_loader = DataLoader(
        TriageDataset(train_texts, lgbm_probs_train, y_train, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        TriageDataset(val_texts, lgbm_probs_val, y_val, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        TriageDataset(test_texts, lgbm_probs_test, y_test, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = MeanPoolHybridModel(MODEL_NAME).to(DEVICE)
    # Enable gradient checkpointing to save VRAM (GatorTron is 3x BioClinicalBERT)
    if hasattr(model.bert, "gradient_checkpointing_enable"):
        model.bert.gradient_checkpointing_enable()
    freeze_all_bert(model)

    bert_dim = model.bert.config.hidden_size
    total_p, train_p = count_parameters(model)
    n_enc_layers = len(_get_encoder_layers(model))
    print(f"\nModel: {MODEL_NAME}")
    print(f"  hidden_size={bert_dim}, encoder_layers={n_enc_layers}")
    print(f"  Total params: {total_p:,} | Trainable: {train_p:,}")

    # ── Class-weighted loss ────────────────────────────────────────────────────
    class_wts = compute_class_weights(y_train, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_wts)

    # ── Two-phase training ─────────────────────────────────────────────────────
    best_val_f1  = -1.0
    bad_epochs   = 0
    global_epoch = 0
    best_path    = "/tmp/best_gatortron.pt"
    stop         = False

    phases = [
        ("head_warmup",                    HEAD_WARMUP_EPOCHS),
        (f"top{UNFREEZE_LAYERS}_finetune", FINETUNE_EPOCHS),
    ]

    for phase_name, phase_epochs in phases:
        if phase_name == "head_warmup":
            freeze_all_bert(model)
            optimizer = AdamW([
                {"params": model.fusion_head.parameters(), "lr": HEAD_LR, "weight_decay": 0.01}
            ])
        else:
            unfreeze_top_layers(model, UNFREEZE_LAYERS)
            layers   = _get_encoder_layers(model)
            selected = list(range(len(layers) - UNFREEZE_LAYERS, len(layers)))
            param_groups = [
                {"params": model.fusion_head.parameters(), "lr": FUSION_LR, "weight_decay": 0.01}
            ]
            for rank, li in enumerate(selected):
                scale = LR_DECAY ** (len(selected) - 1 - rank)
                param_groups.append({
                    "params": layers[li].parameters(),
                    "lr": BERT_BASE_LR * scale, "weight_decay": 0.01,
                })
            optimizer  = AdamW(param_groups)
            bad_epochs = 0

        steps_per_epoch = math.ceil(len(train_loader) / ACCUMULATION_STEPS)
        total_steps     = steps_per_epoch * phase_epochs
        warmup_steps    = max(1, int(0.10 * total_steps))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

        _, trainable = count_parameters(model)
        print(f"\nPhase: {phase_name} | epochs={phase_epochs} | trainable={trainable:,}")

        for _ in range(1, phase_epochs + 1):
            global_epoch += 1
            tr = train_epoch(model, train_loader, optimizer, scheduler, criterion)
            vl = eval_epoch(model, val_loader, criterion)

            improved = vl["macro_f1"] > best_val_f1
            if improved:
                best_val_f1 = vl["macro_f1"]
                bad_epochs  = 0
                torch.save(model.state_dict(), best_path)
                tag = "<- best"
            else:
                bad_epochs += 1
                tag = f"patience {bad_epochs}/{PATIENCE}"

            gap = tr["macro_f1"] - vl["macro_f1"]
            print(
                f"Epoch {global_epoch:02d} | "
                f"train_loss={tr['loss']:.4f} train_f1={tr['macro_f1']:.4f} | "
                f"val_loss={vl['loss']:.4f} val_f1={vl['macro_f1']:.4f} | "
                f"crit_f1={vl['critical_f1']:.4f} gap={gap:.4f} | {tag}"
            )

            if bad_epochs >= PATIENCE:
                print(f"Early stopping in phase '{phase_name}'.")
                stop = True
                break
        if stop:
            break

    # ── Evaluate best model on test set ────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_metrics = eval_epoch(model, test_loader, criterion)

    test_report = classification_report(
        y_test, test_metrics["preds"],
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    print("\n" + "=" * 60)
    print("TEST SET — GatorTron + LightGBM Hybrid")
    print("=" * 60)
    print(classification_report(
        y_test, test_metrics["preds"], target_names=CLASS_NAMES, zero_division=0,
    ))

    wall_time = time.time() - wall_t0
    total_p, _ = count_parameters(model)
    print_comparison_table(
        model_name="GatorTron-base",
        param_count=f"{total_p / 1e6:.0f}M",
        test_report=test_report,
        lgbm_test_report=lgbm_test_report,
        wall_clock_secs=wall_time,
    )


if __name__ == "__main__":
    main()
