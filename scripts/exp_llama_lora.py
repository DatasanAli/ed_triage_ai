"""
exp_llama_lora.py — Llama-3.1-8B + LoRA fine-tuning experiment.

Model:  meta-llama/Llama-3.1-8B + LoRA (rank=16, alpha=32), 4-bit base via bitsandbytes
Why:    Tests whether a general-purpose LLM fine-tuned with LoRA on our triage task
        can learn task-specific text representations that outperform domain-pretrained models.

Method — Two sub-experiments:
  (a) LoRA fine-tune for classification -> extract adapted embeddings ->
      fuse with LightGBM probs -> train fusion head
  (b) Direct LoRA classification: Llama predicts triage from text alone,
      then simple ensemble with LightGBM probs (weighted average, alpha tuned on val)

Run:    python exp_llama_lora.py      (on SageMaker ml.g4dn.2xlarge)
"""

import math
import os
import subprocess
import sys
import time

# Redirect HF cache to instance-store (/tmp) before any HF import.
# On SageMaker the home EBS volume is ~20 GB — not enough for 8B model shards.
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

# `hf auth login` saves the token to ~/.cache/huggingface/token (the default HF_HOME).
# With a custom HF_HOME above, that path is no longer checked — bridge the gap here.
_default_token = os.path.expanduser("~/.cache/huggingface/token")
if "HF_TOKEN" not in os.environ and os.path.exists(_default_token):
    with open(_default_token) as _f:
        os.environ["HF_TOKEN"] = _f.read().strip()

# ── Install dependencies ──────────────────────────────────────────────────────
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "boto3", "transformers", "lightgbm", "scikit-learn", "joblib",
    "torch", "numpy==1.26.4", "pandas",
    "bitsandbytes", "accelerate", "peft", "sentencepiece",
])

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from exp_utils import (
    CLASS_NAMES, NUM_CLASSES, SEED,
    build_lgbm_baseline, compute_class_weights, load_splits_from_s3,
    print_comparison_table, set_seed, train_fusion_head,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_NAME         = "meta-llama/Llama-3.1-8B"
MAX_LEN            = 384
BATCH_SIZE         = 4
ACCUMULATION_STEPS = 8     # effective batch size = 32
LORA_RANK          = 16
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.1
LORA_EPOCHS        = 5
LORA_LR            = 2e-4
PATIENCE           = 2
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE          = torch.float16


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════
class TriageTextDataset(Dataset):
    """Tokenized triage text + labels for LoRA classification."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len   = max_len

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
            "label":          self.labels[idx],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Model wrappers
# ═══════════════════════════════════════════════════════════════════════════════
class LlamaForClassification(nn.Module):
    """
    Wraps a Llama base model with a classification head on mean-pooled hidden states.
    Used instead of AutoModelForSequenceClassification to get clean control over
    pooling and to enable LoRA on the base model only.
    """

    def __init__(self, base_model, hidden_size, num_classes=3, dropout=0.3):
        super().__init__()
        self.base   = base_model
        self.drop   = nn.Dropout(dropout)
        self.head   = nn.Linear(hidden_size, num_classes)

    def masked_mean_pool(self, hidden_states, attention_mask):
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        out    = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.masked_mean_pool(out.last_hidden_state.float(), attention_mask)
        return self.head(self.drop(pooled))

    def extract_embeddings(self, input_ids, attention_mask):
        """Return mean-pooled embeddings (no classification head)."""
        out    = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.masked_mean_pool(out.last_hidden_state.float(), attention_mask)
        return pooled


def load_base_model_4bit(model_name):
    """Load Llama in 4-bit quantization."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )
    return base


def apply_lora(base_model):
    """Apply LoRA adapters to attention projection layers."""
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(base_model, lora_config)


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA fine-tuning loop
# ═══════════════════════════════════════════════════════════════════════════════
def train_lora_classifier(model, train_loader, val_loader, y_val, criterion,
                          epochs=LORA_EPOCHS, patience=PATIENCE):
    """
    Fine-tune LoRA + classification head. BERT base stays frozen (4-bit),
    only LoRA adapters + head train.
    """
    # Only optimize trainable parameters (LoRA + head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LORA_LR, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(train_loader) / ACCUMULATION_STEPS)
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = max(1, int(0.10 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_f1 = -1.0
    bad_epochs  = 0
    best_state  = None

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\nLoRA fine-tuning: {n_train:,} trainable / {n_total:,} total params")
    print(f"  epochs={epochs}, lr={LORA_LR}, accum={ACCUMULATION_STEPS}, "
          f"eff_batch={BATCH_SIZE*ACCUMULATION_STEPS}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            tgt  = batch["label"].to(DEVICE)

            with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE,
                                    enabled=torch.cuda.is_available()):
                logits = model(ids, mask)
                loss   = criterion(logits, tgt) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if step % ACCUMULATION_STEPS == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS

        # Validate
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE,
                                        enabled=torch.cuda.is_available()):
                    logits = model(ids, mask)
                val_preds.extend(logits.float().argmax(1).cpu().numpy())

        val_f1 = f1_score(y_val, val_preds, average="macro")
        improved = val_f1 > best_val_f1

        if improved:
            best_val_f1 = val_f1
            bad_epochs  = 0
            # Save only trainable state (LoRA + head) to avoid serializing 4-bit base
            # Note: state_dict() tensors are always detached, so requires_grad is always False — filter by key name only
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
                if "lora" in k.lower() or "head" in k.lower()
            }
            tag = "<- best"
        else:
            bad_epochs += 1
            tag = f"patience {bad_epochs}/{patience}"

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch} | loss={avg_loss:.4f} | val_f1={val_f1:.4f} | {tag}")

        if bad_epochs >= patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    # Reload best weights
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    return best_val_f1


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding extraction from LoRA-adapted model
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_lora_embeddings(model, dataloader, device):
    """Extract mean-pooled embeddings from the LoRA-adapted model."""
    model.eval()
    all_embs = []
    for i, batch in enumerate(dataloader):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE,
                                enabled=torch.cuda.is_available()):
            emb = model.extract_embeddings(ids, mask)
        all_embs.append(emb.cpu().numpy())
        if (i + 1) % 100 == 0:
            print(f"    Batch {i+1}/{len(dataloader)}")
    return np.vstack(all_embs)


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-experiment (b): Direct classification + ensemble with LightGBM
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_probs(model, dataloader, device):
    """Get softmax probabilities from the LoRA classifier."""
    model.eval()
    all_probs = []
    for batch in dataloader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE,
                                enabled=torch.cuda.is_available()):
            logits = model(ids, mask)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def find_best_ensemble_alpha(llm_probs_val, lgbm_probs_val, y_val):
    """
    Grid search for best alpha: ensemble = alpha * llm_probs + (1-alpha) * lgbm_probs.
    Returns best alpha and corresponding macro-F1.
    """
    best_alpha = 0.5
    best_f1    = -1.0

    for alpha in np.arange(0.0, 1.01, 0.05):
        ensemble = alpha * llm_probs_val + (1 - alpha) * lgbm_probs_val
        preds = ensemble.argmax(axis=1)
        f1 = f1_score(y_val, preds, average="macro")
        if f1 > best_f1:
            best_f1    = f1
            best_alpha = alpha

    print(f"  Best ensemble alpha={best_alpha:.2f} (val macro-F1={best_f1:.4f})")
    return best_alpha, best_f1


def tune_l1_threshold(probs_val, y_val):
    """
    Lower the L1-Critical decision threshold to recover recall on the minority class.
    The alpha grid search maximises overall macro-F1 but can still collapse L1 to zero
    (as seen in the mellama frozen-embedding run). This sweep finds the threshold on
    prob[:,0] that maximises macro-F1 on the val set.
    Returns best threshold and corresponding val macro-F1.
    """
    base_preds = probs_val.argmax(axis=1)
    best_thresh = None          # None → plain argmax
    best_f1     = f1_score(y_val, base_preds, average="macro")

    for t in np.arange(0.10, 0.50, 0.02):
        preds = probs_val.argmax(axis=1).copy()
        preds[probs_val[:, 0] > t] = 0        # override: call L1 if prob[0] > t
        f1 = f1_score(y_val, preds, average="macro")
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = t

    if best_thresh is not None:
        print(f"  L1 threshold tuned: t={best_thresh:.2f} (val macro-F1={best_f1:.4f})")
    else:
        print(f"  L1 threshold tuning: plain argmax already optimal (val macro-F1={best_f1:.4f})")
    return best_thresh, best_f1


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    set_seed(SEED)
    wall_t0 = time.time()

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")

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

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_texts = data["X_train"]["triage_text"].tolist()
    val_texts   = data["X_val"]["triage_text"].tolist()
    test_texts  = data["X_test"]["triage_text"].tolist()

    train_ds = TriageTextDataset(train_texts, y_train, tokenizer, MAX_LEN)
    val_ds   = TriageTextDataset(val_texts,   y_val,   tokenizer, MAX_LEN)
    test_ds  = TriageTextDataset(test_texts,  y_test,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ══════════════════════════════════════════════════════════════════════════
    # Load 4-bit Llama + LoRA
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nLoading {MODEL_NAME} in 4-bit...")
    base_model = load_base_model_4bit(MODEL_NAME)
    hidden_dim = base_model.config.hidden_size

    print("Applying LoRA adapters...")
    lora_base = apply_lora(base_model)
    lora_base.print_trainable_parameters()

    # Wrap with classification head — move only new layers; lora_base is already on CUDA via device_map="auto"
    model = LlamaForClassification(lora_base, hidden_dim, NUM_CLASSES)
    model.head = model.head.to(DEVICE)
    model.drop = model.drop.to(DEVICE)

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after model load: {alloc:.1f} GB")

    # ══════════════════════════════════════════════════════════════════════════
    # LoRA fine-tuning
    # ══════════════════════════════════════════════════════════════════════════
    class_wts = compute_class_weights(y_train, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_wts)

    lora_t0 = time.time()
    best_val_f1 = train_lora_classifier(
        model, train_loader, val_loader, y_val, criterion,
        epochs=LORA_EPOCHS, patience=PATIENCE,
    )
    lora_time = time.time() - lora_t0
    print(f"LoRA fine-tuning time: {lora_time:.0f}s ({lora_time/60:.1f} min)")

    # ══════════════════════════════════════════════════════════════════════════
    # Sub-experiment (b): Direct LoRA classification + LightGBM ensemble
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Sub-experiment (b): Direct LoRA classification + LightGBM ensemble")
    print("=" * 60)

    # Get LoRA classifier probabilities
    llm_probs_val  = predict_probs(model, val_loader,  DEVICE)
    llm_probs_test = predict_probs(model, test_loader, DEVICE)

    # Direct LoRA-only performance
    lora_preds_test = llm_probs_test.argmax(axis=1)
    lora_only_f1 = f1_score(y_test, lora_preds_test, average="macro")
    print(f"\nLoRA-only test macro-F1: {lora_only_f1:.4f}")
    print(classification_report(y_test, lora_preds_test, target_names=CLASS_NAMES, zero_division=0))

    # Find best ensemble alpha on val set
    print("Tuning ensemble alpha on val set...")
    best_alpha, _ = find_best_ensemble_alpha(llm_probs_val, lgbm_probs_val, y_val)

    # Ensemble on test set
    ensemble_probs = best_alpha * llm_probs_test + (1 - best_alpha) * lgbm_probs_test

    # Tune L1-Critical threshold on val to avoid minority-class collapse
    # (mellama frozen run: L1 F1=0.00 despite class weights, fixed by lowering threshold)
    ensemble_val_probs = best_alpha * llm_probs_val + (1 - best_alpha) * lgbm_probs_val
    print("Tuning L1-Critical threshold on val set...")
    best_l1_thresh, _ = tune_l1_threshold(ensemble_val_probs, y_val)

    if best_l1_thresh is not None:
        ensemble_preds = ensemble_probs.argmax(axis=1)
        ensemble_preds[ensemble_probs[:, 0] > best_l1_thresh] = 0
    else:
        ensemble_preds = ensemble_probs.argmax(axis=1)

    ensemble_test_report = classification_report(
        y_test, ensemble_preds,
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    ensemble_f1 = f1_score(y_test, ensemble_preds, average="macro")
    print(f"\nEnsemble (alpha={best_alpha:.2f}, l1_thresh={best_l1_thresh}) test macro-F1: {ensemble_f1:.4f}")
    print(classification_report(y_test, ensemble_preds, target_names=CLASS_NAMES, zero_division=0))

    print_comparison_table(
        model_name=f"Llama-3.1-8B + LoRA (direct ensemble, alpha={best_alpha:.2f})",
        param_count="8B (4-bit) + LoRA",
        test_report=ensemble_test_report,
        lgbm_test_report=lgbm_test_report,
        wall_clock_secs=None,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Sub-experiment (a): Extract LoRA-adapted embeddings + fusion head
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Sub-experiment (a): LoRA-adapted embeddings + fusion head")
    print("=" * 60)

    # Use non-shuffled loader for deterministic embedding extraction
    train_loader_noshuffle = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    embed_t0 = time.time()
    print("\nExtracting LoRA-adapted embeddings...")
    print(f"  Train ({len(train_texts)} samples)...")
    train_embs = extract_lora_embeddings(model, train_loader_noshuffle, DEVICE)
    print(f"  Val ({len(val_texts)} samples)...")
    val_embs   = extract_lora_embeddings(model, val_loader, DEVICE)
    print(f"  Test ({len(test_texts)} samples)...")
    test_embs  = extract_lora_embeddings(model, test_loader, DEVICE)
    embed_time = time.time() - embed_t0
    print(f"  Embedding extraction: {embed_time:.0f}s")
    print(f"  Shapes: train={train_embs.shape}, val={val_embs.shape}, test={test_embs.shape}")

    # Free LLM from GPU
    del model, base_model, lora_base
    torch.cuda.empty_cache()

    # Train fusion head
    head_t0 = time.time()
    # dropout=0.3 (not 0.55): high dropout on 96 hidden units was a contributing factor
    # to L1-Critical collapse in the mellama frozen-embedding run. LoRA embeddings are
    # task-adapted so the head has real signal to work with — don't regularise it away.
    test_report_a, val_report_a, best_val_f1_a = train_fusion_head(
        train_embs, lgbm_probs_train, y_train,
        val_embs,   lgbm_probs_val,   y_val,
        test_embs,  lgbm_probs_test,  y_test,
        embedding_dim=hidden_dim,
        hidden_dim=96,
        dropout=0.3,
        lr=1e-3,
        epochs=60,
        patience=8,
        batch_size=128,
    )
    head_time = time.time() - head_t0

    wall_time = time.time() - wall_t0

    print_comparison_table(
        model_name="Llama-3.1-8B + LoRA (adapted embeddings + fusion head)",
        param_count="8B (4-bit) + LoRA",
        test_report=test_report_a,
        lgbm_test_report=lgbm_test_report,
        wall_clock_secs=None,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — Llama-3.1-8B + LoRA")
    print("=" * 60)
    print(f"Total wall-clock time: {wall_time:.0f}s ({wall_time/60:.1f} min)")
    print(f"\nTiming breakdown:")
    print(f"  LoRA fine-tuning:     {lora_time:.0f}s ({lora_time/60:.1f} min)")
    print(f"  Embedding extraction: {embed_time:.0f}s ({embed_time/60:.1f} min)")
    print(f"  Fusion head training: {head_time:.0f}s")

    # Extract macro-F1 for both sub-experiments
    f1_a = test_report_a["macro avg"]["f1-score"]
    f1_b = ensemble_f1
    print(f"\n  (a) LoRA embeddings + fusion head:  test macro-F1 = {f1_a:.4f}")
    print(f"  (b) Direct LoRA + LightGBM ensemble: test macro-F1 = {f1_b:.4f}")

    bcb_f1 = 0.67
    best_f1 = max(f1_a, f1_b)
    best_label = "(a) fusion" if f1_a >= f1_b else "(b) ensemble"
    print(f"\n  Best: {best_label} = {best_f1:.4f}  "
          f"(vs BioClinicalBERT {bcb_f1:.4f}, delta={best_f1 - bcb_f1:+.4f})")


if __name__ == "__main__":
    main()
