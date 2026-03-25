"""
Inference handler for the arch4 triage model (BioClinicalBERT + LightGBM fusion).

SageMaker's PyTorch serving container calls these four functions in order:
  model_fn   -> load BERT model, LightGBM folds, tokenizer, config
  input_fn   -> parse the incoming JSON request
  predict_fn -> feature engineering, LightGBM ensemble, BERT forward pass, fusion
  output_fn  -> format the JSON response

The MeanPoolHybridModel class and all constants are duplicated here (not imported
from train.py) because the serving container runs this file standalone -- the
training source_dir is not available at inference time, only the model archive is.

Request payload format:
  {
    "triage_text": "Chief complaint: ... Presenting with ... History: ...",
    "heart_rate": 110,       # optional -- all vitals imputed if missing
    "sbp": 90,
    "dbp": 60,
    "resp_rate": 22,
    "spo2": 94,
    "temp_f": 98.6,
    "age": 55,
    "arrival_transport": "AMBULANCE",
    "pain": 8               # omit to auto-set pain_missing=1
  }
"""

import json
import os

import joblib
import numpy as np
import shap
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ── Constants (duplicated from train.py) ──────────────────────────────────────

NUM_CLASSES = 3
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


# ── Clinical scoring (duplicated from train.py) ──────────────────────────────

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


# ── Model (duplicated from train.py) ─────────────────────────────────────────

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


# ── SageMaker handler functions ──────────────────────────────────────────────

def model_fn(model_dir):
    """Load model artifacts from the SageMaker model directory."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    bert_model_name = config.get("bert_model", "emilyalsentzer/Bio_ClinicalBERT")
    hp = config.get("hyperparameters", {})
    max_len = hp.get("max_len", 128)
    fusion_hidden_dim = hp.get("fusion_hidden_dim", 96)
    fusion_dropout = hp.get("fusion_dropout", 0.55)

    # Instantiate model and load trained weights
    model = MeanPoolHybridModel(
        bert_model_name,
        tree_dim=NUM_CLASSES,
        num_classes=NUM_CLASSES,
        hidden_dim=fusion_hidden_dim,
        dropout=fusion_dropout,
    )
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "best_model.pt"),
        map_location=torch.device("cpu"),
    ))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    # Load 5-fold LightGBM models and build one SHAP TreeExplainer per fold
    lgbm_models = []
    shap_explainers = []
    for i in range(1, 6):
        path = os.path.join(model_dir, f"lgbm_fold{i}.joblib")
        m = joblib.load(path)
        lgbm_models.append(m)
        shap_explainers.append(shap.TreeExplainer(m))
    print(f"Loaded {len(lgbm_models)} LightGBM fold models + SHAP explainers")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "max_len": max_len,
        "lgbm_models": lgbm_models,
        "shap_explainers": shap_explainers,
        "config": config,
    }


def input_fn(request_body, content_type):
    """Parse JSON request body."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    return json.loads(request_body)


def predict_fn(input_data, model_dict):
    """Run inference: feature engineering -> LightGBM ensemble -> BERT -> fusion."""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    max_len = model_dict["max_len"]
    lgbm_models = model_dict["lgbm_models"]
    shap_explainers = model_dict["shap_explainers"]
    config = model_dict["config"]

    stats = config.get("structured_stats", {})
    vital_medians = stats.get("vital_medians", {})
    pain_median = stats.get("pain_median", 5.0)
    age_median = stats.get("age_median", 55.0)

    # ── 1. Extract and impute structured features ────────────────────────────
    row = {}
    for vital in RAW_VITALS:
        val = input_data.get(vital)
        if val is None:
            val = vital_medians.get(vital, 0.0)
        lo, hi = CLIP_BOUNDS[vital]
        row[vital] = max(lo, min(hi, float(val)))

    # Pain: auto-detect missing
    pain_val = input_data.get("pain")
    if pain_val is None:
        row["pain"] = pain_median
        row["pain_missing"] = 1
    else:
        row["pain"] = max(0.0, min(10.0, float(pain_val)))
        row["pain_missing"] = int(input_data.get("pain_missing", 0))

    # Age
    age_val = input_data.get("age")
    if age_val is None:
        row["age"] = age_median
    else:
        row["age"] = max(18.0, min(120.0, float(age_val)))

    # Transport
    transport = input_data.get("arrival_transport", "UNKNOWN")
    row["transport_ordinal"] = TRANSPORT_MAP.get(transport.upper(), 1)

    # Derived features
    sbp = row["sbp"]
    row["shock_index"] = row["heart_rate"] / sbp if sbp > 0 else 0.0
    row["map"] = (sbp + 2.0 * row["dbp"]) / 3.0
    row["pulse_pressure"] = sbp - row["dbp"]

    # Clinical scores
    row["news2_score"] = compute_news2(row)
    row["mews_score"] = compute_mews(row)

    # Assemble feature vector in the exact order used during training
    features = np.array(
        [[row[f] for f in STRUCTURED_FEATURES]],
        dtype=np.float32,
    )

    # ── 2. LightGBM ensemble + SHAP ─────────────────────────────────────────
    fold_probs = [m.predict_proba(features) for m in lgbm_models]
    lgbm_avg = np.mean(fold_probs, axis=0)  # shape (1, 3)

    # SHAP values: each explainer returns (n_samples, n_features, n_classes)
    # Average across folds, then extract the single sample (index 0)
    fold_shap = [e.shap_values(features) for e in shap_explainers]  # list of (1, 15, 3)
    avg_shap = np.mean(np.stack(fold_shap, axis=0), axis=0)  # (1, 15, 3)
    shap_per_class = avg_shap[0]  # (15, 3) — one row per feature, one col per class

    lgbm_shap = {
        CLASS_NAMES[c]: {
            STRUCTURED_FEATURES[f]: round(float(shap_per_class[f, c]), 5)
            for f in range(len(STRUCTURED_FEATURES))
        }
        for c in range(NUM_CLASSES)
    }

    # ── 3. Tokenize text ────────────────────────────────────────────────────
    text = input_data.get("triage_text", "")
    enc = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # ── 4. Fusion forward pass ──────────────────────────────────────────────
    tree_probs = torch.tensor(lgbm_avg, dtype=torch.float32)

    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"], tree_probs)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    predicted_class = int(probs.argmax())
    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "probabilities": {
            name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)
        },
        "lgbm_shap": lgbm_shap,
    }


def output_fn(prediction, accept):
    """Return JSON response."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction), "application/json"
