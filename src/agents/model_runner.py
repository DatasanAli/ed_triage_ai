"""
model_runner.py
===============
ModelRunner — loads trained model artifacts from S3 and runs
end-to-end inference for a single patient.

Replicates the preprocessing pipeline from arch4_training_v1.ipynb
exactly so that inference is consistent with training.

Artifacts loaded from s3://ed-triage-capstone-group7/models/arch4_v1/:
  best_model.pt        — MeanPoolHybridModel (BioClinicalBERT + fusion head)
  lgbm_fold1-5.joblib  — 5 LightGBM fold models (averaged at inference)
  config.json          — hyperparams + training-set imputation stats

Patient field mapping (agent dict → notebook schema):
  chief_complaint   → chiefcomplaint
  systolic_bp       → sbp
  diastolic_bp      → dbp
  temperature       → temp_f
  hpi               → HPI          (optional, defaults to "")
  pain              → pain         (optional, defaults to training median 5.0)
  arrival_transport → arrival_transport (optional, defaults to "UNKNOWN")
"""

import io
import json
import os
import tempfile

import boto3
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ── S3 config ──────────────────────────────────────────────────────────────────
S3_BUCKET    = "ed-triage-capstone-group7"
MODEL_PREFIX = "models/arch4_v1/"
BERT_MODEL   = "emilyalsentzer/Bio_ClinicalBERT"

# ── Model architecture constants (must match training) ─────────────────────────
NUM_CLASSES       = 3
MAX_LEN           = 384
FUSION_HIDDEN_DIM = 96
FUSION_DROPOUT    = 0.55

# ── Preprocessing constants (must match training) ──────────────────────────────
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
    # Raw vitals (6)
    "heart_rate", "sbp", "dbp", "resp_rate", "spo2", "temp_f",
    # Derived vitals (3)
    "shock_index", "map", "pulse_pressure",
    # Clinical early warning scores (2)
    "news2_score", "mews_score",
    # Demographics (1)
    "age",
    # Transport (1)
    "transport_ordinal",
    # Pain (2)
    "pain", "pain_missing",
]


# ── Clinical scoring functions (copied verbatim from arch4_training_v1.ipynb) ──

def _compute_news2(row: dict) -> float:
    score = 0
    rr = row["resp_rate"]
    if   rr <= 8:  score += 3
    elif rr <= 11: score += 1
    elif rr <= 20: score += 0
    elif rr <= 24: score += 2
    else:          score += 3

    spo2 = row["spo2"]
    if   spo2 <= 91: score += 3
    elif spo2 <= 93: score += 2
    elif spo2 <= 95: score += 1

    sbp = row["sbp"]
    if   sbp <= 90:  score += 3
    elif sbp <= 100: score += 2
    elif sbp <= 110: score += 1
    elif sbp <= 219: score += 0
    else:            score += 3

    hr = row["heart_rate"]
    if   hr <= 40:  score += 3
    elif hr <= 50:  score += 1
    elif hr <= 90:  score += 0
    elif hr <= 110: score += 1
    elif hr <= 130: score += 2
    else:           score += 3

    temp = row["temp_f"]
    if   temp <= 95.0:  score += 3
    elif temp <= 96.8:  score += 1
    elif temp <= 100.4: score += 0
    elif temp <= 102.2: score += 1
    else:               score += 2

    return float(score)


def _compute_mews(row: dict) -> float:
    score = 0
    sbp = row["sbp"]
    if   sbp < 70:  score += 3
    elif sbp < 81:  score += 2
    elif sbp < 101: score += 1
    elif sbp < 200: score += 0
    else:           score += 2

    hr = row["heart_rate"]
    if   hr < 40:  score += 2
    elif hr < 51:  score += 1
    elif hr < 101: score += 0
    elif hr < 111: score += 1
    elif hr < 130: score += 2
    else:          score += 3

    rr = row["resp_rate"]
    if   rr < 9:  score += 2
    elif rr < 15: score += 0
    elif rr < 21: score += 1
    elif rr < 30: score += 2
    else:         score += 3

    temp = row["temp_f"]
    if   temp < 95.0:   score += 2
    elif temp <= 101.1: score += 0
    else:               score += 2

    return float(score)


def _clip_words(text, max_words: int) -> str:
    if not text or (isinstance(text, float)):
        return ""
    text = str(text).replace("\n", " ").strip()
    return " ".join(text.split()[:max_words])


def _build_triage_text(chiefcomplaint: str, hpi: str) -> str:
    """CC-emphasized text format matching training: CC_2x + HPI."""
    cc  = _clip_words(chiefcomplaint, 24)
    hpi = _clip_words(hpi, 160)
    parts = []
    if cc:  parts.append(f"Chief complaint: {cc}.")
    if cc:  parts.append(f"Presenting with {cc}.")   # CC repeated for emphasis
    if hpi: parts.append(f"History: {hpi}.")
    return " ".join(parts)


def _transform_structured(mapped: dict, stats: dict) -> np.ndarray:
    """
    Build the 15 structured features from a mapped patient dict.
    Uses training-set stats for imputation to prevent leakage.

    Args:
        mapped: dict with notebook-schema field names
        stats:  structured_stats from config.json
                {"vital_medians": {...}, "pain_median": float, "age_median": float}

    Returns:
        np.ndarray of shape (15,), dtype float32
    """
    feat = dict(mapped)

    # Impute + clip raw vitals
    for col in RAW_VITALS:
        val = feat.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = stats["vital_medians"][col]
        lo, hi = CLIP_BOUNDS[col]
        feat[col] = float(np.clip(val, lo, hi))

    # Pain — derive pain_missing flag before imputing
    pain_raw = feat.get("pain")
    if pain_raw is None or (isinstance(pain_raw, float) and np.isnan(pain_raw)):
        feat["pain_missing"] = 1.0
        feat["pain"]         = float(stats["pain_median"])
    else:
        feat["pain_missing"] = 0.0
        feat["pain"]         = float(np.clip(pain_raw, 0.0, 10.0))

    # Age
    age_val = feat.get("age")
    if age_val is None:
        age_val = stats["age_median"]
    feat["age"] = float(np.clip(age_val, 18, 120))

    # Transport ordinal
    transport_str = (feat.get("arrival_transport") or "UNKNOWN").strip().upper()
    feat["transport_ordinal"] = float(TRANSPORT_MAP.get(transport_str, 1))

    # Derived vitals
    sbp = feat["sbp"]
    feat["shock_index"]    = feat["heart_rate"] / sbp if sbp > 0 else 0.0
    feat["map"]            = (sbp + 2.0 * feat["dbp"]) / 3.0
    feat["pulse_pressure"] = sbp - feat["dbp"]

    # Clinical early warning scores
    feat["news2_score"] = _compute_news2(feat)
    feat["mews_score"]  = _compute_mews(feat)

    return np.array([feat[f] for f in STRUCTURED_FEATURES], dtype=np.float32)


# ── Model definition (must match arch4_training_v1.ipynb exactly) ─────────────

class _MeanPoolHybridModel(nn.Module):
    """BioClinicalBERT + LightGBM probs fusion model (arch4 v1)."""

    def __init__(self, bert_model_name: str, tree_dim: int = 3,
                 num_classes: int = 3, hidden_dim: int = FUSION_HIDDEN_DIM,
                 dropout: float = FUSION_DROPOUT):
        super().__init__()
        # Silence the LOAD REPORT for cls.predictions.*/cls.seq_relationship.*
        # — those are pre-training heads absent in AutoModel (encoder-only), which is expected.
        import logging
        logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)
        self.bert    = AutoModel.from_pretrained(bert_model_name)
        bert_dim     = self.bert.config.hidden_size  # 768
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(bert_dim + tree_dim),
            nn.Linear(bert_dim + tree_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _masked_mean_pool(self, last_hidden_state, attention_mask):
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, input_ids, attention_mask, tree_probs):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled   = self._masked_mean_pool(bert_out.last_hidden_state, attention_mask)
        fused    = torch.cat([pooled, tree_probs], dim=1)
        return self.fusion_head(fused)


# ── ModelRunner ───────────────────────────────────────────────────────────

class ModelRunner:
    """
    Singleton that loads model artifacts from S3 once per process and
    runs end-to-end inference for a single patient.

    Usage:
        runner = ModelRunner.get()
        result = runner.predict(patient_dict)
        # {"predicted_class": int, "probabilities": [float, float, float]}

    Cold start: ~30–60 s (S3 download + BioClinicalBERT load).
    Subsequent calls: <1 s on CPU, ~100 ms with GPU.
    """

    _instance: "ModelRunner | None" = None

    def __init__(self):
        self._s3     = boto3.client("s3", region_name="us-east-1")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config       = self._load_config()
        self._stats  = {
            "vital_medians": config["structured_stats"]["vital_medians"],
            "pain_median":   config["structured_stats"]["pain_median"],
            "age_median":    config["structured_stats"]["age_median"],
        }

        self._lgbm_models = self._load_lgbm_folds()
        self._tokenizer   = AutoTokenizer.from_pretrained(BERT_MODEL)
        self._model       = self._load_bert_model(config)

    @classmethod
    def get(cls) -> "ModelRunner":
        """Return the module-level singleton, initializing on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Private loaders ────────────────────────────────────────────────────────

    def _s3_bytes(self, key: str) -> io.BytesIO:
        obj = self._s3.get_object(Bucket=S3_BUCKET, Key=key)
        return io.BytesIO(obj["Body"].read())

    def _load_config(self) -> dict:
        buf = self._s3_bytes(f"{MODEL_PREFIX}config.json")
        return json.loads(buf.read().decode("utf-8"))

    def _load_lgbm_folds(self) -> list:
        """Load all 5 LightGBM fold models via temp files (joblib requirement)."""
        models = []
        for fold in range(1, 6):
            buf = self._s3_bytes(f"{MODEL_PREFIX}lgbm_fold{fold}.joblib")
            # joblib.load requires a real file path, not a BytesIO object
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                tmp.write(buf.read())
                tmp_path = tmp.name
            try:
                models.append(joblib.load(tmp_path))
            finally:
                os.unlink(tmp_path)
        return models

    def _load_bert_model(self, config: dict) -> _MeanPoolHybridModel:
        model = _MeanPoolHybridModel(
            bert_model_name = BERT_MODEL,
            tree_dim        = NUM_CLASSES,
            num_classes     = NUM_CLASSES,
            hidden_dim      = config.get("fusion_hidden_dim", FUSION_HIDDEN_DIM),
            dropout         = config.get("fusion_dropout",    FUSION_DROPOUT),
        ).to(self._device)

        buf        = self._s3_bytes(f"{MODEL_PREFIX}best_model.pt")
        state_dict = torch.load(buf, map_location=self._device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    # ── Field mapping ──────────────────────────────────────────────────────────

    @staticmethod
    def _map_patient_fields(patient: dict) -> dict:
        """
        Translate agent patient dict keys to the notebook's column names.
        Optional fields default to values that trigger median imputation
        or neutral encoding in _transform_structured.
        """
        return {
            # Required vitals — renamed to match notebook schema
            "chiefcomplaint":   patient.get("chief_complaint", ""),
            "sbp":              patient.get("systolic_bp"),
            "dbp":              patient.get("diastolic_bp"),
            "temp_f":           patient.get("temperature"),
            "heart_rate":       patient.get("heart_rate"),
            "resp_rate":        patient.get("resp_rate"),
            "spo2":             patient.get("spo2"),
            "age":              patient.get("age"),
            # Optional fields — None triggers imputation in _transform_structured
            "pain":             patient.get("pain"),
            "arrival_transport": patient.get("arrival_transport", "UNKNOWN"),
            "HPI":              patient.get("hpi", ""),
        }

    # ── Public inference API ───────────────────────────────────────────────────

    def predict(self, patient: dict) -> dict:
        """
        Run end-to-end inference for a single patient.

        Args:
            patient: TriageState patient dict. Required keys: chief_complaint,
                     systolic_bp, diastolic_bp, temperature, heart_rate,
                     resp_rate, spo2, age. Optional: pain, arrival_transport, hpi.

        Returns:
            {"predicted_class": int, "probabilities": [float, float, float]}
            — shape expected by ModelPrediction.normalize() and predict_node.
        """
        mapped = self._map_patient_fields(patient)

        # ── Structured features ───────────────────────────────────────────────
        struct_feats = _transform_structured(mapped, self._stats)  # (15,)
        # Wrap in a DataFrame so LightGBM receives the same named columns it
        # was trained on — avoids the "no valid feature names" warning and
        # ensures positional order can never silently diverge from training.
        struct_df = pd.DataFrame(
            struct_feats.reshape(1, -1), columns=STRUCTURED_FEATURES
        )

        # ── LightGBM ensemble: average predict_proba across all 5 folds ───────
        fold_probs   = np.stack(
            [m.predict_proba(struct_df) for m in self._lgbm_models], axis=0
        )                                    # (5, 1, 3)
        lgbm_probs   = fold_probs.mean(axis=0)  # (1, 3)

        # ── Text construction + tokenization ─────────────────────────────────
        text = _build_triage_text(mapped["chiefcomplaint"], mapped["HPI"])
        enc  = self._tokenizer(
            text,
            max_length  = MAX_LEN,
            padding     = "max_length",
            truncation  = True,
            return_tensors = "pt",
        )

        # ── BioClinicalBERT + fusion forward pass ─────────────────────────────
        input_ids      = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)
        tree_probs_t   = torch.tensor(
            lgbm_probs, dtype=torch.float32
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(input_ids, attention_mask, tree_probs_t)
            probs  = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]

        pred_class = int(probs.argmax())
        return {
            "predicted_class": pred_class,
            "probabilities":   [float(p) for p in probs],
        }
