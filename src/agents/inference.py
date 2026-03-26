"""
inference.py
============
Model Prediction — a stateless normalizer for model output.

This module does NOT load or run the model. The model (BioClinicalBERT +
LightGBM fusion) runs separately (on SageMaker or locally) and its output is
passed into the graph. This wrapper normalizes whatever shape the caller
provides into the canonical dict that predict_node and downstream nodes expect.

Keeping model loading out of the agent graph means:
  - The graph is testable offline with no GPU or S3 access
  - Model inference can be swapped (SageMaker endpoint, local, mock) without
    touching the graph logic
  - Cold-start latency on graph import is zero

Accepted input shapes:
  1. Full dict:   {"predicted_class": 2, "probabilities": [0.12, 0.78, 0.10]}
  2. Probs-only:  {"probabilities": [0.12, 0.78, 0.10]}
  3. Raw list:    [0.12, 0.78, 0.10]

Class index mapping (from model output, target_map):
  0 → L1-Critical
  1 → L2-Emergent
  2 → L3-Urgent/LessUrgent   (L4 collapsed into L3 during training)
"""

LABEL_MAP = {
    0: "L1-Critical",
    1: "L2-Emergent",
    2: "L3-Urgent",
}

# Confidence thresholds below which uncertainty_flag is set True.
# L1-Critical uses a stricter threshold because a missed critical is a
# patient-safety issue — the nurse needs to know the model is uncertain.
CONFIDENCE_THRESHOLDS = {
    0: 0.65,   # L1-Critical — stricter
    1: 0.55,   # L2-Emergent
    2: 0.55,   # L3-Urgent
}


class ModelPrediction:
    """
    Stateless normalizer for model output.
    All methods are @staticmethod — no instantiation needed.

    Usage:
        normalized = ModelPrediction.normalize(raw_prediction)
        # normalized is always the canonical dict shape
    """

    @staticmethod
    def from_probs(probs: list) -> dict:
        """
        Build canonical dict from a raw probability list.
        predicted_class is inferred as argmax.
        """
        if len(probs) != 3:
            raise ValueError(f"Expected 3 class probabilities, got {len(probs)}")
        total = sum(probs)
        if total <= 0:
            raise ValueError("Probability list sums to zero — invalid model output")

        pred_class = int(max(range(3), key=lambda i: probs[i]))
        confidence = float(probs[pred_class])
        threshold  = CONFIDENCE_THRESHOLDS[pred_class]

        return {
            "predicted_class":  pred_class,
            "predicted_label":  LABEL_MAP[pred_class],
            "probabilities":    [float(p) for p in probs],
            "confidence":       round(confidence, 4),
            "uncertainty_flag": confidence < threshold,
            "prob_breakdown": {
                "L1-Critical": f"{probs[0]:.0%}",
                "L2-Emergent": f"{probs[1]:.0%}",
                "L3-Urgent":   f"{probs[2]:.0%}",
            },
        }

    @staticmethod
    def from_dict(prediction: dict) -> dict:
        """
        Normalize a prediction dict that already contains probabilities.
        If predicted_class is missing, infers it from argmax.
        """
        probs = prediction.get("probabilities")
        if probs is None:
            raise ValueError("Prediction dict must contain 'probabilities' key")

        base = ModelPrediction.from_probs(probs)

        # Honour an explicitly supplied predicted_class (e.g. from threshold tuning)
        # only if it's a valid index
        if "predicted_class" in prediction:
            explicit_class = int(prediction["predicted_class"])
            if explicit_class not in LABEL_MAP:
                raise ValueError(f"predicted_class {explicit_class} not in {list(LABEL_MAP)}")
            # Recompute label and uncertainty with the explicit class
            base["predicted_class"]  = explicit_class
            base["predicted_label"]  = LABEL_MAP[explicit_class]
            conf = float(probs[explicit_class])
            base["confidence"]       = round(conf, 4)
            base["uncertainty_flag"] = conf < CONFIDENCE_THRESHOLDS[explicit_class]

        return base

    @staticmethod
    def normalize(prediction) -> dict:
        """
        Auto-detect input type and return the canonical normalized dict.

        Accepts:
          list  → treated as raw probability array
          dict  → must contain "probabilities" key
        """
        if isinstance(prediction, list):
            return ModelPrediction.from_probs(prediction)
        if isinstance(prediction, dict):
            return ModelPrediction.from_dict(prediction)
        raise TypeError(
            f"prediction must be list or dict, got {type(prediction).__name__}"
        )
