"""
Inference handler for the mock triage model.

SageMaker's PyTorch serving container calls these four functions in order:
  model_fn  → load model, tokenizer, config from the model archive
  input_fn  → parse the incoming JSON request
  predict_fn → tokenize text, run forward pass, return probabilities
  output_fn  → format the JSON response

The MockTriageModel class is duplicated here (not imported from train.py)
because the serving container runs this file standalone — the training
source_dir is not available at inference time, only the model archive is.
"""

import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

NUM_CLASSES = 3
CLASS_NAMES = ["L1-Critical", "L2-Emergent", "L3-Urgent/LessUrgent"]


class MockTriageModel(nn.Module):
    """bert-tiny → mean pool → Linear.  Must match the class used during training."""

    def __init__(self, bert_model_name, num_classes=NUM_CLASSES):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size
        self.head = nn.Linear(bert_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.head(pooled)


def model_fn(model_dir):
    """Load model artifacts from the SageMaker model directory."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    bert_model_name = config.get("bert_model", "prajjwal1/bert-tiny")
    max_len = config.get("hyperparameters", {}).get("max_len", 128)

    model = MockTriageModel(bert_model_name, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "model.pt"),
        map_location=torch.device("cpu"),
    ))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    return {"model": model, "tokenizer": tokenizer, "max_len": max_len}


def input_fn(request_body, content_type):
    """Parse JSON request body."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    return json.loads(request_body)


def predict_fn(input_data, model_dict):
    """Run inference on the parsed input."""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    max_len = model_dict["max_len"]

    text = input_data.get("triage_text", "")
    enc = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    predicted_class = int(probs.argmax())
    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "probabilities": {
            name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)
        },
    }


def output_fn(prediction, accept):
    """Return JSON response."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction), "application/json"
