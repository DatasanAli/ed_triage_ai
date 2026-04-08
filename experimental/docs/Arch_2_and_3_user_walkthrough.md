# Architecture 2 & 3 — User Walkthrough

> Plain-English explanation of how each hybrid model works, using a single patient example.

---

## The Core Idea

Both architectures use **two specialists** looking at the same patient, then a third person combines their opinions into a final triage prediction.

| Specialist | What it reads | Output |
|---|---|---|
| **BERT** | Words — chief complaint, HPI, past medical history | 768-number vector (text meaning) |
| **MLP / TabNet** | Numbers — vital signs and demographics | 128 or 64-number vector (vitals summary) |
| **Fusion head** | Both vectors combined | P(ESI Level 1–4) |

---

## Sample Patient Record

| Field | Value |
|---|---|
| Chief complaint | "Chest pain" |
| HPI | "62yo male, sudden crushing chest pain radiating to left arm, started 30min ago, with sweating" |
| Past medical history | "Hypertension, diabetes, prior heart attack 2019" |
| Heart rate | 118 bpm |
| SpO2 | 94% |
| Respiratory rate | 22 breaths/min |
| Systolic BP | 88 mmHg |
| Age | 62 |
| Arrival | Ambulance |

---

## Architecture 3: BERT + MLP

### What BERT does

BERT reads the combined text string:

```
"Chest pain. 62yo male, sudden crushing chest pain radiating to left arm,
started 30min ago, with sweating. PMH: Hypertension, diabetes, prior heart
attack 2019"
```

It processes all the words and outputs one **768-number vector** — a compressed representation of the meaning of that text. Words like "crushing", "radiating", and "prior heart attack" push this vector in a direction that signals high acuity.

### What the MLP does

MLP stands for **Multi-Layer Perceptron** — it's a small neural network made of stacked layers of math (multiply → activate → repeat). It sees only the raw numbers:

```
[118, 94, 22, 88, ..., 62, ...]  ← 15 numbers (vitals + demographics)
```

It passes them through two layers and outputs one **128-number vector**. It processes all 15 features at once, uniformly. Low SpO2 + high HR + low BP together push the vector toward high acuity — but the MLP treats every feature with the same process, with no sense of priority or order.

### Combining them

```
BERT vector    [768 numbers]   ← meaning of the words
MLP vector     [128 numbers]   ← vitals summary
──────────────────────────────
Concatenated   [896 numbers]   →  Fusion head  →  P(ESI 1–4)
```

**Prediction: ESI Level 1 (Critical) — 93% confidence**

---

## Architecture 2: BERT + TabNet

### What BERT does

Identical to Arch 3. Same text goes in, same 768-number vector comes out.

### What TabNet does (the difference)

TabNet replaces the MLP. Instead of looking at all 15 vitals at once, **it works in steps — like a doctor running through a checklist.**

```
Step 1 — Check the most critical vitals first:
  SpO2 = 94%  →  borderline low        [attention weight: 0.82]
  RR   = 22   →  elevated              [attention weight: 0.71]
  Partial decision: "something is wrong with breathing"

Step 2 — Given that, now look at circulation:
  HR   = 118  →  tachycardia           [attention weight: 0.65]
  SBP  = 88   →  hypotensive           [attention weight: 0.48]
  Partial decision: "patient may be in shock"

Step 3 — Check contextual factors:
  Age       = 62        →  elevated risk      [attention weight: 0.55]
  Transport = AMBULANCE →  pre-assessed high  [attention weight: 0.40]
  Partial decision: "elderly, serious arrival"

Step 4 — Pain and temperature:
  Pain  = 9   →  severe                [attention weight: 0.32]
  Temp  = 38.9 →  low-grade fever      [attention weight: 0.28]

Step 5 — Remaining features:
  Gender, race features → minimal contribution  [weights < 0.12]
```

At the end of all 5 steps, TabNet outputs one **64-number vector** summarising the sequential reasoning.

### Combining them

```
BERT vector    [768 numbers]   ← meaning of the words
TabNet vector  [ 64 numbers]   ← sequential vital sign reasoning
──────────────────────────────
Concatenated   [832 numbers]   →  Fusion head  →  P(ESI 1–4)
```

**Prediction: ESI Level 1 (Critical) — 91% confidence**

---

## Side-by-Side Comparison

| | **Architecture 3 (BERT + MLP)** | **Architecture 2 (BERT + TabNet)** |
|---|---|---|
| Text branch | BioClinicalBERT, fine-tuned | BioClinicalBERT, fine-tuned (identical) |
| Structured branch | 2-layer MLP | TabNet with 5 sequential attention steps |
| Looks at vitals | All at once, uniformly | Step by step, prioritised |
| Feature selection | No — all 15 features treated equally | Yes — sparse attention selects a few features per step |
| Structured output vector | 128 numbers | 64 numbers |
| Combined vector | 896 numbers | 832 numbers |
| Built-in reasoning trace | No | Yes — which vitals mattered at which step |
| Explainability | Post-hoc SHAP / Integrated Gradients | Native attention masks + SHAP |
| Training complexity | Simple, fast (~50–80 min on A100) | More complex (~90–120 min on A100) |
| Role in experiment | Benchmark baseline (simplest hybrid) | Mid-complexity model (sequential attention) |

---

## Why Both Are Real Deep Learning (Not Just Embedding Extraction)

A common misconception: "BERT is just generating embeddings, and the real model is the classifier on top."

That is NOT what these architectures do. In both Arch 2 and Arch 3:

- BERT's weights are **updated during training** via backpropagation (lr = 2e-5)
- The structured branch (MLP or TabNet) trains **jointly** with BERT in a single training loop
- One loss → one `backward()` call → gradients flow through everything simultaneously

```
Loss (cross-entropy)
    │
    ├── → updates fusion head
    ├── → updates MLP / TabNet encoder
    └── → updates all BERT transformer layers
```

This is end-to-end fine-tuning, not frozen embedding extraction.

**Contrast with Architecture 1 (BERT + XGBoost):** XGBoost is not differentiable — gradients cannot flow back through it into BERT. That forces a two-stage pipeline: fine-tune BERT first, freeze it, then train XGBoost on fixed embeddings separately. Arch 2 and 3 avoid this limitation entirely.

---

## What Makes the Architectures Different From Each Other

The BERT branch is identical in both. The only difference is **how the 15 structured features are processed:**

- **MLP:** "Here are 15 numbers. Process them all at once through two layers. Give me a summary." — Simple, fast, no feature prioritisation.
- **TabNet:** "Here are 15 numbers. Check the most important ones first. Then, given what you found, check the next most important. Repeat 5 times. Give me a summary." — Slower, but the step-by-step attention mirrors clinical triage reasoning and provides a built-in explanation.

---

*Written March 2026 — Capstone Project: ED Triage Prediction using MIMIC-IV-Ext*
