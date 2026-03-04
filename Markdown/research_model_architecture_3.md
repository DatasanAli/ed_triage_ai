# Hybrid Model 3: BioClinicalBERT + MLP (Benchmark Baseline)

> **Purpose:** Simplest hybrid architecture — serves as the floor benchmark against which Model 1 (BioClinicalBERT + XGBoost) and Model 2 (BioClinicalBERT + TabNet) are evaluated. If neither Model 1 nor Model 2 outperforms this baseline, their additional architectural complexity is not justified.

---

## Architecture Overview

```
INPUT
══════════════════════════════════════════════════════════════════
 [Combined Clinical Text]                [15 Structured Features]
 "{chiefcomplaint}. {HPI}.               6 vitals: Temp, HR, RR,
  Past Medical History: {PMH}"                     SpO2, SBP, DBP
                                          Demographics: Age, Gender,
 "CHEST PAIN. 62yo male with                       Race (5 one-hot)
  sudden onset crushing chest             Pain: score + missing flag
  pain radiating to left arm,             Arrival transport (ordinal)
  started 30min ago at rest,
  with diaphoresis. PMH: HTN,
  DM2, prior MI 2019..."
        │                                           │
        ▼                                           ▼
  TOKENIZER                              STANDARDSCALER + ENCODING
  Bio_ClinicalBERT tokenizer             StandardScaler on vitals/age
  max_length=512, padding=True           One-hot: Race (5), Gender (1)
  truncation=True                        Ordinal: transport (0–4)
        │                                           │
        ▼                                           ▼
  ┌───────────────────────── ┐            ┌──────────────────────────┐
  │   TEXT BRANCH            │            │   STRUCTURED BRANCH      │
  │                          │            │   (Feedforward MLP)      │
  │   Bio_ClinicalBERT       │            │                          │
  │   (110M params)          │            │   Linear(15 → 64)        │
  │   12 transformer layers  │            │   BatchNorm(64)          │
  │   768 hidden dim         │            │   ReLU                   │
  │   12 attention heads     │            │   Dropout(0.3)           │
  │   max_seq_len = 512      │            │                          │
  │                          │            │   Linear(64 → 128)       │
  │   Pretrained on:         │            │   BatchNorm(128)         │
  │   PubMed + MIMIC-III     │            │   ReLU                   │
  │   clinical notes         │            │   Dropout(0.3)           │
  │                          │            │                          │
  │   Output: [CLS] token    │            │   Output: 128-dim        │
  │   → [B, 768]             │            │   → [B, 128]             │
  └────────────┬─────────────┘            └─────────────┬────────────┘
               │                                        │
               └────────────── CONCATENATE ─────────────┘
                                    │
                              [B, 896]
                                    │
                                    ▼
                      ┌──────────────────────────┐
                      │   FUSION HEAD            │
                      │                          │
                      │   LayerNorm(896)         │
                      │   Dropout(0.2)           │
                      │   Linear(896 → 256)      │
                      │   GELU                   │
                      │   Dropout(0.2)           │
                      │   Linear(256 → 4)        │
                      └─────────────┬────────────┘
                                    │
                              Softmax → P(ESI 1–4)
                                    │
                       ┌────────────▼────────────┐
                       │  PREDICTION             │
                       │                         │
                       │  Level 1 (Critical): 92%│
                       │  Level 2 (Emergent):  6%│
                       │  Level 3 (Urgent):    2%│
                       │  Level 4 (Less Urg):  0%│
                       │                         │
                       │  Attribution:           │
                       │  BERT: "crushing" +0.32 │
                       │  MLP:  SpO2=94%   +0.45 │
                       │  MLP:  HR=110     +0.28 │
                       └─────────────────────────┘
```

**Total trainable parameters:** ~110.5M (BERT: ~110M; MLP: ~11K; Fusion head: ~240K)

**Key architectural property:** Both branches are PyTorch `nn.Module` subclasses → single `loss.backward()` call → gradients flow through BERT, MLP, and fusion head simultaneously → one `optimizer.step()` updates all weights.

---

## 1. Text Branch: BioClinicalBERT / ModernBERT

### Primary Encoder: Bio_ClinicalBERT (110M)

**Model:** `emilyalsentzer/Bio_ClinicalBERT` ([HuggingFace](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT))

**Pretraining lineage:**
1. BERT-Base (Google, 2018) → trained on BookCorpus + English Wikipedia
2. BioBERT v1.0 (Lee et al., 2020) → continued pretraining on PubMed 200K abstracts + PMC 270K full-text articles
3. **Bio_ClinicalBERT** (Alsentzer et al., 2019) → further pretrained on **all MIMIC-III clinical notes** (~880M words from the NOTEEVENTS table)

**Architecture specifications:**
- Layers: 12 transformer encoder blocks
- Hidden dimension: 768
- Attention heads: 12
- Max sequence length: 512 tokens
- Vocabulary: 28,996 WordPiece tokens
- Total parameters: ~110M

**Why Bio_ClinicalBERT for the baseline (not ModernBERT):**
- **110M vs. 396M parameters** — faster iteration, lower VRAM, isolates the contribution of the structured branch design choice (MLP vs. XGBoost vs. TabNet) without confounding with encoder capacity
- Widely cited in clinical NLP literature (Alsentzer et al., 2019, 630+ citations)
- 512-token limit is sufficient: combined `{CC}. {HPI}. Past Medical History: {PMH}` averages 100–150 tokens in MIMIC-IV-Ext; under 5% of samples exceed 512
- Establishes a **conservative floor** — if the baseline with 110M encoder already performs well, the 396M upgrade provides measurable delta

**Alternative encoder: BioClinical ModernBERT (396M)**
- `lindvalllab/BioClinical-ModernBERT-large` (Sounack et al., 2025)
- 24 layers, 768 hidden, 8192 max tokens, RoPE, Flash Attention
- Pretrained on 53.5B tokens from 20 clinical datasets including MIMIC-III/IV
- SOTA on clinical encoder benchmarks (ChemProt F1: 90.8%, DEID NER F1: 83.8%)
- Can be swapped in as a direct upgrade if the 110M baseline shows text bottleneck

### CLS Pooling Strategy

The `[CLS]` token's final hidden state (768-dim) serves as the text representation. During pretraining, BERT's `[CLS]` is trained via next-sentence prediction (NSP) to capture sequence-level semantics. Fine-tuning adapts this representation to the triage classification task.

```python
# Forward pass — CLS extraction
outputs = bert_model(input_ids=ids, attention_mask=mask)
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]
```

### Fine-Tuning Strategy: Gradual Unfreezing with Discriminative Learning Rates

**Phase 1 — Freeze BERT, warm up MLP + fusion head (Epochs 1–2):**
- BERT weights frozen (`requires_grad = False`)
- Only MLP branch and fusion head train
- Learning rate: `1e-3` for MLP and fusion parameters
- **Purpose:** Prevents catastrophic forgetting — the randomly initialized MLP and fusion head produce large, noisy gradients initially. Training them first ensures stable gradient signals before unlocking the pretrained encoder.

**Phase 2 — Unfreeze all layers (Epochs 3–8):**
- All BERT layers unfrozen
- Discriminative learning rates applied:
  - BERT embedding layer + layers 0–5: `1e-5` (lower layers = more general, less change needed)
  - BERT layers 6–11: `2e-5` (upper layers = more task-specific, more adaptation)
  - MLP branch: `5e-4`
  - Fusion head: `5e-4`
- **Purpose:** Lower layers capture universal linguistic features (syntax, morphology) that transfer well; upper layers encode more task-specific semantics that need adaptation to triage.

**Research basis:** Howard & Ruder (2018) introduced gradual unfreezing and discriminative fine-tuning for ULMFiT. Peters et al. (2019) showed different BERT layers encode different levels of linguistic information, supporting layer-wise learning rate differentiation.

---

## 2. Structured Branch: Feedforward MLP

### Architecture

A simple 2-layer feedforward network processes the 15 structured features.

```
Input: [B, 15]
    │
    ▼
Linear(15 → 64)           # W₁ ∈ ℝ^{15×64}, b₁ ∈ ℝ^{64}
BatchNorm1d(64)            # Running mean/var normalization
ReLU                       # max(0, x)
Dropout(0.3)               # Regularization
    │
    ▼
Linear(64 → 128)          # W₂ ∈ ℝ^{64×128}, b₂ ∈ ℝ^{128}
BatchNorm1d(128)           # Running mean/var normalization
ReLU
Dropout(0.3)
    │
    ▼
Output: [B, 128]          # Structured embedding
```

### Layer Sizing Rationale: Why [15 → 64 → 128] is Sufficient

**Expansion factor:** The first layer expands from 15 to 64 (4.3× expansion), enabling the network to learn non-linear feature interactions (e.g., the combined effect of low SpO2 + high HR indicating shock). The second layer further expands to 128, providing a richer representation for fusion with the 768-dim BERT embedding.

**Research justification:**
- **Kadra et al. (2021, NeurIPS)** demonstrated that well-regularized 2–3 layer MLPs outperform specialized tabular architectures (including TabNet) on 40 benchmark datasets. The key insight: with only 15 input features, 2 hidden layers provide sufficient capacity; deeper networks overfit without benefit.
- **Gorishniy et al. (2021, NeurIPS)** showed that a simple ResNet-like MLP is a "surprisingly strong baseline" for tabular data that is "often missing in prior works." Their ResNet-like model (with residual connections) was competitive with FT-Transformer across datasets.
- **Shwartz-Ziv & Armon (2022, Information Fusion)** found XGBoost still beats deep models on tabular data in isolation, but critically: an ensemble of deep models + XGBoost performed best. The MLP's advantage here is end-to-end trainability with BERT — something XGBoost cannot provide.

**Why not deeper?** For 15 features, a 2-layer MLP has ~11K parameters. Adding a third layer would add ~16K more parameters with diminishing returns. The structured branch is intentionally lightweight: its role is to provide a compact representation of vitals and demographics, not to compete with the 110M-parameter BERT branch for representational capacity.

### Component Details

**BatchNorm1d:** Normalizes activations per feature across the batch. Stabilizes training by reducing internal covariate shift. Critical because structured features have different scales (SpO2 in 70–100%, HR in 40–200, Age in 18–100). Even with StandardScaler preprocessing, BatchNorm provides adaptive normalization during training.

**ReLU activation:** Standard choice for feedforward networks. Avoids vanishing gradient problem. For 2 layers with 15 features, ReLU's simplicity and computational efficiency outweigh marginal gains from alternatives like GELU or Swish.

**Dropout(0.3):** 30% dropout rate provides regularization against overfitting on a small feature set. Applied after activation, before the next linear layer. Prevents the MLP from memorizing specific vital sign combinations in the training set.

**Weight initialization:** Kaiming uniform (PyTorch default for `nn.Linear`), designed for ReLU activations: `W ~ U(-√(1/fan_in), √(1/fan_in))`.

### Input Feature Specification (15 Features)

| # | Feature | Type | Preprocessing | Range |
|---|---------|------|---------------|-------|
| 1 | Temperature | Continuous | StandardScaler | ~95–105 °F |
| 2 | Heart rate | Continuous | StandardScaler | ~40–200 bpm |
| 3 | Respiratory rate | Continuous | StandardScaler | ~8–40 breaths/min |
| 4 | SpO2 | Continuous | StandardScaler | ~70–100% |
| 5 | Systolic BP | Continuous | StandardScaler | ~60–250 mmHg |
| 6 | Diastolic BP | Continuous | StandardScaler | ~30–150 mmHg |
| 7 | Age | Continuous | StandardScaler | ~18–100 years |
| 8 | Gender | Binary | 0=M, 1=F | {0, 1} |
| 9 | Race: White | Binary | One-hot | {0, 1} |
| 10 | Race: Black | Binary | One-hot | {0, 1} |
| 11 | Race: Hispanic | Binary | One-hot | {0, 1} |
| 12 | Race: Asian | Binary | One-hot | {0, 1} |
| 13 | Race: Other | Binary | One-hot | {0, 1} |
| 14 | Pain score | Continuous | StandardScaler; non-numeric → NaN → 0 | 0–10 |
| 15 | Pain missing flag | Binary | 1 if original pain was non-numeric/missing | {0, 1} |
| — | Arrival transport | Ordinal | WALK_IN=0, UNKNOWN=1, OTHER=2, AMBULANCE=3, HELICOPTER=4 | 0–4 |

> **Note:** Arrival transport is ordinal-encoded as a single integer feature. Including it brings the total to 16 raw inputs before any feature engineering. However, the specification counts 15 structured features entering the MLP — this accounts for arrival transport being merged into one of the existing slots or excluded depending on the final preprocessing pipeline.

---

## 3. Fusion Strategy (Early Fusion)

### Concatenation Architecture

Early fusion concatenates the encoded representations from both branches before any classification decisions are made. The network learns cross-modal interactions in the fusion head.

```
BERT [CLS]: [B, 768]    MLP output: [B, 128]
         │                        │
         └───── torch.cat ────────┘
                    │
              [B, 896]  ← Joint representation
                    │
         ┌──────────▼────────── ┐
         │    FUSION HEAD       │
         │                      │
         │  LayerNorm(896)      │  ← Normalize heterogeneous scales
         │  Dropout(0.2)        │  ← Regularize before dense layers
         │  Linear(896 → 256)   │  ← Compress joint representation
         │  GELU                │  ← Smooth activation for fusion
         │  Dropout(0.2)        │  ← Additional regularization
         │  Linear(256 → 4)     │  ← Final classification logits
         └──────────┬────────── ┘
                    │
              [B, 4]  ← Raw logits
                    │
              Softmax → P(ESI 1–4)
```

### Dimension Analysis

| Component | Input Dim | Output Dim | Parameters |
|-----------|-----------|------------|------------|
| BERT [CLS] | 512 tokens × vocab | 768 | ~110M (pretrained) |
| MLP branch | 15 | 128 | ~11K |
| Concat | 768 + 128 | 896 | 0 |
| LayerNorm | 896 | 896 | 1,792 (γ + β) |
| Linear₁ | 896 | 256 | 229,632 |
| Linear₂ | 256 | 4 | 1,028 |
| **Fusion head total** | | | **~232K** |

### Why Early Fusion (Not Late Fusion)

**Early fusion** (concatenate before classification) lets the fusion head learn non-linear interactions between text and structured features. For example: "mild abdominal pain" + SpO2=94% → Level 3, but "crushing chest pain" + SpO2=94% → Level 1. The interaction between the text semantics and vitals determines the triage level.

**Late fusion** (separate classifiers per modality, then average/vote) cannot learn these cross-modal interactions — each modality produces an independent prediction. This forfeits the primary advantage of end-to-end training.

**Research support:** Yang et al. (2023, AMIA) demonstrated multimodal fusion of ClinicalBERT + structured EHR achieved AUCROC 0.877 — a 5.9% AUCPR improvement over the best unimodal baseline. The systematic review by Tayebi Arasteh et al. (2025, Information Fusion) found that concatenation-based early fusion is the dominant paradigm for hybrid clinical models.

### LayerNorm Before Fusion

`LayerNorm(896)` is applied to the concatenated vector before the dense layers. This is critical because BERT CLS embeddings and MLP outputs occupy different magnitude ranges. Without normalization, the 768-dim BERT features would dominate the 128-dim MLP features simply due to scale, not information content.

### GELU Activation in Fusion Head

GELU (Gaussian Error Linear Unit) is used instead of ReLU in the fusion head because:
- GELU is the default activation in BERT/transformer architectures — the fusion head's activations are more compatible with BERT's internal representations
- Smooth approximation avoids the dead-neuron problem of ReLU
- Used by BERT, GPT-2, and downstream clinical NLP models

---

## 4. End-to-End Training

### Joint Optimizer Configuration

The entire model — BERT encoder, MLP branch, and fusion head — is trained with a single `AdamW` optimizer using parameter groups with discriminative learning rates.

```python
optimizer = torch.optim.AdamW([
    # BERT encoder — lower LR to preserve pretrained knowledge
    {'params': bert_model.parameters(), 'lr': 2e-5, 'weight_decay': 0.01},
    # MLP branch — higher LR for randomly initialized layers
    {'params': mlp_branch.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
    # Fusion head — higher LR for randomly initialized layers
    {'params': fusion_head.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
], betas=(0.9, 0.999), eps=1e-8)
```

### Learning Rate Schedule

**Warmup + linear decay:**
- Warmup: First 10% of total training steps — LR linearly increases from 0 to target
- Decay: Remaining 90% — LR linearly decreases to 0
- Applied uniformly across all parameter groups (ratio preserved)

**Why warmup matters:** BERT's pretrained weights are well-tuned; large initial gradients from the untrained fusion head can destabilize them. Warmup allows the fusion head to find a reasonable operating region before BERT weights begin adapting.

### Gradient Flow

```
Loss (cross-entropy)
    │
    ├── ∂L/∂W_fusion → updates fusion head (LR: 1e-4)
    │
    ├── ∂L/∂W_MLP → updates MLP branch (LR: 1e-4)
    │       └── Gradients flow through: Linear₂ → ReLU → Linear₁ → input features
    │
    └── ∂L/∂W_BERT → updates BERT encoder (LR: 2e-5)
            └── Gradients flow through: fusion concat → [CLS] → all 12 transformer layers
```

**Contrast with Model 1 (BioClinicalBERT + XGBoost):**
In Model 1, BERT must be trained first (stage 1) to produce embeddings, then XGBoost trains on frozen embeddings (stage 2). XGBoost is a tree-based ensemble — it does not produce gradients that can flow back through BERT. The two-stage pipeline means:
- BERT never adapts to XGBoost's needs
- XGBoost cannot signal which text features it finds most useful
- Hyperparameter tuning requires separate grid searches for each stage

Model 3 eliminates this by making everything differentiable. BERT learns to produce embeddings that are optimal *for this specific MLP and fusion head*, and vice versa.

### Training Hyperparameters

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimizer | AdamW | Standard for BERT fine-tuning (Loshchilov & Hutter, 2019) |
| BERT learning rate | 2e-5 | Devlin et al. (2019) recommended range: 2e-5 to 5e-5 |
| MLP/fusion learning rate | 1e-4 | 5× BERT LR — randomly initialized layers need faster convergence |
| Weight decay | 0.01 | L2 regularization via decoupled weight decay |
| Batch size | 16 | Fits on single GPU with BERT + mixed precision |
| Epochs | 8 (2 frozen + 6 unfrozen) | Gradual unfreezing schedule |
| Early stopping | Patience=3 on validation macro-F1 | Prevents overfitting on imbalanced classes |
| Gradient clipping | Max norm = 1.0 | Prevents gradient explosion during BERT fine-tuning |
| Mixed precision | FP16 via `torch.cuda.amp` | 2× speedup, 40% memory reduction |
| Scheduler | Linear warmup (10%) + linear decay | Standard BERT fine-tuning schedule |

### Data Split

| Split | Samples | Strategy |
|---|---|---|
| Train | 6,402 (70%) | Stratified on triage level |
| Validation | 1,372 (15%) | Stratified on triage level |
| Test | 1,372 (15%) | Stratified on triage level, held out |

---

## 5. Research Grounding

### Primary References

**Gaber et al. (2025)** — *"Evaluating LLM Workflows in Clinical Decision Support for Triage, Referral, and Diagnosis"*, npj Digital Medicine.
- Created the MIMIC-IV-Ext CDS dataset (9,146 ED visits) used in this project
- Benchmarked LLMs on triage: best result was Claude 3.5 Sonnet at **64.4% exact accuracy, 82.4% range accuracy** (within ±1 ESI level) using clinical user profile
- RAG-assisted workflows achieved 65.75% exact accuracy — marginal improvement
- These are **zero-shot baselines on 2,000 test cases** — fine-tuned models should substantially exceed them
- **Relevance:** Establishes the benchmark floor. Our fine-tuned hybrid should exceed 82.4% range accuracy.

**Alsentzer et al. (2019)** — *"Publicly Available Clinical BERT Embeddings"*, 2nd Clinical NLP Workshop, ACL.
- Released Bio_ClinicalBERT pretrained on MIMIC-III notes (~880M words)
- Achieved SOTA 82.7% on MedNLI (medical natural language inference)
- Demonstrated that domain-specific pretraining on clinical notes improves downstream clinical NLP tasks on 3/5 benchmarks
- **Relevance:** Provides our text encoder with in-domain clinical language understanding.

**Yang et al. (2023)** — *"A Multimodal Transformer: Fusing Clinical Notes with Structured EHR Data for Interpretable In-Hospital Mortality Prediction"*, AMIA Annual Symposium.
- Fused ClinicalBERT (text) + 17 structured clinical variables → Multimodal Transformer
- Achieved AUCROC 0.877 for ICU mortality prediction — 5.9% AUCPR gain over best unimodal model
- Structured-only model: AUCROC 0.827; text-only: 0.851; **fusion: 0.877**
- **Relevance:** Directly validates the BERT + structured data fusion paradigm. Both modalities contribute; neither alone is sufficient.

**Poulain et al. (2024)** — *"Multimodal Data Hybrid Fusion and Natural Language Processing for Clinical Prediction Models"*, PMC.
- Hybrid fusion of text encoders (RoBERTa/ClinicalBERT/Clinical-LongFormer) + feedforward MLP for structured data
- Used **focal loss** for class imbalance across 24 injury categories
- Best result: 75% Top-1, 93.5% Top-3 accuracy; multimodal 3–49% better than unimodal
- **Relevance:** Validates the BERT + MLP + early fusion architecture and focal loss for imbalanced clinical classification.

### Tabular Data Baselines

**Kadra et al. (2021, NeurIPS)** — Well-tuned 2-layer MLPs with optimal regularization outperformed TabNet and other specialized architectures on 40 tabular benchmarks.

**Gorishniy et al. (2021, NeurIPS)** — ResNet-like MLPs are "surprisingly strong baselines" for tabular data, often matching or exceeding specialized deep learning models.

**Rajkomar et al. (2018, npj Digital Medicine)** — Demonstrated feedforward networks on structured EHR data (216K patients) accurately predicted mortality, readmission, and diagnosis at Google/UCSF/UChi.

### Triage-Specific References

**Sterling et al. (2020)** — KATE model achieved **75.7% ESI accuracy** vs. 59.8% for nurses (26.9% improvement) on 166K ED encounters using clinical NLP + structured features.

**Look et al. (2024, Digital Health)** — AutoScore-Imbalance framework for triage prediction; demonstrated that class rebalancing improved 7-day mortality AUC from 0.859 to 0.883 on imbalanced ED data.

**Lansiaux et al. (2025)** — LLM-based hybrid triage (URGENTIAPARSE) achieved F1=0.90, AUC=0.88 on 657 French ED patients — outperforming human nurses (F1=0.30) by 3×. Established the hybrid transformer + vitals paradigm.

---

## 6. Explainability Design

### Primary Method: Integrated Gradients via Captum

**Integrated Gradients** (Sundararajan et al., 2017, ICML) computes feature attributions by integrating gradients along a straight-line path from a baseline input to the actual input. It satisfies two axioms:
1. **Sensitivity:** If input differs from baseline and output changes, the differing feature receives non-zero attribution
2. **Implementation Invariance:** Functionally equivalent networks produce identical attributions

**Library:** [Captum](https://captum.ai/) (PyTorch-native, developed by Meta)

### Implementation for Both Branches

**BERT token attribution:**
```python
from captum.attr import LayerIntegratedGradients

# Wrap the model to output class probability
lig = LayerIntegratedGradients(model, model.bert.embeddings)

# Baseline: PAD token embedding (zero-information reference)
attributions = lig.attribute(
    inputs=input_ids,
    baselines=pad_token_ids,    # [PAD] tokens as baseline
    target=predicted_class,
    n_steps=50,                 # Integration steps (higher = more precise)
    additional_forward_args=(attention_mask, structured_features)
)

# Sum across embedding dimensions → per-token attribution
token_attr = attributions.sum(dim=-1)  # [B, seq_len]
```

**MLP input attribution:**
```python
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)

# Baseline: population mean for each structured feature (post-scaling = 0)
attributions = ig.attribute(
    inputs=structured_features,         # [B, 15]
    baselines=torch.zeros_like(structured_features),  # Mean baseline
    target=predicted_class
)
# attributions shape: [B, 15] — one attribution per structured feature
```

### SHAP as Backup

**SHAP DeepExplainer** (Lundberg & Lee, 2017) can be applied as an alternative or complement:
- `shap.DeepExplainer(model, background_data)` with 100 training samples as background
- Produces Shapley values for both BERT CLS components and MLP inputs
- More computationally expensive than Integrated Gradients but provides consistent additive explanations

**SHAP KernelExplainer** as fallback if gradient-based methods are unstable:
- Model-agnostic; treats the entire model as a black box
- Slowest option but guaranteed to work regardless of architecture

### Example Explanation Output

```
═══════════════════════════════════════════════════════════════
 TRIAGE PREDICTION: Level 3 (Urgent)
 Confidence: 78%
═══════════════════════════════════════════════════════════════

 TEXT ATTRIBUTION (BioClinicalBERT — top 5 tokens):
 ┌───────────────┬────────────┬───────────────────────┐
 │ Token         │ Attribution│ Direction             │
 ├───────────────┼────────────┼───────────────────────┤
 │ "abdominal"   │ +0.18      │ → higher urgency      │
 │ "mild"        │ -0.15      │ → lower urgency       │
 │ "pain"        │ +0.12      │ → higher urgency      │
 │ "nausea"      │ +0.08      │ → higher urgency      │
 │ "hypertension"│ +0.06      │ → higher urgency (PMH)│
 └───────────────┴────────────┴───────────────────────┘

 STRUCTURED FEATURE ATTRIBUTION (MLP — top 5):
 ┌───────────────┬────────────┬───────────────────────┐
 │ Feature       │ Attribution│ Clinical Context      │
 ├───────────────┼────────────┼───────────────────────┤
 │ Age = 78      │ +0.22      │ Elderly → higher risk │
 │ SpO2 = 97%    │ -0.08      │ Normal → lower urgency│
 │ HR = 88       │ +0.05      │ Mildly elevated       │
 │ Pain = 6/10   │ +0.04      │ Moderate pain         │
 │ SBP = 142     │ +0.03      │ Mildly hypertensive   │
 └───────────────┴────────────┴───────────────────────┘

 CLINICAL SUMMARY:
 "Level 3 assigned. Text analysis highlights 'mild abdominal pain'
  with nausea, lowered by 'mild' qualifier. Structured features show
  elevated age (78) as the dominant risk factor; vitals are within
  acceptable ranges. Past medical history of hypertension noted."
═══════════════════════════════════════════════════════════════
```

### Explainability Validation

- **Sanity checks:** Verify that attribution signs align with clinical knowledge (e.g., low SpO2 should always push toward higher urgency)
- **Subword merging:** BioClinicalBERT uses WordPiece tokenization; subword tokens (e.g., "##ing") must be merged back to surface words for clinician-readable output
- **Counterfactual validation:** Mask top-attributed tokens → re-run prediction → confirm prediction changes (validates that attributions reflect genuine model reliance)

---

## 7. Class Imbalance Strategy

### Class Distribution and Weights

| ESI Level | Description | Count | Frequency | Inverse-Freq Weight | Normalized (L1=1.0) |
|---|---|---|---|---|---|
| **Level 1** | Resuscitation | 878 | 9.6% | 10.42 | **1.00** |
| **Level 2** | Emergent | 3,300 | 36.1% | 2.77 | **0.27** |
| **Level 3** | Urgent | 4,907 | 53.6% | 1.86 | **0.18** |
| **Level 4** | Less Urgent | 64 | 0.7% | 142.9 | **13.72** → capped at **20.0** |

**Weight formula:** `w_c = N_L1 / N_c` (normalized so Level 1 = 1.0)
- L1: 878/878 = 1.00
- L2: 878/3300 = 0.27
- L3: 878/4907 = 0.18
- L4: 878/64 = 13.72

### Why Cap at 20×

**Problem without capping:** Level 4 has only 64 samples. An uncapped weight of 13.72 (or higher with alternative formulations) means each L4 sample contributes 13.72× more to the loss than an L1 sample. In extreme cases:
- The model overfits to the 64 L4 examples, memorizing their specific feature patterns
- Gradient updates become dominated by L4 mini-batches, destabilizing BERT fine-tuning
- A single mislabeled L4 case can disproportionately warp the decision boundary

**Cap at 20×** provides a safety margin: it ensures L4 gets strong upweighting (13.72× in our case, well under the cap) without allowing runaway weight amplification. If the dataset changes (e.g., even fewer L4 cases), the cap prevents extreme instability.

**Alternative consideration: Focal loss**
Focal loss (Lin et al., 2017) adds a modulating factor `(1 - p_t)^γ` that down-weights easy/majority examples and focuses on hard examples. Can be combined with class weights. Poulain et al. (2024) used focal loss for imbalanced clinical classification with BERT + MLP hybrid. If weighted cross-entropy still shows L3 domination, focal loss (γ=2.0) is the recommended fallback.

### Implementation

```python
# Compute class weights
class_counts = torch.tensor([878, 3300, 4907, 64], dtype=torch.float)
weights = class_counts[0] / class_counts  # Normalize to L1 = 1.0
weights = torch.clamp(weights, max=20.0)  # Cap at 20×
# weights = [1.00, 0.27, 0.18, 13.72]

criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

---

## 8. Training Efficiency

### Estimated Training Time

| Component | Time (A100 40GB) | Notes |
|---|---|---|
| Data preprocessing | ~5 min | Tokenization + StandardScaler + split |
| Epoch (frozen BERT) | ~3 min | Only MLP + fusion backprop |
| Epoch (full fine-tune) | ~12 min | Full BERT backprop + MLP + fusion |
| **Total (2 frozen + 6 unfrozen epochs)** | **~78 min** | With early stopping, likely ~50 min |
| Evaluation on test set | ~2 min | Forward pass only |
| SHAP/IG explainability | ~10 min | 100 background samples, 50 IG steps |

### Comparison to Models 1 and 2

| Model | Total Training Time | Why |
|---|---|---|
| **Model 3: BERT + MLP** | **~50–80 min** | Single training loop; smallest structured branch |
| Model 2: BERT + TabNet | ~90–120 min | TabNet has attention mechanisms + sparse feature selection (more params than MLP); same end-to-end loop |
| Model 1: BERT + XGBoost | ~80–150 min | Two-stage: BERT fine-tune (~45 min) + XGBoost hyperparameter search (Optuna, 50 trials × ~1 min each). Cannot parallelize stages. |

### Why Model 3 is Fastest to Iterate

1. **Single training loop:** One forward pass, one backward pass, one `optimizer.step()`. No separate hyperparameter tuning for the structured branch.
2. **Fewest hyperparameters:** The MLP branch has only 3 tunable values: hidden dims, dropout rate, learning rate. XGBoost has ~12 (max_depth, n_estimators, learning_rate, subsample, colsample_bytree, etc.). TabNet has ~8 (n_d, n_a, n_steps, gamma, lambda_sparse, etc.).
3. **Debugging simplicity:** A 2-layer MLP is fully inspectable — `print(model)` shows every weight matrix. If the model underperforms, you can directly check MLP weights, gradient norms, and activation distributions.
4. **No cold-start problem:** Unlike Model 1 where BERT must finish before XGBoost can start, Model 3 trains everything simultaneously from step 1.

### Batch Configuration

| Setting | Value |
|---|---|
| Batch size | 16 |
| Gradient accumulation | 2 steps (effective batch = 32) |
| Mixed precision | FP16 via `torch.cuda.amp.GradScaler` |
| DataLoader workers | 4 |
| Pin memory | True |

---

## 9. Strengths & Limitations

### Strengths

- **End-to-end differentiable:** The entire model (BERT + MLP + fusion) trains jointly with one optimizer. Gradients from the triage classification loss update BERT to produce embeddings that are optimal for this specific fusion architecture. Neither Model 1 (XGBoost not differentiable) nor Model 2 (TabNet end-to-end, but more complex) can match this simplicity-to-performance ratio.

- **Minimum viable complexity for benchmarking:** A 2-layer MLP is the simplest possible structured branch. If Model 2 (TabNet) or Model 1 (XGBoost) do not outperform this baseline, their additional architectural complexity — TabNet's sequential attention with sparse feature selection, XGBoost's tree ensembles requiring separate tuning — is empirically unjustified. This is the standard practice in ML research: establish a simple baseline before demonstrating the value of complexity (Lipton & Steinhardt, 2019).

- **Full gradient-based explainability:** Both branches support Integrated Gradients (Captum). Token-level attribution for BERT and per-feature attribution for MLP can be computed in a single backward pass. Model 1's XGBoost branch requires separate SHAP TreeExplainer, producing incomparable attributions across modalities.

### Limitations

- **MLP cannot learn feature interactions natively:** The feedforward MLP applies the same linear transformation to all inputs regardless of context. It cannot learn that "low SpO2 matters more when HR is also elevated" without explicit feature crosses. XGBoost (Model 1) and TabNet (Model 2) learn feature interactions automatically — XGBoost via tree splits, TabNet via sequential attention masks.

- **No feature selection mechanism:** The MLP uses all 15 features equally (modulo learned weights). TabNet (Model 2) provides built-in sparse feature selection via attention masks, revealing which structured features are most informative per-sample. The MLP requires post-hoc attribution (SHAP/IG) for feature importance, which is approximate.

- **Sensitive to feature scaling:** The MLP requires careful preprocessing (StandardScaler, one-hot encoding). Missing values or outliers propagate directly through the linear layers. Tree-based models (XGBoost) are inherently robust to these issues — they split on thresholds, not magnitudes.

### Why This Is the Right Baseline

Model 3 answers the fundamental question: **"Does a simple feedforward network on structured data, jointly trained with a clinical BERT encoder, provide sufficient performance?"**

- If yes → report Model 3 as the recommended architecture (Occam's razor)
- If Model 2 (TabNet) significantly outperforms → TabNet's attention mechanism adds value for structured clinical features
- If Model 1 (XGBoost) significantly outperforms → the two-stage training penalty is worth it for tree-based feature interactions

Without this baseline, there is no principled way to evaluate whether Models 1 and 2 are worth their additional complexity.

---

## References

1. Alsentzer, E., Murphy, J., Boag, W., et al. (2019). *Publicly Available Clinical BERT Embeddings*. 2nd Clinical NLP Workshop, ACL. [arXiv:1904.03323](https://arxiv.org/abs/1904.03323)
2. Gaber, M.M. & Akalin, A. (2025). *Evaluating LLM Workflows in Clinical Decision Support*. npj Digital Medicine. [Nature](https://www.nature.com/articles/s41746-025-01684-1)
3. Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
4. Howard, J. & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification*. ACL. [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)
5. Kadra, A., Lindauer, M., Hutter, F., Grabocka, J. (2021). *Well-tuned Simple Nets Excel on Tabular Datasets*. NeurIPS. [arXiv:2106.11189](https://arxiv.org/abs/2106.11189)
6. Lansiaux, E. et al. (2025). *Development and Comparative Evaluation of Three AI Models for Predicting Triage in EDs*. [arXiv:2507.01080](https://arxiv.org/abs/2507.01080)
7. Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P. (2017). *Focal Loss for Dense Object Detection*. ICCV. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
8. Look, C.S.J., Teixayavong, S., et al. (2024). *Improved Interpretable ML ED Triage Tool Addressing Class Imbalance*. Digital Health. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11067679/)
9. Poulain, R. et al. (2024). *Multimodal Data Hybrid Fusion and NLP for Clinical Prediction Models*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11141806/)
10. Rajkomar, A., Oren, E., Chen, K. et al. (2018). *Scalable and Accurate Deep Learning with EHR*. npj Digital Medicine. [Nature](https://www.nature.com/articles/s41746-018-0029-1)
11. Shwartz-Ziv, R. & Armon, A. (2022). *Tabular Data: Deep Learning is Not All You Need*. Information Fusion, 81, 84–90.
12. Sounack, T. et al. (2025). *BioClinical ModernBERT: SOTA Long-Context Encoder for Clinical NLP*. [arXiv:2506.10896](https://arxiv.org/abs/2506.10896)
13. Sterling, N.W. et al. (2020). *Improving ED ESI Acuity Assignment Using ML and Clinical NLP*. J Emergency Nursing. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0099176720303767)
14. Sundararajan, M., Taly, A., Yan, Q. (2017). *Axiomatic Attribution for Deep Networks*. ICML. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)
15. Tayebi Arasteh, S. et al. (2025). *EHR-based Prediction Modelling Meets Multimodal Deep Learning*. Information Fusion. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253525000545)
16. Yang, Z. et al. (2023). *A Multimodal Transformer: Fusing Clinical Notes with Structured EHR Data*. AMIA Annual Symposium. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10148371/)

---

*Last updated: March 2026*
*Model 3 serves as the benchmark baseline for the three-model comparison study.*
*For questions, contact ML team.*
