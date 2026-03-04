# Hybrid Model 2: BioClinicalBERT + TabNet

> Research compiled: 2026-03-04
> Task: 4-class ED triage prediction (ESI 1–4) on MIMIC-IV-Ext (n=9,146)

---

## Architecture Overview

```
INPUT
══════════════════════════════════════════════════════════════════════

[Combined Clinical Text]                  [15 Structured Features]
"{CC}. {HPI}. Past Medical               6 vitals: Temp, HR, RR,
 History: {PMH}"                                   SpO2, SBP, DBP
                                          Age, Gender, Race×5 (one-hot)
max_length=512 tokens                     Pain score, Pain missing flag
                                          Arrival transport (ordinal)
        │                                          │
        ▼                                          ▼
  ┌───────────────────── ┐               ┌──────────────────────────┐
  │  TEXT BRANCH         │               │  STRUCTURED BRANCH       │
  │                      │               │                          │
  │  BioClinicalBERT     │               │  TabNet Encoder          │
  │  (110M params)       │               │  (Arik & Pfister 2021)   │
  │  OR                  │               │                          │
  │  BioClinical         │               │  N_steps=5 decision steps│
  │  ModernBERT (396M)   │               │  N_a=64 (attention dim)  │
  │                      │               │  N_d=64 (feature dim)    │
  │  12/24 transformer   │               │  Relaxation γ=1.5        │
  │  layers              │               │  Ghost Batch Norm        │
  │  Bidirectional attn  │               │  Sparse feature masks    │
  │                      │               │                          │
  │  Output: [CLS]       │               │  Output: aggregated      │
  │  → [B, 768]          │               │  embedding → [B, N_d]    │
  └─────────┬────────────┘               └────────────┬─────────────┘
            │                                         │
            │         ┌─────────────────┐             │
            └────────►│  CONCATENATION  │◄────────────┘
                      │  [B, 768+N_d]   │
                      │  = [B, 832]     │
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │  FUSION HEAD    │
                      │                 │
                      │  LayerNorm(832) │
                      │  Dropout(0.2)   │
                      │  Linear(832→256)│
                      │  GELU           │
                      │  Dropout(0.2)   │
                      │  Linear(256→4)  │
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │  SOFTMAX        │
                      │  → P(ESI 1–4)   │
                      └────────┬────────┘
                               │
                 ┌─────────────▼────────────────── ┐
                 │  PREDICTION + EXPLANATION       │
                 │                                 │
                 │  Level 1 (Critical) — 91%       │
                 │                                 │
                 │  TabNet masks (step 1): SpO2 ↑  │
                 │  TabNet masks (step 2): HR ↑    │
                 │  BERT attention: "crushing" ↑   │
                 └─────────────────────────────────┘
```

---

## 1. Text Branch: BioClinicalBERT / ModernBERT

### Encoder Choice (Fixed Across All Hybrid Models)

The text branch uses the same clinical language model across all three proposed architectures (Model 1: +XGBoost, Model 2: +TabNet, Model 3: +MLP). This is deliberate — isolating the text encoder as a constant allows fair comparison of the structured data processing strategy.

**Primary option — BioClinical ModernBERT (396M):**
- Pre-trained on MIMIC-III + MIMIC-IV clinical notes (~53.5B tokens)
- 24 transformer layers, hidden dim 768, 12 attention heads
- 8,192-token context window (our combined text averages 80–150 tokens)
- 2025 SOTA on clinical NLP benchmarks (lindvalllab, Karolinska/US)
- HuggingFace: `lindvalllab/BioClinical-ModernBERT-large`

**Fallback — Bio_ClinicalBERT (110M):**
- Pre-trained on MIMIC-III (~880M words of clinical notes)
- 12 transformer layers, hidden dim 768, 12 attention heads
- 512-token context window (sufficient for our combined text input)
- Widely validated baseline; simpler to train
- HuggingFace: `emilyalsentzer/Bio_ClinicalBERT`

**Why this choice is fixed:** The text encoder processes the same combined text string (`"{CC}. {HPI}. Past Medical History: {PMH}"`) identically in all models. Swapping the text encoder would conflate two variables (text model quality vs. structured model quality), making ablation results uninterpretable. The [CLS] token output is always a 768-dimensional dense vector regardless of which encoder variant is used.

**Training mode:** Full fine-tuning (all encoder weights updated via backpropagation) with a lower learning rate (`lr=2e-5`) than the structured branch, following standard discriminative fine-tuning practice.

---

## 2. Structured Branch: TabNet

### 2.1 What is TabNet?

TabNet (Arik & Pfister, AAAI 2021) is a deep learning architecture purpose-built for tabular data. Unlike standard MLPs that process all features uniformly, TabNet uses **sequential attention** to dynamically select which features to focus on at each decision step — mimicking how a human expert might sequentially consider different clinical signs.

**Core innovation:** Instance-wise sparse feature selection learned end-to-end, without requiring manual feature engineering or external feature selection.

### 2.2 Step-Wise Attention Mechanism

TabNet processes input features through `N_steps` sequential decision steps. At each step:

1. **Attentive Transformer** selects a sparse subset of input features using a learned attention mask
2. **Feature Transformer** processes the selected features through shared and step-specific FC layers
3. **Split Block** splits the transformed output into two parts:
   - One part contributes to the final prediction (aggregated across steps)
   - One part informs the next step's attention decisions

```
Step 1: Select SpO2, HR (critical vitals)
   → Process through feature transformer
   → Partial decision: "vitals suggest high acuity"

Step 2: Select Age, Pain score
   → Process given Step 1's context
   → Partial decision: "elderly patient with high pain"

Step 3: Select Gender, Arrival transport
   → Process given Steps 1-2 context
   → Partial decision: "ambulance arrival supports acuity"

Final: Aggregate all step outputs → [B, N_d] embedding
```

### 2.3 Key Hyperparameters

| Parameter | Symbol | Role | Recommended Range | Our Setting |
|-----------|--------|------|-------------------|-------------|
| Decision steps | `N_steps` | Number of sequential attention rounds | 3–10 | **5** |
| Attention dimension | `N_a` | Width of attentive transformer | 8–128 | **64** |
| Feature dimension | `N_d` | Width of feature transformer output (also final embedding dim) | 8–128 | **64** |
| Relaxation factor | `γ` | Controls feature reuse across steps (1.0 = no reuse, higher = more reuse) | 1.0–2.5 | **1.5** |
| Sparsity coefficient | `λ_sparse` | Regularization strength for attention mask sparsity | 1e-4 to 1e-2 | **1e-3** |
| Virtual batch size | `v_B` | Sub-batch size for Ghost Batch Normalization | 16–128 | **32** |
| BN momentum | `m_B` | Momentum for Ghost Batch Normalization | 0.5–0.98 | **0.7** |

**Why N_d = N_a = 64:** Arik & Pfister recommend N_d ≈ N_a for most datasets. With only 15 input features, larger dimensions risk overfitting. 64 provides sufficient capacity without excess.

**Why N_steps = 5:** With 15 features across vitals, demographics, and clinical indicators, 5 steps allow the model to attend to 3–5 feature groups: (1) critical vitals like SpO2/HR, (2) secondary vitals like BP/RR, (3) demographics, (4) pain/transport, (5) residual interactions. This aligns with clinical triage reasoning, which proceeds through similar prioritized checks.

### 2.4 Ghost Batch Normalization

Standard Batch Normalization performs poorly with large batch sizes (noisy gradient estimates dominate). Ghost Batch Normalization (GBN) splits each training batch into virtual sub-batches of size `v_B` and applies BN independently to each sub-batch.

**Why it matters for our use case:** With only 9,146 total samples and 15 features, regularization is critical. GBN acts as implicit regularization by adding noise through smaller virtual batch statistics, reducing overfitting risk.

### 2.5 TabNet's Embedding Output

Unlike XGBoost (which outputs scalar predictions), TabNet's encoder produces a **dense N_d-dimensional embedding** for each input sample. This embedding captures the model's learned representation of the structured features after all attention steps.

```
Input: [B, 15]  →  TabNet Encoder (5 steps)  →  Output: [B, 64]
```

This 64-dimensional embedding is what gets concatenated with BioClinicalBERT's 768-dimensional [CLS] vector for fusion.

---

## 3. Fusion Strategy

### 3.1 Concatenation-Based Late Fusion

The fusion strategy concatenates the two branch outputs and passes the joint representation through a classification head:

```
BioClinicalBERT [CLS]:  [B, 768]   (text semantics)
TabNet embedding:       [B, 64]    (structured feature representation)
                        ─────────
Concatenated:           [B, 832]

Fusion Head:
  LayerNorm(832)
  Dropout(0.2)
  Linear(832 → 256) + GELU
  Dropout(0.2)
  Linear(256 → 4)         → logits for ESI 1–4
```

### 3.2 Why Concatenation (Not Gating or Cross-Attention)?

- **Simplicity:** Concatenation is the most common fusion strategy in clinical multimodal models and performs competitively with more complex alternatives (Multimodal Data Hybrid Fusion, PMC 2024)
- **Proven in clinical settings:** The KTH multimodal survival prediction study (2024) using BioClinicalBERT + TabTransformer on MIMIC-III used late fusion with aggregation — validating this pattern for clinical EHR + text tasks
- **Preserves interpretability:** Each branch's contribution remains separable. We can zero out one branch to measure the other's standalone performance (ablation)
- **Stable gradients:** No additional attention parameters to train, reducing optimization complexity

### 3.3 Joint vs. Separate Training

**Option A — Fully joint (end-to-end):**
Both BioClinicalBERT and TabNet are trained simultaneously with a single loss function. Gradients flow from the classification head through both branches.

**Option B — Pre-train then fine-tune:**
1. Pre-train TabNet on structured features alone (auxiliary triage task)
2. Pre-train BioClinicalBERT on text alone
3. Concatenate frozen embeddings, train only the fusion head
4. Optionally unfreeze for end-to-end fine-tuning

**Recommendation: Option A (fully joint).** With 9,146 samples, the dataset is large enough for joint training. End-to-end training allows the branches to learn complementary representations — TabNet may learn to focus on features that the text encoder misses, and vice versa. This satisfies the DNN fine-tuning requirement natively.

---

## 4. Research Grounding

### 4.1 TabNet: Attentive Interpretable Tabular Learning (Arik & Pfister, AAAI 2021)

**Citation:** Arik, S.Ö. & Pfister, T. (2021). "TabNet: Attentive Interpretable Tabular Learning." *Proceedings of the AAAI Conference on Artificial Intelligence, 35*(8), 6679–6687.

**Key contributions:**
- First deep learning model to match or exceed tree-based ensembles (XGBoost, LightGBM) on tabular benchmarks while providing native interpretability
- Sequential attention mechanism selects sparse feature subsets at each decision step
- Self-supervised pre-training mode (mask-then-predict) improves performance on small datasets
- Instance-wise feature importance via aggregated attention masks

**Benchmark results from paper:**
- Forest Cover Type: TabNet achieves 96.99% accuracy (vs. XGBoost 89.75%)
- Poker Hand: TabNet 99.2% (vs. XGBoost 71.1%)
- On smaller datasets, XGBoost remains competitive or better

### 4.2 TabNet in Clinical Applications

**Chronic Kidney Disease Detection (2025):**
TabNet achieved 94.06% accuracy for CKD stage prediction in diabetes patients, with attention masks revealing that eGFR and albumin-to-creatinine ratio were consistently selected in early decision steps — matching clinical guidelines for CKD staging.

**Heart Disease Prediction (Frontiers in Physiology, 2025):**
A hybrid TabNet architecture with stacked ensemble learning achieved 80.70% accuracy and F1-score of 77.52% for cardiovascular disease prediction. TabNet alone yielded 77.40% accuracy, outperforming standalone XGBoost. Importantly, combining TabNet + XGBoost via meta-learning produced the best results — suggesting the two models capture complementary patterns.

**Multi-Disease Risk Prediction (ScienceDirect, 2025):**
GAT-Enhanced TabNet improved prediction accuracy by ~10% over baseline models on clinical data from a tertiary hospital, demonstrating TabNet's compatibility with graph-based augmentation for structured health records.

### 4.3 Multimodal Clinical Fusion Precedents

**BioClinicalBERT + TabTransformer on MIMIC-III (IEEE Big Data, December 2024):**
A KTH study combined BioClinicalBERT for clinical text with TabTransformer for structured EHR data in a multimodal survival prediction system on MIMIC-III. The fused system demonstrated "superior performance on evaluation metrics, highlighting the system's ability to identify high-risk patients." This directly validates the BERT + tabular-attention-model fusion pattern we propose, on the same MIMIC data family.

**Multimodal Data Hybrid Fusion (PMC, 2024):**
Comprehensive framework integrating clinical notes (via clinical BERT models) with structured EHR data, demonstrating that late fusion of separately encoded modalities consistently outperforms single-modality baselines for clinical prediction tasks.

### 4.4 TabNet: Deep Learning Is Not All You Need? (Shwartz-Ziv & Armon, 2022)

An important counterpoint: Shwartz-Ziv & Armon (2022) found that XGBoost outperformed deep learning models (including TabNet) on 8 of 11 tabular datasets. However, their evaluation used relatively small datasets and did not include multimodal fusion settings. In our hybrid architecture, TabNet's advantage is its differentiable, embedding-producing design — not raw standalone tabular performance.

---

## 5. Interpretability Design

### 5.1 TabNet's Built-In Attention Masks

TabNet produces sparse attention masks at each decision step, showing exactly which features were selected and with what weight. These masks are available without any post-hoc computation.

**Mask aggregation formula:**

```
M_agg[i, j] = Σ_{step=1}^{N_steps} η_step · M_step[i, j]
```

Where `η_step` is the proportion of the prediction contributed by each step, `M_step[i, j]` is the attention weight for feature `j` at step `step` for sample `i`.

### 5.2 SHAP Compatibility

TabNet's attention masks provide local (instance-wise) feature importance natively. For global importance and theoretical guarantees (additivity, consistency), SHAP can be applied on top:

- **KernelSHAP**: Model-agnostic, works with TabNet's predict function directly
- **TreeSHAP**: Not applicable (TabNet is not tree-based)
- **GradientSHAP**: Applicable since TabNet is differentiable — use `shap.GradientExplainer`

The combination of TabNet masks + SHAP provides two complementary views:
1. **TabNet masks**: "Which features did the model attend to, and at which decision step?"
2. **SHAP values**: "What is each feature's marginal contribution to this specific prediction?"

### 5.3 Example Clinical Explanation Output

```
Patient: 72yo female, chief complaint "difficulty breathing"
Prediction: ESI Level 1 (Critical) — Confidence: 93%

── TabNet Structured Feature Analysis ──
Step 1 (critical vitals): SpO2=88% [mask=0.82], RR=28 [mask=0.71]
   → "Hypoxemia and tachypnea detected as primary acuity drivers"
Step 2 (secondary vitals): HR=118 [mask=0.65], SBP=92 [mask=0.48]
   → "Compensatory tachycardia with borderline hypotension"
Step 3 (demographics):    Age=72 [mask=0.55], Transport=AMBULANCE [mask=0.40]
   → "Elderly patient arriving by ambulance — higher acuity prior"
Step 4 (pain/clinical):   Pain=8 [mask=0.32], Temperature=38.9 [mask=0.28]
   → "Moderate pain with low-grade fever"
Step 5 (residual):        Gender=F [mask=0.12], Race features [mask<0.1]
   → "Minimal contribution from demographic features"

── BioClinicalBERT Text Attention ──
High-attention tokens: "difficulty breathing" (0.89), "worsening" (0.72),
                       "three days" (0.45), "productive cough" (0.38)

── SHAP Feature Contributions (top 5) ──
  SpO2 = 88%          → +0.52 (strongest driver toward Level 1)
  RR = 28             → +0.38
  Text: "difficulty breathing" → +0.35
  HR = 118            → +0.22
  Age = 72            → +0.15

Clinical Summary: Low oxygen saturation (88%) and elevated respiratory rate (28)
were the primary drivers, reinforced by the text description of worsening dyspnea
in an elderly patient. TabNet selected critical vitals in its first attention step,
consistent with clinical triage protocols that prioritize airway/breathing assessment.
```

### 5.4 Interpretability Advantage Over Model 1 (XGBoost)

| Aspect | Model 1 (XGBoost) | Model 2 (TabNet) |
|--------|-------------------|-------------------|
| Feature importance | TreeSHAP (post-hoc) | **Native attention masks + SHAP** |
| Per-step reasoning | Not available | **Step-wise attention shows reasoning progression** |
| Instance-wise selection | SHAP provides this | **Both masks and SHAP provide this** |
| Clinical narrative | Feature rankings only | **Step-by-step reasoning mirrors clinical assessment** |
| Computation cost | TreeSHAP is very fast | Masks are free; SHAP adds overhead |

The step-wise attention narrative is TabNet's strongest interpretability advantage — it mirrors how clinicians perform triage: first check airway/breathing (SpO2, RR), then circulation (HR, BP), then contextual factors (age, transport mode).

---

## 6. Class Imbalance Strategy

### 6.1 Class Distribution

| ESI Level | Count | Proportion | Inverse Frequency Weight |
|-----------|-------|------------|--------------------------|
| Level 1 (Resuscitation) | 878 | 9.6% | 2.60 |
| Level 2 (Emergent) | 3,300 | 36.1% | 0.69 |
| Level 3 (Urgent) | 4,907 | 53.6% | 0.47 |
| Level 4 (Less Urgent) | 64 | 0.7% | **35.70** |

### 6.2 TabNet's Built-In Class Weighting

The `pytorch_tabnet` library natively supports class weighting via the `weights` parameter in `.fit()`:

```python
# Option 1: Automatic inverse-frequency balancing
clf.fit(X_train, y_train, weights=1)  # auto-balances

# Option 2: Custom weights dict
clf.fit(X_train, y_train, weights={
    0: 2.60,   # Level 1
    1: 0.69,   # Level 2
    2: 0.47,   # Level 3
    3: 20.00   # Level 4 — capped at 20× to prevent instability
})
```

**Level 4 weight capping:** The raw inverse frequency for L4 is 35.7×, but weights above ~20× can destabilize training (gradient explosions on the 64 L4 samples). Capping at 20× is a practical compromise. This matches the strategy used in Model 1 and the team's overall approach.

### 6.3 Joint Model Class Weighting

In the hybrid model, class weights are applied to the **joint cross-entropy loss** (after fusion), not separately to each branch:

```python
# Class weights for the joint loss function
class_weights = torch.tensor([2.60, 0.69, 0.47, 20.00])
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Applied to fusion head output
logits = fusion_head(concat(bert_cls, tabnet_embedding))  # [B, 4]
loss = criterion(logits, labels)
```

### 6.4 Additional Imbalance Mitigation for Level 4

With only 64 Level 4 samples, class weighting alone may be insufficient. Additional strategies:

1. **Stratified sampling:** Ensure every training batch contains at least 1 L4 sample (custom sampler)
2. **SMOTE on structured features:** Generate synthetic L4 samples in the 15-dim structured feature space (applied before TabNet, not to text)
3. **Focal loss alternative:** Replace cross-entropy with focal loss (γ=2.0) to down-weight easy L3 samples and focus on hard minority cases
4. **Data augmentation for text:** Paraphrase L4 patient texts using back-translation to increase text diversity

**Recommended approach:** Start with class-weighted cross-entropy + stratified sampling. Add focal loss only if L4 recall remains below 60% after initial training.

---

## 7. Strengths & Limitations vs Model 1 (XGBoost)

### Strengths of TabNet Over XGBoost (Model 2 vs. Model 1)

- **End-to-end differentiable:** TabNet can be trained jointly with BioClinicalBERT via backpropagation through a single loss function. XGBoost requires a two-stage pipeline (freeze BERT → extract embeddings → train XGBoost separately), which prevents the text encoder from adapting to the structured features and vice versa. Joint training enables complementary representation learning.

- **Native interpretability with step-wise reasoning:** TabNet's attention masks provide a sequential narrative of feature selection that mirrors clinical triage protocols (check vitals first → demographics → contextual factors). XGBoost's feature importance requires post-hoc SHAP computation and does not show a reasoning progression.

- **Produces dense embeddings for fusion:** TabNet outputs a learned N_d-dimensional embedding that can be concatenated with the BERT [CLS] vector for a unified classifier. XGBoost outputs scalar predictions or leaf indices — fusing XGBoost with BERT requires the less elegant "stacking" approach where XGBoost's prediction becomes just another feature.

### Limitations of TabNet Compared to XGBoost

- **Higher computational cost:** TabNet requires GPU training and more hyperparameter tuning (N_steps, N_a, N_d, γ, λ_sparse, virtual batch size). XGBoost trains in minutes on CPU with fewer hyperparameters to tune. For our 9,146-sample dataset with 15 features, XGBoost's training efficiency is a meaningful advantage.

- **Less robust on small/noisy tabular data:** Multiple benchmarks (Shwartz-Ziv & Armon 2022; Grinsztajn et al. 2022) show XGBoost outperforming TabNet on datasets with <10K samples and mixed feature types. Our dataset sits at the boundary (9,146 samples) where this concern is real but not decisive.

- **Overfitting risk with 15 features and sparse attention:** With only 15 input features and 5 attention steps, TabNet's feature selection mechanism may not provide the same benefit as on high-dimensional datasets (e.g., 100+ features) where selecting relevant subsets is critical. With 15 features, most are clinically relevant, and the selection overhead may not pay off.

### When Each Model Is Preferred

| Scenario | Prefer XGBoost (Model 1) | Prefer TabNet (Model 2) |
|----------|--------------------------|--------------------------|
| Training speed priority | ✅ Minutes on CPU | ❌ Requires GPU |
| Standalone structured performance | ✅ Strong on small tabular data | ❌ May overfit with n=9K |
| End-to-end joint training | ❌ Not differentiable | ✅ Native backprop |
| Step-wise interpretability | ❌ No step narrative | ✅ Attention masks |
| Production simplicity | ✅ Fewer dependencies | ❌ PyTorch + TabNet stack |
| Feature interaction learning | ✅ Tree splits naturally capture interactions | ✅ Multi-step attention captures interactions differently |

---

## 8. Training Configuration (Recommended)

### 8.1 TabNet-Specific Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_d` | 64 | Feature transformer output dimension; moderate for 15 features |
| `n_a` | 64 | Attention dimension; matched to n_d per paper recommendation |
| `n_steps` | 5 | 5 sequential attention steps for 15 features (~3 features/step) |
| `gamma` (relaxation) | 1.5 | Allow moderate feature reuse across steps |
| `lambda_sparse` | 1e-3 | Sparsity regularization on attention masks |
| `n_independent` | 2 | Number of step-specific FC layers in feature transformer |
| `n_shared` | 2 | Number of shared FC layers across steps |
| `virtual_batch_size` | 32 | Ghost Batch Normalization sub-batch size |
| `momentum` | 0.7 | BN momentum for Ghost Batch Normalization |
| `mask_type` | "sparsemax" | Sparse attention (default); alternative: "entmax" |

### 8.2 Joint Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | AdamW | Two parameter groups |
| **LR — BERT encoder** | 2e-5 | Low LR for pre-trained weights |
| **LR — TabNet encoder** | 1e-3 | TabNet default; higher since training from scratch |
| **LR — Fusion head** | 1e-3 | Same as TabNet (random init) |
| **LR schedule** | Cosine annealing with warm-up (500 steps) | Stabilizes early training |
| **Weight decay** | 0.01 (BERT), 1e-5 (TabNet) | Standard per architecture |
| **Batch size** | 64 | Divisible by virtual_batch_size=32 |
| **Epochs** | 20 | TabNet typically needs more epochs than XGBoost |
| **Early stopping** | Patience=5 on validation macro-F1 | Prevent overfitting |
| **Gradient clipping** | Max norm = 1.0 | Stabilize BERT + TabNet joint training |
| **Precision** | FP16 mixed precision | Reduce memory, speed up training |
| **Loss function** | Cross-entropy with class weights [2.60, 0.69, 0.47, 20.00] | Address L1/L4 imbalance |
| **Data split** | 70/15/15 stratified on triage level | Match Model 1 splits |

### 8.3 TabNet Pre-Training (Optional)

TabNet supports self-supervised pre-training via a mask-then-predict objective (similar to masked language modeling but for tabular data). This can improve performance on small datasets:

```python
# Self-supervised pre-training (optional, may help with n=9,146)
unsupervised_model = TabNetPretrainer(
    n_d=64, n_a=64, n_steps=5,
    mask_type='sparsemax',
    pretraining_ratio=0.8  # mask 80% of features, predict them
)
unsupervised_model.fit(X_train_unlabeled)

# Transfer learned weights to supervised TabNet
supervised_model = TabNetClassifier(...)
supervised_model.load_weights_from_unsupervised(unsupervised_model)
```

**Recommendation:** Try pre-training as an ablation. If it improves validation macro-F1 by ≥1%, keep it. The MIMIC-IV-Ext dataset has ~9K labeled samples — pre-training on all 15 features may help TabNet learn better feature interactions before the supervised task.

### 8.4 Implementation Notes

**Library:** `pytorch_tabnet` (dreamquark-ai) — the standard PyTorch implementation
- Install: `pip install pytorch-tabnet`
- GitHub: [dreamquark-ai/tabnet](https://github.com/dreamquark-ai/tabnet)

**Custom integration required:** The default `pytorch_tabnet.TabNetClassifier` is a standalone model. For our hybrid architecture, we need to extract the TabNet **encoder** (without its built-in classification head) and wrap it in a custom PyTorch `nn.Module` alongside BioClinicalBERT:

```python
from pytorch_tabnet.tab_network import TabNet

class HybridBERTTabNet(nn.Module):
    def __init__(self, bert_model, tabnet_params, num_classes=4):
        super().__init__()
        self.bert = bert_model  # BioClinicalBERT / ModernBERT
        self.tabnet = TabNet(
            input_dim=15,
            output_dim=num_classes,  # will discard final layer
            **tabnet_params
        )
        # Fusion head
        fusion_dim = 768 + tabnet_params['n_d']  # 768 + 64 = 832
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, structured_features):
        # Text branch
        bert_out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        cls_embedding = bert_out.last_hidden_state[:, 0, :]  # [B, 768]

        # Structured branch (TabNet encoder only)
        tabnet_out, attention_masks = self.tabnet(structured_features)
        # tabnet_out: [B, N_d=64], attention_masks: list of [B, 15] per step

        # Fusion
        combined = torch.cat([cls_embedding, tabnet_out], dim=1)  # [B, 832]
        logits = self.fusion(combined)  # [B, 4]
        return logits, attention_masks
```

### 8.5 Estimated Training Time

| Component | Time (A100 40GB) |
|-----------|-----------------|
| TabNet self-supervised pre-training (optional) | ~5 min |
| Joint model training (20 epochs, batch=64) | ~1.5 hours |
| Inference on test set (1,372 samples) | <30 seconds |
| SHAP explanation generation (100 samples) | ~10 min |

**Total: ~2 hours** (including optional pre-training)

---

## Sources

### Primary Papers
- [Arik & Pfister (2021) — TabNet: Attentive Interpretable Tabular Learning (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/16826)
- [Arik & Pfister — TabNet (arXiv)](https://arxiv.org/abs/1908.07442)
- [Shwartz-Ziv & Armon (2022) — Tabular Data: Deep Learning Is Not All You Need](https://arxiv.org/pdf/2106.03253)

### Clinical TabNet Applications
- [CKD Detection with TabNet (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0933365725000880)
- [Heart Disease Prediction using Hybrid TabNet (Frontiers in Physiology, 2025)](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1665128/full)
- [GAT-Enhanced TabNet for Multi-Disease Prediction (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0169260725004973)

### Multimodal Fusion Precedents
- [BioClinicalBERT + TabTransformer on MIMIC-III (KTH / IEEE Big Data, 2024)](https://kth.diva-portal.org/smash/record.jsf?pid=diva2:1940625)
- [Multimodal Data Hybrid Fusion for Clinical Prediction (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11141806/)
- [MedPatch: Confidence-Guided Multi-Stage Fusion (2025)](https://arxiv.org/html/2508.09182v1)

### TabNet vs XGBoost Comparisons
- [TabNet vs XGBoost — Comparative Study (ResearchGate, 2025)](https://www.researchgate.net/publication/392509412_A_Comparative_Study_of_TabNet_and_XGBoost_for_Tabular_Data_Classification)
- [When Do Neural Nets Outperform Boosted Trees? (arXiv, 2023)](https://arxiv.org/pdf/2305.02997)

### Implementation
- [pytorch_tabnet (dreamquark-ai) — GitHub](https://github.com/dreamquark-ai/tabnet)
- [pytorch_tabnet documentation](https://dreamquark-ai.github.io/tabnet/generated_docs/pytorch_tabnet.html)

---

*Research compiled 2026-03-04 for Capstone Project: ED Triage Prediction using MIMIC-IV-Ext.*
