# RAG System Design: Hybrid Retrieval for ED Triage Explainability
## BioClinical Embeddings + BM25 + Cross-Encoder Reranking

**Status:** Proposed
**Project:** Explainable AI for Emergency Department Triage
**Component:** Retrieval-Augmented Generation (RAG)
**Date:** March 8, 2026

---

## 1. Overview

This document proposes the RAG component of the ED triage explainability system. The goal is to retrieve clinically relevant historical cases from MIMIC-IV-Ext at inference time, then pass those cases — alongside model predictions and SHAP feature importances — to a large language model that generates a grounded, evidence-based clinical explanation.

The core design principle is **architectural coherence**: the retrieval system uses the same clinical language model already loaded for triage prediction (`BioClinical ModernBERT`), so the embedding space for retrieval and the embedding space for prediction are identical. This eliminates the semantic mismatch that would arise from using a general-purpose commercial embedding model.

Retrieval is **hybrid**: dense semantic search over BioClinical embeddings is fused with BM25 sparse keyword retrieval via Reciprocal Rank Fusion (RRF). This ensures that critical medical terms (`"NSTEMI"`, `"SpO2 88%"`, `"tachypnea"`) are matched exactly, not dissolved into approximate semantic similarity. A cross-encoder reranker then rescores the top candidates before the final top-5 are passed to the LLM.

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                        PATIENT INPUT                             │
│   "{CC}. {HPI}. Past Medical History: {PMH}" + Structured Vitals │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   TRIAGE PREDICTION MODEL                        │
│   BioClinical ModernBERT (text) + TabNet/MLP (vitals)            │
│   → ESI Level (1–4), confidence, SHAP feature importances        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   HYBRID RAG RETRIEVAL                           │
│                                                                  │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │  DENSE RETRIEVAL        │  │  SPARSE RETRIEVAL (BM25)     │  │
│  │                         │  │                              │  │
│  │  BioClinical ModernBERT │  │  BM25 over indexed case text │  │
│  │  → query embedding      │  │  → keyword match scores      │  │
│  │  → ANN search (FAISS)   │  │  → top-50 candidates         │  │
│  │  → top-50 candidates    │  │                              │  │
│  └────────────┬────────────┘  └───────────────┬──────────────┘  │
│               │                               │                  │
│               └──────────────┬────────────────┘                  │
│                              ▼                                   │
│                   Reciprocal Rank Fusion (RRF)                   │
│                   → merged top-50 candidates                     │
│                              │                                   │
│                              ▼                                   │
│                   Cross-Encoder Reranker                         │
│                   → top-50 → top-5 final cases                   │
│                              │                                   │
│                              ▼                                   │
│         Supporting Cases (same ESI) + Contrastive Cases          │
│                        (adjacent ESI)                            │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                  CLINICAL REASONING (LLM)                        │
│                                                                  │
│  Input:  Patient data + ESI prediction + SHAP importances        │
│          + Retrieved supporting cases + Contrastive cases        │
│                                                                  │
│  Output: Evidence-grounded clinical explanation (300–400 words)  │
│          citing specific retrieved cases by ID                   │
│                                                                  │
│  Model:  Claude 3.5 Sonnet (via Amazon Bedrock)                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Dense Embedding** | BioClinical ModernBERT (`lindvalllab/BioClinical-ModernBERT-large`) | Clinical semantic vectors — same encoder as prediction model |
| **Sparse Retrieval** | BM25 (`rank_bm25`) | Exact medical keyword matching |
| **Vector Index** | FAISS (`faiss-cpu`) | ANN search over 9K–30K dense vectors |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Merge dense + sparse ranked lists |
| **Reranker** | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | Rescore top-50 → top-5 |
| **Historical Cases** | MIMIC-IV-Ext (~9,146 labeled ED visits) | Retrieval corpus |
| **LLM** | Claude 3.5 Sonnet (Bedrock) | Generate grounded clinical reasoning |

---

## 3. Embedding Strategy: Why BioClinical ModernBERT

### 3.1 Architectural Coherence

The triage prediction model encodes patient text using `BioClinical ModernBERT` to produce a [CLS] embedding that drives classification. If the RAG system uses a different embedding model (e.g., a general-purpose commercial embedding API), the retrieval and prediction live in entirely different vector spaces.

The consequence: a case that the prediction model finds semantically critical may score low in retrieval, and vice versa. Retrieved "similar" cases may not actually represent the same clinical reasoning the model applied.

**Solution:** Reuse the BioClinical ModernBERT encoder — already loaded in memory for prediction — as the embedding model for both indexing and query encoding. No additional model deployment. No semantic mismatch.

### 3.2 Query and Index Text Format

Both query construction and historical case indexing use the identical text format as the prediction model:

```
"{CC}. {HPI}. Past Medical History: {PMH}"
```

This is not optional. If the query omits HPI and PMH — using only chief complaint and vitals — the retrieval operates on a degraded representation compared to what the model actually saw. The HPI is often the most discriminative text: `"crushing chest pain radiating to left arm for 20 minutes"` carries acuity signal that `"chest pain"` alone does not.

Historical cases in the index must include HPI where available in MIMIC-IV. For cases where HPI is missing, the chief complaint alone is used and flagged in metadata.

### 3.3 Embedding Dimension

BioClinical ModernBERT outputs a 768-dimensional [CLS] vector. FAISS index uses `IndexFlatIP` (inner product, equivalent to cosine on normalized vectors) for exact search at our corpus size (~9K vectors), or `IndexIVFFlat` for approximate search if the corpus grows to 30K+.

---

## 4. BM25 Sparse Retrieval

### 4.1 Why Sparse Retrieval Matters for Medical Text

Dense embeddings encode semantic meaning but can conflate clinically distinct concepts. `"NSTEMI"` and `"chest pain"` are semantically related but are not the same — a model predicting ESI 1 for a confirmed NSTEMI should retrieve other NSTEMI cases, not just any chest-pain presentation.

BM25 operates on exact token matches weighted by term frequency and inverse document frequency. For clinical text where precise terminology matters — diagnosis codes, vital sign thresholds, drug names, procedure names — BM25 reliably surfaces exact matches that dense retrieval may rank lower.

### 4.2 BM25 Configuration

```python
from rank_bm25 import BM25Okapi
import nltk

def tokenize_clinical(text: str) -> list[str]:
    """
    Clinical-aware tokenizer:
    - Lowercase
    - Preserve numeric values (e.g. "88%" stays as token)
    - Preserve medical abbreviations (SpO2, HR, NSTEMI)
    - No stemming (clinical terms should match exactly)
    """
    tokens = nltk.word_tokenize(text.lower())
    return tokens

# Build BM25 index over all case documents
corpus = [tokenize_clinical(case['text']) for case in all_cases]
bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)  # Okapi BM25 defaults
```

**Parameter choices:**
- `k1 = 1.5`: Term frequency saturation. Standard value; clinical notes tend to repeat key terms (e.g., "chest pain" mentioned multiple times) so saturation is appropriate.
- `b = 0.75`: Document length normalization. Prevents long case summaries from being unfairly penalized vs. short ones.
- No stemming: `"tachycardic"` and `"tachycardia"` should be treated as distinct tokens. Clinical abbreviations are case-preserved at tokenization then lowercased, so `SpO2` and `spo2` map to the same token.

---

## 5. Reciprocal Rank Fusion

After dense retrieval and BM25 each return an ordered list of top-50 candidates, RRF merges the two ranked lists into a single unified ranking without requiring score calibration between the two systems.

### 5.1 RRF Formula

```
RRF_score(doc) = Σ_ranker  1 / (k + rank_in_ranker(doc))
```

Where `k = 60` (standard constant that dampens the influence of very high ranks).

### 5.2 Implementation

```python
def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],   # [(case_id, score), ...]
    sparse_results: list[tuple[str, float]],  # [(case_id, score), ...]
    k: int = 60
) -> list[tuple[str, float]]:
    """
    Merge dense and sparse ranked lists via RRF.
    Returns unified ranked list of (case_id, rrf_score).
    """
    scores = {}
    for rank, (case_id, _) in enumerate(dense_results, start=1):
        scores[case_id] = scores.get(case_id, 0) + 1 / (k + rank)
    for rank, (case_id, _) in enumerate(sparse_results, start=1):
        scores[case_id] = scores.get(case_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

The top-50 candidates from RRF are passed to the reranker. The RRF step adds negligible latency (~1ms) and requires no learned parameters.

---

## 6. Cross-Encoder Reranking

### 6.1 Why Rerank

Both dense retrieval and BM25 are **bi-encoders**: query and candidate are encoded independently, and similarity is measured post-hoc. This is computationally efficient but misses fine-grained interaction between query and candidate.

A **cross-encoder** encodes the `(query, candidate)` pair jointly, allowing full attention between query tokens and candidate tokens. This produces more accurate relevance scores at the cost of running one inference per candidate.

At top-50 candidates this is acceptable: 50 × 6ms ≈ 300ms on CPU, or ~50ms on GPU.

### 6.2 Reranker Choice

`cross-encoder/ms-marco-MiniLM-L-6-v2` — 22M parameters, fast, strong performance on passage retrieval benchmarks. Not domain-specific to clinical text, but it operates on the already-filtered top-50 where rough clinical relevance has been established. A domain-specific clinical cross-encoder would be the upgrade path if precision targets are not met.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query_text: str, candidates: list[dict]) -> list[dict]:
    """Rerank top-50 candidates, return top-5."""
    pairs = [(query_text, c['text']) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:5]]
```

---

## 7. Contrastive Retrieval

### 7.1 Design

Beyond retrieving cases that support the predicted ESI level, the system runs a second retrieval pass targeting the **adjacent ESI level** — cases with similar presentations but a different triage outcome. These contrastive cases enable the LLM to explain the decision boundary explicitly.

```
Query patient: dyspnea, HR 110, SpO2 94% → Predicted ESI 2

Supporting cases (ESI 2):  similar vitals, emergent outcome → "cases like this are Level 2"
Contrastive cases (ESI 1): similar vitals, critical outcome → "what would push to Level 1 is..."
```

This is structurally the most valuable addition for explanation quality. Without contrastive cases, the LLM can only affirm the prediction. With them, it can articulate the reasoning boundary.

### 7.2 Implementation

The contrastive retrieval uses the same hybrid pipeline with a metadata pre-filter applied before ANN search:

```python
# Supporting: filter to predicted ESI level
supporting = retrieve(
    query,
    metadata_filter={"esi_level": {"$eq": predicted_esi}}
)

# Contrastive: filter to adjacent ESI level
contrastive_esi = predicted_esi - 1 if predicted_esi > 1 else predicted_esi + 1
contrastive = retrieve(
    query,
    metadata_filter={"esi_level": {"$eq": contrastive_esi}},
    top_k=3
)
```

Two Pinecone queries. The contrastive results are explicitly labelled in the LLM prompt.

---

## 8. Vector Database: FAISS + Pinecone

### 8.1 Development: FAISS (Local)

During development and evaluation, FAISS runs locally:

```python
import faiss
import numpy as np

# Build index (cosine similarity via normalized inner product)
dimension = 768  # BioClinical ModernBERT output
index = faiss.IndexFlatIP(dimension)

# Add case embeddings
embeddings = np.array([case['embedding'] for case in all_cases], dtype='float32')
faiss.normalize_L2(embeddings)  # normalize for cosine similarity
index.add(embeddings)

# Query
query_vec = encode_query(patient_text)  # [1, 768]
faiss.normalize_L2(query_vec)
distances, indices = index.search(query_vec, k=50)
```

FAISS for 9K vectors with 768 dimensions fits in ~28MB RAM. Query latency: <5ms.

### 8.2 Production: Pinecone Serverless

For production deployment, FAISS is replaced with Pinecone serverless (no pod management, scales with usage).

**Pinecone API (current v3+ syntax):**

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)

# Create index (once)
pc.create_index(
    name="ed-triage-cases",
    dimension=768,           # BioClinical ModernBERT output dim
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Query
index = pc.Index("ed-triage-cases")
results = index.query(
    vector=query_embedding.tolist(),
    top_k=50,
    filter={"esi_level": {"$eq": predicted_esi}},
    include_metadata=True
)
```

Note: `pinecone.init()` is deprecated. All Pinecone operations use the `Pinecone()` client class.

---

## 9. Data Pipeline

### 9.1 Case Document Structure

Each MIMIC-IV-Ext case is indexed with a rich text representation and structured metadata:

```json
{
  "case_id": "stay_28451",
  "text": "67-year-old male presenting with chest pain. History of present illness: crushing substernal chest pain radiating to left arm, onset 45 minutes prior to arrival. Past Medical History: hypertension, hyperlipidemia. HR 118 bpm, BP 145/92 mmHg, RR 20, Temp 98.4°F, SpO2 97%.",
  "embedding": [0.032, -0.118, ...],
  "bm25_tokens": ["67-year-old", "male", "chest", "pain", "crushing", ...],
  "metadata": {
    "esi_level": 2,
    "age": 67,
    "gender": "male",
    "chief_complaint": "chest pain",
    "heart_rate": 118,
    "systolic_bp": 145,
    "diastolic_bp": 92,
    "respiratory_rate": 20,
    "temperature": 98.4,
    "spo2": 97,
    "diagnosis": "NSTEMI",
    "disposition": "admitted",
    "los_hours": 48.2,
    "has_hpi": true,
    "year": 2015
  }
}
```

Key differences from a minimal case representation:
- Full `{CC}. {HPI}. PMH` text (matches model input format)
- `has_hpi` flag distinguishes complete vs. CC-only cases
- `year` field enables temporal filtering (prefer post-2015 cases to reduce protocol drift)
- Pre-computed `bm25_tokens` stored to avoid re-tokenization at query time

### 9.2 Indexing Pipeline

```
1. Load MIMIC-IV-Ext from S3
2. For each case (n ≈ 9,146):
   a. Construct text: "{CC}. {HPI}. PMH: {PMH}"
   b. Encode with BioClinical ModernBERT → [768] embedding
   c. Tokenize for BM25 corpus
   d. Upsert to Pinecone (production) or FAISS (dev) with metadata
3. Build BM25 index from tokenized corpus
4. Serialize BM25 index to disk (pickle)
5. Validate: run 20 test queries, manual spot-check
```

**Estimated indexing time (A100):** ~15 minutes for 9K cases
**Storage:** 9K × 768 × 4 bytes ≈ 28MB (FAISS) + BM25 index ~10MB

---

## 10. Query Construction

The query text uses the identical format as the prediction model input:

```python
def build_query_text(patient: dict) -> str:
    parts = [patient['chief_complaint']]
    if patient.get('hpi'):
        parts.append(patient['hpi'])
    if patient.get('pmh'):
        parts.append(f"Past Medical History: {patient['pmh']}")

    vitals = (
        f"HR {patient['heart_rate']} bpm, "
        f"BP {patient['systolic_bp']}/{patient['diastolic_bp']} mmHg, "
        f"RR {patient['respiratory_rate']}, "
        f"Temp {patient['temperature']}°F, "
        f"SpO2 {patient['spo2']}%."
    )
    parts.append(vitals)
    return " ".join(parts)
```

The structured vitals are appended as natural language text so BM25 can match on numeric values and unit strings (e.g., `"SpO2 88%"` matches cases with the same SpO2 value).

---

## 11. LLM Integration: Grounded Clinical Reasoning

### 11.1 Hallucination Prevention

The LLM prompt is structured so that the model is constrained to the retrieved evidence:

```python
SYSTEM_PROMPT = """
You are a clinical decision support assistant. Your task is to explain an
emergency department triage prediction using evidence from historical cases.

Rules:
1. Only reference evidence from the retrieved cases provided below.
2. Do not introduce clinical knowledge not present in the patient data,
   SHAP features, or retrieved cases.
3. When citing a case, reference it by its Case ID (e.g., Case #28451).
4. If the retrieved evidence does not support a claim, do not make the claim.
5. Acknowledge uncertainty explicitly when the evidence is mixed.
"""

def build_prompt(patient, prediction, shap_features, supporting, contrastive):
    supporting_text = "\n".join([
        f"Case #{c['case_id']}: {c['text']} → ESI {c['metadata']['esi_level']}, "
        f"Diagnosis: {c['metadata']['diagnosis']}, Disposition: {c['metadata']['disposition']}"
        for c in supporting
    ])
    contrastive_text = "\n".join([
        f"Case #{c['case_id']}: {c['text']} → ESI {c['metadata']['esi_level']}, "
        f"Diagnosis: {c['metadata']['diagnosis']}, Disposition: {c['metadata']['disposition']}"
        for c in contrastive
    ])
    top_shap = "\n".join([
        f"  {feat}: {val:+.3f}" for feat, val in shap_features[:5]
    ])

    return f"""
PATIENT:
{build_query_text(patient)}

TRIAGE PREDICTION: ESI Level {prediction['esi_level']} — Confidence: {prediction['confidence']:.0%}

MODEL FEATURE IMPORTANCES (SHAP):
{top_shap}

SUPPORTING CASES (similar presentations, same ESI level):
{supporting_text}

CONTRASTIVE CASES (similar presentations, adjacent ESI level):
{contrastive_text}

Generate a clinical explanation (300–400 words) for this triage prediction.
Cite specific cases by ID. Explain what the contrastive cases reveal about
the decision boundary.
"""
```

### 11.2 Model Choice

**Claude 3.5 Sonnet** via Amazon Bedrock. The structured grounding prompt reduces the output token count, keeping cost low. Haiku is viable for cost optimization if output quality is acceptable; Sonnet is the default for clinical reasoning quality.

---

## 12. Embedding Query Cache

In an ED setting, the same chief complaint patterns dominate: chest pain, dyspnea, abdominal pain, altered mental status. Encoding the same query text repeatedly wastes compute and Bedrock API calls.

An LRU cache on query embeddings, keyed by the normalized query text, eliminates redundant encoding for semantically identical queries:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=512)
def encode_cached(text_hash: str, text: str) -> np.ndarray:
    return model.encode(text)

def get_query_embedding(text: str) -> np.ndarray:
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return encode_cached(text_hash, text)
```

Cache size of 512 covers the most common ED presentations. No persistent storage needed — in-memory cache per session.

---

## 13. API Design

```python
class EDTriageRAG:
    """
    Hybrid RAG system for ED triage explainability.
    Dense (BioClinical ModernBERT) + BM25 + Cross-Encoder Reranking.
    """

    def __init__(
        self,
        encoder_model,          # BioClinical ModernBERT (shared with prediction model)
        faiss_index,            # FAISS index (dev) or Pinecone index (prod)
        bm25_index,             # BM25Okapi instance
        case_store: dict,       # case_id → case dict lookup
        reranker_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        bedrock_region: str = 'us-east-1'
    ):
        ...

    def retrieve(
        self,
        patient: dict,
        predicted_esi: int,
        top_k_dense: int = 50,
        top_k_sparse: int = 50,
        top_k_final: int = 5
    ) -> dict:
        """
        Returns:
            {
                'supporting': [case_dict × top_k_final],
                'contrastive': [case_dict × 3],
                'retrieval_time_ms': int
            }
        """
        ...

    def explain(
        self,
        patient: dict,
        prediction: dict,
        shap_features: list[tuple[str, float]]
    ) -> dict:
        """
        Full explainability pipeline.

        Returns:
            {
                'retrieved': {'supporting': [...], 'contrastive': [...]},
                'clinical_reasoning': str,
                'retrieval_time_ms': int,
                'reasoning_time_ms': int
            }
        """
        ...
```

---

## 14. Evaluation Strategy

### 14.1 Retrieval Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Precision@5** | Are top-5 retrieved cases clinically similar? (manual review) | ≥ 0.75 |
| **ESI Concordance@5** | What fraction of top-5 share the query's ground-truth ESI? | ≥ 0.70 |
| **Recall@10 (ESI-filtered)** | Of all ESI-matched cases in corpus, how many appear in top-10? | ≥ 0.50 |
| **Query latency p95** | End-to-end retrieval (dense + BM25 + RRF + rerank) | < 500ms |
| **BM25 lift** | Does hybrid beat dense-only on Precision@5? | > 5% relative |

Note: `Precision@5` and `ESI Concordance@5` measure different things and should be tracked separately. High semantic similarity does not imply same ESI level — a 67yo with chest pain at ESI 3 (stable, non-cardiac) is semantically similar to a 67yo with NSTEMI at ESI 2.

### 14.2 Reasoning Quality Metrics

**Automated:**
- Citation accuracy: Does the LLM's reasoning reference case IDs that actually appear in the retrieved set?
- ESI label consistency: Does the LLM's explanation agree with the predicted ESI level?
- Length compliance: Output within 300–400 word target?

**Manual (100-case evaluation):**
- Clinical accuracy (per team clinician review)
- Actionability: Does the reasoning suggest clear next steps?
- Boundary explanation: Does the reasoning articulate why the contrastive cases differ?

### 14.3 Evaluation Dataset

100 held-out cases from MIMIC-IV-Ext test split:
- 25 cases per ESI level (1–4), stratified
- Mix of common (chest pain, dyspnea) and less common (overdose, trauma) chief complaints
- Include borderline cases where ESI 1/2 and 2/3 boundaries are clinically ambiguous

For each case: run retrieval, record metrics, pass to LLM, evaluate reasoning output.

### 14.4 Ablation

Run the following configurations on the 100-case eval set to confirm each component adds value:

| Configuration | Precision@5 | ESI Concordance@5 | Notes |
|---------------|-------------|-------------------|-------|
| Dense only (BioClinical) | baseline | baseline | |
| BM25 only | ? | ? | |
| Dense + BM25 (RRF) | ? | ? | Expected improvement |
| Dense + BM25 + Reranker | ? | ? | Expected best |

If the reranker does not improve over RRF by ≥ 2% Precision@5, drop it (latency vs. accuracy tradeoff).

---

## 15. Dependencies

```txt
# Core retrieval
rank-bm25==0.2.2
faiss-cpu==1.8.0
sentence-transformers==3.0.0    # cross-encoder reranker

# Production vector DB
pinecone==5.0.0                 # use Pinecone() client, not pinecone.init()

# AWS
boto3==1.34.0

# Utilities
numpy==1.26.0
pandas==2.1.0
nltk==3.8.1
python-dotenv==1.0.0
```

Note: `BioClinical ModernBERT` is loaded via the `transformers` library already required by the prediction model. No additional model dependencies.

---

## 16. Cost Analysis

### 16.1 Development (6 weeks)

| Component | Usage | Cost |
|-----------|-------|------|
| FAISS (local) | All retrieval | $0 |
| BioClinical ModernBERT (local) | All embedding | $0 |
| BM25 (local) | All sparse retrieval | $0 |
| Cross-encoder (local) | All reranking | $0 |
| Bedrock Claude 3.5 Sonnet | 1,000 reasoning calls × ~1.5K tokens output | ~$4.50 |
| **Total** | | **~$5** |

### 16.2 Production Estimate (1,000 patients/day)

| Component | Monthly Usage | Cost |
|-----------|---------------|------|
| Pinecone Serverless | 30K queries × 50 candidates | ~$15 |
| BioClinical Embeddings (SageMaker endpoint) | 30K queries × 300 tokens | ~$8 |
| Bedrock Claude 3.5 Sonnet | 30K reasoning × 1.5K tokens | ~$135 |
| **Total/month** | | **~$158** |

---

## 17. Implementation Sequence

| Step | Task | Deliverable | Success Criteria |
|------|------|-------------|-----------------|
| 1 | Data preparation | Cleaned cases with CC + HPI + PMH in `{CC}. {HPI}. PMH: {PMH}` format | All cases have complete text; `has_hpi` flag accurate |
| 2 | FAISS + BM25 index build | `build_index.py` | 9K cases indexed; test queries return sensible results |
| 3 | Hybrid retrieval implementation | `retrieval.py` with RRF fusion | Retrieval latency < 500ms p95 |
| 4 | Cross-encoder reranker integration | Reranker added to pipeline | Top-5 quality confirmed via spot-check |
| 5 | Contrastive retrieval | Second query pass with ESI filter | Contrastive cases are clinically distinct from supporting cases |
| 6 | LLM prompt + reasoning | `generate_reasoning.py` | LLM cites actual case IDs; no detectable hallucinations |
| 7 | Ablation evaluation | Evaluation notebook | BM25 + reranker each contribute positive Precision@5 lift |
| 8 | End-to-end integration | `EDTriageRAG` class | Full pipeline < 2 seconds p95; clean API for Streamlit |

---

## 18. Open Questions

1. **BM25 tokenizer:** Standard NLTK word tokenizer vs. clinical-specific tokenizer (e.g., `scispaCy`)? Recommendation: start with NLTK, evaluate if clinical abbreviation handling is a bottleneck.
2. **Reranker domain gap:** `ms-marco-MiniLM` is not clinical-domain. If ablation shows it hurts, replace with a BioClinical cross-encoder fine-tuned on MIMIC pairs.
3. **Optimal top_k for RRF input:** Top-50 per ranker is standard. With 9K corpus this may be reducible to 20 without recall loss — test empirically.
4. **Contrastive ESI direction:** For ESI 1 predictions, there is no ESI 0. Contrastive direction is always toward less acute (ESI 2). Handle edge case in code.
5. **Temporal filter threshold:** `year >= 2015` proposed to reduce protocol drift. Validate that this doesn't disproportionately reduce representation of rare ESI levels.

---

## 19. References

### Retrieval Methods
- Robertson, S. & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
- Cormack, G.V., Clarke, C.L.A. & Buettcher, S. (2009). "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods." *SIGIR 2009*.
- Nogueira, R. & Cho, K. (2019). "Passage Re-ranking with BERT." arXiv:1901.04085.

### Hybrid RAG
- Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
- Ma, X. et al. (2022). "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods." *ACM Computing Surveys*.
- Gao, Y. et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey." arXiv:2312.10997.

### Clinical Embedding
- Lindvall, M. et al. (2025). "BioClinical ModernBERT: A Clinical Language Model." Karolinska / lindvalllab, HuggingFace: `lindvalllab/BioClinical-ModernBERT-large`.
- Alsentzer, E. et al. (2019). "Publicly Available Clinical BERT Embeddings." *NAACL Clinical NLP Workshop*.

### Vector Search
- Johnson, J., Douze, M. & Jégou, H. (2021). "Billion-Scale Similarity Search with GPUs." *IEEE Transactions on Big Data*.

---

*Compiled March 8, 2026 — Capstone Project: Explainable AI for ED Triage using MIMIC-IV-Ext.*
