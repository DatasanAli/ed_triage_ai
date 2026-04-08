# RAG System Design — ED Triage AI

**Status:** Implemented and deployed
**Last updated:** 2026-04-07

---

## Overview

The RAG component retrieves the top-5 most similar historical ED cases from a Pinecone vector index for each incoming patient. These cases are passed to the LLM reasoning node as grounding evidence, and surfaced in the UI alongside SHAP attributions and clinical rationale.

---

## Architecture

```
New Patient
     │
     ▼
build_query_text()          ← structured text: demographics + CC + vitals + HPI
     │
     ▼
Bedrock Titan Embed          ← amazon.titan-embed-text-v2:0  (1024-dim)
     │
     ▼
Pinecone Query               ← cosine similarity, top_k=5, index: ed-triage-cases
     │
     ▼
Retrieved Cases              ← case_id, score, metadata (ESI, vitals, diagnosis, outcome)
     │
     ├──→ format_cases_for_prompt()  →  analyze_node (LLM grounding)
     └──→ TriageResponse.similar_cases  →  Streamlit UI
```

---

## Components

| Component | Technology | Detail |
|-----------|------------|--------|
| Embedding model | Amazon Bedrock Titan Embed v2 | `amazon.titan-embed-text-v2:0`, 1024-dim, cosine similarity |
| Vector database | Pinecone (serverless) | Index: `ed-triage-cases`, ~8,383 vectors |
| Historical cases | MIMIC-IV-Ext ED encounters | De-identified, ESI 1–3 |
| Auth | AWS Secrets Manager | Pinecone API key at `prod/pinecone/api_key` |
| Client | `src/retreival/retrieval.py` | `EDTriageRAG` class, lazy-initialized |

---

## Query Text Format

The query text must exactly mirror the format used at index time — semantic alignment between stored and query vectors depends on this.

```
PATIENT: Gender: Female, Age: 68
CHIEF COMPLAINT: CHEST PAIN
VITALS: HR 110 bpm, BP 135/85 mmHg, RR 18, Temp 98.6F, SpO2 99%
HISTORY: <hpi if present>
PAST MEDICAL HISTORY: <pmh if present>
```

Built by `EDTriageRAG.build_query_text()` in `src/retreival/retrieval.py`, which mirrors `build_embedding_text()` from the indexing pipeline.

---

## Retrieval

`EDTriageRAG.retrieve_cases(patient, top_k=5)` returns a list of dicts:

```python
[
    {
        "case_id": "stay_28451",
        "score": 0.89,           # cosine similarity
        "metadata": {
            "triage_level": 2,
            "chief_complaint": "CHEST PAIN",
            "heart_rate": 118.0,
            "sbp": 145.0,
            "dbp": 92.0,
            "spo2": 97.0,
            "icd_title": "NSTEMI",
            "disposition": "ADMITTED",
            "patient_info": "Gender: Male, Age: 67",
            ...
        }
    },
    ...
]
```

Optional parameters:
- `exclude_id` — skip a case already in the index (prevents self-match)
- `min_score` — cosine similarity threshold (default 0.0)
- `filter` — Pinecone metadata filter, e.g. `{"triage_level": {"$in": [1, 2]}}`

---

## Integration with LangGraph

`retrieve_node` in `src/agents/nodes.py` runs in **parallel** with `predict_node` (fan-out from START). It calls `retrieve_cases()` and `format_cases_for_prompt()`, storing results in `TriageState`:

- `state["similar_cases"]` — structured list passed to `synthesize_node` → `TriageResponse`
- `state["cases_text"]` — formatted text block injected into the LLM prompt in `analyze_node`

The LLM sees cases as:
```
SIMILAR CASE 1 (similarity: 0.89):
  Patient: Gender: Male, Age: 67 | ESI: 2 | Chief Complaint: CHEST PAIN
  Vitals: HR 118 bpm, BP 145/92 mmHg, SpO2 97%
  Diagnosis: NSTEMI
  Outcome: ADMITTED
```

---

## Index Build

Embeddings were generated offline using `src/embeddings/`:

1. `data_prep.py` — structured ~8,383 MIMIC-IV-Ext cases into the text format above
2. `generate_embeddings.py` — batched Titan embedding calls, output to `embeddings_output.jsonl`
3. `upload_to_pinecone.py` — upserted vectors + metadata to `ed-triage-cases`

The raw embedding files are excluded from git (`.gitignore`) and archived in S3.

---

## Configuration

| Env Var | Default | Notes |
|---------|---------|-------|
| `PINECONE_INDEX_NAME` | `ed-triage-cases` | Set in `.env` |
| `AWS_REGION` | `us-east-1` | Titan + Secrets Manager region |
| `AWS_PROFILE` | — | Local dev only; SageMaker uses instance role |

Pinecone API key is **not** stored in `.env` — it is fetched at runtime from AWS Secrets Manager (`prod/pinecone/api_key`).
