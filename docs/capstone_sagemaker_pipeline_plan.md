# SageMaker Training Pipeline Plan

## Purpose
Agreed MVP plan for the capstone training pipeline. Intended for team review and PR approval.

## Goal
Build a **reproducible SageMaker training pipeline** that proves the workflow works end to end using a mock model on CPU instances. Once validated, teammates swap in real training scripts (e.g., arch4: BioClinicalBERT + LightGBM) and GPU instances.

The pipeline uses a **single pipeline definition** — different architectures are supported by swapping the training script, not by creating separate pipelines.

## MVP Scope
1. Read a small dataset from S3
2. Run a preprocessing step
3. Run a mock training step on a CPU instance
4. Save a model artifact to S3
5. Be executable from any environment with a configured AWS profile (local machine or Studio)
6. Be easy for a teammate to reuse by swapping the training script

## Out of Scope
- Real LLM fine-tuning or GPU training
- Inference endpoints or explainability
- RAG integration
- GitHub Actions or automated triggers
- Production security hardening

## High-Level Approach

- **Region:** `us-east-1`
- **Dev environment:** SageMaker JupyterLab (default instance: `ml.c5.2xlarge`)
- **Orchestration:** SageMaker Pipelines via the Python SDK
- **Source of truth:** GitHub
- **Current execution mode:** JupyterLab (scripts run locally via notebook)
- **Target execution mode:** Managed SageMaker Pipeline jobs (blocked on quota — see Phase 6)

### Execution Modes

The pipeline scripts support two execution modes with zero code changes:

| | JupyterLab (current) | Pipeline jobs (target) |
|---|---|---|
| **How** | `python scripts/preprocess.py` from notebook/terminal | `pipeline.start()` → SageMaker launches containers |
| **Compute** | Studio app instance (`ml.c5.2xlarge`) | Ephemeral job instances (`ml.m5.xlarge` etc.) |
| **Data I/O** | Local filesystem + boto3 to S3 | SageMaker mounts S3 ↔ `/opt/ml/` automatically |
| **Quota** | Studio JupyterLab app quotas (available) | Processing/Training job quotas (**currently 0**) |

Scripts read paths from environment variables. JupyterLab sets local overrides; managed jobs use `/opt/ml/` defaults.

### Pipeline Shape
1. **ProcessingStep** — reads feature CSV from S3, selects columns, builds triage text, creates 3-class target, splits into train/val/test, writes to S3
2. **TrainingStep** — reads splits, runs training script (mock or real), writes model artifacts to S3

### Architecture Support
One pipeline, swap scripts. To train a different architecture:
- Point the TrainingStep at a different script file (e.g., `train_mock.py` → `train_arch4.py`)
- Change `TrainingInstanceType` parameter (CPU → GPU)
- Pass different hyperparameters

The pipeline definition itself does not change.

## Team Handoff Model

**Pipeline Lead** owns: domain setup, Shared Space, pipeline structure, script scaffolds, documentation.

**Fine-Tuning Lead** owns: replacing mock training with real model loading, fine-tuning logic, hyperparameters, and GPU instance type.

The pipeline structure should **not** change significantly when real training logic is added.

## Naming Convention
Pattern: `[service]-[project]-[env]-[purpose]`

| Resource | Name |
|---|---|
| Domain | `sm-edtriage-dev-domain` |
| Shared space | `sm-edtriage-shared` |
| Pipeline | `edtriage-train-pipeline` |
| S3 bucket (existing) | `ed-triage-capstone-group7` |

## S3 Layout
Within `s3://ed-triage-capstone-group7/`:
- `Data_Output/consolidated_dataset_features.csv` — input (existing)
- `Data_Output/splits/` — preprocessing output (train/val/test CSVs)
- `models/` — training artifacts

## Repository Structure
Additive — lives alongside the existing `src/` directory:

```text
ed_triage_ai/
  pipeline/
    pipeline_definition.py   # defines the SageMaker Pipeline
    run_pipeline.py           # upserts and starts an execution
  scripts/
    preprocess.py             # entry point for ProcessingStep
    train_mock.py             # mock training script (CPU smoke test)
    train_arch4.py            # real arch4 training script (swapped in later)
  notebooks/
    smoke_test_local.ipynb    # runs preprocess + train locally from JupyterLab
  src/                        # existing code
  docs/
  README.md
```

`scripts/` holds entry-point scripts that work in both JupyterLab (via env-var overrides) and SageMaker managed containers (via `/opt/ml/` defaults). Different training scripts live side by side — `run_pipeline.py` selects which one to use.

## Phase Plan

### Phase 1 — AWS Foundation
Set up the environment needed to start coding.

Tasks:
- Create SageMaker Domain (`sm-edtriage-dev-domain`)
- Create or identify the SageMaker execution role for the domain
- Create Shared Space (`sm-edtriage-shared`) (optional — for shared Studio use)
- Ensure each team member has a configured AWS profile with credentials that can assume the execution role
- Confirm the pipeline can be triggered from a local machine using the AWS profile
- Verify SageMaker instance quotas for `ml.m5.xlarge` (processing and training)
  - **Status:** quota is 0 for both Processing and Training jobs. JupyterLab app quotas are available as interim workaround.

#### Execution Role Permissions

| Service | Access | Reason |
|---|---|---|
| S3 | Read/Write on project bucket | Data and artifact I/O |
| SageMaker | Full access | Pipelines, training, processing |
| ECR | Pull | Pre-built container images |
| CloudWatch Logs | Write | Step-level debugging |
| IAM PassRole | Pass role to SageMaker | Required for job execution |

For development: **AmazonSageMakerFullAccess** + S3 access scoped to the project bucket.

Completion gate:
- [x] Domain active
- [ ] Each team member can run `aws sts get-caller-identity` with their configured profile
- [x] Execution role ARN known: `arn:aws:iam::478502030741:role/service-role/SageMaker-ExecutionRole-20260311T231755`
- [x] Instance quotas checked — job quotas are 0; JupyterLab quotas available as interim
- [x] Data in `s3://ed-triage-capstone-group7/Data_Output/consolidated_dataset_features.csv`

### Phase 2 — Preprocessing Script
Write a standalone script for a SageMaker Processing job.

Input: `consolidated_dataset_features.csv` (8,383 rows, existing in S3).

- Select BASE_COLUMNS (`stay_id`, `triage`, raw vitals, `pain`, `pain_missing`, `age`, `arrival_transport`, `chiefcomplaint`, `HPI`)
- Build `triage_text` using CC-emphasized format (CC_2x + HPI, no PMH)
- Create `triage_3class` target (merge ESI 4 into ESI 3 → 3-class)
- Stratified train/val/test split (80/10/10)
- Write splits to `/opt/ml/processing/output/train/`, `/opt/ml/processing/output/validation/`, `/opt/ml/processing/output/test/`

This script does **not** do feature scaling or imputation — the training script handles that using train-set statistics to prevent leakage.

Completion gate:
- [ ] Runs locally in Studio on sample data
- [ ] Output splits contain expected columns and correct row counts

### Phase 3 — Mock Training Script
Write a standalone mock training script for a SageMaker Training job.

The mock script proves the SageMaker training contract works on CPU. It reads the same input format as the real arch4 script but uses a trivial model.

- Read train/val CSVs from `SM_CHANNEL_TRAIN` / `SM_CHANNEL_VALIDATION`
- Fit structured stats on train set, transform features via `transform_structured()`
- Tokenize `triage_text` with `prajjwal1/bert-tiny` (4M params, 128-dim)
- Train a trivial model: BERT-tiny → mean pool → Linear(128 → 3), 1 epoch, CPU
- Skip LightGBM cross-fitting (not needed for smoke test)
- Save `model.pt` to `SM_MODEL_DIR`

The real `train_arch4.py` (swapped in later) adds: BioClinicalBERT, LightGBM 5-fold cross-fitting, two-phase training, mixed precision, GPU.

Completion gate:
- [ ] Runs locally in Studio on processed sample data
- [ ] Model artifact produced in expected path

### Phase 4 — Pipeline Assembly
Wire both scripts into a SageMaker Pipeline.

- Define `ProcessingStep` and `TrainingStep`
- Pass processed data from step 1 to step 2
- Expose **Pipeline Parameters** (see below)
- One Python entry point to upsert and execute

#### Pipeline Parameters
These allow the team to change behavior at execution time without editing the pipeline definition:

| Parameter | Mock Default | Real (arch4) |
|---|---|---|
| `InputDataUri` | `s3://ed-triage-capstone-group7/Data_Output/consolidated_dataset_features.csv` | same or different dataset |
| `ProcessingInstanceType` | `ml.m5.xlarge` | same or larger |
| `TrainingInstanceType` | `ml.m5.xlarge` | `ml.g5.xlarge`+ |
| `TrainingInstanceCount` | `1` | 1+ |
| `TrainingScript` | `train_mock.py` | `train_arch4.py` |

Completion gate:
- [ ] Pipeline upserts without errors
- [ ] Both steps wired correctly in `pipeline.definition()`
- [ ] Pipeline visible in Studio Pipelines UI

### Phase 5 — Smoke Test (JupyterLab)
Prove the scripts work end to end from JupyterLab, using `notebooks/smoke_test_local.ipynb`.

This is the interim smoke test while job quotas are 0.

- [ ] `preprocess.py` produces train/val/test splits with correct row counts and columns
- [ ] `train_mock.py` produces `model.pt` and `structured_stats.json`
- [ ] Both scripts run to completion without errors
- [ ] A teammate can reproduce by running the notebook

### Phase 6 — Quota Request & Pipeline Execution (target state)
Request managed-job quotas and run the full SageMaker Pipeline.

Quota requests (Service Quotas → SageMaker, `us-east-1`):
- `ml.m5.large for processing job usage` → request ≥ 1
- `ml.m5.large for training job usage` → request ≥ 1
- (optional) `ml.g4dn.2xlarge for training job usage` → for arch4 GPU training later

Once approved:
- [ ] `python pipeline/run_pipeline.py` executes the pipeline with managed jobs
- [ ] Processed data and model artifact appear in S3
- [ ] Logs visible in CloudWatch
- [ ] Pipeline visible in Studio Pipelines UI
- [ ] A second run with different parameters also succeeds

## Definition of Done

**Interim (Phase 5):**
- Studio environment usable by the team
- Scripts run end to end from JupyterLab notebook
- Mock model artifact produced locally
- Training script is swappable without changing the pipeline definition

**Target (Phase 6):**
- Pipeline can be created/updated from Python
- Pipeline runs end to end as managed SageMaker jobs without console intervention
- Mock model artifact produced in S3
- Logs visible in CloudWatch

## Risks
| Risk | Mitigation |
|---|---|
| Environment setup takes too long | Keep scope to Studio + S3 + two pipeline steps |
| Real training adds complexity | Mock script mirrors the real script's I/O contract |
| Team confusion about ownership | This document + PR approval = agreed plan |
| Instance quota is 0 for managed jobs | **Active blocker.** Using JupyterLab execution as interim. Quota request planned in Phase 6. Scripts work in both modes — no code changes needed when quotas arrive. |
| Different architectures need different pipelines | One pipeline, swap scripts — architecture lives in `scripts/`, not in pipeline definition |

## Resolved Decisions
- **Mock model:** `prajjwal1/bert-tiny` (HuggingFace transformer, not sklearn) — keeps the handoff structurally identical
- **Data format:** CSV (matches existing pipeline)
- **Reference architecture:** Arch4 v1 (BioClinicalBERT + LightGBM fusion, pruned 15 features, CC_2x+HPI text)
- **Architecture swapping:** One pipeline definition, swap training scripts via `TrainingScript` parameter
- **Preprocessing input:** `consolidated_dataset_features.csv` (already cleaned, feature engineering partially done in preprocessing and partially in training script)
- **Dual execution:** Scripts support both JupyterLab (env-var path overrides) and managed SageMaker jobs (`/opt/ml/` defaults) with zero code changes
- **Interim compute:** JupyterLab on `ml.c5.2xlarge` while managed-job quotas are 0
- **Default JupyterLab instance:** `ml.c5.2xlarge` (4 vCPU, 16 GB — sufficient for mock and structured features)

## Current Status

| Phase | Status |
|---|---|
| Phase 1 — AWS Foundation | ✅ Done (quota blocker noted) |
| Phase 2 — Preprocessing Script | ✅ Done (`scripts/preprocess.py`) |
| Phase 3 — Mock Training Script | ✅ Done (`scripts/train_mock.py`) |
| Phase 4 — Pipeline Assembly | ✅ Done (`pipeline/pipeline_definition.py`, `pipeline/run_pipeline.py`) |
| Phase 5 — Smoke Test (JupyterLab) | ⏳ Next — run `notebooks/smoke_test_local.ipynb` |
| Phase 6 — Quota Request & Pipeline Execution | Not started |

## Next Step
Run the JupyterLab smoke test (Phase 5) using `notebooks/smoke_test_local.ipynb` on `ml.c5.2xlarge`.
