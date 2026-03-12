# SageMaker Training Pipeline Plan

## Purpose
Agreed MVP plan for the capstone training pipeline. Intended for team review and PR approval.

## Goal
Build a **reproducible SageMaker training pipeline** that proves the workflow works end to end using a mock model on CPU instances. Once validated, the Fine-Tuning Lead swaps in real LLM training logic and GPU instances.

## MVP Scope
1. Read a small dataset from S3
2. Run a preprocessing step
3. Run a mock training step on a CPU instance
4. Save a model artifact to S3
5. Be executable from SageMaker Studio
6. Be easy for a teammate to reuse by swapping the training script

## Out of Scope
- Real LLM fine-tuning or GPU training
- Inference endpoints or explainability
- RAG integration
- GitHub Actions or automated triggers
- Production security hardening

## High-Level Approach

- **Region:** `us-east-1`
- **Dev environment:** SageMaker Studio (Shared Space)
- **Orchestration:** SageMaker Pipelines via the Python SDK
- **Source of truth:** GitHub

### Pipeline Shape
1. **ProcessingStep** — reads raw data from S3, preprocesses, writes train/validation splits to S3
2. **TrainingStep** — reads processed data, runs mock training, writes model artifact to S3

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
| Pipeline | `edtriage-mock-train-pipeline` |
| S3 bucket (existing) | `ed-triage-capstone-group7` |

## S3 Layout
Within `s3://ed-triage-capstone-group7/`:
- `data/raw/` — raw input
- `data/processed/` — preprocessing output
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
    train_mock.py             # entry point for TrainingStep
  src/                        # existing code
  docs/
  README.md
```

`scripts/` is separate from `src/` because SageMaker uploads these independently to ephemeral containers.

## Phase Plan

### Phase 1 — AWS Foundation
Set up the environment needed to start coding.

Tasks:
- Create SageMaker Domain (`sm-edtriage-dev-domain`)
- Create or identify the SageMaker execution role for the domain
- Create Shared Space (`sm-edtriage-shared`)
- Confirm all team members can access Studio
- Verify SageMaker instance quotas for `ml.m5.xlarge` (processing and training); if admin access is required, mark quota request status as TBD

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
- [ ] Domain active; all members can open Studio
- [ ] Execution role ARN known with required permissions
- [ ] Instance quotas confirmed, or quota request status recorded as TBD
- [ ] Sample data in `s3://ed-triage-capstone-group7/data/raw/`

### Phase 2 — Preprocessing Script
Write a standalone script for a SageMaker Processing job.

- Read raw CSV from `/opt/ml/processing/input`
- Clean/sample, split into train and validation sets
- Write train split to `/opt/ml/processing/output/train`
- Write validation split to `/opt/ml/processing/output/validation`

Completion gate:
- [ ] Runs locally in Studio on sample data
- [ ] Outputs land in expected paths

### Phase 3 — Mock Training Script
Write a standalone script for a SageMaker Training job.

- Read from `SM_CHANNEL_TRAIN`
- Run trivial training logic
- Write artifact to `SM_MODEL_DIR`

Completion gate:
- [ ] Runs locally in Studio on processed sample data
- [ ] Model artifact produced

### Phase 4 — Pipeline Assembly
Wire both scripts into a SageMaker Pipeline.

- Define `ProcessingStep` and `TrainingStep`
- Pass processed data from step 1 to step 2
- Expose **Pipeline Parameters** (see below)
- One Python entry point to upsert and execute

#### Pipeline Parameters
These allow the Fine-Tuning Lead to change behavior at execution time without editing code:

| Parameter | Mock Default | Real Fine-Tuning |
|---|---|---|
| `InputDataUri` | `s3://ed-triage-capstone-group7/data/raw/` | full dataset path |
| `ProcessingInstanceType` | `ml.m5.xlarge` | same or larger |
| `TrainingInstanceType` | `ml.m5.xlarge` | `ml.g5.xlarge`+ |
| `TrainingInstanceCount` | `1` | 1+ |
| `Epochs` | `1` | tuned per experiment |

Completion gate:
- [ ] Pipeline upserts without errors
- [ ] Both steps wired correctly in `pipeline.definition()`
- [ ] Pipeline visible in Studio Pipelines UI

### Phase 5 — Smoke Test
Prove the pipeline works end to end.

- [ ] Pipeline executes successfully from Studio
- [ ] Processed data and model artifact appear in S3
- [ ] Logs visible in CloudWatch
- [ ] A second run with different parameters also succeeds
- [ ] A teammate can trigger the pipeline following the README

## Definition of Done
- Studio environment usable by the team
- Pipeline can be created/updated from Python
- Pipeline runs end to end without console intervention
- Mock model artifact produced in S3
- Training script is swappable without redesigning the pipeline

## Risks
| Risk | Mitigation |
|---|---|
| Environment setup takes too long | Keep scope to Studio + S3 + two pipeline steps |
| Real fine-tuning adds complexity | Keep mock pipeline structurally close to final |
| Team confusion about ownership | This document + PR approval = agreed plan |
| Instance quota is 0 | Check on Day 1; if admin access is needed, record quota request status as TBD |

## Open Questions
1. Mock training: tiny sklearn model or tiny transformer (GPT-2)?
2. Processed data format: CSV, JSONL, or Parquet?
3. Bucket prefixes as listed, or adjusted to match existing conventions?

## Next Step
After approval: **create and test the standalone preprocessing script (Phase 2).**
