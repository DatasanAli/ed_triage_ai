# SageMaker Pipeline

Automated ML pipeline for the ED Triage AI project. Trains, evaluates, and deploys triage prediction models to a SageMaker real-time endpoint.

## Pipeline Flow

```
Preprocess → Train → Evaluate → CheckChampion
                                    ├─ Yes → Register → Deploy
                                    └─ No  → pipeline ends
```

1. **Preprocess** — Splits raw data into train/val/test sets (skipped if splits already exist in S3)
2. **Train** — Trains the specified architecture (dispatched via `steps/train.py`)
3. **Evaluate** — Compares the new model's macro-F1 against the current champion in S3
4. **CheckChampion** — If the new model is strictly better, proceeds to register and deploy
5. **Register** — Registers the model in SageMaker Model Registry
6. **Deploy** — Creates or updates the `edtriage-live` endpoint with the new champion

## Quick Start

```bash
# From the ed_triage_ai/ directory

# Smoke test with mock model (CPU, 1 epoch, ~5 min)
python sagemaker/pipeline/run_pipeline.py \
    --architecture mock \
    --training-instance-type ml.m5.xlarge \
    --epochs 1

# Real training with arch4 (GPU, 20 epochs)
python sagemaker/pipeline/run_pipeline.py \
    --architecture arch4 \
    --training-instance-type ml.g5.xlarge \
    --epochs 20

# Force re-run preprocessing
python sagemaker/pipeline/run_pipeline.py \
    --architecture mock \
    --training-instance-type ml.m5.xlarge \
    --epochs 1 \
    --force-preprocess
```

## CLI Options

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--architecture` | Yes | — | Model architecture (`mock`, `arch4`) |
| `--training-instance-type` | Yes | — | Instance for training (`ml.m5.xlarge`, `ml.g5.xlarge`) |
| `--epochs` | Yes | — | Number of training epochs |
| `--force-preprocess` | No | false | Re-run preprocessing even if splits exist |
| `--endpoint-name` | No | `edtriage-live` | SageMaker endpoint name |
| `--endpoint-instance-type` | No | `ml.m5.xlarge` | Instance for the inference endpoint |
| `--role` | No | project default | SageMaker execution role ARN |
| `--region` | No | `us-east-1` | AWS region |
| `--bucket` | No | `ed-triage-capstone-group7` | S3 bucket |

## Directory Structure

```
sagemaker/
  pipeline/
    pipeline_definition.py   # Pipeline DAG definition (runs locally)
    run_pipeline.py          # CLI to upsert and execute the pipeline (runs locally)
  steps/
    preprocess.py            # Data splitting (runs in SageMaker container)
    evaluate.py              # Champion comparison (runs in SageMaker container)
    deploy.py                # Endpoint management (runs in SageMaker container)
    train.py                 # Training dispatcher (runs in SageMaker container)
  models/
    mock/                    # bert-tiny smoke test model
      train.py                 # Training logic
      inference.py             # Serving handler
    arch4/                   # BioClinicalBERT + LightGBM fusion model
      train.py                 # Training logic
  requirements.txt           # Shared training dependencies
```

## How the Dispatcher Works

`steps/train.py` is always the SageMaker training entry point. It receives `--architecture` as a hyperparameter, imports the corresponding `models/<arch>/train.py`, and calls its `main()` function. This means:

- The pipeline definition never changes when you add a new architecture
- Each architecture owns its training logic, inference handler, and dependencies
- Architecture-specific hyperparameters are passed through transparently

## Adding a New Architecture

1. Create the model directory:
   ```
   sagemaker/models/<name>/
     __init__.py      # empty
     train.py         # must export main()
     inference.py     # model_fn, input_fn, predict_fn, output_fn
   ```

2. In `train.py`, your `main()` function must:
   - Read data from `SM_CHANNEL_TRAIN` and `SM_CHANNEL_VALIDATION`
   - Save model artifacts to `SM_MODEL_DIR`
   - Write `config.json` with at least `{"architecture": "<name>", "val_metrics": {"best_val_macro_f1": <float>}}`
   - Copy `inference.py` into `SM_MODEL_DIR`
   - Write a `requirements.txt` into `SM_MODEL_DIR` listing inference-time dependencies

3. Run the pipeline:
   ```bash
   python sagemaker/pipeline/run_pipeline.py \
       --architecture <name> \
       --training-instance-type <instance> \
       --epochs <n>
   ```

No changes to pipeline orchestration code are needed.

## Champion Model

Only one model is deployed at a time — the best across all architectures. The champion is tracked in S3:

```
s3://ed-triage-capstone-group7/champion/best_metrics.json
```

A new model replaces the champion only if its validation macro-F1 is strictly greater. Ties keep the incumbent.

## Testing the Endpoint

```bash
# Invoke the live endpoint
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name edtriage-live \
    --content-type application/json \
    --body '{"triage_text": "Chief complaint: chest pain. Presenting with chest pain. History: 55 yo male with acute onset chest pain radiating to left arm"}' \
    /tmp/response.json && cat /tmp/response.json
```

Expected response:
```json
{
  "predicted_class": 0,
  "predicted_label": "L1-Critical",
  "probabilities": {
    "L1-Critical": 0.7,
    "L2-Emergent": 0.2,
    "L3-Urgent/LessUrgent": 0.1
  }
}
```

## Monitoring

```bash
# Check pipeline execution status
aws sagemaker list-pipeline-executions \
    --pipeline-name edtriage-train-pipeline \
    --sort-by CreationTime --sort-order Descending --max-results 3

# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name edtriage-live

# View step-level details for an execution
aws sagemaker list-pipeline-execution-steps \
    --pipeline-execution-arn <execution-arn>

# Check endpoint logs
aws logs tail /aws/sagemaker/Endpoints/edtriage-live --since 30m
```
