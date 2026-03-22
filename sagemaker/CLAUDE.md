# SageMaker Pipeline — Agent Rules

## Directory Structure (enforced)

```
sagemaker/
  pipeline/           # Orchestration code — runs locally, NOT inside SageMaker containers
    pipeline_definition.py   # Defines the DAG: Preprocess → Train → Evaluate → Condition → Register → Deploy
    run_pipeline.py          # CLI entry point to upsert and execute the pipeline
  steps/               # Shared pipeline step entry points — run inside SageMaker containers
    preprocess.py        # ProcessingStep: splits raw data into train/val/test
    evaluate.py          # ProcessingStep: compares new model against champion
    deploy.py            # ProcessingStep: creates/updates SageMaker endpoint
    train.py             # TrainingStep: dispatcher — resolves and delegates to models/<arch>/train.py
  models/              # Architecture-specific code — one directory per model architecture
    __init__.py
    mock/
      __init__.py
      train.py           # Training logic (must export main())
      inference.py       # PyTorch serving handler (model_fn, input_fn, predict_fn, output_fn)
    arch4/
      __init__.py
      train.py           # Training logic (must export main())
      inference.py       # PyTorch serving handler (BERT + LightGBM + feature engineering)
  requirements.txt     # Shared deps installed by SageMaker before running training entry point
```

## Critical Rules

### Never put training logic in steps/train.py
`steps/train.py` is a dispatcher only. It extracts `--architecture`, imports `models.<arch>.train`, and calls `main()`. All training logic belongs in `models/<arch>/train.py`.

### Every model directory must have train.py with a main() function
The dispatcher calls `module.main()`. If `main()` doesn't exist, the pipeline will fail. The function takes no arguments — use `argparse` inside `main()` for model-specific hyperparameters.

### inference.py must be copied into SM_MODEL_DIR during training
The training script must copy `inference.py` (and a `requirements.txt` listing inference-time dependencies) into `SM_MODEL_DIR` so they are bundled into `model.tar.gz`. Without this, the endpoint will fail to serve requests. Pattern:
```python
import shutil
shutil.copy2(os.path.join(os.path.dirname(__file__), "inference.py"),
             os.path.join(MODEL_DIR, "inference.py"))
```

### inference.py must duplicate model class definitions
The serving container only has access to the model archive — not the training source_dir. Any model classes used in `inference.py` must be defined directly in that file, not imported from `train.py`.

### inference.py must write a requirements.txt into SM_MODEL_DIR
The PyTorch inference container does not have `transformers` or `lightgbm` pre-installed. The training script must write a `requirements.txt` to `SM_MODEL_DIR` listing inference-time dependencies. The serving container installs these before loading the model. Dependencies are model-specific: mock needs only `transformers`; arch4 also needs `lightgbm`, `joblib`, and `scikit-learn`.

### Do not hardcode architecture names in pipeline_definition.py or run_pipeline.py
Architecture is passed as a parameter (`--architecture`). The pipeline code is architecture-agnostic. To add a new architecture, only create files in `models/<name>/` — never modify pipeline orchestration code.

### config.json must be saved to SM_MODEL_DIR by every training script
The evaluate step reads `config.json` from the model archive to extract `val_metrics.best_val_macro_f1`. Every model's `train.py` must write this file. Required schema:
```json
{
  "architecture": "<name>",
  "val_metrics": {
    "best_val_macro_f1": 0.78
  }
}
```

### One endpoint, one champion
There is a single endpoint (`edtriage-live`) serving the best model across all architectures. The champion is tracked in `s3://ed-triage-capstone-group7/champion/best_metrics.json`. A new model deploys only if its macro-F1 is strictly greater than the current champion's. Do not create architecture-specific endpoints.

### source_dir is sagemaker/
The entire `sagemaker/` directory is packaged as the training source. Entry point is always `steps/train.py`. Do not change `source_dir` or `entry_point` in the estimator config.

## S3 Layout

```
s3://ed-triage-capstone-group7/
  Data_Output/
    consolidated_dataset_features.csv    # Raw input
    splits/train/                        # Preprocessed train split
    splits/validation/                   # Preprocessed validation split
    splits/test/                         # Preprocessed test split
  models/                                # Training artifacts (model.tar.gz)
  evaluation/evaluation.json             # Latest evaluation results
  champion/best_metrics.json             # Current champion metadata
```

## Adding a New Architecture

1. Create `sagemaker/models/<name>/__init__.py` (empty)
2. Create `sagemaker/models/<name>/train.py` with a `main()` function
3. Create `sagemaker/models/<name>/inference.py` with `model_fn`, `input_fn`, `predict_fn`, `output_fn`
4. In `train.py`: save `config.json`, `inference.py`, and `requirements.txt` to `SM_MODEL_DIR`
5. Run: `python sagemaker/pipeline/run_pipeline.py --architecture <name> --training-instance-type <type> --epochs <n>`

No changes to pipeline code are needed.
