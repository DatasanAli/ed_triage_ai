"""
Evaluate step for SageMaker Pipeline.

Compares the newly trained model's validation metrics against the current
champion model (tracked in S3).  Writes an evaluation.json PropertyFile
so the downstream ConditionStep can decide whether to register and deploy.

SageMaker contract (ProcessingStep):
  /opt/ml/processing/input/model/  - model.tar.gz from TrainingStep
  /opt/ml/processing/input/test/   - test CSV (reserved for future use)
  /opt/ml/processing/output/evaluation/evaluation.json  - PropertyFile output

Environment variables:
  BUCKET_NAME  - S3 bucket for champion metrics
"""

import json
import os
import tarfile

import boto3
from botocore.exceptions import ClientError

MODEL_INPUT_DIR = "/opt/ml/processing/input/model"
TEST_INPUT_DIR = "/opt/ml/processing/input/test"
EVAL_OUTPUT_DIR = "/opt/ml/processing/output/evaluation"

CHAMPION_KEY = "champion/best_metrics.json"


def extract_model_tar(model_dir):
    """Extract model.tar.gz to access config.json and other artifacts."""
    tar_path = os.path.join(model_dir, "model.tar.gz")
    extract_dir = os.path.join(model_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)
    return extract_dir


def load_new_model_metrics(extract_dir):
    """Read val_metrics from config.json inside extracted model artifacts."""
    config_path = os.path.join(extract_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json not found in model archive at {extract_dir}. "
            "Every training script must write config.json to SM_MODEL_DIR."
        )

    with open(config_path) as f:
        config = json.load(f)

    val_metrics = config.get("val_metrics", {})
    return {
        "macro_f1": val_metrics.get("best_val_macro_f1", 0.0),
        "architecture": config.get("architecture", "unknown"),
    }


def load_champion_metrics(bucket):
    """Fetch current champion metrics from S3.  Returns None on first run."""
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    s3 = boto3.client("s3", region_name=region)
    try:
        resp = s3.get_object(Bucket=bucket, Key=CHAMPION_KEY)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            print("No champion metrics found — first run, new model wins.")
            return None
        raise


def main():
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    bucket = os.environ.get("BUCKET_NAME", "ed-triage-capstone-group7")

    # 1. Extract model artifacts
    extract_dir = extract_model_tar(MODEL_INPUT_DIR)

    # 2. Get new model metrics
    new_metrics = load_new_model_metrics(extract_dir)
    new_f1 = new_metrics["macro_f1"]

    # 3. Get champion metrics
    champion = load_champion_metrics(bucket)
    champion_f1 = champion["macro_f1"] if champion else 0.0

    # 4. Compare (strictly greater — ties keep the incumbent)
    is_new_champion = new_f1 > champion_f1

    print(f"New model macro_f1:  {new_f1}")
    print(f"Champion macro_f1:   {champion_f1}")
    print(f"Is new champion:     {is_new_champion}")

    # 5. Get model S3 URI from processing job config so deploy.py can use it
    model_data_uri = ""
    job_config_path = "/opt/ml/config/processingjobconfig.json"
    if os.path.exists(job_config_path):
        with open(job_config_path) as f:
            job_config = json.load(f)
        for inp in job_config.get("ProcessingInputs", []):
            if inp.get("InputName") == "model":
                model_data_uri = inp["S3Input"]["S3Uri"]
                break
    print(f"Model data URI: {model_data_uri}")

    # 6. Write evaluation.json (PropertyFile output)
    #    is_new_champion is a string because SageMaker ConditionEquals
    #    operates on strings when reading from JsonGet.
    evaluation = {
        "macro_f1": new_f1,
        "champion_macro_f1": champion_f1,
        "is_new_champion": "true" if is_new_champion else "false",
        "model_data_uri": model_data_uri,
    }
    eval_path = os.path.join(EVAL_OUTPUT_DIR, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"Wrote {eval_path}")


if __name__ == "__main__":
    main()
