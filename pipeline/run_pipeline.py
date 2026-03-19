"""
Upsert and execute the SageMaker Pipeline.

Usage (smoke test — auto-skips preprocessing if splits already exist):
    python pipeline/run_pipeline.py

Force preprocessing even if splits exist:
    python pipeline/run_pipeline.py --force-preprocess

Real arch4 training on GPU:
    python pipeline/run_pipeline.py \\
        --training-script train_arch4.py \\
        --training-instance-type ml.g5.xlarge
"""

import argparse

import boto3

from pipeline_definition import get_pipeline, DEFAULT_SPLITS_PREFIX

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ROLE   = "arn:aws:iam::478502030741:role/service-role/SageMaker-ExecutionRole-20260311T231755"
DEFAULT_REGION = "us-east-1"
DEFAULT_BUCKET = "ed-triage-capstone-group7"


def splits_exist(bucket: str, prefix: str = DEFAULT_SPLITS_PREFIX) -> bool:
    """Return True if processed splits are already present in S3."""
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return resp.get("KeyCount", 0) > 0


def main():
    parser = argparse.ArgumentParser(description="Run the edtriage training pipeline")
    parser.add_argument("--role", default=DEFAULT_ROLE, help="SageMaker execution role ARN")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket")
    parser.add_argument("--pipeline-name", default="edtriage-train-pipeline")
    parser.add_argument("--training-script", default="train_mock.py", help="Training script name")
    parser.add_argument("--training-instance-type", required=True, help="Training instance type (e.g.: ml.g5.xlarge for GPU)")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--input-data-uri", default=None, help="Override InputDataUri for preprocessing")
    parser.add_argument("--force-preprocess", action="store_true",
                        help="Re-run preprocessing even if splits already exist in S3")
    args = parser.parse_args()

    # ── Decide whether to skip preprocessing ─────────────────────────────────
    if args.force_preprocess:
        skip_preprocessing = False
        print("Preprocessing forced — running full pipeline.")
    elif splits_exist(args.bucket):
        skip_preprocessing = True
        print(f"Splits found at s3://{args.bucket}/{DEFAULT_SPLITS_PREFIX} — skipping preprocessing.")
    else:
        skip_preprocessing = False
        print("No splits found — running full pipeline including preprocessing.")

    pipeline = get_pipeline(
        role=args.role,
        epochs=args.epochs,
        training_instance_type_str=args.training_instance_type,
        region=args.region,
        default_bucket=args.bucket,
        pipeline_name=args.pipeline_name,
        training_script=args.training_script,
        skip_preprocessing=skip_preprocessing,
    )

    # ── Upsert (create or update) ─────────────────────────────────────────────
    print(f"Upserting pipeline: {args.pipeline_name}")
    pipeline.upsert(role_arn=args.role)

    # ── Build execution parameters from CLI overrides ─────────────────────────
    execution_params = {}
    if args.training_instance_type:
        execution_params["TrainingInstanceType"] = args.training_instance_type
    if args.input_data_uri and not skip_preprocessing:
        execution_params["InputDataUri"] = args.input_data_uri

    # ── Start execution ───────────────────────────────────────────────────────
    print(f"Starting execution with overrides: {execution_params or '(defaults)'}")
    execution = pipeline.start(parameters=execution_params)
    print(f"Execution ARN: {execution.arn}")
    print("Pipeline started. Monitor in SageMaker Studio → Pipelines UI.")


if __name__ == "__main__":
    main()
