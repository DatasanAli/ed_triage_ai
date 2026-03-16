"""
Upsert and execute the SageMaker Pipeline.

Usage (smoke test with defaults):
    python pipeline/run_pipeline.py

Override for real arch4 training:
    python pipeline/run_pipeline.py \\
        --training-script train_arch4.py \\
        --training-instance-type ml.g5.xlarge
"""

import argparse

from pipeline_definition import get_pipeline

# ── Defaults for smoke test (edit ROLE_ARN before first run) ──────────────────
DEFAULT_ROLE   = "arn:aws:iam::478502030741:role/service-role/SageMaker-ExecutionRole-20260311T231755"
DEFAULT_REGION = "us-east-1"
DEFAULT_BUCKET = "ed-triage-capstone-group7"


def main():
    parser = argparse.ArgumentParser(description="Run the edtriage training pipeline")
    parser.add_argument("--role", default=DEFAULT_ROLE, help="SageMaker execution role ARN")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket")
    parser.add_argument("--pipeline-name", default="edtriage-train-pipeline")
    parser.add_argument("--training-script", default="train_mock.py", help="Training script name")
    parser.add_argument("--training-instance-type", default="ml.m5.xlarge", help="Training instance type")
    parser.add_argument("--input-data-uri", default=None, help="Override InputDataUri")
    args = parser.parse_args()

    pipeline = get_pipeline(
        role=args.role,
        region=args.region,
        default_bucket=args.bucket,
        pipeline_name=args.pipeline_name,
    )

    # ── Upsert (create or update) ─────────────────────────────────────────────
    print(f"Upserting pipeline: {args.pipeline_name}")
    pipeline.upsert(role_arn=args.role)

    # ── Build execution parameters from CLI overrides ─────────────────────────
    execution_params = {}
    if args.training_script:
        execution_params["TrainingScript"] = args.training_script
    if args.training_instance_type:
        execution_params["TrainingInstanceType"] = args.training_instance_type
    if args.input_data_uri:
        execution_params["InputDataUri"] = args.input_data_uri

    # ── Start execution ───────────────────────────────────────────────────────
    print(f"Starting execution with overrides: {execution_params or '(defaults)'}")
    execution = pipeline.start(parameters=execution_params)
    print(f"Execution ARN: {execution.arn}")
    print("Pipeline started. Monitor in SageMaker Studio → Pipelines UI.")


if __name__ == "__main__":
    main()
