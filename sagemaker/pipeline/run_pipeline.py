"""
Upsert and execute the SageMaker Pipeline.

Usage (smoke test — auto-skips preprocessing if splits already exist):
    python sagemaker/pipeline/run_pipeline.py \\
        --architecture mock \\
        --training-instance-type ml.m5.xlarge \\
        --epochs 1

Force preprocessing even if splits exist:
    python sagemaker/pipeline/run_pipeline.py \\
        --architecture mock \\
        --training-instance-type ml.m5.xlarge \\
        --epochs 1 \\
        --force-preprocess

Real arch4 training on GPU:
    python sagemaker/pipeline/run_pipeline.py \\
        --architecture arch4 \\
        --training-instance-type ml.g5.xlarge \\
        --epochs 20

Deploy to a specific endpoint:
    python sagemaker/pipeline/run_pipeline.py \\
        --architecture arch4 \\
        --training-instance-type ml.g5.xlarge \\
        --epochs 20 \\
        --endpoint-name edtriage-live \\
        --endpoint-instance-type ml.m5.xlarge
"""

import argparse
import os
import sys

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
    parser.add_argument("--architecture", required=True,
                        help="Model architecture to train (e.g. mock, arch4)")
    parser.add_argument("--training-instance-type", required=True,
                        help="Training instance type (e.g.: ml.g5.xlarge for GPU)")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--input-data-uri", default=None,
                        help="Override InputDataUri for preprocessing")
    parser.add_argument("--force-preprocess", action="store_true",
                        help="Re-run preprocessing even if splits already exist in S3")
    parser.add_argument("--endpoint-instance-type", default="ml.m5.xlarge",
                        help="Instance type for the inference endpoint (default: ml.m5.xlarge)")
    parser.add_argument("--endpoint-name", default="edtriage-live",
                        help="Name of the SageMaker endpoint to create/update (default: edtriage-live)")
    parser.add_argument("--processing-instance-type", default="ml.t3.medium",
                        help="Instance type for evaluate/deploy processing steps (default: ml.t3.medium)")
    args = parser.parse_args()

    # ── Validate architecture exists ────────────────────────────────────────
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    arch_dir = os.path.join(models_dir, args.architecture)
    if not os.path.isdir(arch_dir):
        available = [
            d for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d)) and d != "__pycache__"
        ]
        print(
            f"Error: architecture '{args.architecture}' not found. "
            f"Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

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
        architecture=args.architecture,
        region=args.region,
        default_bucket=args.bucket,
        pipeline_name=args.pipeline_name,
        skip_preprocessing=skip_preprocessing,
        endpoint_instance_type_str=args.endpoint_instance_type,
        processing_instance_type_str=args.processing_instance_type,
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
    if args.endpoint_instance_type:
        execution_params["EndpointInstanceType"] = args.endpoint_instance_type
    if args.endpoint_name:
        execution_params["EndpointName"] = args.endpoint_name

    # ── Start execution ───────────────────────────────────────────────────────
    print(f"Starting execution with overrides: {execution_params or '(defaults)'}")
    execution = pipeline.start(parameters=execution_params)
    print(f"Execution ARN: {execution.arn}")
    print("Pipeline started. Monitor in SageMaker Studio → Pipelines UI.")


if __name__ == "__main__":
    main()
