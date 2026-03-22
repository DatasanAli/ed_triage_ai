"""
Deploy step for SageMaker Pipeline.

Creates or updates a SageMaker real-time endpoint with the new champion model,
then writes the updated champion metrics to S3.

Repacks the model archive with a code/ directory containing inference.py and
requirements.txt.  The PyTorch serving container automatically installs deps
from code/requirements.txt — no SAGEMAKER_TS_INSTALL_PY_DEP_PER_MODEL needed.

SageMaker contract (ProcessingStep):
  /opt/ml/processing/input/model/  - model.tar.gz (needed so that
        processingjobconfig.json contains the S3 URI)

Environment variables:
  BUCKET_NAME              - S3 bucket (default: ed-triage-capstone-group7)
  ENDPOINT_NAME            - endpoint to create/update (default: edtriage-live)
  ENDPOINT_INSTANCE_TYPE   - instance type (default: ml.m5.xlarge)
  ROLE_ARN                 - SageMaker execution role
  CONTAINER_IMAGE          - PyTorch inference container image URI
  ARCHITECTURE             - architecture name (e.g. arch4, mock)
"""

import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

CHAMPION_KEY = "champion/best_metrics.json"
EVALUATION_KEY = "evaluation/evaluation.json"


def load_evaluation_metrics(bucket):
    """Read evaluation.json written by the Evaluate step."""
    s3 = boto3.client("s3")
    resp = s3.get_object(Bucket=bucket, Key=EVALUATION_KEY)
    return json.loads(resp["Body"].read().decode("utf-8"))


def repack_model_with_code_dir(model_data_uri, region):
    """Download model.tar.gz, add a code/ directory, and re-upload.

    The PyTorch serving container looks for code/inference.py and
    code/requirements.txt and installs deps automatically.  This
    avoids the unreliable SAGEMAKER_TS_INSTALL_PY_DEP_PER_MODEL env var.

    Returns the S3 URI of the repacked model archive.
    """
    s3 = boto3.client("s3", region_name=region)
    parts = model_data_uri.replace("s3://", "").split("/", 1)
    src_bucket, src_key = parts[0], parts[1]

    tmp_dir = tempfile.mkdtemp()
    tar_path = os.path.join(tmp_dir, "model.tar.gz")
    extract_dir = os.path.join(tmp_dir, "model")
    os.makedirs(extract_dir)

    # Download and extract
    s3.download_file(src_bucket, src_key, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    # Move inference.py and requirements.txt into code/
    code_dir = os.path.join(extract_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    for filename in ["inference.py", "requirements.txt"]:
        src = os.path.join(extract_dir, filename)
        if os.path.exists(src):
            os.rename(src, os.path.join(code_dir, filename))

    print(f"Repacked model contents: {os.listdir(extract_dir)}")
    print(f"  code/ contents: {os.listdir(code_dir)}")

    # Re-pack
    repacked_path = os.path.join(tmp_dir, "repacked.tar.gz")
    with tarfile.open(repacked_path, "w:gz") as tar:
        for item in os.listdir(extract_dir):
            tar.add(os.path.join(extract_dir, item), arcname=item)

    # Upload alongside original
    repacked_key = src_key.replace("model.tar.gz", "model-repacked.tar.gz")
    s3.upload_file(repacked_path, src_bucket, repacked_key)
    repacked_uri = f"s3://{src_bucket}/{repacked_key}"
    print(f"Uploaded repacked model to {repacked_uri}")
    return repacked_uri


def main():
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    sm = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    bucket = os.environ.get("BUCKET_NAME", "ed-triage-capstone-group7")
    endpoint_name = os.environ.get("ENDPOINT_NAME", "edtriage-live")
    instance_type = os.environ.get("ENDPOINT_INSTANCE_TYPE", "ml.m5.xlarge")
    role_arn = os.environ["ROLE_ARN"]
    container_image = os.environ["CONTAINER_IMAGE"]
    architecture = os.environ.get("ARCHITECTURE", "unknown")

    evaluation = load_evaluation_metrics(bucket)
    new_macro_f1 = evaluation["macro_f1"]
    model_data_uri = evaluation["model_data_uri"]
    if not model_data_uri:
        raise ValueError("model_data_uri missing from evaluation.json")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_name = f"edtriage-{timestamp}"
    config_name = f"edtriage-config-{timestamp}"

    # Repack model archive with code/ directory for automatic dep installation
    repacked_uri = repack_model_with_code_dir(model_data_uri, region)

    # 1. Create SageMaker Model
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": container_image,
            "ModelDataUrl": repacked_uri,
        },
        ExecutionRoleArn=role_arn,
    )
    print(f"Created model: {model_name}")

    # 2. Create Endpoint Configuration
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "InstanceType": instance_type,
            "InitialInstanceCount": 1,
        }],
    )
    print(f"Created endpoint config: {config_name}")

    # 3. Create or Update Endpoint
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
        print(f"Updating existing endpoint: {endpoint_name}")
    except ClientError as e:
        if "Could not find endpoint" in str(e):
            sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )
            print(f"Created new endpoint: {endpoint_name}")
        else:
            raise

    # 4. Wait for endpoint to become InService
    print("Waiting for endpoint to become InService...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 60},  # 30 min max
    )
    print(f"Endpoint {endpoint_name} is InService.")

    # 5. Update champion metrics in S3
    champion_metrics = {
        "macro_f1": new_macro_f1,
        "architecture": architecture,
        "model_data_uri": repacked_uri,
        "model_name": model_name,
        "endpoint_name": endpoint_name,
        "endpoint_config_name": config_name,
        "timestamp": timestamp,
    }
    s3.put_object(
        Bucket=bucket,
        Key=CHAMPION_KEY,
        Body=json.dumps(champion_metrics, indent=2),
        ContentType="application/json",
    )
    print(f"Updated champion metrics at s3://{bucket}/{CHAMPION_KEY}")
    print("Deploy complete.")


if __name__ == "__main__":
    main()
