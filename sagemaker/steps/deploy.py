"""
Deploy step for SageMaker Pipeline.

Creates or updates a SageMaker real-time endpoint with the new champion model,
then writes the updated champion metrics to S3.

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

    # 1. Create SageMaker Model
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": container_image,
            "ModelDataUrl": model_data_uri,
            "Environment": {
                "SAGEMAKER_TS_INSTALL_PY_DEP_PER_MODEL": "True",
            },
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
        "model_data_uri": model_data_uri,
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
