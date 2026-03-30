"""
Repack the existing model archive with the updated inference.py and deploy
to the edtriage-live endpoint — no retraining required.

Usage (local Mac):
    AWS_PROFILE=ed-triage python sagemaker/scripts/repack_and_deploy.py

Usage (SageMaker terminal):
    python sagemaker/scripts/repack_and_deploy.py
"""

import json
import os
import shutil
import tarfile
import tempfile
import time

import boto3
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────────────────────
BUCKET          = "ed-triage-capstone-group7"
ENDPOINT_NAME   = "edtriage-live"
INSTANCE_TYPE   = "ml.m5.xlarge"
REGION          = "us-east-1"
ROLE_ARN        = "arn:aws:iam::478502030741:role/service-role/SageMaker-ExecutionRole-20260311T231755"
CHAMPION_KEY    = "champion/best_metrics.json"
EVALUATION_KEY  = "evaluation/evaluation.json"

# Path to the updated inference.py in this repo
INFERENCE_SRC = os.path.join(
    os.path.dirname(__file__), "..", "models", "arch4", "inference.py"
)


def get_model_uri():
    """Read the current champion model URI from S3."""
    s3 = boto3.client("s3", region_name=REGION)
    # Try champion first, fall back to evaluation.json
    for key in [CHAMPION_KEY, EVALUATION_KEY]:
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            data = json.loads(resp["Body"].read())
            uri  = data.get("model_data_uri") or data.get("model_data_uri")
            if uri:
                print(f"Model URI from s3://{BUCKET}/{key}: {uri}")
                return uri
        except ClientError:
            continue
    raise RuntimeError("Could not find model_data_uri in champion or evaluation.json")


def repack(model_uri):
    """Download model.tar.gz, swap inference.py, repack, upload."""
    s3 = boto3.client("s3", region_name=REGION)
    parts = model_uri.replace("s3://", "").split("/", 1)
    src_bucket, src_key = parts[0], parts[1]

    tmp = tempfile.mkdtemp()
    tar_path     = os.path.join(tmp, "model.tar.gz")
    extract_dir  = os.path.join(tmp, "model")
    repacked_path = os.path.join(tmp, "repacked.tar.gz")
    os.makedirs(extract_dir)

    print("Downloading model archive...")
    s3.download_file(src_bucket, src_key, tar_path)

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    print(f"Archive contents: {os.listdir(extract_dir)}")

    # Ensure code/ directory exists and swap inference.py
    code_dir = os.path.join(extract_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    # If inference.py is at root level, move it to code/
    root_inf = os.path.join(extract_dir, "inference.py")
    if os.path.exists(root_inf):
        os.remove(root_inf)

    shutil.copy2(INFERENCE_SRC, os.path.join(code_dir, "inference.py"))
    print(f"Swapped inference.py from {INFERENCE_SRC}")

    # Also update requirements.txt in code/ to include shap
    req_path = os.path.join(code_dir, "requirements.txt")
    with open(req_path, "w") as f:
        f.write("transformers==4.40.2\n")
        f.write("lightgbm==4.3.0\n")
        f.write("joblib>=1.3.0\n")
        f.write("scikit-learn>=1.3.0\n")
        f.write("shap>=0.44.0\n")
    print("Updated code/requirements.txt (added shap)")

    # Repack
    print("Repacking...")
    with tarfile.open(repacked_path, "w:gz") as tar:
        for item in os.listdir(extract_dir):
            tar.add(os.path.join(extract_dir, item), arcname=item)

    # Upload
    repacked_key = src_key.replace("model.tar.gz", "model-repacked-v2.tar.gz")
    print(f"Uploading to s3://{src_bucket}/{repacked_key}...")
    s3.upload_file(repacked_path, src_bucket, repacked_key)
    repacked_uri = f"s3://{src_bucket}/{repacked_key}"
    print(f"Uploaded: {repacked_uri}")

    shutil.rmtree(tmp)
    return repacked_uri


def deploy(repacked_uri):
    """Create a new SageMaker model + endpoint config, then update the endpoint."""
    import sagemaker
    sm = boto3.client("sagemaker", region_name=REGION)

    # Get PyTorch inference image
    session = sagemaker.Session()
    container_image = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=REGION,
        version="2.1.0",
        py_version="py310",
        instance_type=INSTANCE_TYPE,
        image_scope="inference",
    )
    print(f"Container image: {container_image}")

    timestamp  = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"edtriage-repacked-{timestamp}"
    config_name = f"edtriage-config-repacked-{timestamp}"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": container_image, "ModelDataUrl": repacked_uri},
        ExecutionRoleArn=ROLE_ARN,
    )
    print(f"Created model: {model_name}")

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "InstanceType": INSTANCE_TYPE,
            "InitialInstanceCount": 1,
        }],
    )
    print(f"Created endpoint config: {config_name}")

    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        print(f"Updating existing endpoint: {ENDPOINT_NAME}")
    except ClientError as e:
        if "Could not find endpoint" in str(e):
            sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
            print(f"Created new endpoint: {ENDPOINT_NAME}")
        else:
            raise

    print("Waiting for endpoint to become InService (this takes ~5-10 min)...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=ENDPOINT_NAME, WaiterConfig={"Delay": 30, "MaxAttempts": 40})
    print(f"Endpoint {ENDPOINT_NAME} is InService.")


if __name__ == "__main__":
    model_uri    = get_model_uri()
    repacked_uri = repack(model_uri)
    deploy(repacked_uri)
    print("\nDone. Test with:")
    print(f'  runtime.invoke_endpoint(EndpointName="{ENDPOINT_NAME}", ...)')
