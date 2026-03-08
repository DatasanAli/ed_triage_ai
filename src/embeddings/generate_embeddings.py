"""
generate_embeddings.py
======================
Reads cases_for_embedding.jsonl, calls Amazon Bedrock Titan Embeddings
for each case, and writes the resulting vectors to embeddings_output.jsonl.

What this script does:
  1. Reads each case from the data_prep output (cases_for_embedding.jsonl)
  2. Sends the embedding_text to Bedrock Titan Text Embeddings v2
  3. Gets back a 1024-dimensional vector
  4. Writes {id, vector, metadata} to embeddings_output.jsonl

Why Titan Text Embeddings v2 (not v1)?
  - v1: 1536 dims, older, higher cost
  - v2: 1024 dims, better quality, lower cost ($0.00002 per 1K tokens vs $0.0001)
  - v2 is the current recommended model for new projects

Batching strategy:
  We call Bedrock one case at a time (no batch API for Titan).
  To be safe against network errors or throttling, we:
    - Save progress every CHECKPOINT_EVERY cases
    - Skip already-embedded cases on resume (resume-safe)
    - Respect Bedrock rate limits with a small delay between calls

Cost estimate for 9147 cases:
  Average ~200 tokens per embedding text
  9147 cases x 200 tokens = ~1.83M tokens
  At $0.00002/1K tokens = ~$0.037 (less than 4 cents)

Run:
  python embeddings/generate_embeddings.py

  With AWS credentials in environment:
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... python embeddings/generate_embeddings.py

  Or if you have a named profile:
  AWS_PROFILE=your-profile python embeddings/generate_embeddings.py
"""

import json
import time
import boto3
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = ROOT / "embeddings"

INPUT_JSONL = EMBEDDINGS_DIR / "cases_for_embedding.jsonl"
OUTPUT_JSONL = EMBEDDINGS_DIR / "embeddings_output.jsonl"
PROGRESS_FILE = EMBEDDINGS_DIR / ".embed_progress.json"

# Bedrock model ID for Titan Text Embeddings v2
# Returns 1024-dim vectors by default
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"

# AWS region — Bedrock is not available in all regions
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# AWS named profile — credentials live in ~/.aws/credentials, never in this repo
# Set up with: aws configure --profile ed-triage
AWS_PROFILE = os.getenv("AWS_PROFILE", "ed-triage")

# Pinecone secret name in AWS Secrets Manager
PINECONE_SECRET_NAME = "prod/pinecone/api_key"

# How often to write a progress checkpoint
# If the script crashes at case 5000, we can resume from the last checkpoint
CHECKPOINT_EVERY = 100

# Delay between API calls in seconds
# Bedrock has a default throttle limit of ~10 req/sec for Titan
# 0.1s delay keeps us comfortably under that
DELAY_SECONDS = 0.1


# ---------------------------------------------------------------------------
# AWS session — shared across all clients
# ---------------------------------------------------------------------------

def get_session() -> boto3.Session:
    """
    Create a boto3 Session using the named AWS profile.

    Why a Session instead of calling boto3.client() directly?
      A Session holds the credentials and region config once, and you can
      create multiple clients (Bedrock, Secrets Manager, S3) from the same
      session without re-authenticating each time.

    How AWS_PROFILE works:
      boto3 reads ~/.aws/credentials and ~/.aws/config, finds the section
      matching the profile name, and uses those credentials. No keys ever
      touch this codebase.
    """
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Secrets Manager — fetch Pinecone API key at runtime
# ---------------------------------------------------------------------------

def get_pinecone_api_key(session: boto3.Session) -> str:
    """
    Fetch the Pinecone API key from AWS Secrets Manager.

    Why Secrets Manager instead of .env?
      - The key never lives on disk in the repo
      - Access is logged in AWS CloudTrail (audit trail)
      - You can rotate the key in one place and all services pick it up
      - Teammates with the right IAM permissions get it automatically —
        no need to share a .env file over Slack

    The secret is stored as a JSON string: {"PINECONE_API_KEY": "pcsk_..."}
    We parse it and return just the key value.
    """
    client = session.client("secretsmanager")
    response = client.get_secret_value(SecretId=PINECONE_SECRET_NAME)
    secret = json.loads(response["SecretString"])
    return secret["PINECONE_API_KEY"]


# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------

def get_bedrock_client(session: boto3.Session):
    """
    Create a Bedrock Runtime client from the shared session.

    bedrock-runtime is the endpoint for model inference (embeddings, text gen).
    bedrock (without -runtime) is for management operations (listing models, etc.)
    """
    return session.client("bedrock-runtime")


# ---------------------------------------------------------------------------
# Embedding function
# ---------------------------------------------------------------------------

def get_embedding(client, text: str) -> list[float]:
    """
    Call Bedrock Titan to embed a single text string.

    How Bedrock API works:
      - You call invoke_model() with a JSON body
      - The body format is model-specific (each model has its own schema)
      - Titan Embeddings v2 expects: {"inputText": "...", "dimensions": 1024}
      - It returns a JSON body with: {"embedding": [...], "inputTextTokenCount": N}

    We use the minimal request body (just inputText) for broadest compatibility.
    Titan v2 returns 1024-dim vectors by default, already normalized for cosine similarity.
    """
    body = json.dumps({
        "inputText": text,
    })

    response = client.invoke_model(
        modelId=TITAN_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    response_body = json.loads(response["body"].read())
    return response_body["embedding"]


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def load_already_embedded() -> set:
    """
    Load the set of case IDs we've already embedded.

    Why resume support?
      9147 API calls at 0.1s each = ~15 minutes.
      If the script crashes halfway through, we don't want to re-embed
      the first 4000 cases and pay for them twice.

    We track progress by reading the output file and collecting all IDs
    that already have an entry. On resume, we skip those IDs.
    """
    embedded = set()
    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    embedded.add(record["id"])
                except json.JSONDecodeError:
                    pass
    return embedded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Bedrock region : {AWS_REGION}")
    print(f"Model          : {TITAN_MODEL_ID}")
    print(f"Input          : {INPUT_JSONL}")
    print(f"Output         : {OUTPUT_JSONL}")
    print()

    # Count total cases
    with open(INPUT_JSONL) as f:
        total = sum(1 for _ in f)
    print(f"Total cases to embed: {total}")

    # Check which cases are already done (resume support)
    already_done = load_already_embedded()
    if already_done:
        print(f"Resuming — {len(already_done)} cases already embedded, skipping them")
    remaining = total - len(already_done)
    print(f"Cases to process: {remaining}")
    print()

    if remaining == 0:
        print("All cases already embedded. Nothing to do.")
        return

    # Initialize AWS session using named profile (~/.aws/credentials)
    # This will raise an error immediately if the profile doesn't exist
    session = get_session()
    print(f"AWS session initialized (profile: {AWS_PROFILE})")

    # Fetch Pinecone key from Secrets Manager — not from .env
    print(f"Fetching Pinecone API key from Secrets Manager ({PINECONE_SECRET_NAME})...")
    pinecone_api_key = get_pinecone_api_key(session)
    print("Pinecone API key retrieved.")

    # Initialize Bedrock client from the same session
    client = get_bedrock_client(session)
    print("Bedrock client initialized.")
    print()

    # Open output file in append mode so we don't overwrite existing progress
    success = 0
    errors = 0
    skipped = 0

    with open(OUTPUT_JSONL, "a") as out_f:
        with open(INPUT_JSONL) as in_f:
            for i, line in enumerate(in_f):
                record = json.loads(line)
                case_id = record["id"]

                # Skip already-embedded cases
                if case_id in already_done:
                    skipped += 1
                    continue

                try:
                    embedding = get_embedding(client, record["embedding_text"])

                    output_record = {
                        "id": case_id,
                        "values": embedding,      # "values" is Pinecone's expected key
                        "metadata": record["metadata"],
                    }

                    out_f.write(json.dumps(output_record) + "\n")
                    out_f.flush()  # Write immediately, don't buffer
                    success += 1

                except Exception as e:
                    # Log errors but don't crash — continue with remaining cases
                    print(f"  ERROR on {case_id}: {e}")
                    errors += 1

                # Progress report every 100 cases
                if (success + errors) % CHECKPOINT_EVERY == 0:
                    total_done = success + errors + skipped
                    pct = (total_done / total) * 100
                    print(f"  Progress: {total_done}/{total} ({pct:.1f}%) | "
                          f"success={success} errors={errors}")

                # Rate limiting — be polite to the API
                time.sleep(DELAY_SECONDS)

    print()
    print(f"Done.")
    print(f"  Embedded successfully : {success}")
    print(f"  Errors                : {errors}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Output file           : {OUTPUT_JSONL}")

    if errors > 0:
        print(f"\n  {errors} cases failed. Re-run the script to retry them.")
        print(f"  (Resume logic will skip the {success} already completed cases)")


if __name__ == "__main__":
    main()
