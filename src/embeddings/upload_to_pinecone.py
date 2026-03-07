"""
upload_to_pinecone.py
=====================
Reads embeddings_output.jsonl and upserts all vectors into a Pinecone index.
Also uploads the embeddings file to S3 as a backup.

What this script does:
  1. Fetches the Pinecone API key from AWS Secrets Manager
  2. Creates the Pinecone index if it doesn't exist yet
  3. Reads embeddings_output.jsonl in batches of 100
  4. Upserts each batch to Pinecone
  5. Uploads embeddings_output.jsonl to S3 as a permanent backup

Why batches of 100?
  Pinecone's upsert API accepts up to 100 vectors per call.
  Sending one vector at a time would mean 9,146 API calls.
  Sending 100 at a time means only ~92 calls — much faster.
  This is the key reason upload is much faster than embedding generation.

Why upsert and not insert?
  "Upsert" = update if exists, insert if not.
  This makes the script safe to re-run — if you run it twice, you won't
  get duplicate vectors. Pinecone deduplicates by ID.

Run:
  python embeddings/upload_to_pinecone.py
"""

import json
import os
import boto3
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = ROOT / "embeddings"

INPUT_JSONL = EMBEDDINGS_DIR / "embeddings_output.jsonl"

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "ed-triage")

PINECONE_SECRET_NAME = "prod/pinecone/api_key"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ed-triage-cases")

# S3 bucket for backing up the embeddings file
S3_BUCKET = os.getenv("S3_BUCKET", "ed-triage-capstone-group7")
S3_KEY = "embeddings/embeddings_output.jsonl"

# Pinecone upsert batch size — max allowed is 100
BATCH_SIZE = 100

# Vector dimensions — must match what Titan v2 produced
VECTOR_DIMS = 1024


# ---------------------------------------------------------------------------
# AWS session
# ---------------------------------------------------------------------------

def get_session() -> boto3.Session:
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def get_pinecone_api_key(session: boto3.Session) -> str:
    client = session.client("secretsmanager")
    response = client.get_secret_value(SecretId=PINECONE_SECRET_NAME)
    secret = json.loads(response["SecretString"])
    return secret["PINECONE_API_KEY"]


# ---------------------------------------------------------------------------
# Pinecone index setup
# ---------------------------------------------------------------------------

def get_or_create_index(pc: Pinecone) -> object:
    """
    Get the Pinecone index, creating it first if it doesn't exist.

    Index configuration explained:
      name: "ed-triage-cases" — the index name you query against
      dimension: 1024 — MUST match the embedding size from Titan v2.
                        If this doesn't match, every upsert will fail.
      metric: "cosine" — the similarity function used at query time.
              Cosine measures the angle between two vectors, ignoring magnitude.
              It's the standard choice for text embeddings because it focuses
              on semantic direction, not vector length.
              Other options: "euclidean" (distance), "dotproduct" (speed).

    ServerlessSpec:
      Pinecone has two deployment modes:
        - Serverless: scales to zero, pay per query, no always-on cost.
                      Best for development and low-traffic production.
        - Pod-based: always-on, predictable latency, higher cost.
                     Best for high-traffic production.
      We use Serverless on AWS us-east-1 — same region as our Bedrock calls,
      which minimizes latency when we later query from Lambda.
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating it...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VECTOR_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    return pc.Index(PINECONE_INDEX_NAME)


# ---------------------------------------------------------------------------
# Upload to Pinecone
# ---------------------------------------------------------------------------

def clean_metadata(metadata: dict) -> dict:
    """
    Remove any metadata fields with None/null values.

    Pinecone only accepts str, int, float, bool, or list of strings.
    null (Python None) is not allowed and causes a 400 Bad Request.
    Missing vitals show up as None — we just drop those fields entirely
    rather than substituting a fake value like -1.
    """
    return {k: v for k, v in metadata.items() if v is not None}


def upload_to_pinecone(index) -> dict:
    """
    Read embeddings_output.jsonl and upsert to Pinecone in batches.

    Each record in the JSONL has exactly the shape Pinecone expects:
      {"id": "stay_32822973", "values": [...1024 floats...], "metadata": {...}}

    We accumulate BATCH_SIZE records, then upsert the whole batch at once.
    After the loop, we flush any remaining records that didn't fill a full batch.
    """
    batch = []
    total_upserted = 0
    batch_count = 0

    with open(INPUT_JSONL) as f:
        for line in f:
            record = json.loads(line)
            record["metadata"] = clean_metadata(record["metadata"])
            batch.append(record)

            if len(batch) == BATCH_SIZE:
                index.upsert(vectors=batch)
                total_upserted += len(batch)
                batch_count += 1
                batch = []

                if batch_count % 10 == 0:
                    print(f"  Upserted {total_upserted} vectors...")

    # Flush remaining records (the last partial batch)
    if batch:
        index.upsert(vectors=batch)
        total_upserted += len(batch)

    return {"total_upserted": total_upserted}


# ---------------------------------------------------------------------------
# S3 backup
# ---------------------------------------------------------------------------

def backup_to_s3(session: boto3.Session):
    """
    Upload embeddings_output.jsonl to S3.

    Why back up to S3?
      - embeddings_output.jsonl is ~90MB and gitignored
      - If the Pinecone index is deleted, teammates can rebuild from S3
        without re-calling Bedrock ($0.04 but ~30 mins)
      - S3 is durable (11 nines) and cheap (~$0.002/month for 90MB)

    The file goes to: s3://ed-triage/embeddings/embeddings_output.jsonl
    """
    s3 = session.client("s3")
    file_size_mb = INPUT_JSONL.stat().st_size / (1024 * 1024)
    print(f"Uploading {file_size_mb:.1f}MB to s3://{S3_BUCKET}/{S3_KEY} ...")
    s3.upload_file(str(INPUT_JSONL), S3_BUCKET, S3_KEY)
    print(f"S3 backup complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Input file : {INPUT_JSONL}")
    print(f"Index name : {PINECONE_INDEX_NAME}")
    print(f"Batch size : {BATCH_SIZE}")
    print()

    # Count total records
    with open(INPUT_JSONL) as f:
        total = sum(1 for _ in f)
    print(f"Total vectors to upload: {total}")
    print()

    # AWS session
    session = get_session()
    print(f"AWS session initialized (profile: {AWS_PROFILE})")

    # Pinecone API key from Secrets Manager
    print(f"Fetching Pinecone API key from Secrets Manager...")
    api_key = get_pinecone_api_key(session)
    print("Pinecone API key retrieved.")
    print()

    # Initialize Pinecone client (v3 API — no environment param needed)
    pc = Pinecone(api_key=api_key)

    # Get or create the index
    index = get_or_create_index(pc)
    print()

    # Upload vectors
    print("Uploading vectors to Pinecone...")
    result = upload_to_pinecone(index)
    print(f"Done. {result['total_upserted']} vectors upserted.")
    print()

    # Verify — ask Pinecone how many vectors are in the index
    stats = index.describe_index_stats()
    print(f"Index stats:")
    print(f"  Total vectors in index : {stats['total_vector_count']}")
    print(f"  Dimensions             : {stats['dimension']}")
    print()

    # S3 backup
    print("Backing up embeddings file to S3...")
    backup_to_s3(session)
    print()

    print("All done. Pinecone index is ready for querying.")


if __name__ == "__main__":
    main()
