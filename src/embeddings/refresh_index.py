"""
refresh_index.py
================
Full refresh of the Pinecone index with updated embeddings.

Run this when cases_for_embedding.jsonl has changed (e.g. new data source,
updated metadata fields) and you want Pinecone to reflect the latest data.

What this does:
  1. Deletes the existing Pinecone index (removes stale vectors)
  2. Recreates the index fresh
  3. Runs generate_embeddings.py to re-embed all cases
  4. Runs upload_to_pinecone.py to upload fresh vectors

Why delete instead of upsert?
  We switched from consolidated_dataset_PMH.csv (9,147 cases) to
  consolidated_dataset_features.csv (8,383 cases). If we just upsert,
  the 766 cases that dropped out remain as stale vectors in the index.
  A full delete+recreate ensures the index exactly mirrors the current dataset.

Run on SageMaker (has AWS credentials):
  python src/embeddings/refresh_index.py
"""

import json
import os
import time
import boto3
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

ROOT           = Path(__file__).resolve().parent.parent.parent
EMBEDDINGS_DIR = ROOT / "src" / "embeddings"

AWS_REGION           = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE          = os.getenv("AWS_PROFILE", "ed-triage")
PINECONE_SECRET_NAME = "prod/pinecone/api_key"
PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME", "ed-triage-cases")
VECTOR_DIMS          = 1024
BATCH_SIZE           = 100
S3_BUCKET            = os.getenv("S3_BUCKET", "ed-triage-capstone-group7")


def get_session():
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def get_pinecone_api_key(session):
    client = session.client("secretsmanager")
    response = client.get_secret_value(SecretId=PINECONE_SECRET_NAME)
    return json.loads(response["SecretString"])["PINECONE_API_KEY"]


def clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if v is not None}


def delete_and_recreate_index(pc: Pinecone):
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing:
        print(f"Deleting existing index '{PINECONE_INDEX_NAME}'...")
        pc.delete_index(PINECONE_INDEX_NAME)
        # Wait for deletion to complete
        while PINECONE_INDEX_NAME in [idx.name for idx in pc.list_indexes()]:
            print("  Waiting for deletion...")
            time.sleep(3)
        print("  Deleted.")

    print(f"Creating fresh index '{PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=VECTOR_DIMS,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index to be ready
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("  Waiting for index to be ready...")
        time.sleep(3)
    print("  Index ready.")
    return pc.Index(PINECONE_INDEX_NAME)


def embed_all_cases(bedrock_client):
    """Re-embed all cases from cases_for_embedding.jsonl."""
    input_jsonl  = EMBEDDINGS_DIR / "cases_for_embedding.jsonl"
    output_jsonl = EMBEDDINGS_DIR / "embeddings_output.jsonl"

    with open(input_jsonl) as f:
        total = sum(1 for _ in f)
    print(f"Embedding {total} cases...")

    # Overwrite output file fresh
    success = 0
    errors  = 0
    with open(output_jsonl, "w") as out_f:
        with open(input_jsonl) as in_f:
            for i, line in enumerate(in_f, start=1):
                record = json.loads(line)
                try:
                    body = json.dumps({"inputText": record["embedding_text"]})
                    response = bedrock_client.invoke_model(
                        modelId="amazon.titan-embed-text-v2:0",
                        contentType="application/json",
                        accept="application/json",
                        body=body,
                    )
                    embedding = json.loads(response["body"].read())["embedding"]
                    out_f.write(json.dumps({
                        "id":       record["id"],
                        "values":   embedding,
                        "metadata": record["metadata"],
                    }) + "\n")
                    out_f.flush()
                    success += 1
                except Exception as e:
                    print(f"  ERROR on {record['id']}: {e}")
                    errors += 1

                if i % 100 == 0:
                    print(f"  Progress: {i}/{total} ({100*i/total:.1f}%) | "
                          f"success={success} errors={errors}")
                time.sleep(0.1)

    print(f"Embedding complete: {success} success, {errors} errors.")
    return output_jsonl


def upload_to_pinecone(index, output_jsonl: Path):
    print(f"Uploading vectors to Pinecone...")
    batch          = []
    total_upserted = 0

    with open(output_jsonl) as f:
        for line in f:
            record = json.loads(line)
            record["metadata"] = clean_metadata(record["metadata"])
            batch.append(record)

            if len(batch) == BATCH_SIZE:
                index.upsert(vectors=batch)
                total_upserted += len(batch)
                batch = []
                if total_upserted % 1000 == 0:
                    print(f"  Upserted {total_upserted} vectors...")

    if batch:
        index.upsert(vectors=batch)
        total_upserted += len(batch)

    print(f"Upload complete: {total_upserted} vectors upserted.")
    return total_upserted


def backup_to_s3(session, output_jsonl: Path):
    s3   = session.client("s3")
    key  = "embeddings/embeddings_output.jsonl"
    size = output_jsonl.stat().st_size / (1024 * 1024)
    print(f"Backing up {size:.1f}MB to s3://{S3_BUCKET}/{key}...")
    s3.upload_file(str(output_jsonl), S3_BUCKET, key)
    print("S3 backup complete.")


def main():
    print("=== Pinecone Full Refresh ===")
    print(f"Index : {PINECONE_INDEX_NAME}")
    print(f"Region: {AWS_REGION}")
    print()

    session        = get_session()
    api_key        = get_pinecone_api_key(session)
    pc             = Pinecone(api_key=api_key)
    bedrock_client = session.client("bedrock-runtime")

    # Step 1: Delete and recreate index
    index = delete_and_recreate_index(pc)
    print()

    # Step 2: Re-embed all cases
    output_jsonl = embed_all_cases(bedrock_client)
    print()

    # Step 3: Upload to Pinecone
    total = upload_to_pinecone(index, output_jsonl)
    print()

    # Step 4: Verify
    stats = index.describe_index_stats()
    print(f"Index verified: {stats['total_vector_count']} vectors, "
          f"{stats['dimension']} dims")
    print()

    # Step 5: S3 backup
    backup_to_s3(session, output_jsonl)
    print()

    print("Refresh complete. Pinecone index is ready.")


if __name__ == "__main__":
    main()
