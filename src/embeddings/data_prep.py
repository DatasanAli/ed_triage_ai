"""
data_prep.py
============
Prepares the MIMIC-IV-Ext ED dataset for embedding generation.

What this script does:
  1. Loads the raw CSV
  2. Cleans and validates each case
  3. Builds a structured "embedding text" per case — the text we send to
     Bedrock Titan to get a vector representation
  4. Assembles metadata per case — the structured fields we store alongside
     the vector in Pinecone (for filtering and display)
  5. Writes a JSONL file where each line is one case ready to embed

Why JSONL (JSON Lines)?
  Each line is a self-contained JSON object. This makes it easy to stream
  through 9K+ records without loading everything into memory at once —
  important when we later send each record to Bedrock one at a time.

Run:
  python embeddings/data_prep.py
"""

import json
import re
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # project root (ed_triage_ai/)
DATA_DIR = ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent  # src/embeddings/

INPUT_CSV = DATA_DIR / "consolidated_dataset_PMH.csv"
OUTPUT_JSONL = OUTPUT_DIR / "cases_for_embedding.jsonl"
STATS_FILE = OUTPUT_DIR / "data_prep_stats.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(value) -> str:
    """
    Convert any value to a clean string.

    Raw fields from MIMIC often contain:
      - NaN (pandas missing value)
      - Redacted tokens like '___' (MIMIC de-identification placeholder)
      - HTML tags like <br>
      - Extra whitespace / newlines

    We strip all of that so the embedding model sees clean prose.
    """
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)        # remove HTML tags
    text = re.sub(r"\b___\b", "[redacted]", text) # mark de-id placeholders
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    return text.strip()


def parse_vitals(raw: str) -> dict:
    """
    Parse the initial_vitals string into a dict of floats.

    Raw format: "Temperature: 100.2, Heartrate: 95.0, resprate: 28.0,
                 o2sat: 100.0, sbp: 99.0, dbp: 47.0"

    We parse it so we can:
      a) Store individual vital values as Pinecone metadata (for filtering)
      b) Write them into the embedding text in a consistent, human-readable way
    """
    vitals = {}
    if not raw:
        return vitals
    # Each segment looks like "Key: value"
    for segment in raw.split(","):
        parts = segment.strip().split(":")
        if len(parts) == 2:
            key = parts[0].strip().lower()
            try:
                vitals[key] = float(parts[1].strip())
            except ValueError:
                pass
    return vitals


def format_vitals_text(vitals: dict) -> str:
    """
    Turn the vitals dict into readable prose for the embedding text.

    Why readable prose?
      Titan Embeddings is a language model. It understands "HR 95 bpm" better
      than a raw dict. Prose also matches how clinical notes are written,
      which is what Titan was trained on.
    """
    if not vitals:
        return "Vitals not recorded."
    parts = []
    if "heartrate" in vitals:
        parts.append(f"HR {vitals['heartrate']:.0f} bpm")
    if "sbp" in vitals and "dbp" in vitals:
        parts.append(f"BP {vitals['sbp']:.0f}/{vitals['dbp']:.0f} mmHg")
    if "resprate" in vitals:
        parts.append(f"RR {vitals['resprate']:.0f}")
    if "temperature" in vitals:
        parts.append(f"Temp {vitals['temperature']:.1f}F")
    if "o2sat" in vitals:
        parts.append(f"SpO2 {vitals['o2sat']:.0f}%")
    return ", ".join(parts) if parts else "Vitals not recorded."


def build_embedding_text(row: pd.Series) -> str:
    """
    Build the text we will send to Bedrock Titan for embedding.

    IMPORTANT — only triage-time data is included here.

    Research basis (JMIR 2026 - Multi-Evidence Clinical Reasoning RAG for ED Triage):
    Embeddings should only use information available at the moment of triage.
    Including post-triage data (HPI, diagnosis, PMH, disposition) causes
    spurious similarity — two unrelated patients match because they share
    common comorbidities or similar discharge note language, not because
    their presentations were clinically similar.

    What a triage nurse knows at triage time:
      ✅ Chief complaint (why did you come in?)
      ✅ Vitals (measured on arrival)
      ✅ Demographics (age, gender — from registration)
      ✅ ESI level (assigned during triage)
      ✅ Arrival transport (ambulance vs walk-in — observable)

    What is NOT yet known at triage time:
      ❌ HPI — comes from physician interview later
      ❌ Diagnosis — determined after workup
      ❌ Disposition — decided after treatment
      ❌ PMH — often unknown or unverified at triage

    Diagnosis and disposition are kept in metadata (for the LLM to use
    when explaining retrieved cases) but NOT in the embedding text.
    """
    sections = []

    # Demographics — from registration, available at triage
    patient_info = clean_text(row["patient_info"])
    if patient_info:
        sections.append(f"PATIENT: {patient_info}")

    # Chief complaint — the single most important triage signal
    cc = clean_text(row["chiefcomplaint"])
    sections.append(f"CHIEF COMPLAINT: {cc if cc else 'Not recorded'}")

    # ESI triage level — assigned by the triage nurse
    sections.append(f"ESI TRIAGE LEVEL: {int(row['triage'])}")

    # Vitals — measured on arrival, core triage data
    vitals = parse_vitals(clean_text(row["initial_vitals"]))
    sections.append(f"VITALS: {format_vitals_text(vitals)}")

    # Arrival transport — observable at triage, clinically meaningful
    # (ambulance arrival signals pre-hospital concern)
    arrival = clean_text(row["arrival_transport"])
    if arrival:
        sections.append(f"ARRIVAL: {arrival}")

    return "\n".join(sections)


def build_metadata(row: pd.Series, vitals: dict) -> dict:
    """
    Build the metadata dict stored alongside each vector in Pinecone.

    Why metadata matters:
      Pinecone stores vectors + metadata. After we retrieve top-k similar
      vectors, we use the metadata to:
        - Display case details in the UI
        - Filter queries (e.g., "only retrieve ESI 2-3 cases")
        - Feed into the LLM reasoning prompt

    Pinecone metadata rules:
      - Values must be str, int, float, bool, or list of those
      - Keep it flat (no nested dicts)
      - Don't store huge text blobs here — that's what the vector is for
    """
    return {
        "stay_id": int(row["stay_id"]),
        "triage_level": int(row["triage"]),
        "chief_complaint": clean_text(row["chiefcomplaint"]) or "unknown",
        "disposition": clean_text(row["disposition"]),
        "primary_diagnosis": clean_text(row["primary_diagnosis"]),
        "icd_code": str(row["icd_code"]),
        "icd_title": clean_text(row["icd_title"]),
        "patient_info": clean_text(row["patient_info"]),
        # Individual vitals as floats for range filtering in Pinecone
        "heart_rate": vitals.get("heartrate"),
        "sbp": vitals.get("sbp"),
        "dbp": vitals.get("dbp"),
        "resp_rate": vitals.get("resprate"),
        "temp": vitals.get("temperature"),
        "spo2": vitals.get("o2sat"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading dataset from {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Counters for the stats report
    stats = {
        "total_rows": len(df),
        "skipped_missing_triage": 0,
        "skipped_missing_cc": 0,
        "missing_vitals": int(df["initial_vitals"].isna().sum()),
        "output_cases": 0,
        "triage_distribution": df["triage"].value_counts().sort_index().to_dict(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with open(OUTPUT_JSONL, "w") as f:
        for _, row in df.iterrows():

            # Skip rows with no triage label — we need this for metadata
            if pd.isna(row["triage"]):
                stats["skipped_missing_triage"] += 1
                skipped += 1
                continue

            # Skip rows with no chief complaint — too little signal to embed
            if pd.isna(row["chiefcomplaint"]):
                stats["skipped_missing_cc"] += 1
                skipped += 1
                continue

            vitals = parse_vitals(clean_text(row["initial_vitals"]))
            embedding_text = build_embedding_text(row)
            metadata = build_metadata(row, vitals)

            record = {
                # Pinecone requires a unique string ID per vector
                "id": f"stay_{int(row['stay_id'])}",
                "embedding_text": embedding_text,
                "metadata": metadata,
            }

            f.write(json.dumps(record) + "\n")
            written += 1

    stats["output_cases"] = written

    # Save stats so we have a record of what was processed
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone.")
    print(f"  Cases written : {written}")
    print(f"  Cases skipped : {skipped}")
    print(f"  Output file   : {OUTPUT_JSONL}")
    print(f"  Stats file    : {STATS_FILE}")
    print(f"\nTriage distribution in output:")
    for level, count in stats["triage_distribution"].items():
        print(f"  ESI {level}: {count} cases")


if __name__ == "__main__":
    main()
