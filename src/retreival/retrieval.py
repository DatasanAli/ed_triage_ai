"""
retrieval.py
============
EDTriageRAG class — the main interface your teammates will use.

Usage:
    from embeddings.retrieval import EDTriageRAG

    rag = EDTriageRAG()
    cases = rag.retrieve_cases(patient_data, top_k=5)

    # Or with the full explain pipeline (after LLM reasoning is built):
    result = rag.explain(patient_data, prediction, shap_features)

What this module does:
  1. Takes a new patient's data (chief complaint + vitals + demographics)
  2. Builds the same structured text format we used during indexing
  3. Embeds it with Bedrock Titan (one API call)
  4. Queries Pinecone for top-k most similar historical cases
  5. Returns the cases with their metadata and similarity scores

Why the same text format matters:
  The query text MUST be built the same way as the indexed documents.
  If the index was built with "CHIEF COMPLAINT: chest pain" but the query
  sends "chief_complaint: chest pain", the vectors will be in different
  semantic spaces and similarity scores will be meaningless.
  We reuse build_query_text() which mirrors data_prep.py's build_embedding_text().
"""

import json
import os
import time
import boto3
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "ed-triage")
PINECONE_SECRET_NAME = "prod/pinecone/api_key"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ed-triage-cases")
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"


# ---------------------------------------------------------------------------
# EDTriageRAG
# ---------------------------------------------------------------------------

class EDTriageRAG:
    """
    RAG system for ED triage explainability.

    Initialization is lazy about credentials — it only calls AWS/Pinecone
    when you first make a retrieve_cases() call, not when you instantiate
    the class. This makes it easy to import and test without live credentials.

    Attributes:
        _session: boto3 Session (initialized on first use)
        _bedrock: Bedrock Runtime client (initialized on first use)
        _index: Pinecone Index object (initialized on first use)
    """

    def __init__(self):
        self._session = None
        self._bedrock = None
        self._index = None

    def _init_clients(self):
        """
        Initialize AWS and Pinecone clients on first use.

        Why lazy initialization?
          If you import this class at the top of a file, you don't want it
          making network calls just from the import. Lazy init means clients
          are only created when actually needed, and only once.
        """
        if self._bedrock is not None:
            return  # Already initialized

        self._session = boto3.Session(
            profile_name=AWS_PROFILE,
            region_name=AWS_REGION,
        )

        # Fetch Pinecone API key from Secrets Manager
        sm = self._session.client("secretsmanager")
        secret = json.loads(
            sm.get_secret_value(SecretId=PINECONE_SECRET_NAME)["SecretString"]
        )
        pinecone_api_key = secret["PINECONE_API_KEY"]

        # Bedrock client for embedding queries
        self._bedrock = self._session.client("bedrock-runtime")

        # Pinecone index for similarity search
        pc = Pinecone(api_key=pinecone_api_key)
        self._index = pc.Index(PINECONE_INDEX_NAME)

    # -----------------------------------------------------------------------
    # Query text builder
    # -----------------------------------------------------------------------

    def build_query_text(self, patient: dict) -> str:
        """
        Convert patient data dict into the embedding text format.

        This MUST mirror the format used in data_prep.py's build_embedding_text().
        The patient dict should have these keys (all optional except chief_complaint):
          - age (int or str)
          - gender (str)
          - race (str)
          - chief_complaint (str)
          - heart_rate (float)
          - systolic_bp (float)
          - diastolic_bp (float)
          - resp_rate (float)
          - temperature (float)
          - spo2 (float)
          - hpi (str) — history of present illness narrative
          - past_medical_history (str)

        Example:
            patient = {
                "age": 68,
                "gender": "Female",
                "chief_complaint": "chest pain",
                "heart_rate": 110,
                "systolic_bp": 135,
                "diastolic_bp": 85,
                "spo2": 99,
            }
        """
        sections = []

        # Patient demographics
        demo_parts = []
        if patient.get("gender"):
            demo_parts.append(f"Gender: {patient['gender']}")
        if patient.get("race"):
            demo_parts.append(f"Race: {patient['race']}")
        if patient.get("age"):
            demo_parts.append(f"Age: {patient['age']}")
        if demo_parts:
            sections.append(f"PATIENT: {', '.join(demo_parts)}")

        # Chief complaint
        cc = patient.get("chief_complaint", "")
        sections.append(f"CHIEF COMPLAINT: {cc if cc else 'Not recorded'}")

        # Vitals
        vitals_parts = []
        if patient.get("heart_rate"):
            vitals_parts.append(f"HR {patient['heart_rate']:.0f} bpm")
        if patient.get("systolic_bp") and patient.get("diastolic_bp"):
            vitals_parts.append(
                f"BP {patient['systolic_bp']:.0f}/{patient['diastolic_bp']:.0f} mmHg"
            )
        if patient.get("resp_rate"):
            vitals_parts.append(f"RR {patient['resp_rate']:.0f}")
        if patient.get("temperature"):
            vitals_parts.append(f"Temp {patient['temperature']:.1f}F")
        if patient.get("spo2"):
            vitals_parts.append(f"SpO2 {patient['spo2']:.0f}%")

        vitals_text = ", ".join(vitals_parts) if vitals_parts else "Vitals not recorded."
        sections.append(f"VITALS: {vitals_text}")

        # HPI narrative
        hpi = patient.get("hpi", "")
        if hpi:
            sections.append(f"HISTORY: {hpi[:500]}")

        # Past medical history
        pmh = patient.get("past_medical_history", "")
        if pmh:
            sections.append(f"PAST MEDICAL HISTORY: {pmh[:300]}")

        return "\n".join(sections)

    # -----------------------------------------------------------------------
    # Embedding
    # -----------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """
        Embed a text string using Bedrock Titan.
        Same call as generate_embeddings.py — consistency is critical.
        """
        body = json.dumps({"inputText": text})
        response = self._bedrock.invoke_model(
            modelId=TITAN_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        return json.loads(response["body"].read())["embedding"]

    # -----------------------------------------------------------------------
    # Core retrieval
    # -----------------------------------------------------------------------

    def retrieve_cases(
        self,
        patient: dict,
        top_k: int = 5,
        exclude_id: str = None,
        min_score: float = 0.0,
        filter: dict = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most similar historical cases for a patient.

        Args:
            patient: Dict with patient data (see build_query_text for keys)
            top_k: Number of similar cases to return (default 5)
            exclude_id: Pinecone vector ID to exclude from results.
                        Use this when querying with a case that's already
                        in the index (e.g. "stay_32822973") to avoid
                        returning the case itself as its own best match.
            min_score: Minimum cosine similarity threshold (0.0 to 1.0).
                       Cases below this score are filtered out.
                       0.7 is a reasonable starting point for clinical similarity.
            filter: Optional Pinecone metadata filter dict.
                    Example: {"triage_level": {"$in": [2, 3]}}
                    See Pinecone filter docs for full syntax.

        Returns:
            List of dicts, each containing:
              - case_id (str): Pinecone vector ID e.g. "stay_32822973"
              - score (float): Cosine similarity score (0.0 to 1.0)
              - metadata (dict): All stored metadata fields

        Example return value:
            [
                {
                    "case_id": "stay_28451",
                    "score": 0.89,
                    "metadata": {
                        "triage_level": 2,
                        "chief_complaint": "CHEST PAIN",
                        "disposition": "ADMITTED",
                        "primary_diagnosis": "['NSTEMI']",
                        ...
                    }
                },
                ...
            ]
        """
        self._init_clients()

        start = time.time()

        # Build and embed the query
        query_text = self.build_query_text(patient)
        query_vector = self._embed(query_text)

        # Request more results than needed if we're excluding one
        # e.g. if top_k=5 and we exclude 1, request 6 so we still get 5 back
        fetch_k = top_k + 1 if exclude_id else top_k

        # Query Pinecone
        # include_metadata=True tells Pinecone to return the metadata alongside
        # each vector — without this we'd only get IDs and scores
        query_kwargs = {
            "vector": query_vector,
            "top_k": fetch_k,
            "include_metadata": True,
        }
        if filter:
            query_kwargs["filter"] = filter

        response = self._index.query(**query_kwargs)

        elapsed_ms = (time.time() - start) * 1000

        # Format results
        results = []
        for match in response["matches"]:
            # Skip the excluded ID (self-match)
            if exclude_id and match["id"] == exclude_id:
                continue

            # Apply minimum score threshold
            if match["score"] < min_score:
                continue

            results.append({
                "case_id": match["id"],
                "score": round(match["score"], 4),
                "metadata": match.get("metadata", {}),
            })

            if len(results) == top_k:
                break

        return results, elapsed_ms

    # -----------------------------------------------------------------------
    # Convenience: format cases for display or LLM prompt
    # -----------------------------------------------------------------------

    def format_cases_for_prompt(self, cases: list[dict]) -> str:
        """
        Format retrieved cases as readable text for an LLM prompt.

        The LLM reasoning step (generate_reasoning.py) will call this to
        build the "evidence" section of the prompt. Each case is formatted
        as a concise clinical summary.

        Example output:
            SIMILAR CASE 1 (similarity: 0.89):
            Patient: Gender: Female, Age: 80 | ESI: 2 | Chief Complaint: LETHARGY/SOB
            Vitals: HR 95 bpm, BP 99/47 mmHg
            Diagnosis: Acute Respiratory Failure
            Outcome: ADMITTED

            SIMILAR CASE 2 (similarity: 0.85):
            ...
        """
        lines = []
        for i, case in enumerate(cases, 1):
            m = case["metadata"]
            lines.append(f"SIMILAR CASE {i} (similarity: {case['score']}):")
            lines.append(
                f"  Patient: {m.get('patient_info', 'unknown')} | "
                f"ESI: {m.get('triage_level', '?')} | "
                f"Chief Complaint: {m.get('chief_complaint', 'unknown')}"
            )

            # Vitals summary from individual metadata fields
            vitals = []
            if m.get("heart_rate"):
                vitals.append(f"HR {m['heart_rate']:.0f} bpm")
            if m.get("sbp") and m.get("dbp"):
                vitals.append(f"BP {m['sbp']:.0f}/{m['dbp']:.0f} mmHg")
            if m.get("spo2"):
                vitals.append(f"SpO2 {m['spo2']:.0f}%")
            if vitals:
                lines.append(f"  Vitals: {', '.join(vitals)}")

            lines.append(f"  Diagnosis: {m.get('icd_title', 'unknown')}")
            lines.append(f"  Outcome: {m.get('disposition', 'unknown')}")
            if m.get("hpi"):
                lines.append(f"  History: {m['hpi'][:300]}...")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick test — run directly to verify retrieval is working
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rag = EDTriageRAG()

    # Test patient — 68yo female with chest pain
    test_patient = {
        "age": 68,
        "gender": "Female",
        "chief_complaint": "CHEST PAIN",
        "heart_rate": 110,
        "systolic_bp": 135,
        "diastolic_bp": 85,
        "resp_rate": 18,
        "temperature": 98.6,
        "spo2": 99,
    }

    print("Querying Pinecone for similar cases...")
    print(f"Patient: {test_patient}")
    print()

    cases, elapsed_ms = rag.retrieve_cases(test_patient, top_k=5)

    print(f"Retrieved {len(cases)} cases in {elapsed_ms:.0f}ms")
    print()
    print(rag.format_cases_for_prompt(cases))
