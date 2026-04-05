"""
eval_e2e_pipeline.py
====================
Runs the full end-to-end inference pipeline (SageMaker → LangGraph → LLM)
over the held-out test set (839 records) and computes per-class F1 scores
for both model-only and reconciled (model + LLM) predictions.

Usage:
    python scripts/eval_e2e_pipeline.py [--workers N] [--output results/e2e_eval.json]

    --workers   Concurrent async workers (default 10). At ~10s/call that gives
                ~840s / 10 workers ≈ 14 minutes total.
    --output    Path to write JSON results (default: results/e2e_eval.json)
    --limit     Optional: evaluate only first N records (for smoke testing)

Output files:
    <output>.json       Full per-record results + aggregate metrics
    <output>_summary.txt  Human-readable classification reports
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ── Add project root to path so src.agents imports work ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SAGEMAKER_ENDPOINT = os.getenv("TRIAGE_SAGEMAKER_ENDPOINT_NAME", "edtriage-live")
AWS_PROFILE        = os.getenv("TRIAGE_AWS_PROFILE", "ed-triage")
AWS_REGION         = os.getenv("TRIAGE_AWS_REGION", "us-east-1")
DATA_PATH          = ROOT / "data" / "consolidated_dataset_features.csv"
LABEL_MAP          = {0: "L1-Critical", 1: "L2-Emergent", 2: "L3-Urgent/LessUrgent"}
TRIAGE_MAP         = {1: 0, 2: 1, 3: 2, 4: 2}


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_test_set() -> pd.DataFrame:
    """Recreate the exact 839-record test split used during training."""
    df = pd.read_csv(DATA_PATH)
    df["triage_3class"] = df["triage"].map(TRIAGE_MAP)
    df = df.dropna(subset=["triage_3class"])
    df["triage_3class"] = df["triage_3class"].astype(int)

    # Build triage_text (CC_2x + HPI format — matches arch4 training)
    df["triage_text"] = (
        df["chiefcomplaint"].fillna("") + ". "
        + df["chiefcomplaint"].fillna("") + ". "
        + df["HPI"].fillna("")
    )

    train_val, test = train_test_split(
        df, test_size=0.10, stratify=df["triage_3class"], random_state=42
    )
    logger.info(
        "Test set: %d records | class dist: %s",
        len(test),
        test["triage_3class"].value_counts().sort_index().to_dict(),
    )
    return test.reset_index(drop=True)


def row_to_sagemaker_payload(row: pd.Series) -> dict:
    """Convert a dataset row to the SageMaker endpoint payload format."""
    return {
        "triage_text":       str(row.get("triage_text", "")),
        "arrival_transport": str(row.get("arrival_transport", "WALK IN")).upper(),
        "age":               float(row["age"])        if pd.notna(row.get("age"))        else None,
        "heart_rate":        float(row["heart_rate"]) if pd.notna(row.get("heart_rate")) else None,
        "resp_rate":         float(row["resp_rate"])  if pd.notna(row.get("resp_rate"))  else None,
        "sbp":               float(row["sbp"])        if pd.notna(row.get("sbp"))        else None,
        "dbp":               float(row["dbp"])        if pd.notna(row.get("dbp"))        else None,
        "spo2":              float(row["spo2"])        if pd.notna(row.get("spo2"))       else None,
        "temp_f":            float(row["temp_f"])     if pd.notna(row.get("temp_f"))     else None,
        "pain":              float(row["pain"])       if pd.notna(row.get("pain"))       else None,
    }


def row_to_patient(row: pd.Series) -> dict:
    """Convert a dataset row to the patient dict the LangGraph analyze_node expects."""
    return {
        "chief_complaint": str(row.get("chiefcomplaint", "")),
        "age":             float(row["age"])        if pd.notna(row.get("age"))        else None,
        "gender":          str(row.get("gender", "unknown")),
        "heart_rate":      float(row["heart_rate"]) if pd.notna(row.get("heart_rate")) else None,
        "systolic_bp":     float(row["sbp"])        if pd.notna(row.get("sbp"))        else None,
        "diastolic_bp":    float(row["dbp"])        if pd.notna(row.get("dbp"))        else None,
        "resp_rate":       float(row["resp_rate"])  if pd.notna(row.get("resp_rate"))  else None,
        "temperature":     float(row["temp_f"])     if pd.notna(row.get("temp_f"))     else None,
        "spo2":            float(row["spo2"])        if pd.notna(row.get("spo2"))       else None,
        "pain":            float(row["pain"])       if pd.notna(row.get("pain"))       else None,
        "arrival_transport": str(row.get("arrival_transport", "WALK IN")).upper(),
        "hpi":             str(row.get("HPI", "")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SageMaker invocation
# ─────────────────────────────────────────────────────────────────────────────

def make_sagemaker_client():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client("sagemaker-runtime")


def invoke_sagemaker(client, payload: dict) -> dict:
    response = client.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps({k: v for k, v in payload.items() if v is not None}),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


# Per-record timeout (seconds) — kills hung threads instead of blocking forever
RECORD_TIMEOUT_S = 60


# ─────────────────────────────────────────────────────────────────────────────
# Per-record inference
# ─────────────────────────────────────────────────────────────────────────────

def run_one_record(row: pd.Series, sm_client) -> dict:
    """
    Run the full pipeline for a single test record.
    Returns a result dict with model and reconciled predictions.
    """
    from src.agents.graph import triage_graph

    stay_id   = int(row["stay_id"])
    true_class = int(row["triage_3class"])
    t0         = time.perf_counter()

    result = {
        "stay_id":          stay_id,
        "true_class":       true_class,
        "true_label":       LABEL_MAP[true_class],
        "model_class":      None,
        "model_label":      None,
        "reconciled_class": None,
        "reconciled_label": None,
        "llm_esi":          None,
        "llm_agreement":    None,
        "confidence_pct":   None,
        "rag_cases_retrieved": None,
        "rag_best_similarity": None,
        "rag_mean_similarity": None,
        "rag_retrieval_ms":    None,
        "error":            None,
        "elapsed_s":        None,
    }

    try:
        # 1. SageMaker model prediction
        sm_payload     = row_to_sagemaker_payload(row)
        sm_response    = invoke_sagemaker(sm_client, sm_payload)

        # Normalize probabilities dict → list for the graph
        probs_dict = sm_response.get("probabilities", {})
        if isinstance(probs_dict, dict):
            sm_response["probabilities"] = [
                probs_dict.get("L1-Critical", 0.0),
                probs_dict.get("L2-Emergent", 0.0),
                probs_dict.get("L3-Urgent/LessUrgent", probs_dict.get("L3-Urgent", 0.0)),
            ]

        # 2. Full LangGraph pipeline (RAG + LLM + reconciliation)
        patient     = row_to_patient(row)
        thread_id   = str(uuid.uuid4())
        graph_result = triage_graph.invoke(
            {"patient": patient, "prediction": sm_response},
            config={"configurable": {"thread_id": thread_id}},
        )
        report = graph_result["final_report"]

        # RAG similarity stats
        similar_cases = report.get("similar_cases") or []
        sim_scores = [c["similarity"] for c in similar_cases if c.get("similarity") is not None]

        result.update({
            "model_class":       report["triage_class"],
            "model_label":       report["triage_level"],
            "reconciled_class":  report["reconciled_class"],
            "reconciled_label":  report["reconciled_level"],
            "llm_esi":           report["llm_esi"],
            "llm_agreement":     report["llm_agreement"],
            "confidence_pct":    report["confidence_pct"],
            "rag_cases_retrieved": report.get("cases_retrieved"),
            "rag_best_similarity": round(max(sim_scores), 4) if sim_scores else None,
            "rag_mean_similarity": round(sum(sim_scores) / len(sim_scores), 4) if sim_scores else None,
            "rag_retrieval_ms":    report.get("retrieval_ms"),
        })

    except Exception as exc:
        result["error"] = str(exc)
        logger.warning("stay_id=%s failed: %s", stay_id, exc)

    result["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(test_df: pd.DataFrame, workers: int, limit: int | None) -> list[dict]:
    """Run inference over the test set with a thread pool."""
    rows = [test_df.iloc[i] for i in range(len(test_df))]
    if limit:
        rows = rows[:limit]
        logger.info("Limiting evaluation to first %d records", limit)

    logger.info(
        "Starting batch evaluation: %d records, %d workers", len(rows), workers
    )

    # One SageMaker client per thread (boto3 clients are thread-safe)
    sm_clients = [make_sagemaker_client() for _ in range(workers)]

    results   = []
    done      = 0
    errors    = 0
    t_start   = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one_record, row, sm_clients[i % workers]): i
            for i, row in enumerate(rows)
        }
        for future in as_completed(futures, timeout=None):
            try:
                rec = future.result(timeout=RECORD_TIMEOUT_S)
            except Exception as exc:
                i = futures[future]
                rec = {
                    "stay_id": int(rows[i]["stay_id"]),
                    "true_class": int(rows[i]["triage_3class"]),
                    "true_label": LABEL_MAP[int(rows[i]["triage_3class"])],
                    "model_class": None, "model_label": None,
                    "reconciled_class": None, "reconciled_label": None,
                    "llm_esi": None, "llm_agreement": None,
                    "confidence_pct": None, "elapsed_s": RECORD_TIMEOUT_S,
                    "error": f"timeout/exception: {exc}",
                }
            results.append(rec)
            done += 1
            if rec["error"]:
                errors += 1
            if done % 50 == 0 or done == len(rows):
                elapsed   = time.perf_counter() - t_start
                rate      = done / elapsed
                remaining = (len(rows) - done) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d  errors=%d  rate=%.1f rec/s  ETA=%.0fs",
                    done, len(rows), errors, rate, remaining,
                )

    results.sort(key=lambda r: r["stay_id"])
    logger.info(
        "Batch complete: %d records in %.1fs  (%d errors)",
        len(results), time.perf_counter() - t_start, errors,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute model-only and reconciled F1 metrics from batch results."""
    # Filter out errored records
    valid = [r for r in results if r["error"] is None and r["model_class"] is not None]
    logger.info(
        "Computing metrics on %d valid records (%d skipped due to errors)",
        len(valid), len(results) - len(valid),
    )

    y_true       = [r["true_class"]       for r in valid]
    y_model      = [r["model_class"]      for r in valid]
    y_reconciled = [r["reconciled_class"] for r in valid]

    labels     = [0, 1, 2]
    label_names = [LABEL_MAP[l] for l in labels]

    def metrics_block(y_pred):
        return {
            "macro_f1":    round(f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0), 4),
            "per_class":   {
                LABEL_MAP[l]: {
                    "precision": round(float(classification_report(y_true, y_pred, labels=labels, target_names=label_names, output_dict=True, zero_division=0)[LABEL_MAP[l]]["precision"]), 4),
                    "recall":    round(float(classification_report(y_true, y_pred, labels=labels, target_names=label_names, output_dict=True, zero_division=0)[LABEL_MAP[l]]["recall"]),    4),
                    "f1":        round(float(classification_report(y_true, y_pred, labels=labels, target_names=label_names, output_dict=True, zero_division=0)[LABEL_MAP[l]]["f1-score"]),  4),
                    "support":   int(  classification_report(y_true, y_pred, labels=labels, target_names=label_names, output_dict=True, zero_division=0)[LABEL_MAP[l]]["support"]),
                }
                for l in labels
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "report_text":      classification_report(y_true, y_pred, labels=labels, target_names=label_names, zero_division=0),
        }

    model_metrics      = metrics_block(y_model)
    reconciled_metrics = metrics_block(y_reconciled)

    # LLM agreement stats
    agreements = [r["llm_agreement"] for r in valid if r["llm_agreement"] is not None]
    agree_rate = sum(agreements) / len(agreements) if agreements else None

    # Escalation rate: cases where reconciled is more urgent than model
    escalations = [
        r for r in valid
        if r["reconciled_class"] is not None
        and r["model_class"] is not None
        and r["reconciled_class"] < r["model_class"]
    ]
    escalation_rate = len(escalations) / len(valid) if valid else None

    # Escalation accuracy: of escalated cases, how many had the escalated class correct?
    escalation_correct = [
        e for e in escalations if e["reconciled_class"] == e["true_class"]
    ]
    escalation_precision = (
        len(escalation_correct) / len(escalations) if escalations else None
    )

    # RAG aggregate stats
    best_sims = [r["rag_best_similarity"] for r in valid if r.get("rag_best_similarity") is not None]
    mean_sims = [r["rag_mean_similarity"] for r in valid if r.get("rag_mean_similarity") is not None]
    ret_ms    = [r["rag_retrieval_ms"]    for r in valid if r.get("rag_retrieval_ms")    is not None]
    low_sim_count = sum(1 for s in best_sims if s < 0.65)

    rag_stats = {
        "n_with_retrieval":       len(best_sims),
        "mean_best_similarity":   round(sum(best_sims) / len(best_sims), 4) if best_sims else None,
        "min_best_similarity":    round(min(best_sims), 4) if best_sims else None,
        "max_best_similarity":    round(max(best_sims), 4) if best_sims else None,
        "mean_top5_similarity":   round(sum(mean_sims) / len(mean_sims), 4) if mean_sims else None,
        "mean_retrieval_ms":      round(sum(ret_ms) / len(ret_ms), 1) if ret_ms else None,
        "n_below_threshold_0_65": low_sim_count,
    }

    return {
        "n_records":           len(valid),
        "n_errors":            len(results) - len(valid),
        "model_only":          model_metrics,
        "reconciled":          reconciled_metrics,
        "llm_agreement_rate":  round(agree_rate, 4) if agree_rate is not None else None,
        "escalation_rate":     round(escalation_rate, 4) if escalation_rate is not None else None,
        "n_escalations":       len(escalations),
        "escalation_precision": round(escalation_precision, 4) if escalation_precision is not None else None,
        "rag":                 rag_stats,
    }


def print_summary(metrics: dict):
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE EVALUATION  —  Full Test Set")
    print("=" * 70)
    print(f"\nRecords evaluated : {metrics['n_records']}  ({metrics['n_errors']} errors skipped)")
    print(f"LLM agreement rate: {metrics['llm_agreement_rate']:.1%}")
    print(f"Escalation rate   : {metrics['escalation_rate']:.1%}  ({metrics['n_escalations']} cases escalated by LLM)")
    print(f"Escalation precision: {metrics['escalation_precision']:.1%}  (of escalated cases, fraction where escalation was correct)")

    for label, m_dict in [("MODEL ONLY", metrics["model_only"]), ("RECONCILED (model + LLM)", metrics["reconciled"])]:
        print(f"\n── {label} ──")
        print(f"  Macro-F1    : {m_dict['macro_f1']:.4f}")
        print(f"  Weighted-F1 : {m_dict['weighted_f1']:.4f}")
        print()
        print(m_dict["report_text"])
        print("  Confusion matrix (rows=true, cols=pred):")
        print(f"  {'':30s}  {'L1-Crit':>8}  {'L2-Emrg':>8}  {'L3-Urg':>8}")
        for i, row_name in enumerate(["L1-Critical", "L2-Emergent", "L3-Urgent/LessUrgent"]):
            cm_row = m_dict["confusion_matrix"][i]
            print(f"  {row_name:30s}  {cm_row[0]:>8}  {cm_row[1]:>8}  {cm_row[2]:>8}")

    delta_macro    = metrics["reconciled"]["macro_f1"]    - metrics["model_only"]["macro_f1"]
    delta_critical = metrics["reconciled"]["per_class"]["L1-Critical"]["f1"] - metrics["model_only"]["per_class"]["L1-Critical"]["f1"]
    print(f"\n── Delta (reconciled − model) ──")
    print(f"  Macro-F1   : {delta_macro:+.4f}")
    print(f"  Critical-F1: {delta_critical:+.4f}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E2E pipeline batch evaluation")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent workers (default: 10)")
    parser.add_argument("--output",  type=str, default="results/e2e_eval.json",
                        help="Output JSON path (default: results/e2e_eval.json)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Evaluate only first N records (for smoke testing)")
    args = parser.parse_args()

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load test set
    test_df = load_test_set()

    # Run batch
    record_results = run_batch(test_df, workers=args.workers, limit=args.limit)

    # Compute metrics
    metrics = compute_metrics(record_results)
    print_summary(metrics)

    # Save full results
    output = {
        "metadata": {
            "sagemaker_endpoint": SAGEMAKER_ENDPOINT,
            "n_test_records":     len(test_df),
            "workers":            args.workers,
            "limit":              args.limit,
        },
        "metrics":  metrics,
        "records":  record_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    # Also save human-readable summary
    summary_path = output_path.with_suffix(".txt")
    with open(summary_path, "w") as f:
        f.write(f"Model-only classification report:\n{metrics['model_only']['report_text']}\n\n")
        f.write(f"Reconciled classification report:\n{metrics['reconciled']['report_text']}\n")
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
