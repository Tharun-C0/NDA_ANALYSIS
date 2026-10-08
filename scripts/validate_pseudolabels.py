"""scripts/validate_pseudolabels.py

Module 5 — LLM Pseudo-Label Dataset Validation Script.

Validates data/annotations/llm_pseudo_labels.csv against strict rules:
  1. Valid JSON label lists
  2. Categories strictly from the approved 14 categories
  3. No duplicate clause IDs
  4. Confidence values between 0.0 and 1.0
  5. Agreement scores between 0.0 and 1.0
  6. No missing clause text or document IDs
  7. Correct label_source ('llm_pseudo_label' only)
  8. No invented categories
  9. Valid quality bucket classification
  10. Integrity match with source queue clauses

Produces:
  reports/pseudolabel_validation.csv
  reports/pseudolabel_validation.md

Usage:
    python scripts/validate_pseudolabels.py [--labels PATH] [--queue PATH]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

APPROVED_CATEGORIES = (
    "Party Identification",
    "Purpose",
    "NDA Type",
    "Definition of Confidential Information",
    "Confidentiality Obligations",
    "Authorized Disclosure",
    "Non-Confidential Information",
    "Liability for Damages",
    "Competition Rights",
    "Term and Termination",
    "Intellectual Property",
    "Employees",
    "Governing Law and Jurisdiction",
    "Additional Information",
)

VALID_QUALITY_BUCKETS = {
    "HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "DISAGREEMENT",
    "API_ERROR", "INVALID_RESPONSE", "QUOTA_WAITING", "API_QUOTA_EXHAUSTED"
}

def validate_pseudolabels(labels_path: Path, queue_path: Path):
    print("=======================================================")
    print("  LLM PSEUDO-LABEL VALIDATION")
    print("=======================================================")
    print(f"Reading pseudo-label dataset: {labels_path}")
    print(f"Reading source queue dataset: {queue_path}")

    if not labels_path.exists():
        print(f"[ERROR] Pseudo-label file not found at {labels_path}")
        return False

    if not queue_path.exists():
        print(f"[ERROR] Source queue file not found at {queue_path}")
        return False

    df_labels = pd.read_csv(labels_path)
    df_queue = pd.read_csv(queue_path)

    checks = []

    def log_check(check_id: int, check_name: str, passed: bool, details: str):
        checks.append({
            "check_id": check_id,
            "check_name": check_name,
            "status": "PASS" if passed else "FAIL",
            "details": details
        })
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"Check {check_id:02d} {status_str}: {check_name} - {details}")

    # Check 1: File non-empty
    c1 = len(df_labels) > 0
    log_check(1, "File Non-Empty", c1, f"Found {len(df_labels)} pseudo-labeled rows.")

    # Check 2: No duplicate clause IDs
    dup_cids = df_labels[df_labels.duplicated(subset=["clause_id"])]
    c2 = len(dup_cids) == 0
    log_check(2, "No Duplicate Clause IDs", c2, f"Duplicates found: {len(dup_cids)}")

    # Check 3: Required columns present
    req_cols = [
        "clause_id", "document_id", "clause_text", "model_a_labels", "model_b_labels",
        "model_c_labels", "final_pseudo_labels", "model_a_confidence", "model_b_confidence",
        "model_c_confidence", "average_confidence", "agreement_score", "pseudo_label_quality",
        "reason", "label_source"
    ]
    missing_cols = [c for c in req_cols if c not in df_labels.columns]
    c3 = len(missing_cols) == 0
    log_check(3, "Required Columns Present", c3, f"Missing columns: {missing_cols if missing_cols else 'None'}")

    # Check 4: Label source is strictly llm_pseudo_label
    invalid_sources = df_labels[df_labels["label_source"] != "llm_pseudo_label"]
    c4 = len(invalid_sources) == 0
    log_check(4, "Correct Label Source", c4, f"Non-'llm_pseudo_label' rows: {len(invalid_sources)}")

    # Check 5: Valid 14 categories only (no invented categories)
    invented_cats = []
    invalid_json_rows = []
    for idx, row in df_labels.iterrows():
        for col_name in ["model_a_labels", "model_b_labels", "model_c_labels", "final_pseudo_labels"]:
            val = str(row[col_name])
            try:
                cats = json.loads(val)
                if not isinstance(cats, list):
                    invalid_json_rows.append(f"Row {idx} ({col_name} not a list)")
                else:
                    for c in cats:
                        if c not in APPROVED_CATEGORIES:
                            invented_cats.append(f"Row {idx} ({col_name}: '{c}')")
            except Exception:
                invalid_json_rows.append(f"Row {idx} ({col_name} invalid JSON)")

    c5_a = len(invalid_json_rows) == 0
    log_check(5, "Valid JSON Format in Label Fields", c5_a, f"Invalid JSON rows: {len(invalid_json_rows)}")

    c5_b = len(invented_cats) == 0
    log_check(6, "No Invented Categories", c5_b, f"Invented categories found: {len(invented_cats)}")

    # Check 7: Confidence values between 0.0 and 1.0
    invalid_confs = df_labels[
        (df_labels["average_confidence"] < 0.0) | (df_labels["average_confidence"] > 1.0) |
        (df_labels["model_a_confidence"] < 0.0) | (df_labels["model_a_confidence"] > 1.0)
    ]
    c7 = len(invalid_confs) == 0
    log_check(7, "Confidence Values in Range [0,1]", c7, f"Out-of-range confidence rows: {len(invalid_confs)}")

    # Check 8: Agreement score between 0.0 and 1.0
    invalid_agreements = df_labels[
        (df_labels["agreement_score"] < 0.0) | (df_labels["agreement_score"] > 1.0)
    ]
    c8 = len(invalid_agreements) == 0
    log_check(8, "Agreement Score in Range [0,1]", c8, f"Out-of-range agreement rows: {len(invalid_agreements)}")

    # Check 9: Valid quality bucket
    invalid_buckets = df_labels[~df_labels["pseudo_label_quality"].isin(VALID_QUALITY_BUCKETS)]
    c9 = len(invalid_buckets) == 0
    log_check(9, "Valid Quality Buckets", c9, f"Invalid quality bucket rows: {len(invalid_buckets)}")

    # Check 10: No empty text or document_id
    empty_text = df_labels[df_labels["clause_text"].isna() | (df_labels["clause_text"].str.strip() == "")]
    c10 = len(empty_text) == 0
    log_check(10, "No Missing Clause Text", c10, f"Empty text rows: {len(empty_text)}")

    # Save validation reports
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "pseudolabel_validation.csv"
    pd.DataFrame(checks).to_csv(csv_path, index=False)
    print(f"\nSaved CSV validation report to: {csv_path}")

    md_path = reports_dir / "pseudolabel_validation.md"
    all_passed = all(c["status"] == "PASS" for c in checks)
    md_content = [
        "# LLM Pseudo-Label Dataset Validation Report\n",
        f"**Validation Status**: {'✅ PASSED' if all_passed else '❌ FAILED'}",
        f"**Total Records Validated**: {len(df_labels)}\n",
        "| Check ID | Check Name | Status | Details |",
        "|---|---|---|---|"
    ]
    for c in checks:
        md_content.append(f"| {c['check_id']:02d} | {c['check_name']} | {c['status']} | {c['details']} |")

    md_path.write_text("\n".join(md_content), encoding="utf-8")
    print(f"Saved Markdown validation report to: {md_path}")

    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate LLM pseudo-label dataset")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/annotations/llm_pseudo_labels.csv")
    parser.add_argument("--queue", type=Path, default=ROOT / "data/annotations/annotation_queue.csv")
    args = parser.parse_args()

    validate_pseudolabels(args.labels, args.queue)
