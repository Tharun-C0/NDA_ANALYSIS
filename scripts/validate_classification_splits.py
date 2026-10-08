"""
scripts/validate_classification_splits.py

Module 6 — Classification Split Leakage & Integrity Validation.

Validates:
  1. No duplicate clause_ids across the classification dataset
  2. Strict document-disjoint splits (0 document overlap between Train, Val, Test)
  3. No empty clause text or missing document IDs
  4. No unknown or unapproved categories
  5. Valid JSON formatting in final_pseudo_labels
  6. No API_ERROR or API_QUOTA_EXHAUSTED rows present
  7. Multi-hot binary label completeness (14 label columns)
  8. Non-empty splits (Train, Val, Test all > 0 clauses)

Produces:
  reports/classification_split_validation.csv
  reports/classification_split_validation.md
"""

import argparse
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

APPROVED_CATEGORIES = [
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
]

EXCLUDED_QUALITY_BUCKETS = {
    "API_ERROR",
    "API_QUOTA_EXHAUSTED",
    "INVALID_RESPONSE",
}


def validate_classification_splits(data_dir: Path, reports_dir: Path):
    print("=======================================================")
    print("  MODULE 6 — LEAKAGE & INTEGRITY VALIDATION")
    print("=======================================================")
    print(f"Data directory: {data_dir}")

    train_file = data_dir / "classifier_dataset_train.csv"
    val_file = data_dir / "classifier_dataset_val.csv"
    test_file = data_dir / "classifier_dataset_test.csv"
    all_file = data_dir / "classifier_dataset_all_valid.csv"

    for f in [train_file, val_file, test_file, all_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required classification dataset file not found: {f}")

    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)
    df_all = pd.read_csv(all_file)

    checks = []

    def log_check(check_id: int, name: str, passed: bool, details: str):
        checks.append({
            "check_id": check_id,
            "check_name": name,
            "status": "PASS" if passed else "FAIL",
            "details": details
        })
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"Check {check_id:02d} {status_str}: {name} - {details}")

    # Check 1: Non-empty splits
    c1 = len(df_train) > 0 and len(df_val) > 0 and len(df_test) > 0
    log_check(1, "Non-Empty Splits", c1, f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Check 2: No duplicate clause IDs in dataset
    dup_cids = df_all[df_all.duplicated(subset=["clause_id"])]
    c2 = len(dup_cids) == 0
    log_check(2, "No Duplicate Clause IDs", c2, f"Duplicates found in all_valid: {len(dup_cids)}")

    # Check 3: Document-disjoint Train / Val / Test (NO document overlap)
    train_docs = set(df_train["document_id"].unique())
    val_docs = set(df_val["document_id"].unique())
    test_docs = set(df_test["document_id"].unique())

    tv_overlap = train_docs.intersection(val_docs)
    tt_overlap = train_docs.intersection(test_docs)
    vt_overlap = val_docs.intersection(test_docs)
    total_overlap = len(tv_overlap) + len(tt_overlap) + len(vt_overlap)

    c3 = total_overlap == 0
    log_check(3, "Document-Disjoint Splits (Zero Leakage)", c3,
              f"Overlap count: {total_overlap} (Train-Val: {len(tv_overlap)}, Train-Test: {len(tt_overlap)}, Val-Test: {len(vt_overlap)})")

    # Check 4: No empty clause text or missing document IDs
    empty_text = df_all[df_all["clause_text"].isna() | (df_all["clause_text"].str.strip() == "")]
    empty_doc = df_all[df_all["document_id"].isna() | (df_all["document_id"].str.strip() == "")]
    c4 = len(empty_text) == 0 and len(empty_doc) == 0
    log_check(4, "No Missing Text or Document IDs", c4, f"Empty text: {len(empty_text)}, Empty doc: {len(empty_doc)}")

    # Check 5: No API_ERROR or API_QUOTA_EXHAUSTED rows
    errored_rows = df_all[df_all["pseudo_label_quality"].isin(EXCLUDED_QUALITY_BUCKETS)]
    c5 = len(errored_rows) == 0
    log_check(5, "Zero API_ERROR / Quota Rows", c5, f"Errored/Quota rows found: {len(errored_rows)}")

    # Check 6: Valid JSON and strictly 14 approved categories
    invalid_cats = []
    for idx, row in df_all.iterrows():
        try:
            cats = json.loads(row["final_pseudo_labels"])
            for c in cats:
                if c not in APPROVED_CATEGORIES:
                    invalid_cats.append(f"Row {idx} ({c})")
        except Exception:
            invalid_cats.append(f"Row {idx} (JSON parse error)")

    c6 = len(invalid_cats) == 0
    log_check(6, "Valid 14 Category Vocabulary", c6, f"Invalid categories or JSON errors: {len(invalid_cats)}")

    # Check 7: Multi-hot binary columns completeness
    expected_label_cols = [f"label_{cat.lower().replace(' ', '_')}" for cat in APPROVED_CATEGORIES]
    missing_cols = [col for col in expected_label_cols if col not in df_all.columns]
    c7 = len(missing_cols) == 0
    log_check(7, "14 Multi-Hot Binary Columns Present", c7, f"Missing columns: {missing_cols if missing_cols else 'None'}")

    # Check 8: Total clause count preservation
    sum_splits = len(df_train) + len(df_val) + len(df_test)
    c8 = sum_splits == len(df_all)
    log_check(8, "Total Clause Count Integrity", c8, f"Train+Val+Test ({sum_splits}) == All Valid ({len(df_all)})")

    # Save validation outputs
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "classification_split_validation.csv"
    pd.DataFrame(checks).to_csv(csv_path, index=False)
    print(f"\nSaved CSV report to: {csv_path}")

    all_passed = all(c["status"] == "PASS" for c in checks)
    md_path = reports_dir / "classification_split_validation.md"
    md_content = [
        "# Module 6 — Classification Split Leakage & Integrity Validation Report\n",
        f"**Validation Status**: {'✅ PASSED (0 Leakage Found)' if all_passed else '❌ FAILED'}",
        f"**Total Valid Clauses**: {len(df_all)}\n",
        "| Check ID | Check Name | Status | Details |",
        "|---|---|---|---|"
    ]
    for c in checks:
        md_content.append(f"| {c['check_id']:02d} | {c['check_name']} | {c['status']} | {c['details']} |")

    md_content.extend([
        "\n## Document Split Breakdown",
        f"- **Train**: {len(df_train)} clauses across {len(train_docs)} documents ({[d[:8] for d in train_docs]})",
        f"- **Validation**: {len(df_val)} clauses across {len(val_docs)} documents ({[d[:8] for d in val_docs]})",
        f"- **Test**: {len(df_test)} clauses across {len(test_docs)} documents ({[d[:8] for d in test_docs]})",
    ])

    md_path.write_text("\n".join(md_content), encoding="utf-8")
    print(f"Saved Markdown report to: {md_path}\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate classification split leakage and integrity")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/classification")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    validate_classification_splits(args.data_dir, args.reports_dir)
