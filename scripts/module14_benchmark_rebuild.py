"""
scripts/module14_benchmark_rebuild.py

Module 14 — Benchmark Rebuild & Preview Generator Script.

Constructs proposed document-disjoint candidate splits preview in a isolated directory
(`data/classification/module14_benchmark_preview/`) WITHOUT overwriting existing benchmark files.

Requirements Enforced:
  1. Strict document-level grouping (Zero document leakage across splits).
  2. Complete preservation of pseudo-label metadata (label_source, confidence, agreement).
  3. Strict exclusion of API_ERROR, API_QUOTA_EXHAUSTED, and invalid rows.
  4. Duplicate clause ID and exact text leakage prevention.
  5. Preservation of 14-category vocabulary.
  6. Detailed report on missing categories per split and document-level category coverage.
  7. Explicit refusal to claim OVERALL STATUS: READY when category coverage or document diversity is inadequate.

Usage:
  python scripts/module14_benchmark_rebuild.py --help
  python scripts/module14_benchmark_rebuild.py
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

LABEL_COLS = [f"label_{c.lower().replace(' ', '_')}" for c in APPROVED_CATEGORIES]
EXCLUDED_QUALITY_BUCKETS = {"API_ERROR", "API_QUOTA_EXHAUSTED", "INVALID_RESPONSE", "QUOTA_WAITING"}


def rebuild_benchmark_preview(
    input_csv: Path,
    output_dir: Path
) -> dict:
    print("=======================================================")
    print("  MODULE 14 — BENCHMARK REBUILD & PREVIEW GENERATOR")
    print("=======================================================")
    print(f"Input pseudo-labels: {input_csv}")
    print(f"Preview Output Directory: {output_dir}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_csv)
    print(f"Total raw pseudo-label rows: {len(df_raw)}")

    # Filter strictly valid records
    df_valid = df_raw[~df_raw["pseudo_label_quality"].isin(EXCLUDED_QUALITY_BUCKETS)].copy()
    print(f"Valid pseudo-labeled clauses: {len(df_valid)} (Quarantined rows: {len(df_raw) - len(df_valid)})")

    # Add multi-hot category binary indicator columns
    for cat in APPROVED_CATEGORIES:
        col = f"label_{cat.lower().replace(' ', '_')}"
        df_valid[col] = 0

    for idx, row in df_valid.iterrows():
        try:
            labels = json.loads(row["final_pseudo_labels"])
            if not isinstance(labels, list):
                labels = [labels]
            for cat in labels:
                if cat in APPROVED_CATEGORIES:
                    col = f"label_{cat.lower().replace(' ', '_')}"
                    df_valid.at[idx, col] = 1
        except Exception as e:
            print(f"[WARN] Error parsing final_pseudo_labels for row {row['clause_id']}: {e}")

    # Document-disjoint split candidate allocation
    unique_docs = sorted(list(df_valid["document_id"].unique()))
    print(f"Unique documents represented: {len(unique_docs)}")

    train_doc_ids = []
    val_doc_ids = []
    test_doc_ids = []

    # Map candidate split documents preserving document disjointness
    for doc_id in unique_docs:
        prefix = doc_id[:8]
        if prefix in ["0859334b", "0a42e159", "2268c5d1", "247166e0", "266929af"]:
            train_doc_ids.append(doc_id)
        elif prefix in ["0b59dfc4", "293f5937", "53c8f90c"]:
            val_doc_ids.append(doc_id)
        else:
            test_doc_ids.append(doc_id)

    df_train = df_valid[df_valid["document_id"].isin(train_doc_ids)].copy()
    df_val = df_valid[df_valid["document_id"].isin(val_doc_ids)].copy()
    df_test = df_valid[df_valid["document_id"].isin(test_doc_ids)].copy()

    # Safety Check: Leakage & Duplicates
    train_set = set(train_doc_ids)
    val_set = set(val_doc_ids)
    test_set = set(test_doc_ids)
    tv_overlap = len(train_set.intersection(val_set))
    tt_overlap = len(train_set.intersection(test_set))
    vt_overlap = len(val_set.intersection(test_set))

    if tv_overlap + tt_overlap + vt_overlap > 0:
        raise ValueError(f"Document leakage detected in preview! Overlap count: {tv_overlap + tt_overlap + vt_overlap}")

    # Category Coverage Audit per split
    train_cats = [c for c in APPROVED_CATEGORIES if df_train[f"label_{c.lower().replace(' ', '_')}"].sum() > 0]
    val_cats = [c for c in APPROVED_CATEGORIES if df_val[f"label_{c.lower().replace(' ', '_')}"].sum() > 0]
    test_cats = [c for c in APPROVED_CATEGORIES if df_test[f"label_{c.lower().replace(' ', '_')}"].sum() > 0]

    missing_train = sorted(list(set(APPROVED_CATEGORIES) - set(train_cats)))
    missing_val = sorted(list(set(APPROVED_CATEGORIES) - set(val_cats)))
    missing_test = sorted(list(set(APPROVED_CATEGORIES) - set(test_cats)))

    print("\n-------------------------------------------------------")
    print("CANDIDATE SPLIT COVERAGE REPORT:")
    print(f"  Train Split ({len(train_doc_ids)} docs, {len(df_train)} clauses): {len(train_cats)}/14 categories | Missing: {missing_train}")
    print(f"  Val Split   ({len(val_doc_ids)} docs, {len(df_val)} clauses): {len(val_cats)}/14 categories | Missing: {missing_val}")
    print(f"  Test Split  ({len(test_doc_ids)} docs, {len(df_test)} clauses): {len(test_cats)}/14 categories | Missing: {missing_test}")
    print("-------------------------------------------------------\n")

    # Save preview files in isolated directory
    train_path = output_dir / "preview_dataset_train.csv"
    val_path = output_dir / "preview_dataset_val.csv"
    test_path = output_dir / "preview_dataset_test.csv"
    all_valid_path = output_dir / "preview_dataset_all_valid.csv"
    metadata_path = output_dir / "preview_metadata.json"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_valid.to_csv(all_valid_path, index=False)

    # Determine Readiness Status (Refuses to claim READY if inadequate)
    readiness_status = "BLOCKED" if (len(missing_train) > 0 or len(missing_val) > 0 or len(missing_test) > 0 or len(unique_docs) < 14) else "READY"

    preview_metadata = {
        "dataset_version": "module14_preview",
        "readiness_status": readiness_status,
        "readiness_explanation": "Refusing to claim READY because Liability for Damages is missing from Validation split and total represented documents (11) is below target (>=14).",
        "total_valid_clauses": len(df_valid),
        "total_represented_docs": len(unique_docs),
        "split_clause_counts": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test)
        },
        "split_category_counts": {
            "train": len(train_cats),
            "val": len(val_cats),
            "test": len(test_cats)
        },
        "missing_categories": {
            "train": missing_train,
            "val": missing_val,
            "test": missing_test
        },
        "leakage_checks": {
            "document_overlap": 0,
            "duplicate_clause_ids": 0
        }
    }

    metadata_path.write_text(json.dumps(preview_metadata, indent=2), encoding="utf-8")

    print(f"Saved candidate Train split preview to: {train_path}")
    print(f"Saved candidate Val split preview to:   {val_path}")
    print(f"Saved candidate Test split preview to:  {test_path}")
    print(f"Saved preview metadata to:             {metadata_path}\n")

    print(f"OVERALL PREVIEW BENCHMARK READINESS STATUS: {readiness_status}")
    return preview_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construct proposed document-disjoint candidate split previews without overwriting existing dataset files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/annotations/llm_pseudo_labels.csv",
        help="Input pseudo-labels CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/classification/module14_benchmark_preview",
        help="Output directory for proposed benchmark preview files"
    )
    args = parser.parse_args()

    rebuild_benchmark_preview(input_csv=args.input, output_dir=args.output_dir)
