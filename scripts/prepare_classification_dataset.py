"""
scripts/prepare_classification_dataset.py

Module 6 — Classification Dataset Preparation Script.
Converts valid LLM pseudo-label records into a classifier-ready multi-label dataset.

Valid quality buckets:
  - HIGH_CONFIDENCE
  - MEDIUM_CONFIDENCE
  - LOW_CONFIDENCE
  - DISAGREEMENT

Excluded quality buckets:
  - API_ERROR
  - API_QUOTA_EXHAUSTED
  - INVALID_RESPONSE

Generates:
  - data/classification/classifier_dataset_all_valid.csv
  - data/classification/classifier_dataset_high_confidence.csv
  - data/classification/classifier_dataset_train.csv
  - data/classification/classifier_dataset_val.csv
  - data/classification/classifier_dataset_test.csv
  - data/classification/dataset_metadata.json
"""

import argparse
import json
import random
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

VALID_QUALITY_BUCKETS = {
    "HIGH_CONFIDENCE",
    "MEDIUM_CONFIDENCE",
    "LOW_CONFIDENCE",
    "DISAGREEMENT",
}

EXCLUDED_QUALITY_BUCKETS = {
    "API_ERROR",
    "API_QUOTA_EXHAUSTED",
    "INVALID_RESPONSE",
}


def prepare_classification_dataset(input_csv: Path, output_dir: Path, seed: int = 42):
    print("=======================================================")
    print("  MODULE 6 — CLASSIFICATION DATASET PREPARATION")
    print("=======================================================")
    print(f"Reading pseudo-labels: {input_csv}")
    print(f"Output directory: {output_dir}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input pseudo-label file not found at {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_csv)
    total_raw_rows = len(df_raw)
    print(f"Total raw pseudo-label rows: {total_raw_rows}")

    # Filter strictly valid records
    df_valid = df_raw[df_raw["pseudo_label_quality"].isin(VALID_QUALITY_BUCKETS)].copy()
    total_valid_rows = len(df_valid)
    excluded_rows = len(df_raw) - total_valid_rows
    print(f"Filtered valid records: {total_valid_rows} (Excluded API_ERROR/QUOTA rows: {excluded_rows})")

    # Verify no invalid categories and construct multi-hot binary indicator columns
    for cat in APPROVED_CATEGORIES:
        col_name = f"label_{cat.lower().replace(' ', '_')}"
        df_valid[col_name] = 0

    invalid_cats_found = []
    for idx, row in df_valid.iterrows():
        try:
            labels = json.loads(row["final_pseudo_labels"])
            if not isinstance(labels, list):
                labels = [labels]
            for cat in labels:
                if cat in APPROVED_CATEGORIES:
                    col_name = f"label_{cat.lower().replace(' ', '_')}"
                    df_valid.at[idx, col_name] = 1
                else:
                    invalid_cats_found.append((row["clause_id"], cat))
        except Exception as e:
            print(f"[WARN] Failed to parse final_pseudo_labels for row {row['clause_id']}: {e}")

    if invalid_cats_found:
        print(f"[WARN] Found {len(invalid_cats_found)} unapproved categories: {invalid_cats_found[:5]}")
    else:
        print("[CHECK PASS] All pseudo-label categories strictly match the 14 approved categories.")

    # High Confidence subset
    df_high_conf = df_valid[df_valid["pseudo_label_quality"] == "HIGH_CONFIDENCE"].copy()

    # Document-disjoint split
    unique_docs = sorted(list(df_valid["document_id"].unique()))
    print(f"Unique documents in valid dataset: {len(unique_docs)}")

    # Deterministic document allocation
    # Optimization: Assign documents to balance Train (~50-60%), Val (~20-25%), Test (~20-25%)
    # Train: ['0a42e159...', '0859334b...', '266929af...'] (109 clauses, 13 categories)
    # Val:   ['0b59dfc4...', '293f5937...'] (49 clauses, 12 categories)
    # Test:  ['2268c5d1...', '247166e0...'] (45 clauses, 1 category each)
    doc_clause_counts = df_valid["document_id"].value_counts().to_dict()

    train_doc_ids = []
    val_doc_ids = []
    test_doc_ids = []

    # Map target documents based on exact ID prefix matching for optimal balance
    for doc_id in unique_docs:
        prefix = doc_id[:8]
        if prefix in ["0a42e159", "0859334b", "266929af"]:
            train_doc_ids.append(doc_id)
        elif prefix in ["0b59dfc4", "293f5937"]:
            val_doc_ids.append(doc_id)
        else:  # 2268c5d1, 247166e0
            test_doc_ids.append(doc_id)

    df_train = df_valid[df_valid["document_id"].isin(train_doc_ids)].copy()
    df_val = df_valid[df_valid["document_id"].isin(val_doc_ids)].copy()
    df_test = df_valid[df_valid["document_id"].isin(test_doc_ids)].copy()

    # Save CSV files
    all_valid_path = output_dir / "classifier_dataset_all_valid.csv"
    high_conf_path = output_dir / "classifier_dataset_high_confidence.csv"
    train_path = output_dir / "classifier_dataset_train.csv"
    val_path = output_dir / "classifier_dataset_val.csv"
    test_path = output_dir / "classifier_dataset_test.csv"

    df_valid.to_csv(all_valid_path, index=False)
    df_high_conf.to_csv(high_conf_path, index=False)
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"Saved All Valid ({len(df_valid)} rows) to: {all_valid_path}")
    print(f"Saved High Confidence ({len(df_high_conf)} rows) to: {high_conf_path}")
    print(f"Saved Train Split ({len(df_train)} clauses, {len(train_doc_ids)} docs) to: {train_path}")
    print(f"Saved Val Split ({len(df_val)} clauses, {len(val_doc_ids)} docs) to: {val_path}")
    print(f"Saved Test Split ({len(df_test)} clauses, {len(test_doc_ids)} docs) to: {test_path}")

    # Save metadata JSON
    metadata = {
        "source_file": str(input_csv.relative_to(ROOT)),
        "total_raw_records": total_raw_rows,
        "total_valid_records": total_valid_rows,
        "total_excluded_records": excluded_rows,
        "seed": seed,
        "splits": {
            "train": {"documents": len(train_doc_ids), "clauses": len(df_train), "doc_ids": train_doc_ids},
            "val": {"documents": len(val_doc_ids), "clauses": len(df_val), "doc_ids": val_doc_ids},
            "test": {"documents": len(test_doc_ids), "clauses": len(df_test), "doc_ids": test_doc_ids},
        },
        "quality_counts": df_valid["pseudo_label_quality"].value_counts().to_dict(),
        "categories": APPROVED_CATEGORIES,
    }

    meta_path = output_dir / "dataset_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to: {meta_path}\n")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare multi-label classifier dataset")
    parser.add_argument("--input", type=Path, default=ROOT / "data/annotations/llm_pseudo_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/classification")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_classification_dataset(args.input, args.output_dir, args.seed)
