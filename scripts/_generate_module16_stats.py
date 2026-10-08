"""
scripts/_generate_module16_stats.py

Generates Task 5 statistics CSV/MD and Task 6 benchmark manifest JSON for Module 16.
"""

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

def generate_module16_stats_and_manifest():
    bm_dir = ROOT / "data/classification/module16_benchmark"
    df_train = pd.read_csv(bm_dir / "train.csv")
    df_val = pd.read_csv(bm_dir / "validation.csv")
    df_test = pd.read_csv(bm_dir / "test.csv")
    df_valid = pd.read_csv(bm_dir / "preview_dataset_all_valid.csv") if (bm_dir / "preview_dataset_all_valid.csv").exists() else pd.concat([df_train, df_val, df_test])

    total_clauses = len(df_valid)
    total_docs = df_valid["document_id"].nunique()

    # Category statistics rows
    cat_stats = []
    for cat in APPROVED_CATEGORIES:
        col = f"label_{cat.lower().replace(' ', '_')}"
        tot_clauses = int(df_valid[col].sum())
        doc_cnt = int(df_valid[df_valid[col] == 1]["document_id"].nunique())
        tr_cnt = int(df_train[col].sum())
        va_cnt = int(df_val[col].sum())
        te_cnt = int(df_test[col].sum())

        cat_stats.append({
            "category": cat,
            "total_valid_clauses": tot_clauses,
            "category_doc_count": doc_cnt,
            "train_support": tr_cnt,
            "val_support": va_cnt,
            "test_support": te_cnt,
            "train_present": "Yes" if tr_cnt > 0 else "No",
            "val_present": "Yes" if va_cnt > 0 else "No",
            "test_present": "Yes" if te_cnt > 0 else "No"
        })

    df_cat_stats = pd.DataFrame(cat_stats)
    stats_csv = ROOT / "reports/module16_final_dataset_statistics.csv"
    df_cat_stats.to_csv(stats_csv, index=False)
    print(f"Saved dataset statistics CSV to: {stats_csv}")

    # Generate Markdown Report
    md_lines = [
        "# Module 16: Final Dataset Statistics & Coverage Report\n",
        "## Executive Summary\n",
        f"- **Total Valid Clauses**: {total_clauses}",
        f"- **Total Represented Documents**: {total_docs}",
        f"- **Train Split**: {len(df_train)} clauses ({df_train['document_id'].nunique()} docs) | Categories Present: 13/14",
        f"- **Validation Split**: {len(df_val)} clauses ({df_val['document_id'].nunique()} docs) | Categories Present: 13/14",
        f"- **Test Split**: {len(df_test)} clauses ({df_test['document_id'].nunique()} docs) | Categories Present: 14/14\n",
        "## 1. Category Support & Split Breakdown Table\n",
        "| Category Name | Total Valid Clauses | Document Count | Train Support | Val Support | Test Support | Train Present? | Val Present? | Test Present? |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in cat_stats:
        md_lines.append(
            f"| **{r['category']}** | {r['total_valid_clauses']} | {r['category_doc_count']} | "
            f"{r['train_support']} | {r['val_support']} | {r['test_support']} | "
            f"{r['train_present']} | {r['val_present']} | {r['test_present']} |"
        )

    stats_md = ROOT / "reports/module16_final_dataset_statistics.md"
    stats_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved dataset statistics MD to:  {stats_md}")

    # Generate Task 6 Benchmark Release Manifest JSON
    manifest = {
        "manifest_version": "1.0.0",
        "dataset_name": "NDA Multi-Label Clause Classification Benchmark",
        "dataset_source": "Kleister-NDA PDF Corpus + Gemini 3-Model Ensemble",
        "benchmark_status": "BLOCKED",
        "status_reason": "Liability for Damages occurs in only 2 documents (target >= 3), preventing complete 3-way split representation.",
        "random_seed": 42,
        "split_method": "Deterministic Document-Disjoint Partitioning",
        "total_valid_clauses": total_clauses,
        "total_represented_documents": total_docs,
        "category_vocabulary": APPROVED_CATEGORIES,
        "splits": {
            "train": {
                "clause_count": len(df_train),
                "document_count": int(df_train["document_id"].nunique()),
                "category_count": 13,
                "missing_categories": ["Employees"],
                "document_ids": sorted(list(df_train["document_id"].unique()))
            },
            "validation": {
                "clause_count": len(df_val),
                "document_count": int(df_val["document_id"].nunique()),
                "category_count": 13,
                "missing_categories": ["Liability for Damages"],
                "document_ids": sorted(list(df_val["document_id"].unique()))
            },
            "test": {
                "clause_count": len(df_test),
                "document_count": int(df_test["document_id"].nunique()),
                "category_count": 14,
                "missing_categories": [],
                "document_ids": sorted(list(df_test["document_id"].unique()))
            }
        },
        "known_limitations": [
            "Liability for Damages is present in only 2 documents, forcing zero support in Validation split.",
            "Total represented document count (11) is below target (>= 14).",
            "Labels are LLM pseudo-labels and require human verification on the test split prior to published model claims."
        ]
    }

    manifest_json = ROOT / "reports/module16_benchmark_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved benchmark release manifest to: {manifest_json}")

if __name__ == "__main__":
    generate_module16_stats_and_manifest()
