"""
scripts/_generate_module12_matrix.py
Generates Task 2 reports for Module 12.
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

def generate_module12_matrices():
    labels_csv = ROOT / "data/annotations/llm_pseudo_labels.csv"
    df = pd.read_csv(labels_csv)
    dfv = df[df["pseudo_label_quality"].isin(["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "DISAGREEMENT"])].copy()

    doc_matrix = []
    doc_groups = dfv.groupby("document_id")

    for doc_id, group in doc_groups:
        row = {"document_id": doc_id[:8], "full_document_id": doc_id, "valid_clauses": len(group)}
        cat_counts = {c: 0 for c in APPROVED_CATEGORIES}
        for _, r in group.iterrows():
            try:
                labels = json.loads(r["final_pseudo_labels"])
                for cat in labels:
                    if cat in cat_counts:
                        cat_counts[cat] += 1
            except Exception:
                pass
        row.update(cat_counts)
        row["categories_present"] = sum(1 for v in cat_counts.values() if v > 0)
        doc_matrix.append(row)

    df_matrix = pd.DataFrame(doc_matrix).sort_values(by="valid_clauses", ascending=False)
    cols = ["document_id", "valid_clauses", "categories_present"] + APPROVED_CATEGORIES
    df_matrix_csv = df_matrix[cols]
    df_matrix_csv.to_csv(ROOT / "reports/module12_document_category_matrix.csv", index=False)

    # Create Markdown matrix
    md_lines = [
        "# Module 12: Document × Category Matrix\n",
        "## 1. Document × Category Matrix Table\n",
        "| Document ID | Valid Clauses | Cats Present | " + " | ".join(APPROVED_CATEGORIES) + " |",
        "| :--- | :---: | :---: | " + " | ".join([":---:"] * len(APPROVED_CATEGORIES)) + " |"
    ]
    for _, r in df_matrix.iterrows():
        counts_str = " | ".join([str(r[c]) for c in APPROVED_CATEGORIES])
        md_lines.append(f"| **{r['document_id']}** | {r['valid_clauses']} | **{r['categories_present']}** | {counts_str} |")

    (ROOT / "reports/module12_document_category_matrix.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Category Document Coverage CSV
    cat_cov = []
    total_docs = dfv["document_id"].nunique()
    for cat in APPROVED_CATEGORIES:
        cat_clauses = 0
        cat_docs = set()
        for _, r in dfv.iterrows():
            try:
                labels = json.loads(r["final_pseudo_labels"])
                if cat in labels:
                    cat_clauses += 1
                    cat_docs.add(r["document_id"][:8])
            except Exception:
                pass
        num_docs = len(cat_docs)
        cat_cov.append({
            "category": cat,
            "valid_clauses": cat_clauses,
            "document_count": num_docs,
            "pct_represented_docs": round((num_docs / total_docs) * 100, 2),
            "ge_2_docs": "Yes" if num_docs >= 2 else "No",
            "ge_3_docs": "Yes" if num_docs >= 3 else "No"
        })

    df_cov = pd.DataFrame(cat_cov).sort_values(by=["document_count", "valid_clauses"], ascending=[True, True])
    df_cov.to_csv(ROOT / "reports/module12_category_document_coverage.csv", index=False)
    print("Successfully generated module12_document_category_matrix.csv, module12_document_category_matrix.md, and module12_category_document_coverage.csv")

if __name__ == "__main__":
    generate_module12_matrices()
