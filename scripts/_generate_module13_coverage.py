"""
scripts/_generate_module13_coverage.py

Generates Task 2 category coverage CSV for Module 13.
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

def generate_coverage_csv():
    df_labels = pd.read_csv(ROOT / "data/annotations/llm_pseudo_labels.csv")
    df_valid = df_labels[df_labels["pseudo_label_quality"].isin(["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "DISAGREEMENT"])].copy()
    total_docs = df_valid["document_id"].nunique()

    cat_rows = []
    for cat in APPROVED_CATEGORIES:
        cat_clauses = 0
        cat_docs = set()
        for _, r in df_valid.iterrows():
            try:
                labels = json.loads(r["final_pseudo_labels"])
                if cat in labels:
                    cat_clauses += 1
                    cat_docs.add(r["document_id"][:8])
            except Exception:
                pass
        num_docs = len(cat_docs)
        pct_docs = round((num_docs / total_docs) * 100, 2)
        cat_rows.append({
            "category": cat,
            "valid_clause_count": cat_clauses,
            "document_count": num_docs,
            "pct_represented_documents": pct_docs,
            "ge_2_docs": "Yes" if num_docs >= 2 else "No",
            "ge_3_docs": "Yes" if num_docs >= 3 else "No"
        })

    df_cov = pd.DataFrame(cat_rows).sort_values(by=["document_count", "valid_clause_count"], ascending=[True, True])
    output_path = ROOT / "reports/module13_category_coverage.csv"
    df_cov.to_csv(output_path, index=False)
    print(f"Saved category coverage CSV to: {output_path}")

if __name__ == "__main__":
    generate_coverage_csv()
