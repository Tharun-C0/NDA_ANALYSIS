"""
scripts/analyze_class_imbalance.py

Module 6 — Class Imbalance & Pseudo-Label Quality Analysis Script.

Performs:
  1. Per-category clause frequency & percentage across valid pseudo-labels
  2. Document frequency for each category
  3. Positive/Negative ratio for each category
  4. Pseudo-label quality breakdown (HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, LOW_CONFIDENCE, DISAGREEMENT)
  5. Category distribution grouped by quality bucket

Produces:
  data/classification/class_imbalance_analysis.csv
  reports/class_imbalance_report.md
  reports/pseudolabel_quality_analysis.md
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


def analyze_class_imbalance(data_dir: Path, reports_dir: Path):
    print("=======================================================")
    print("  MODULE 6 — CLASS IMBALANCE & QUALITY ANALYSIS")
    print("=======================================================")
    print(f"Data directory: {data_dir}")

    all_file = data_dir / "classifier_dataset_all_valid.csv"
    if not all_file.exists():
        raise FileNotFoundError(f"Classification dataset not found at {all_file}")

    df = pd.read_csv(all_file)
    total_clauses = len(df)
    total_docs = df["document_id"].nunique()

    print(f"Analyzing {total_clauses} clauses across {total_docs} documents.")

    # Per-category metrics
    cat_stats = []
    total_assigned_labels = 0

    for cat in APPROVED_CATEGORIES:
        col = f"label_{cat.lower().replace(' ', '_')}"
        if col in df.columns:
            pos_count = int(df[col].sum())
        else:
            # Fallback parse from JSON
            pos_count = 0
            for l in df["final_pseudo_labels"]:
                try:
                    if cat in json.loads(l):
                        pos_count += 1
                except Exception:
                    pass

        total_assigned_labels += pos_count
        neg_count = total_clauses - pos_count
        pct = (pos_count / total_clauses * 100) if total_clauses > 0 else 0.0

        # Document frequency (number of distinct documents containing this category)
        if col in df.columns:
            doc_freq = df[df[col] == 1]["document_id"].nunique()
        else:
            doc_freq = 0

        pos_neg_ratio = f"1:{neg_count / pos_count:.1f}" if pos_count > 0 else "N/A"

        cat_stats.append({
            "category": cat,
            "clause_count": pos_count,
            "percentage": round(pct, 2),
            "doc_frequency": doc_freq,
            "pos_neg_ratio": pos_neg_ratio,
            "negative_count": neg_count
        })

    df_imbalance = pd.DataFrame(cat_stats)

    # Save imbalance CSV
    csv_imbalance_path = data_dir / "class_imbalance_analysis.csv"
    df_imbalance.to_csv(csv_imbalance_path, index=False)
    print(f"Saved Class Imbalance CSV to: {csv_imbalance_path}")

    # Generate Class Imbalance Markdown Report
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_imbalance_path = reports_dir / "class_imbalance_report.md"

    # Identify majority and minority categories
    sorted_stats = sorted(cat_stats, key=lambda x: x["clause_count"], reverse=True)
    majority_cats = [s["category"] for s in sorted_stats if s["clause_count"] >= 15]
    minority_cats = [s["category"] for s in sorted_stats if s["clause_count"] < 15]

    md_imbalance_content = [
        "# Module 6 — Category Distribution & Class Imbalance Analysis\n",
        "> [!NOTE]",
        "> **Dataset Note**: All label statistics are calculated over **203 usable LLM pseudo-labeled clauses** derived from 7 NDA documents.\n",
        f"- **Total Usable Clauses**: {total_clauses}",
        f"- **Total Unique Documents**: {total_docs}",
        f"- **Total Category Assignments**: {total_assigned_labels} (Avg: {total_assigned_labels / total_clauses:.2f} labels/clause)\n",
        "## 1. 14-Category Frequency & Distribution Table\n",
        "| Category Name | Positive Clauses | Percentage | Document Frequency | Pos:Neg Ratio | Imbalance Tier |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for s in sorted_stats:
        tier = "Majority (>=15)" if s["clause_count"] >= 15 else ("Minority (<15)" if s["clause_count"] > 0 else "Zero Frequency")
        md_imbalance_content.append(
            f"| **{s['category']}** | {s['clause_count']} | {s['percentage']}% | {s['doc_frequency']} / {total_docs} | {s['pos_neg_ratio']} | {tier} |"
        )

    md_imbalance_content.extend([
        "\n## 2. Minority Category Identification",
        f"- **Majority Categories (>= 15 clauses)**: {', '.join(majority_cats)}",
        f"- **Minority Categories (< 15 clauses)**: {', '.join(minority_cats)}",
        "\n> [!IMPORTANT]",
        "> **Methodology Rule**: Minority categories are preserved in the multi-label taxonomy without removal or premature oversampling during dataset preparation. Class-weighted loss functions (e.g. Focal Loss, Weighted BCE) will address imbalance during training.",
    ])

    md_imbalance_path.write_text("\n".join(md_imbalance_content), encoding="utf-8")
    print(f"Saved Class Imbalance Report to: {md_imbalance_path}")

    # Generate Pseudo-Label Quality Analysis Markdown Report
    quality_counts = df["pseudo_label_quality"].value_counts().to_dict()
    high_conf = quality_counts.get("HIGH_CONFIDENCE", 0)
    med_conf = quality_counts.get("MEDIUM_CONFIDENCE", 0)
    low_conf = quality_counts.get("LOW_CONFIDENCE", 0)
    disagree = quality_counts.get("DISAGREEMENT", 0)

    avg_conf = round(float(df["average_confidence"].mean()), 4)
    avg_agree = round(float(df["agreement_score"].mean()), 4)

    md_quality_path = reports_dir / "pseudolabel_quality_analysis.md"
    md_quality_content = [
        "# Module 6 — LLM Pseudo-Label Quality Analysis\n",
        "> [!IMPORTANT]",
        "> **TERMINOLOGY DEFINITION**: Annotations in this dataset are **LLM pseudo-labels** generated by a 3-model ensemble (Gemini 3.6 Flash, Gemini 3.5 Flash, Gemini 3.1 Flash Lite). They are automated pre-annotations, **NOT human ground truth**.\n",
        "## 1. Usable Quality Breakdown\n",
        f"- **Total Usable Pseudo-Labels**: {total_clauses}",
        f"- **HIGH_CONFIDENCE**: {high_conf} ({high_conf / total_clauses * 100:.2f}%)",
        f"- **MEDIUM_CONFIDENCE**: {med_conf} ({med_conf / total_clauses * 100:.2f}%)",
        f"- **LOW_CONFIDENCE**: {low_conf} ({low_conf / total_clauses * 100:.2f}%)",
        f"- **DISAGREEMENT**: {disagree} ({disagree / total_clauses * 100:.2f}%)",
        f"- **Mean Ensemble Confidence**: {avg_conf}",
        f"- **Mean Agreement Score**: {avg_agree}\n",
        "## 2. Category Distribution by Quality Bucket\n",
        "| Category Name | HIGH_CONF | MED_CONF | LOW_CONF | DISAGREEMENT | Total Usable |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for cat in APPROVED_CATEGORIES:
        col = f"label_{cat.lower().replace(' ', '_')}"
        if col in df.columns:
            h = df[(df[col] == 1) & (df["pseudo_label_quality"] == "HIGH_CONFIDENCE")].shape[0]
            m = df[(df[col] == 1) & (df["pseudo_label_quality"] == "MEDIUM_CONFIDENCE")].shape[0]
            l = df[(df[col] == 1) & (df["pseudo_label_quality"] == "LOW_CONFIDENCE")].shape[0]
            d = df[(df[col] == 1) & (df["pseudo_label_quality"] == "DISAGREEMENT")].shape[0]
            tot = h + m + l + d
            md_quality_content.append(f"| **{cat}** | {h} | {m} | {l} | {d} | {tot} |")

    md_quality_path.write_text("\n".join(md_quality_content), encoding="utf-8")
    print(f"Saved Quality Analysis Report to: {md_quality_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze class imbalance and pseudo-label quality")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/classification")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    analyze_class_imbalance(args.data_dir, args.reports_dir)
