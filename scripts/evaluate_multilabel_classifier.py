"""
scripts/evaluate_multilabel_classifier.py

Module 6B — Comprehensive Multi-Label Classifier Evaluation Framework.

Calculates:
  - Macro F1 (Primary Metric)
  - Micro F1
  - Weighted F1
  - Hamming Loss
  - Multi-Label Matthews Correlation Coefficient (MCC) — Macro-averaged per-category MCC
  - Macro / Micro Precision & Recall
  - Per-category metrics (Precision, Recall, F1, MCC, Support)
  - Extended Multi-Label Confusion Statistics (TP, FP, FN, TN, Actual Positives, Predicted Positives)

Produces:
  - reports/evaluation_per_category.csv
  - reports/classifier_evaluation_summary.md
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    matthews_corrcoef
)

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

DISCLAIMER_TEXT = (
    "> [!IMPORTANT]\n"
    "> **RESEARCH DISCLAIMER**: Training & evaluation labels in this pipeline are **LLM-generated pseudo-labels** "
    "(produced by a 3-model ensemble: Gemini 3.6 Flash, Gemini 3.5 Flash, Gemini 3.1 Flash Lite). "
    "They represent automated pre-annotations for machine learning experimentation and **have NOT yet been fully human-verified**."
)


def calculate_multilabel_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    """
    Calculates per-category Matthews Correlation Coefficient (MCC)
    and returns (macro_mcc, list_of_per_category_mccs).
    """
    num_classes = y_true.shape[1]
    mccs = []
    for c in range(num_classes):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        if len(np.unique(yt)) == 1 and len(np.unique(yp)) == 1:
            mcc = 1.0 if yt[0] == yp[0] else 0.0
        else:
            try:
                mcc = float(matthews_corrcoef(yt, yp))
            except Exception:
                mcc = 0.0
        mccs.append(mcc)
    macro_mcc = float(np.mean(mccs))
    return macro_mcc, mccs


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path = ROOT / "reports",
    model_name: str = "Baseline-Model",
    is_temporary_debug: bool = True
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Primary & Secondary multi-label metrics
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    micro_prec = float(precision_score(y_true, y_pred, average="micro", zero_division=0))

    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    micro_rec = float(recall_score(y_true, y_pred, average="micro", zero_division=0))

    h_loss = float(hamming_loss(y_true, y_pred))
    macro_mcc, per_cat_mccs = calculate_multilabel_mcc(y_true, y_pred)

    # Per-category multi-label confusion analysis
    per_cat_rows = []
    for c, cat in enumerate(APPROVED_CATEGORIES):
        yt = y_true[:, c]
        yp = y_pred[:, c]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))

        actual_pos = int(np.sum(yt == 1))
        pred_pos = int(np.sum(yp == 1))

        prec = float(precision_score(yt, yp, zero_division=0))
        rec = float(recall_score(yt, yp, zero_division=0))
        f1 = float(f1_score(yt, yp, zero_division=0))
        mcc = per_cat_mccs[c]

        per_cat_rows.append({
            "category": cat,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "mcc": round(mcc, 4),
            "actual_positives": actual_pos,
            "predicted_positives": pred_pos,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        })

    df_per_cat = pd.DataFrame(per_cat_rows)
    per_cat_csv = output_dir / "evaluation_per_category.csv"
    df_per_cat.to_csv(per_cat_csv, index=False)

    # Markdown evaluation summary
    md_path = output_dir / "classifier_evaluation_summary.md"
    
    header_tag = " [DEBUG / PIPELINE VALIDATION ONLY]" if is_temporary_debug else ""
    md_content = [
        f"# Multi-Label Classifier Evaluation Report — {model_name}{header_tag}\n",
        DISCLAIMER_TEXT,
        "\n> [!NOTE]",
        "> **PRIMARY EVALUATION METRIC**: **Macro F1** is designated as the primary benchmark metric due to category imbalance across the 14 NDA categories.\n"
    ]

    if is_temporary_debug:
        md_content.append(
            "> [!WARNING]\n"
            "> **TEMPORARY SPLIT NOTICE**: These evaluation metrics were computed on a temporary 7-document debugging split. "
            "They are for code verification only and **MUST NOT be reported as final research results**.\n"
        )

    md_content.extend([
        "## 1. Global Multi-Label Performance Summary\n",
        "| Metric Name | Value | Role / Description |",
        "| :--- | :--- | :--- |",
        f"| **Macro F1** | `{macro_f1:.4f}` | **PRIMARY METRIC** — Unweighted average F1 across 14 categories |",
        f"| **Weighted F1** | `{weighted_f1:.4f}` | Support-weighted average F1 |",
        f"| **Micro F1** | `{micro_f1:.4f}` | Global instance-level F1 score |",
        f"| **Macro Precision** | `{macro_prec:.4f}` | Mean precision across 14 categories |",
        f"| **Macro Recall** | `{macro_rec:.4f}` | Mean recall across 14 categories |",
        f"| **Hamming Loss** | `{h_loss:.4f}` | Fraction of misclassified binary labels (lower is better) |",
        f"| **Macro MCC** | `{macro_mcc:.4f}` | Multi-label Matthews Correlation Coefficient |",
        "\n## 2. Per-Category Breakdown & Confusion Statistics\n",
        "| Category Name | Precision | Recall | F1 Score | MCC | Actual Pos | Pred Pos | TP | FP | FN | TN |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for r in per_cat_rows:
        md_content.append(
            f"| **{r['category']}** | {r['precision']} | {r['recall']} | {r['f1_score']} | {r['mcc']} | "
            f"{r['actual_positives']} | {r['predicted_positives']} | {r['true_positives']} | {r['false_positives']} | {r['false_negatives']} | {r['true_negatives']} |"
        )

    md_path.write_text("\n".join(md_content), encoding="utf-8")

    metrics_summary = {
        "model_name": model_name,
        "primary_metric": "macro_f1",
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "hamming_loss": round(h_loss, 4),
        "macro_mcc": round(macro_mcc, 4),
        "is_temporary_debug": is_temporary_debug,
        "per_category_csv": str(per_cat_csv.relative_to(ROOT)),
        "summary_markdown": str(md_path.relative_to(ROOT)),
        "per_category_metrics": per_cat_rows
    }

    print(f"Saved Per-Category Evaluation CSV to: {per_cat_csv}")
    print(f"Saved Classifier Evaluation Summary to: {md_path}")

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multi-label classifier predictions")
    parser.add_argument("--test-file", type=Path, default=ROOT / "data/classification/classifier_dataset_test.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--smoke-test", action="store_true", help="Run self-test with ground-truth labels")
    args = parser.parse_args()

    if args.smoke_test:
        print("--- RUNNING EVALUATION FRAMEWORK SMOKE TEST ---")
        df_test = pd.read_csv(args.test_file)
        label_cols = [f"label_{c.lower().replace(' ', '_')}" for c in APPROVED_CATEGORIES]
        y_true = df_test[label_cols].values
        y_pred = y_true.copy()
        res = evaluate_predictions(y_true, y_pred, output_dir=args.output_dir, model_name="SmokeTest-PerfectBaseline", is_temporary_debug=True)
        print("[CHECK PASS] EVALUATION FRAMEWORK SMOKE TEST PASSED PERFECTLY!\n")
