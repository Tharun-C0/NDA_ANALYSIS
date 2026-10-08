"""
scripts/compare_experiments.py

Module 6C — Automated Experiment Comparison & Objective Reporting Script.

Reads:
  - reports/experiments/*.json
  - data/classification/experiment_results/experiment_history.csv

Produces:
  - reports/experiments/model_comparison.csv
  - reports/experiments/model_comparison.md

Methodological Constraint:
  - Objective reporting ONLY. Does NOT use subjective ranking terms (e.g. 'best', 'winner', 'superior').
  - Discloses LLM pseudo-label limitation explicitly.
"""

import argparse
import glob
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DISCLAIMER_TEXT = (
    "> [!IMPORTANT]\n"
    "> **RESEARCH METHODOLOGY DISCLOSURE**: All metrics reported in this table are calculated on **LLM-generated pseudo-labels** "
    "(produced by a 3-model ensemble: Gemini 3.6 Flash, Gemini 3.5 Flash, Gemini 3.1 Flash Lite). "
    "They represent automated experimental pre-annotations and **have NOT yet been fully human-verified**."
)


def generate_experiment_comparison(reports_dir: Path, output_dir: Path):
    print("=======================================================")
    print("  MODULE 6C — AUTOMATED EXPERIMENT COMPARISON")
    print("=======================================================")
    print(f"Reading logs from: {reports_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = glob.glob(str(reports_dir / "*.json"))
    history_csv = ROOT / "data/classification/experiment_results/experiment_history.csv"

    records = []

    if json_files:
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                exp_id = data.get("experiment_id", "UNKNOWN")
                model = data.get("model", "UNKNOWN")
                loss = data.get("loss_function", "UNKNOWN")
                label_mode = data.get("label_mode", "all_valid")
                status = data.get("status", "DEBUG_SMOKE_TEST")
                ts = data.get("timestamp", "")
                metrics = data.get("metrics", {})

                records.append({
                    "experiment_id": exp_id,
                    "timestamp": ts,
                    "model": model,
                    "loss_function": loss,
                    "label_mode": label_mode,
                    "status": status,
                    "macro_f1": metrics.get("macro_f1", 0.0),
                    "weighted_f1": metrics.get("weighted_f1", 0.0),
                    "micro_f1": metrics.get("micro_f1", 0.0),
                    "macro_precision": metrics.get("macro_precision", 0.0),
                    "macro_recall": metrics.get("macro_recall", 0.0),
                    "hamming_loss": metrics.get("hamming_loss", 0.0),
                    "macro_mcc": metrics.get("macro_mcc", 0.0),
                    "log_file": str(Path(jf).name)
                })
            except Exception as e:
                print(f"[WARN] Failed to parse {jf}: {e}")

    df_comp = pd.DataFrame(records)

    if len(df_comp) == 0:
        # Create empty template structure if no logs exist
        df_comp = pd.DataFrame(columns=[
            "experiment_id", "timestamp", "model", "loss_function", "label_mode", "status",
            "macro_f1", "weighted_f1", "micro_f1", "macro_precision", "macro_recall",
            "hamming_loss", "macro_mcc", "log_file"
        ])

    csv_path = output_dir / "model_comparison.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"Saved Comparison CSV to: {csv_path}")

    # Generate objective markdown report
    md_path = output_dir / "model_comparison.md"
    md_content = [
        "# Empirical Classifier Experiment Comparison Report\n",
        DISCLAIMER_TEXT,
        "\n> [!NOTE]",
        "> **PRIMARY EVALUATION METRIC**: **Macro F1** is designated as the primary benchmark metric for multi-label clause classification due to category imbalance across the 14 NDA categories.\n",
        "## 1. Measured Empirical Performance Table\n",
        "| Experiment ID | Model Identifier | Loss Function | Label Mode | Status | Macro F1 | Weighted F1 | Micro F1 | Macro Precision | Macro Recall | Hamming Loss | Macro MCC |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    if len(df_comp) > 0:
        for _, r in df_comp.iterrows():
            md_content.append(
                f"| `{r['experiment_id']}` | `{r['model']}` | `{r['loss_function']}` | `{r['label_mode']}` | "
                f"`{r['status']}` | `{r['macro_f1']:.4f}` | `{r['weighted_f1']:.4f}` | `{r['micro_f1']:.4f}` | "
                f"`{r['macro_precision']:.4f}` | `{r['macro_recall']:.4f}` | `{r['hamming_loss']:.4f}` | `{r['macro_mcc']:.4f}` |"
            )
    else:
        md_content.append("| *No experiments executed yet* | — | — | — | — | — | — | — | — | — | — | — |")

    md_content.extend([
        "\n## 2. Objective Metric Interpretation Guidelines",
        "- **Macro F1**: Unweighted arithmetic mean of per-category F1 scores across 14 categories. Evaluates performance equally on minority and majority categories.",
        "- **Weighted F1**: Category F1 scores weighted by true positive support.",
        "- **Micro F1**: Global instance-level F1 score over all binary decisions.",
        "- **Hamming Loss**: Fraction of misclassified binary labels (lower indicates higher classification accuracy).",
        "- **Macro MCC**: Multi-label Matthews Correlation Coefficient reflecting binary correlation across all categories.",
        "\n> [!CAUTION]",
        "> **EVALUATION SPLIT NOTICE**: All runs executed on temporary 7-document debugging splits are strictly labeled as `DEBUG_SMOKE_TEST`. Definitive scientific comparison will occur after additional pseudo-labeling is completed across all 717 V3 clauses."
    ])

    md_path.write_text("\n".join(md_content), encoding="utf-8")
    print(f"Saved Comparison Markdown Report to: {md_path}\n")

    return csv_path, md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate empirical experiment comparison report")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports/experiments")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments")
    args = parser.parse_args()

    generate_experiment_comparison(args.reports_dir, args.output_dir)
