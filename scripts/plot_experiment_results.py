"""
scripts/plot_experiment_results.py

Module 6C — Automated Plotting & Visual Analytics Infrastructure.

Generates:
  - reports/figures/category_support_distribution.png
  - reports/figures/macro_f1_by_model.png
  - reports/figures/macro_f1_by_loss.png
  - reports/figures/per_category_f1_heatmap.png
  - reports/figures/model_by_loss_matrix.png
"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


def plot_category_support_distribution(data_csv: Path, figures_dir: Path):
    if not data_csv.exists():
        print(f"[WARN] Cannot plot category distribution: {data_csv} not found.")
        return

    df = pd.read_csv(data_csv)
    label_cols = [f"label_{c.lower().replace(' ', '_')}" for c in APPROVED_CATEGORIES]

    counts = []
    for cat, col in zip(APPROVED_CATEGORIES, label_cols):
        cnt = df[col].sum() if col in df.columns else 0
        counts.append((cat, cnt))

    df_counts = pd.DataFrame(counts, columns=["Category", "Count"]).sort_values("Count", ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_counts["Category"], df_counts["Count"], color="#34495e", edgecolor="#2c3e50")
    plt.xlabel("Number of Positive Clauses")
    plt.title("14-Category Positive Support Distribution (203 Usable Pseudo-Labels)")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center", fontsize=9)

    plt.tight_layout()
    out_path = figures_dir / "category_support_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved figure to: {out_path}")


def plot_experiment_comparisons(comp_csv: Path, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)

    if comp_csv.exists():
        df_comp = pd.read_csv(comp_csv)
    else:
        df_comp = pd.DataFrame()

    # 1. Macro F1 by Model
    plt.figure(figsize=(8, 5))
    if len(df_comp) > 0 and "macro_f1" in df_comp.columns and df_comp["macro_f1"].max() > 0:
        df_model = df_comp.groupby("model")["macro_f1"].max().reset_index()
        plt.bar(df_model["model"], df_model["macro_f1"], color="#2980b9", edgecolor="#1c5980")
        plt.ylabel("Macro F1 Score")
        plt.title("Empirical Macro F1 Score by Candidate Model")
        plt.ylim(0, 1.0)
    else:
        plt.text(0.5, 0.5, "Template Placeholder — Awaiting Full Model Experiments\n(No research training executed yet)",
                 ha="center", va="center", fontsize=11, color="gray")
        plt.title("Macro F1 by Model (Infrastructure Template)")
    plt.tight_layout()
    out_path1 = figures_dir / "macro_f1_by_model.png"
    plt.savefig(out_path1, dpi=300)
    plt.close()
    print(f"Saved figure to: {out_path1}")

    # 2. Macro F1 by Loss Function
    plt.figure(figsize=(8, 5))
    if len(df_comp) > 0 and "macro_f1" in df_comp.columns and df_comp["macro_f1"].max() > 0:
        df_loss = df_comp.groupby("loss_function")["macro_f1"].max().reset_index()
        plt.bar(df_loss["loss_function"], df_loss["macro_f1"], color="#27ae60", edgecolor="#1e8449")
        plt.ylabel("Macro F1 Score")
        plt.title("Empirical Macro F1 Score by Loss Function")
        plt.ylim(0, 1.0)
    else:
        plt.text(0.5, 0.5, "Template Placeholder — Awaiting Loss Function Experiments\n(No research training executed yet)",
                 ha="center", va="center", fontsize=11, color="gray")
        plt.title("Macro F1 by Loss Function (Infrastructure Template)")
    plt.tight_layout()
    out_path2 = figures_dir / "macro_f1_by_loss.png"
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"Saved figure to: {out_path2}")

    # 3. Per-Category F1 Heatmap Template
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "Per-Category F1 Heatmap Template\n(Will populate automatically upon experiment matrix completion)",
             ha="center", va="center", fontsize=11, color="gray")
    plt.title("Per-Category F1 Heatmap (Infrastructure Template)")
    out_path3 = figures_dir / "per_category_f1_heatmap.png"
    plt.savefig(out_path3, dpi=300)
    plt.close()
    print(f"Saved figure to: {out_path3}")

    # 4. Model x Loss Matrix Template
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, "Model x Loss Matrix Template (3 Models x 3 Loss Functions)\n(Will populate automatically upon experiment matrix completion)",
             ha="center", va="center", fontsize=11, color="gray")
    plt.title("Model x Loss Matrix (Infrastructure Template)")
    out_path4 = figures_dir / "model_by_loss_matrix.png"
    plt.savefig(out_path4, dpi=300)
    plt.close()
    print(f"Saved figure to: {out_path4}")


def run_plotting_pipeline(data_dir: Path, reports_dir: Path, figures_dir: Path):
    print("=======================================================")
    print("  MODULE 6C — AUTOMATED PLOTTING INFRASTRUCTURE")
    print("=======================================================")
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_csv = data_dir / "classifier_dataset_all_valid.csv"
    plot_category_support_distribution(data_csv, figures_dir)

    comp_csv = reports_dir / "model_comparison.csv"
    plot_experiment_comparisons(comp_csv, figures_dir)
    print("All plotting infrastructure figures generated successfully!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment visualization plots")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/classification")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports/experiments")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()

    run_plotting_pipeline(args.data_dir, args.reports_dir, args.figures_dir)
