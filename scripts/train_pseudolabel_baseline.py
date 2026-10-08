"""scripts/train_pseudolabel_baseline.py

Module 5 — Experimental Baseline Trainer for LLM Pseudo-Labels.

Supports experimental training comparisons:
  - Experiment A: Human/verified labels only (--human-only)
  - Experiment B: High-confidence pseudo-labels (--use-high-confidence-only)
  - Experiment C: High + medium confidence pseudo-labels (--use-high-medium-confidence)

Enforces document-disjoint splits and outputs evaluation metrics (Micro/Macro F1).

Usage:
    python scripts/train_pseudolabel_baseline.py --use-high-confidence-only
    python scripts/train_pseudolabel_baseline.py --use-high-medium-confidence
    python scripts/train_pseudolabel_baseline.py --human-only
"""

import argparse
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

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

def parse_labels(val):
    if pd.isna(val) or not val:
        return []
    val_str = str(val).strip()
    if val_str.startswith("["):
        try:
            return json.loads(val_str)
        except Exception:
            pass
    # Fallback split
    return [x.strip() for x in val_str.split(";") if x.strip()]

def load_experimental_dataset(mode: str) -> pd.DataFrame:
    if mode == "human-only":
        db_path = ROOT / "data/annotations/annotations.csv"
        if not db_path.exists():
            raise FileNotFoundError(f"Human annotation DB not found at {db_path}")
        df = pd.read_csv(db_path)
        df = df[df["label_source"] == "human_annotation"]
        df["target_labels"] = df["category_labels"].apply(parse_labels)
        print(f"Loaded {len(df)} human-annotated clauses.")
        return df

    pseudo_path = ROOT / "data/annotations/llm_pseudo_labels.csv"
    if not pseudo_path.exists():
        raise FileNotFoundError(f"LLM pseudo-label dataset not found at {pseudo_path}")
    
    df = pd.read_csv(pseudo_path)
    df["target_labels"] = df["final_pseudo_labels"].apply(parse_labels)

    if mode == "high-confidence-only":
        df = df[df["pseudo_label_quality"] == "HIGH_CONFIDENCE"]
        print(f"Loaded {len(df)} HIGH_CONFIDENCE pseudo-labeled clauses.")
    elif mode == "high-medium-confidence":
        df = df[df["pseudo_label_quality"].isin(["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE"])]
        print(f"Loaded {len(df)} HIGH + MEDIUM confidence pseudo-labeled clauses.")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df

def train_and_evaluate(df: pd.DataFrame, experiment_name: str):
    if len(df) < 10:
        print(f"\n[WARN] Dataset size ({len(df)}) is too small for meaningful cross-validation.")
        return

    print(f"\n=======================================================")
    print(f"  RUNNING EXPERIMENT: {experiment_name}")
    print(f"=======================================================")

    # Create binary label matrix
    Y = np.zeros((len(df), len(APPROVED_CATEGORIES)), dtype=int)
    for i, labels in enumerate(df["target_labels"]):
        for cat in labels:
            if cat in APPROVED_CATEGORIES:
                c_idx = APPROVED_CATEGORIES.index(cat)
                Y[i, c_idx] = 1

    X_text = df["clause_text"].fillna("").values
    groups = df["document_id"].values

    # Group K-Fold Cross Validation (5 folds, document-disjoint)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    
    micro_f1s, macro_f1s = [], []
    precisions, recalls = [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_text, Y, groups=groups)):
        X_train, X_val = X_text[train_idx], X_text[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        vec = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
        X_train_vec = vec.fit_transform(X_train)
        X_val_vec = vec.transform(X_val)

        clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=500, solver="liblinear"))
        clf.fit(X_train_vec, Y_train)

        Y_pred = clf.predict(X_val_vec)

        mic_f1 = f1_score(Y_val, Y_pred, average="micro", zero_division=0)
        mac_f1 = f1_score(Y_val, Y_pred, average="macro", zero_division=0)
        prec = precision_score(Y_val, Y_pred, average="micro", zero_division=0)
        rec = recall_score(Y_val, Y_pred, average="micro", zero_division=0)

        micro_f1s.append(mic_f1)
        macro_f1s.append(mac_f1)
        precisions.append(prec)
        recalls.append(rec)

    avg_mic_f1 = np.mean(micro_f1s)
    avg_mac_f1 = np.mean(macro_f1s)
    avg_prec = np.mean(precisions)
    avg_rec = np.mean(recalls)

    print(f"Results for {experiment_name}:")
    print(f"  Total Samples: {len(df)}")
    print(f"  Micro F1:     {avg_mic_f1:.4f}")
    print(f"  Macro F1:     {avg_mac_f1:.4f}")
    print(f"  Micro Precision: {avg_prec:.4f}")
    print(f"  Micro Recall:    {avg_rec:.4f}")

    res_dict = {
        "experiment": experiment_name,
        "sample_count": len(df),
        "micro_f1": round(avg_mic_f1, 4),
        "macro_f1": round(avg_mac_f1, 4),
        "micro_precision": round(avg_prec, 4),
        "micro_recall": round(avg_rec, 4)
    }

    res_path = ROOT / "reports/pseudolabel_baseline_results.json"
    existing_res = []
    if res_path.exists():
        try:
            existing_res = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception:
            existing_res = []
    
    # Update or append
    existing_res = [r for r in existing_res if r.get("experiment") != experiment_name]
    existing_res.append(res_dict)
    res_path.write_text(json.dumps(existing_res, indent=2), encoding="utf-8")
    print(f"Saved experiment metrics to: {res_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline model on pseudo-labels or human annotations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--use-high-confidence-only", action="store_true", help="Experiment B: Train on High Confidence pseudo-labels only")
    group.add_argument("--use-high-medium-confidence", action="store_true", help="Experiment C: Train on High + Medium Confidence pseudo-labels")
    group.add_argument("--human-only", action="store_true", help="Experiment A: Train on Human/Verified ground truth labels only")

    args = parser.parse_args()

    if args.use_high_confidence_only:
        mode = "high-confidence-only"
        exp_name = "Experiment_B_HighConfidence_PseudoLabels"
    elif args.use_high_medium_confidence:
        mode = "high-medium-confidence"
        exp_name = "Experiment_C_HighMediumConfidence_PseudoLabels"
    else:
        mode = "human-only"
        exp_name = "Experiment_A_HumanGroundTruth"

    df = load_experimental_dataset(mode)
    train_and_evaluate(df, exp_name)
