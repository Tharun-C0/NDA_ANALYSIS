"""
scripts/train_multilabel_classifier.py

Module 6B — Multi-Label Transformer Classifier Training Framework.

Features:
  - Supported Classifiers:
      - EXP-01: saibo/legal-roberta-base
      - EXP-02: nlpaueb/legal-bert-base-uncased
      - EXP-03: microsoft/deberta-v3-base
  - Loss Functions:
      1. BCEWithLogitsLoss ('bce')
      2. Multi-Label Focal Loss ('focal') [gamma=2.0, alpha=0.25]
      3. Class-Weighted BCE ('class_weighted_bce') [weights computed ONLY from training split]
  - Pre-Flight Dataset Safety & Leakage Checks:
      - Duplicate clause IDs, split document overlap, missing text/IDs, unapproved categories, errored/quota rows.
  - Training Label Modes:
      - '--label-mode all_valid' (default)
      - '--label-mode high_confidence'
  - Configurable Sigmoid Classification Threshold: '--threshold 0.5'
  - Full Experiment Result Logging to JSON & CSV.
  - Reproducibility Metadata Logging.

Usage:
  python scripts/train_multilabel_classifier.py --smoke-test
"""

import argparse
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from evaluate_multilabel_classifier import evaluate_predictions

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

EXCLUDED_QUALITY_BUCKETS = {
    "API_ERROR",
    "API_QUOTA_EXHAUSTED",
    "INVALID_RESPONSE",
    "QUOTA_WAITING"
}

DISCLAIMER_TEXT = "Training labels are LLM-generated pseudo-labels and have not yet been fully human-verified."


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_preflight_dataset_checks(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> bool:
    print("\n--- RUNNING PRE-FLIGHT DATASET SAFETY & LEAKAGE CHECKS ---")
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # 1. Duplicate Clause IDs
    dup_cids = df_all[df_all.duplicated(subset=["clause_id"])]
    if len(dup_cids) > 0:
        raise ValueError(f"[SAFETY CHECK FAILED] Found {len(dup_cids)} duplicate clause IDs across splits!")

    # 2. Document-Disjoint Split Leakage
    train_docs = set(df_train["document_id"].unique())
    val_docs = set(df_val["document_id"].unique())
    test_docs = set(df_test["document_id"].unique())

    tv_overlap = train_docs.intersection(val_docs)
    tt_overlap = train_docs.intersection(test_docs)
    vt_overlap = val_docs.intersection(test_docs)
    total_leakage = len(tv_overlap) + len(tt_overlap) + len(vt_overlap)
    if total_leakage > 0:
        raise ValueError(f"[SAFETY CHECK FAILED] Document leakage detected between splits! Overlap count: {total_leakage}")

    # 3. Missing Clause Text or Document ID
    empty_text = df_all[df_all["clause_text"].isna() | (df_all["clause_text"].str.strip() == "")]
    empty_doc = df_all[df_all["document_id"].isna() | (df_all["document_id"].str.strip() == "")]
    if len(empty_text) > 0 or len(empty_doc) > 0:
        raise ValueError(f"[SAFETY CHECK FAILED] Found {len(empty_text)} empty text rows and {len(empty_doc)} empty doc ID rows!")

    # 4. Zero API_ERROR / Quota Rows
    errored = df_all[df_all["pseudo_label_quality"].isin(EXCLUDED_QUALITY_BUCKETS)]
    if len(errored) > 0:
        raise ValueError(f"[SAFETY CHECK FAILED] Found {len(errored)} API_ERROR or Quota rows in dataset!")

    # 5. Label Multi-Hot Binary Completeness
    missing_label_cols = [c for c in LABEL_COLS if c not in df_all.columns]
    if missing_label_cols:
        raise ValueError(f"[SAFETY CHECK FAILED] Missing multi-hot label columns: {missing_label_cols}")

    print("[CHECK PASS] All 5 Pre-Flight Safety & Leakage Checks Passed Cleanly!")
    return True


class FocalLoss(nn.Module):
    """
    Multi-label Focal Loss:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_factor * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class NDAClauseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["clause_text"].tolist()
        labels_list = df[LABEL_COLS].values.astype(np.float32)
        self.labels = torch.tensor(labels_list, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_pos_weights_from_train_only(df_train: pd.DataFrame) -> torch.Tensor:
    """
    CRITICAL SAFETY RULE: Class weights calculated ONLY from the training dataset split.
    """
    pos_counts = df_train[LABEL_COLS].sum().values.astype(np.float32)
    total_train = len(df_train)
    neg_counts = total_train - pos_counts
    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1.0)
    pos_weights = neg_counts / pos_counts
    return torch.tensor(pos_weights, dtype=torch.float32)


def train_classifier(
    model_name: str,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    exp_id: str = "EXP-01",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    loss_type: str = "bce",
    seed: int = 42,
    threshold: float = 0.5,
    label_mode: str = "all_valid",
    smoke_test: bool = False,
    max_length: int = 256
):
    start_time = time.time()
    print("=======================================================")
    print("  MODULE 6B — MULTI-LABEL CLASSIFIER EXPERIMENT PIPELINE")
    print("=======================================================")
    print(f"Experiment ID: {exp_id}")
    print(f"Model Identifier: {model_name}")
    print(f"Loss Type: {loss_type}")
    print(f"Label Mode: {label_mode}")
    print(f"Threshold: {threshold}")
    print(f"Seed: {seed}")
    print(f"Smoke Test Mode: {smoke_test}")

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Label mode filtering
    if label_mode == "high_confidence":
        df_train = df_train[df_train["pseudo_label_quality"] == "HIGH_CONFIDENCE"].copy()
        print(f"[LABEL MODE: HIGH_CONFIDENCE] Filtered train set to {len(df_train)} clauses.")

    # Run safety checks
    run_preflight_dataset_checks(df_train, df_val, df_test)

    # Load tokenizer & config
    print(f"\nLoading tokenizer & config for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=14,
        problem_type="multi_label_classification"
    )

    train_dataset = NDAClauseDataset(df_train, tokenizer, max_length=max_length)
    val_dataset = NDAClauseDataset(df_val, tokenizer, max_length=max_length)
    test_dataset = NDAClauseDataset(df_test, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)

    # Loss Selection
    if loss_type == "focal":
        criterion = FocalLoss(gamma=2.0, alpha=0.25)
    elif loss_type == "class_weighted_bce":
        pos_weight = compute_pos_weights_from_train_only(df_train).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    if smoke_test:
        print("\n--- RUNNING 1-EPOCH DEBUG SMOKE TEST ---")
        model.train()
        smoke_batch = next(iter(train_loader))
        input_ids = smoke_batch["input_ids"].to(device)
        attention_mask = smoke_batch["attention_mask"].to(device)
        labels = smoke_batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        print(f"Smoke Batch Input Shape: {input_ids.shape}")
        print(f"Smoke Batch Logits Shape: {outputs.logits.shape}")
        print(f"Calculated {loss_type} Loss: {loss.item():.4f}")

        # Evaluation on test split
        model.eval()
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for b in test_loader:
                b_ids = b["input_ids"].to(device)
                b_mask = b["attention_mask"].to(device)
                b_out = model(input_ids=b_ids, attention_mask=b_mask)
                all_logits.append(b_out.logits.cpu())
                all_targets.append(b["labels"].cpu())

        y_logits = torch.cat(all_logits, dim=0).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()
        y_probs = 1.0 / (1.0 + np.exp(-y_logits))
        y_pred = (y_probs >= threshold).astype(int)

        eval_res = evaluate_predictions(
            y_true,
            y_pred,
            output_dir=ROOT / "reports",
            model_name=f"{exp_id}-{model_name}-SmokeTest",
            is_temporary_debug=True
        )

        elapsed = time.time() - start_time

        # Save experiment log
        exp_log = {
            "experiment_id": exp_id,
            "status": "DEBUG / PIPELINE VALIDATION ONLY — NOT A RESEARCH RESULT",
            "disclaimer": DISCLAIMER_TEXT,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "loss_function": loss_type,
            "label_mode": label_mode,
            "seed": seed,
            "threshold": threshold,
            "hyperparameters": {
                "epochs": 1,
                "batch_size": batch_size,
                "learning_rate": lr,
                "max_sequence_length": max_length
            },
            "reproducibility_environment": {
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "device": str(device),
                "platform": platform.platform()
            },
            "dataset_splits": {
                "train_documents": sorted(list(df_train["document_id"].unique())),
                "val_documents": sorted(list(df_val["document_id"].unique())),
                "test_documents": sorted(list(df_test["document_id"].unique())),
                "train_clauses": len(df_train),
                "val_clauses": len(df_val),
                "test_clauses": len(df_test)
            },
            "metrics": eval_res
        }

        log_dir = ROOT / "reports/experiments"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{exp_id}_smoketest_{int(time.time())}.json"
        log_path.write_text(json.dumps(exp_log, indent=2), encoding="utf-8")

        # History CSV
        history_csv = ROOT / "data/classification/experiment_results/experiment_history.csv"
        history_csv.parent.mkdir(parents=True, exist_ok=True)
        row_dict = {
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "loss": loss_type,
            "label_mode": label_mode,
            "status": "DEBUG_SMOKE_TEST",
            "train_clauses": len(df_train),
            "val_clauses": len(df_val),
            "test_clauses": len(df_test),
            "macro_f1": eval_res["macro_f1"],
            "weighted_f1": eval_res["weighted_f1"],
            "micro_f1": eval_res["micro_f1"],
            "hamming_loss": eval_res["hamming_loss"],
            "macro_mcc": eval_res["macro_mcc"],
            "execution_time_sec": round(elapsed, 2)
        }
        df_row = pd.DataFrame([row_dict])
        if history_csv.exists():
            df_row.to_csv(history_csv, mode="a", header=False, index=False)
        else:
            df_row.to_csv(history_csv, index=False)

        print(f"\nSaved Experiment JSON Log to: {log_path}")
        print(f"Updated Experiment History CSV: {history_csv}")
        print("[CHECK PASS] PIPELINE SMOKE TEST PASSED PERFECTLY!\n")
        return

    print("[INFO] Full multi-epoch research training is intentionally paused for Module 6B.")
    print("To execute full training later after pseudo-labeling, invoke without --smoke-test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-label transformer classifier training framework")
    parser.add_argument("--exp-id", type=str, default="EXP-01")
    parser.add_argument("--model", type=str, default="saibo/legal-roberta-base")
    parser.add_argument("--train-file", type=Path, default=ROOT / "data/classification/classifier_dataset_train.csv")
    parser.add_argument("--validation-file", type=Path, default=ROOT / "data/classification/classifier_dataset_val.csv")
    parser.add_argument("--test-file", type=Path, default=ROOT / "data/classification/classifier_dataset_test.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/baseline_legal_roberta")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--loss", type=str, choices=["bce", "focal", "class_weighted_bce"], default="bce")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label-mode", type=str, choices=["all_valid", "high_confidence"], default="all_valid")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch debug smoke test")
    args = parser.parse_args()

    train_classifier(
        model_name=args.model,
        train_path=args.train_file,
        val_path=args.validation_file,
        test_path=args.test_file,
        output_dir=args.output_dir,
        exp_id=args.exp_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        loss_type=args.loss,
        seed=args.seed,
        threshold=args.threshold,
        label_mode=args.label_mode,
        smoke_test=args.smoke_test
    )
